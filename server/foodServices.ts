import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';

export {
  buildDeterministicAnswer,
  deriveConversationTitle,
  detectQueryIntent,
  triagePantryItems,
} from './freshoBuddyLogic.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_DB_PATH = path.resolve(__dirname, '..', 'backend', 'app', 'database', 'fresho_buddy.db');

export interface PublicUser {
  id: string;
  email: string;
  fullName: string;
  householdType: string | null;
  avatarUrl: string | null;
  status: string;
  temperatureUnit: string;
  remindersEnabled: boolean;
  eatFirstRoadmapEnabled: boolean;
  freshoBuddyEnabled: boolean;
}

export interface ChatMessageRecord {
  messageId: string;
  conversationId: string;
  role: 'user' | 'assistant';
  content: string;
  analysisContext: Record<string, any> | null;
  createdAt: string;
}

export interface ConversationRecord {
  conversationId: string;
  userId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageRecord[];
}

export interface FoodKeeperEntry {
  canonicalName: string;
  category: string;
  pantryMinDays: number | null;
  pantryMaxDays: number | null;
  fridgeMinDays: number | null;
  fridgeMaxDays: number | null;
  tips: string;
}

// ============================================================================
// PERSISTENT SHARED SQLITE DATABASE (SAME SCHEMA AS backend/app/database/chat_db.py)
// ============================================================================
let cachedDb: DatabaseSync | null = null;
let cachedDbPath = '';

export function resolveDbPath(): string {
  const override = (process.env.FOODFRESH_DB_PATH || '').trim();
  return override || DEFAULT_DB_PATH;
}

export function resetDbConnectionForTests(): void {
  if (cachedDb) {
    try {
      cachedDb.close();
    } catch {
      // ignore
    }
    cachedDb = null;
    cachedDbPath = '';
  }
}

function columnExists(db: DatabaseSync, table: string, column: string): boolean {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  return rows.some((r) => r.name === column);
}

function initSchema(db: DatabaseSync): void {
  db.exec(`
    PRAGMA foreign_keys = ON;
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS conversations (
      conversation_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      title TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations(user_id, updated_at DESC);

    CREATE TABLE IF NOT EXISTS messages (
      message_id TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      analysis_context TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conv
    ON messages(conversation_id, created_at ASC);

    CREATE TABLE IF NOT EXISTS pantry_history (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      food_name TEXT NOT NULL,
      cultivar TEXT,
      status TEXT NOT NULL,
      status_category TEXT,
      quality_score INTEGER DEFAULT 80,
      quality_period TEXT,
      remaining_days INTEGER,
      storage_environment TEXT,
      storage_type TEXT,
      eat_first_priority TEXT,
      eat_first_score INTEGER,
      eat_first_reason TEXT,
      guidance TEXT,
      image_src TEXT,
      raw_metadata TEXT,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_pantry_user
    ON pantry_history(user_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS pantry_meta (
      user_id TEXT PRIMARY KEY,
      has_seeded INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS users (
      user_id TEXT PRIMARY KEY,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      full_name TEXT NOT NULL,
      household_type TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS auth_sessions (
      token TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
    ON auth_sessions(user_id, expires_at DESC);

    CREATE TABLE IF NOT EXISTS ml_feedback (
      feedback_id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      analysis_id TEXT,
      image_path TEXT,
      predicted_food TEXT,
      predicted_food_confidence REAL,
      predicted_food_form TEXT,
      predicted_freshness TEXT,
      predicted_freshness_confidence REAL,
      predicted_shelf_life TEXT,
      user_corrected_food TEXT,
      user_corrected_food_form TEXT,
      user_corrected_freshness TEXT,
      user_corrected_shelf_life TEXT,
      storage_type TEXT,
      days_stored INTEGER,
      feedback_source TEXT DEFAULT 'user',
      review_status TEXT DEFAULT 'COLLECTED',
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ml_feedback_user ON ml_feedback(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_feedback_status ON ml_feedback(review_status);
  `);

  const pantryColumns: Array<[string, string]> = [
    ['analysis_id', 'TEXT'],
    ['added_at', 'TEXT'],
    ['estimated_quality_days', 'INTEGER'],
    ['estimated_end_at', 'TEXT'],
    ['freshness_state', 'TEXT'],
    ['freshness_confidence', 'REAL'],
    ['updated_at', 'TEXT'],
    ['days_stored_at_analysis', 'INTEGER'],
    ['reference_min_days', 'INTEGER'],
    ['reference_max_days', 'INTEGER'],
    ['food_form', 'TEXT'],
    ['legacy_timing', 'INTEGER DEFAULT 0'],
    ['food_model_version', 'TEXT'],
    ['freshness_model_version', 'TEXT'],
    ['shelf_life_model_version', 'TEXT'],
  ];
  for (const [col, colType] of pantryColumns) {
    if (!columnExists(db, 'pantry_history', col)) {
      db.exec(`ALTER TABLE pantry_history ADD COLUMN ${col} ${colType}`);
    }
  }
}

export function getDb(): DatabaseSync {
  const targetPath = resolveDbPath();
  if (cachedDb && cachedDbPath === targetPath) {
    return cachedDb;
  }
  if (cachedDb) {
    try {
      cachedDb.close();
    } catch {
      // ignore
    }
  }
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  const db = new DatabaseSync(targetPath);
  initSchema(db);
  cachedDb = db;
  cachedDbPath = targetPath;
  return db;
}

// ============================================================================
// AUTHENTICATION & SESSIONS (COMPATIBLE WITH PYTHON PBKDF2-SHA256)
// ============================================================================
const PBKDF2_ITERATIONS = 120_000;

function hashPassword(password: string, salt: string): string {
  return crypto.pbkdf2Sync(password, salt, PBKDF2_ITERATIONS, 32, 'sha256').toString('hex');
}

function toPublicUser(u: {
  user_id: string;
  email: string;
  full_name: string;
  household_type: string | null;
}): PublicUser {
  return {
    id: u.user_id,
    email: u.email,
    fullName: u.full_name,
    householdType: u.household_type ?? null,
    avatarUrl: null,
    status: 'Pantry Guard Active',
    temperatureUnit: 'C',
    remindersEnabled: true,
    eatFirstRoadmapEnabled: true,
    freshoBuddyEnabled: true,
  };
}

export function registerUser(
  email: string,
  password: string,
  fullName: string,
  householdType: string | null = null,
): { user: PublicUser; token: string } {
  const cleanEmail = (email || '').trim().toLowerCase();
  if (!cleanEmail || !cleanEmail.includes('@')) {
    throw new Error('Please enter a valid email address.');
  }
  if (!password || password.length < 6) {
    throw new Error('Password must be at least 6 characters.');
  }
  if (!fullName || !fullName.trim()) {
    throw new Error('Please enter your full name.');
  }

  const db = getDb();
  const existing = db.prepare('SELECT user_id FROM users WHERE email = ?').get(cleanEmail);
  if (existing) {
    throw new Error('An account with this email already exists.');
  }

  const userId = `usr_${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`;
  const salt = crypto.randomBytes(16).toString('hex');
  const pwdHash = hashPassword(password, salt);
  const nowIso = new Date().toISOString();

  db.prepare(
    `INSERT INTO users (user_id, email, password_hash, full_name, household_type, created_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(userId, cleanEmail, `${salt}$${pwdHash}`, fullName.trim(), householdType ?? null, nowIso);

  const token = createSession(userId);
  return {
    user: toPublicUser({
      user_id: userId,
      email: cleanEmail,
      full_name: fullName.trim(),
      household_type: householdType ?? null,
    }),
    token,
  };
}

export function loginUser(email: string, password: string): { user: PublicUser; token: string } {
  const cleanEmail = (email || '').trim().toLowerCase();
  if (!cleanEmail || !password) {
    throw new Error('Please enter both email and password.');
  }

  const db = getDb();
  const stored = db
    .prepare('SELECT user_id, email, password_hash, full_name, household_type FROM users WHERE email = ?')
    .get(cleanEmail) as
    | { user_id: string; email: string; password_hash: string; full_name: string; household_type: string | null }
    | undefined;

  if (!stored || !stored.password_hash.includes('$')) {
    throw new Error('Invalid email or password.');
  }

  const [salt, expectedHex] = stored.password_hash.split('$', 2);
  const actualHex = hashPassword(password, salt);
  const a = Buffer.from(actualHex, 'hex');
  const b = Buffer.from(expectedHex, 'hex');
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    throw new Error('Invalid email or password.');
  }

  const token = createSession(stored.user_id);
  return { user: toPublicUser(stored), token };
}

export function createSession(userId: string, ttlDays = 30): string {
  const db = getDb();
  const token = `sess_${crypto.randomUUID().replace(/-/g, '')}`;
  const nowIso = new Date().toISOString();
  const expiresAt = new Date(Date.now() + ttlDays * 86_400_000).toISOString();
  db.prepare(
    `INSERT INTO auth_sessions (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)`,
  ).run(token, userId, expiresAt, nowIso);
  return token;
}

export function resolveSession(token: string | null | undefined): PublicUser | null {
  if (!token || !token.trim()) return null;
  const db = getDb();
  const session = db
    .prepare('SELECT token, user_id, expires_at FROM auth_sessions WHERE token = ?')
    .get(token.trim()) as { token: string; user_id: string; expires_at: string } | undefined;
  if (!session) return null;
  if (Date.parse(session.expires_at) < Date.now()) {
    db.prepare('DELETE FROM auth_sessions WHERE token = ?').run(token.trim());
    return null;
  }
  const user = db
    .prepare('SELECT user_id, email, full_name, household_type FROM users WHERE user_id = ?')
    .get(session.user_id) as
    | { user_id: string; email: string; full_name: string; household_type: string | null }
    | undefined;
  return user ? toPublicUser(user) : null;
}

export function logoutUser(token: string | null | undefined): boolean {
  if (!token) return false;
  const db = getDb();
  const res = db.prepare('DELETE FROM auth_sessions WHERE token = ?').run(token.trim());
  return Number(res.changes || 0) > 0;
}

// ============================================================================
// USDA FOODKEEPER CANONICAL REFERENCE TABLE
// ============================================================================
const FOODKEEPER_CATALOG: Record<string, FoodKeeperEntry> = {
  apple: {
    canonicalName: 'Apples',
    category: 'Fruit',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 21,
    fridgeMaxDays: 28,
    tips: 'Store in the refrigerator crisper drawer away from ethylene-sensitive greens.',
  },
  banana: {
    canonicalName: 'Bananas',
    category: 'Fruit',
    pantryMinDays: 2,
    pantryMaxDays: 5,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Keep at room temperature until ripe, then refrigerate to slow further ripening.',
  },
  tomato: {
    canonicalName: 'Tomatoes',
    category: 'Vegetables',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 7,
    fridgeMaxDays: 10,
    tips: 'Store stem-side down on the counter away from direct sunlight; chill only once fully ripe.',
  },
  'cherry tomato': {
    canonicalName: 'Cherry tomatoes',
    category: 'Vegetables',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 7,
    fridgeMaxDays: 10,
    tips: 'Keep unwashed in a breathable container until ready to eat.',
  },
  orange: {
    canonicalName: 'Citrus fruit',
    category: 'Fruit',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Refrigerate in a mesh bag or crisper drawer for maximum moisture retention.',
  },
  lemon: {
    canonicalName: 'Citrus fruit',
    category: 'Fruit',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Store whole lemons in the crisper drawer to prevent rind drying.',
  },
  lime: {
    canonicalName: 'Citrus fruit',
    category: 'Fruit',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Keep chilled in the refrigerator crisper for longest juiciness.',
  },
  grape: {
    canonicalName: 'Grapes',
    category: 'Fruit',
    pantryMinDays: 2,
    pantryMaxDays: 3,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Keep unwashed in a ventilated bag in the coldest part of the refrigerator.',
  },
  strawberry: {
    canonicalName: 'Strawberries',
    category: 'Fruit',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 3,
    fridgeMaxDays: 5,
    tips: 'Do not wash until ready to eat; store in a paper-towel-lined ventilated container.',
  },
  blueberry: {
    canonicalName: 'Blueberries',
    category: 'Fruit',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 7,
    fridgeMaxDays: 10,
    tips: 'Refrigerate dry and unwashed in original ventilated clamshell.',
  },
  pomegranate: {
    canonicalName: 'Pomegranate',
    category: 'Fruit',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 14,
    fridgeMaxDays: 30,
    tips: 'Whole pomegranates keep well in cool dark areas or much longer in the refrigerator.',
  },
  mango: {
    canonicalName: 'Papaya, mango, feijoa, passionfruit',
    category: 'Fruit',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Ripen on the countertop, then move to the refrigerator once soft to preserve peak sweetness.',
  },
  papaya: {
    canonicalName: 'Papaya, mango, feijoa, passionfruit',
    category: 'Fruit',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Ripen at room temperature; refrigerate whole ripe papaya for up to a week.',
  },
  chickoo: {
    canonicalName: 'Papaya, mango, feijoa, passionfruit',
    category: 'Fruit',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 7,
    fridgeMaxDays: 10,
    tips: 'Allow firm chickoo to soften at ambient temperature before chilling.',
  },
  avocado: {
    canonicalName: 'Avocados',
    category: 'Fruit',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Ripen on the counter; refrigerate once ripe to hold peak texture for several extra days.',
  },
  pineapple: {
    canonicalName: 'Pineapple',
    category: 'Fruit',
    pantryMinDays: 2,
    pantryMaxDays: 3,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Store whole pineapple at room temperature 1–2 days, then chill to prevent over-fermentation.',
  },
  watermelon: {
    canonicalName: 'Watermelon',
    category: 'Fruit',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 10,
    fridgeMaxDays: 14,
    tips: 'Keep whole watermelon cool; refrigerate promptly once cut.',
  },
  pear: {
    canonicalName: 'Peaches, nectarines, plums, pears',
    category: 'Fruit',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Ripen on countertop and chill once fruit yields gently near the stem.',
  },
  peach: {
    canonicalName: 'Peaches, nectarines, plums, pears',
    category: 'Fruit',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Keep in a single layer on the counter to ripen without bruising, then chill.',
  },
  plum: {
    canonicalName: 'Peaches, nectarines, plums, pears',
    category: 'Fruit',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Refrigerate ripe plums in the crisper drawer.',
  },
  guava: {
    canonicalName: 'Guava',
    category: 'Fruit',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Store ripe guavas chilled in a breathable bag.',
  },
  cherry: {
    canonicalName: 'Cherries',
    category: 'Fruit',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Refrigerate unwashed with stems attached in a breathable container.',
  },
  'bell pepper': {
    canonicalName: 'Peppers',
    category: 'Vegetables',
    pantryMinDays: 2,
    pantryMaxDays: 3,
    fridgeMinDays: 7,
    fridgeMaxDays: 14,
    tips: 'Keep dry in the crisper drawer; moisture accelerates softening.',
  },
  'green chilli': {
    canonicalName: 'Hot peppers',
    category: 'Vegetables',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 7,
    fridgeMaxDays: 14,
    tips: 'Wrap dry chillies in a paper towel inside a breathable bag in the crisper.',
  },
  potato: {
    canonicalName: 'Potatoes',
    category: 'Vegetables',
    pantryMinDays: 7,
    pantryMaxDays: 14,
    fridgeMinDays: null,
    fridgeMaxDays: null,
    tips: 'Store in a cool, dark, ventilated pantry away from onions; refrigeration converts starch to sugar.',
  },
  'sweet potato': {
    canonicalName: 'Yams/sweet potatoes',
    category: 'Vegetables',
    pantryMinDays: 7,
    pantryMaxDays: 14,
    fridgeMinDays: null,
    fridgeMaxDays: null,
    tips: 'Keep in a cool dark cupboard rather than the refrigerator.',
  },
  onion: {
    canonicalName: 'Onions',
    category: 'Vegetables',
    pantryMinDays: 7,
    pantryMaxDays: 14,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Store whole dry onions in a cool, dry, well-ventilated basket away from potatoes.',
  },
  garlic: {
    canonicalName: 'Garlic',
    category: 'Vegetables',
    pantryMinDays: 7,
    pantryMaxDays: 21,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Store whole bulbs in a dry, dark, ventilated space at room temperature.',
  },
  carrot: {
    canonicalName: 'Carrots, parsnips',
    category: 'Vegetables',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Remove leafy tops and store in the crisper drawer with slight moisture.',
  },
  cucumber: {
    canonicalName: 'Cucumbers',
    category: 'Vegetables',
    pantryMinDays: 2,
    pantryMaxDays: 3,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Wrap loosely and store away from high-ethylene fruits like apples and tomatoes.',
  },
  eggplant: {
    canonicalName: 'Eggplant',
    category: 'Vegetables',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Store in a cool spot or crisper drawer; use promptly before skin wrinkles.',
  },
  broccoli: {
    canonicalName: 'Broccoli',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Keep unwashed in a loosely perforated bag in the refrigerator.',
  },
  cauliflower: {
    canonicalName: 'Cauliflower',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Store stem-side down in the crisper drawer to prevent moisture pooling.',
  },
  cabbage: {
    canonicalName: 'Cabbage',
    category: 'Vegetables',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 10,
    fridgeMaxDays: 15,
    tips: 'Keep whole head tightly wrapped in the refrigerator crisper.',
  },
  spinach: {
    canonicalName: 'Greens',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 1,
    fridgeMinDays: 3,
    fridgeMaxDays: 5,
    tips: 'Store dry with a paper towel in the crisper to absorb excess condensation.',
  },
  lettuce: {
    canonicalName: 'Lettuce',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Keep chilled and dry in the high-humidity crisper drawer.',
  },
  beetroot: {
    canonicalName: 'Beets',
    category: 'Vegetables',
    pantryMinDays: 3,
    pantryMaxDays: 5,
    fridgeMinDays: 10,
    fridgeMaxDays: 14,
    tips: 'Trim greens leaving 1 inch of stem and refrigerate unwashed roots.',
  },
  corn: {
    canonicalName: 'Corn on the cob',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 3,
    fridgeMaxDays: 5,
    tips: 'Refrigerate unhusked for sweetest flavor.',
  },
  radish: {
    canonicalName: 'Radishes',
    category: 'Vegetables',
    pantryMinDays: 2,
    pantryMaxDays: 3,
    fridgeMinDays: 7,
    fridgeMaxDays: 14,
    tips: 'Remove tops and refrigerate in a sealed breathable bag.',
  },
  ginger: {
    canonicalName: 'Ginger root',
    category: 'Vegetables',
    pantryMinDays: 5,
    pantryMaxDays: 7,
    fridgeMinDays: 14,
    fridgeMaxDays: 21,
    tips: 'Keep unpeeled ginger dry in a paper bag in the crisper drawer.',
  },
  'green beans': {
    canonicalName: 'Beans and peas',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 2,
    fridgeMinDays: 4,
    fridgeMaxDays: 7,
    tips: 'Store unwashed in a breathable bag in the refrigerator.',
  },
  mushroom: {
    canonicalName: 'Mushrooms',
    category: 'Vegetables',
    pantryMinDays: 1,
    pantryMaxDays: 1,
    fridgeMinDays: 4,
    fridgeMaxDays: 7,
    tips: 'Keep in a paper bag in the refrigerator to avoid sliminess.',
  },
  bread: {
    canonicalName: 'Commercial bread products',
    category: 'Bakery',
    pantryMinDays: 4,
    pantryMaxDays: 7,
    fridgeMinDays: 7,
    fridgeMaxDays: 14,
    tips: 'Keep tightly sealed at cool room temperature, or freeze slices for longer storage.',
  },
  cake: {
    canonicalName: 'Commercial cakes and muffins',
    category: 'Bakery',
    pantryMinDays: 2,
    pantryMaxDays: 4,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Cover tightly to prevent staling; refrigerate cream-filled varieties.',
  },
  milk: {
    canonicalName: 'Milk',
    category: 'Dairy',
    pantryMinDays: null,
    pantryMaxDays: null,
    fridgeMinDays: 5,
    fridgeMaxDays: 7,
    tips: 'Store in the main body of the refrigerator rather than the warmer door.',
  },
  cheese: {
    canonicalName: 'Cheese',
    category: 'Dairy',
    pantryMinDays: null,
    pantryMaxDays: null,
    fridgeMinDays: 7,
    fridgeMaxDays: 15,
    tips: 'Wrap in wax or parchment paper inside a loose bag in the cheese drawer.',
  },
  yogurt: {
    canonicalName: 'Yogurt',
    category: 'Dairy',
    pantryMinDays: null,
    pantryMaxDays: null,
    fridgeMinDays: 7,
    fridgeMaxDays: 14,
    tips: 'Keep sealed and refrigerated at 4°C or below.',
  },
};

const ALIAS_MAP: Record<string, string> = {
  apples: 'apple',
  bananas: 'banana',
  tomatoes: 'tomato',
  'cherry tomatoes': 'cherry tomato',
  oranges: 'orange',
  citrus: 'orange',
  grapefruit: 'orange',
  lemons: 'lemon',
  limes: 'lime',
  grapes: 'grape',
  strawberries: 'strawberry',
  blueberries: 'blueberry',
  pomegranates: 'pomegranate',
  mangoes: 'mango',
  mangos: 'mango',
  papayas: 'papaya',
  chiku: 'chickoo',
  sapota: 'chickoo',
  sapodilla: 'chickoo',
  avocados: 'avocado',
  pineapples: 'pineapple',
  watermelons: 'watermelon',
  pears: 'pear',
  peaches: 'peach',
  plums: 'plum',
  guavas: 'guava',
  cherries: 'cherry',
  peppers: 'bell pepper',
  pepper: 'bell pepper',
  capsicum: 'bell pepper',
  'sweet pepper': 'bell pepper',
  'red pepper': 'bell pepper',
  'yellow pepper': 'bell pepper',
  'green chili': 'green chilli',
  chilli: 'green chilli',
  chili: 'green chilli',
  'chili pepper': 'green chilli',
  'hot pepper': 'green chilli',
  jalapeno: 'green chilli',
  potatoes: 'potato',
  'sweet potatoes': 'sweet potato',
  yam: 'sweet potato',
  onions: 'onion',
  carrots: 'carrot',
  cucumbers: 'cucumber',
  aubergine: 'eggplant',
  brinjal: 'eggplant',
  greens: 'spinach',
  beets: 'beetroot',
  beet: 'beetroot',
  'sweet corn': 'corn',
  radishes: 'radish',
  'green bean': 'green beans',
  mushrooms: 'mushroom',
  loaf: 'bread',
  'white bread': 'bread',
  'whole wheat bread': 'bread',
  sourdough: 'bread',
  toast: 'bread',
  bun: 'bread',
  buns: 'bread',
  croissant: 'bread',
  bagel: 'bread',
  cakes: 'cake',
  muffin: 'cake',
};

function normalizeFoodKey(name: string): string {
  return (name || '')
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findFoodKeeperEntry(foodName: string): FoodKeeperEntry | null {
  const norm = normalizeFoodKey(foodName);
  if (!norm) return null;
  if (FOODKEEPER_CATALOG[norm]) return FOODKEEPER_CATALOG[norm];
  if (ALIAS_MAP[norm] && FOODKEEPER_CATALOG[ALIAS_MAP[norm]]) {
    return FOODKEEPER_CATALOG[ALIAS_MAP[norm]];
  }
  for (const [alias, target] of Object.entries(ALIAS_MAP)) {
    if (norm.includes(alias) && FOODKEEPER_CATALOG[target]) {
      return FOODKEEPER_CATALOG[target];
    }
  }
  for (const key of Object.keys(FOODKEEPER_CATALOG)) {
    if (norm.includes(key)) {
      return FOODKEEPER_CATALOG[key];
    }
  }
  return null;
}

export function inferFoodForm(foodName: string | null | undefined, freshnessLabel?: string | null): string {
  const name = (foodName || '').toLowerCase();
  const blob = `${name} ${(freshnessLabel || '').toLowerCase()}`;
  if (/dried|dehydrated|sun-dried|sun dried/.test(name)) return 'dried';
  if (/powder|ground spice|chili powder|chilli powder/.test(name)) return 'powdered';
  if (/cooked|roasted|fried|grilled/.test(name)) return 'cooked';
  if (/canned|pickled|preserved|jarred/.test(name)) return 'processed';
  if (/\bdiced\b/.test(name)) return 'diced';
  if (/\bsliced\b/.test(name)) return 'sliced';
  if (/\b(cut|chopped|halved)\b|-cut/.test(name)) return 'cut';
  if (/\b(fresh|raw|green)\b/.test(blob) && !name.includes('dried')) return 'fresh';
  if (name.includes('bell pepper') || name.includes('capsicum')) return 'fresh';
  if (name.includes('chilli') || name.includes('chili') || name.includes('pepper')) {
    if (name.includes('dried') || (name.includes('red chilli') && !name.includes('green'))) return 'dried';
    if (name.includes('green')) return 'fresh';
    return 'unknown';
  }
  return name ? 'fresh' : 'unknown';
}

const BAKERY_FOOD_TERMS = new Set([
  'bread',
  'bun',
  'buns',
  'toast',
  'roti',
  'cake',
  'cakes',
  'muffin',
  'muffins',
  'croissant',
  'bagel',
  'pastry',
  'cookie',
  'cookies',
  'loaf',
  'sourdough',
  'white bread',
  'whole wheat bread',
]);

const PRODUCE_FRESHNESS_TERMS = new Set([
  'apple',
  'banana',
  'tomato',
  'cherry tomato',
  'orange',
  'lemon',
  'lime',
  'grape',
  'strawberry',
  'blueberry',
  'pomegranate',
  'mango',
  'papaya',
  'chickoo',
  'sapodilla',
  'chiku',
  'sapota',
  'avocado',
  'pineapple',
  'watermelon',
  'pear',
  'peach',
  'plum',
  'guava',
  'cherry',
  'bell pepper',
  'green chilli',
  'chili pepper',
  'potato',
  'sweet potato',
  'onion',
  'garlic',
  'carrot',
  'cucumber',
  'eggplant',
  'broccoli',
  'cauliflower',
  'cabbage',
  'spinach',
  'lettuce',
  'beetroot',
  'corn',
  'radish',
  'ginger',
  'green beans',
  'mushroom',
  'kiwi',
  'cantaloupe',
  'melon',
  'raspberry',
  'blackberry',
  'fig',
  'coconut',
  'dragon fruit',
  'passion fruit',
  'jackfruit',
  'lychee',
  'custard apple',
  'zucchini',
  'peas',
  'celery',
  'asparagus',
  'bitter gourd',
  'bottle gourd',
]);

export function isFreshnessSupportedFood(
  foodName: string | null | undefined,
  foodForm: string | null | undefined = 'fresh',
): boolean {
  const form = String(foodForm || 'unknown').toLowerCase().trim();
  if (['dried', 'powdered', 'cooked', 'processed'].includes(form)) {
    return false;
  }
  const norm = normalizeFoodKey(foodName || '');
  if (!norm) return false;
  const words = norm.split(' ');
  if (BAKERY_FOOD_TERMS.has(norm) || words.some((w) => BAKERY_FOOD_TERMS.has(w))) {
    return false;
  }
  const entry = findFoodKeeperEntry(norm);
  if (entry && (entry.category === 'Bakery' || entry.category === 'Dairy')) {
    return false;
  }
  if (entry && (entry.category === 'Fruit' || entry.category === 'Vegetables')) {
    return true;
  }
  if (PRODUCE_FRESHNESS_TERMS.has(norm)) return true;
  if (norm.endsWith('es') && PRODUCE_FRESHNESS_TERMS.has(norm.slice(0, -2))) return true;
  if (norm.endsWith('s') && PRODUCE_FRESHNESS_TERMS.has(norm.slice(0, -1))) return true;
  for (const term of PRODUCE_FRESHNESS_TERMS) {
    if (norm.includes(term)) return true;
  }
  return false;
}

export function evaluateFreshnessPolicy(params: {
  detectedFood: string | null | undefined;
  foodForm?: string | null;
  rawFreshnessLabel?: unknown;
  rawFreshnessScore?: unknown;
  isLowConfidence?: boolean;
  source?: string;
}): {
  status: 'success' | 'unavailable';
  label: string;
  confidence: number | null;
  score: number | null;
  freshnessUncertain: boolean;
  source: string;
  modelVersion: string;
} {
  const source = params.source || 'meta-llama/llama-4-scout-17b-16e-instruct';
  if (
    params.isLowConfidence ||
    !params.detectedFood ||
    !isFreshnessSupportedFood(params.detectedFood, params.foodForm)
  ) {
    return {
      status: 'unavailable',
      label: 'Freshness assessment unavailable.',
      confidence: null,
      score: null,
      freshnessUncertain: true,
      source,
      modelVersion: source,
    };
  }

  const rawLabel = typeof params.rawFreshnessLabel === 'string' ? params.rawFreshnessLabel.trim() : '';
  if (!rawLabel || /unavailable|unknown|n\/a/i.test(rawLabel)) {
    return {
      status: 'unavailable',
      label: 'Freshness assessment unavailable.',
      confidence: null,
      score: null,
      freshnessUncertain: true,
      source,
      modelVersion: source,
    };
  }

  let mappedLabel: 'Fresh' | 'Slightly Spoiled' | 'Rotten' | null = null;
  if (/rotten|decay|moldy/i.test(rawLabel)) {
    mappedLabel = 'Rotten';
  } else if (/slightly|spoiled|bruised|wilted/i.test(rawLabel)) {
    mappedLabel = 'Slightly Spoiled';
  } else if (/fresh/i.test(rawLabel)) {
    mappedLabel = 'Fresh';
  }

  if (!mappedLabel) {
    return {
      status: 'unavailable',
      label: 'Freshness assessment unavailable.',
      confidence: null,
      score: null,
      freshnessUncertain: true,
      source,
      modelVersion: source,
    };
  }

  const numericScore =
    typeof params.rawFreshnessScore === 'number' && Number.isFinite(params.rawFreshnessScore)
      ? Math.min(100, Math.max(0, Number(params.rawFreshnessScore)))
      : null;

  return {
    status: 'success',
    label: mappedLabel,
    confidence: numericScore,
    score: numericScore,
    freshnessUncertain: numericScore !== null ? numericScore < 50 : true,
    source,
    modelVersion: source,
  };
}

export function normalizeFreshnessLabel(label: string | null | undefined): string {
  if (!label) return 'fresh';
  const raw = String(label)
    .toLowerCase()
    .replace(/\s*\([^)]*\)/g, '')
    .trim();
  if (raw.includes('rotten')) return 'rotten';
  if (raw.includes('slightly') || raw.includes('spoiled') || raw.includes('stale') || raw.includes('mold')) {
    return 'slightly_spoiled';
  }
  if (raw.includes('uncertain') || raw.includes('unavailable')) return 'uncertain';
  if (raw.includes('fresh')) return 'fresh';
  return raw;
}

export function estimateShelfLife(
  foodName: string | null | undefined,
  storageType = 'countertop',
  daysStored = 0,
  freshnessLabel?: string | null,
  foodForm = 'unknown',
): Record<string, any> {
  const formNorm = (foodForm || 'unknown').toLowerCase().trim();
  if (['dried', 'powdered', 'processed', 'cooked'].includes(formNorm)) {
    return {
      status: 'unavailable',
      food: foodName,
      foodForm: formNorm,
      canonicalFood: null,
      matchType: 'form_unsupported',
      storageType,
      reason:
        'Remaining quality cannot currently be estimated for this food form. USDA FoodKeeper references in this pipeline target fresh whole produce forms.',
      remaining: null,
      referenceDuration: null,
      source: 'USDA FoodKeeper',
      isEstimate: false,
      isMlRegression: false,
    };
  }

  if (!foodName || ['none', 'null', 'uncertain', ''].includes(foodName.trim().toLowerCase())) {
    return {
      status: 'unavailable',
      food: foodName,
      canonicalFood: null,
      matchType: 'none',
      storageType,
      reason: 'No food detected for shelf-life reference lookup.',
      remaining: null,
      referenceDuration: null,
      source: 'USDA FoodKeeper',
      isEstimate: false,
      isMlRegression: false,
    };
  }

  const entry = findFoodKeeperEntry(foodName);
  const freshnessNorm = normalizeFreshnessLabel(freshnessLabel);
  if (!entry) {
    return {
      status: 'unavailable',
      food: foodName,
      canonicalFood: null,
      matchType: 'unmatched',
      storageType,
      reason: `No matching FoodKeeper reference guideline found for '${foodName}'.`,
      remaining: null,
      referenceDuration: null,
      source: 'USDA FoodKeeper',
      isEstimate: false,
      isMlRegression: false,
    };
  }

  const normStorage = storageType.toLowerCase().trim();
  const isCrisper = ['fridge', 'crisper', 'refrigerate', 'refrigerator'].includes(normStorage);
  const storageLabel = isCrisper ? 'Crisper Chill' : 'Countertop';

  const rawMin = isCrisper ? entry.fridgeMinDays : entry.pantryMinDays;
  const rawMax = isCrisper ? entry.fridgeMaxDays : entry.pantryMaxDays;

  if (rawMin === null && rawMax === null) {
    const altMin = isCrisper ? entry.pantryMinDays : entry.fridgeMinDays;
    const altLabel = isCrisper ? 'Countertop (Ambient)' : 'Crisper Chill (Refrigerated)';
    const explanation =
      altMin !== null
        ? `USDA FoodKeeper does not specify a ${storageLabel} quality window for ${entry.canonicalName}. Recommended storage is ${altLabel} (${entry.tips}).`
        : `No numerical duration guideline in USDA FoodKeeper for ${entry.canonicalName} under ${storageLabel} storage.`;
    return {
      status: 'unavailable',
      food: foodName,
      canonicalFood: entry.canonicalName,
      matchType: 'canonical_match',
      storageType,
      reason: explanation,
      referenceDuration: null,
      remaining: null,
      source: 'USDA FoodKeeper',
      isEstimate: false,
      isMlRegression: false,
    };
  }

  let refMin = Number(rawMin ?? rawMax);
  let refMax = Number(rawMax ?? rawMin);
  if (refMin > refMax) [refMin, refMax] = [refMax, refMin];

  const isProduceCategory = entry.category === 'Fruit' || entry.category === 'Vegetables';
  const isCutProduce = ['cut', 'sliced', 'diced'].includes(formNorm) && isProduceCategory;

  let formNote: string | null = null;
  let sourceLabelForm: string | null = null;
  if (isCutProduce) {
    if (isCrisper) {
      refMin = Math.min(refMin, 3);
      refMax = Math.min(refMax, 5);
      formNote = 'Form-adjusted estimate: cut/sliced refrigerated produce has an accelerated 3–5 day window.';
    } else {
      refMin = Math.min(refMin, 1);
      refMax = Math.min(refMax, 2);
      formNote = 'Form-adjusted estimate: ambient fresh-cut produce must be used rapidly (1–2 day window).';
    }
    sourceLabelForm = 'USDA FoodKeeper (form-adjusted)';
  }

  let prototypeNote: string | null = null;
  if (!isCutProduce) {
    if (!isCrisper && refMax > 7) {
      refMax = 7;
      refMin = Math.min(refMin, 7);
      prototypeNote = 'Constrained by FoodFresh AI countertop prototype policy (up to 7 days ambient window).';
    } else if (isCrisper && refMax > 15) {
      refMax = 15;
      refMin = Math.min(refMin, 15);
      prototypeNote = 'Constrained by FoodFresh AI refrigerated prototype policy (up to 15 days crisper window).';
    }
  }

  const daysPast = Math.max(0, Math.floor(Number(daysStored) || 0));
  const rawRemMin = refMin - daysPast;
  const rawRemMax = refMax - daysPast;

  let remMin = Math.max(0, rawRemMin);
  let remMax = Math.max(0, rawRemMax);
  let adjustmentApplied = false;
  let heuristicNote: string | null = null;

  if (freshnessNorm === 'rotten') {
    remMin = 0;
    remMax = 0;
    adjustmentApplied = true;
    heuristicNote = 'Visible signs of decomposition indicate that the quality window has expired.';
  } else if (freshnessNorm === 'slightly_spoiled') {
    remMin = Math.max(0, rawRemMin * 0.5);
    remMax = Math.max(rawRemMax >= 1 ? 1 : 0, rawRemMax * 0.5);
    adjustmentApplied = true;
    heuristicNote = 'Remaining quality window reduced due to visible surface wear.';
  }

  if (freshnessNorm === 'fresh') {
    if (daysPast === 0) {
      remMin = Math.max(1, remMin);
      remMax = Math.max(1, remMax);
    } else if (daysPast < refMax) {
      remMax = Math.max(1, remMax);
    }
  }

  let intRemMin = Math.round(remMin);
  let intRemMax = Math.round(remMax);
  let zeroDayReason: string | null = null;

  if (intRemMax <= 0 || rawRemMax <= 0) {
    intRemMin = 0;
    intRemMax = 0;
    if (freshnessNorm === 'rotten') {
      zeroDayReason = 'Visible deterioration indicates that this item is past its consumable quality window.';
    } else if (daysPast >= refMax) {
      zeroDayReason = `Stored for ${daysPast} days, exceeding the USDA FoodKeeper reference window of ${Math.round(refMin)}–${Math.round(refMax)} days under ${storageLabel} storage.`;
    } else {
      zeroDayReason = 'Estimated remaining quality has reached 0 days.';
    }
  }

  const notes = [heuristicNote, formNote, prototypeNote].filter(Boolean);
  const sourceLabel =
    sourceLabelForm || (adjustmentApplied ? 'USDA FoodKeeper + freshness observation' : 'USDA FoodKeeper');

  return {
    status: 'available',
    food: foodName,
    canonicalFood: entry.canonicalName,
    category: entry.category,
    matchType: 'canonical_match',
    matchConfidence: 1.0,
    storageType,
    daysStored: daysPast,
    referenceDuration: {
      minDays: Math.round(refMin),
      maxDays: Math.round(refMax),
    },
    remaining: {
      minDays: intRemMin,
      maxDays: intRemMax,
    },
    unit: 'days',
    source: sourceLabel,
    tips: entry.tips,
    isEstimate: true,
    heuristicApplied: adjustmentApplied,
    isMlRegression: false,
    methodology: 'USDA FoodKeeper reference lookup and explicit rules',
    heuristicAdjustment: notes.length ? notes.join(' ') : null,
    zeroDayReason,
  };
}

// ============================================================================
// EAT FIRST DETERMINISTIC PRIORITY ENGINE
// ============================================================================
export function evaluateEatFirstPriority(
  shelfLifeData: Record<string, any> | null | undefined,
  freshnessLabel?: string | null,
): Record<string, any> {
  if (!shelfLifeData || shelfLifeData.status !== 'available') {
    return {
      status: 'unavailable',
      priority: null,
      score: 0,
      tier: 'UNAVAILABLE',
      reason: 'Eat First requires an available shelf-life estimate.',
      urgency: 'Guidance unavailable',
      urgencyLabel: 'Guidance unavailable',
    };
  }

  const remaining = shelfLifeData.remaining || {};
  const remMin = remaining.minDays;
  const remMax = remaining.maxDays;
  if (remMin == null && remMax == null) {
    return {
      status: 'unavailable',
      priority: null,
      score: 0,
      tier: 'UNAVAILABLE',
      reason: 'Shelf-life window bounds are undefined.',
      urgency: 'Guidance unavailable',
      urgencyLabel: 'Guidance unavailable',
    };
  }

  const effectiveMin = Number(remMin ?? remMax);
  const effectiveMax = Number(remMax ?? remMin);
  const effectiveRem = (effectiveMin + effectiveMax) / 2;
  const freshnessNorm = normalizeFreshnessLabel(freshnessLabel);

  if (freshnessNorm === 'rotten' || effectiveMax <= 0 || effectiveRem <= 1 || effectiveMax <= 1) {
    let reason: string;
    let urgency: string;
    if (effectiveMax <= 0) {
      reason = 'Very high priority because the estimated remaining quality is 0 days.';
      urgency = 'Consume immediately or discard';
    } else if (freshnessNorm === 'rotten') {
      reason = 'Very high priority because visible surface degradation indicates the quality window has expired.';
      urgency = 'Do not consume / dispose';
    } else {
      reason = `Very high priority because the estimated remaining quality window is critically short (${Math.round(effectiveMin)}–${Math.round(effectiveMax)} days).`;
      urgency = 'Consume immediately (within 24 hours)';
    }
    return {
      status: 'available',
      priority: 'VERY_HIGH',
      score: 4,
      tier: 'VERY_HIGH',
      reason,
      urgency,
      urgencyLabel: urgency,
      isEstimate: true,
    };
  }

  if (freshnessNorm === 'slightly_spoiled' || effectiveRem <= 3 || effectiveMax <= 3) {
    const reason =
      freshnessNorm === 'slightly_spoiled'
        ? `High priority because the estimated remaining quality window is short (${Math.round(effectiveMin)}–${Math.round(effectiveMax)} days) and visible freshness is declining.`
        : `High priority because the estimated remaining quality window is short (${Math.round(effectiveMin)}–${Math.round(effectiveMax)} days).`;
    return {
      status: 'available',
      priority: 'HIGH',
      score: 3,
      tier: 'HIGH',
      reason,
      urgency: 'Consume within 1–3 days',
      urgencyLabel: 'Consume within 1–3 days',
      isEstimate: true,
    };
  }

  if (effectiveRem <= 7 || effectiveMax <= 7) {
    return {
      status: 'available',
      priority: 'MEDIUM',
      score: 2,
      tier: 'MEDIUM',
      reason: `Estimated remaining quality is moderate (${effectiveMin}–${effectiveMax} days) under proper storage.`,
      urgency: 'Consume within a week',
      urgencyLabel: 'Consume within a week',
      isEstimate: true,
    };
  }

  return {
    status: 'available',
    priority: 'LOW',
    score: 1,
    tier: 'LOW',
    reason: `Good visible freshness with ample estimated shelf-life (${effectiveMin}–${effectiveMax} days).`,
    urgency: 'Ample quality window remaining',
    urgencyLabel: 'Ample quality window remaining',
    isEstimate: true,
  };
}

// ============================================================================
// PANTRY HISTORY & DYNAMIC UTC REMAINING QUALITY
// ============================================================================
function computeEstimatedEndAt(addedAt: Date, estimatedQualityDays: number): Date {
  const y = addedAt.getUTCFullYear();
  const m = addedAt.getUTCMonth();
  const d = addedAt.getUTCDate();
  const offset = estimatedQualityDays <= 0 ? 0 : Math.floor(estimatedQualityDays);
  return new Date(Date.UTC(y, m, d + offset, 23, 59, 59));
}

function computeRemainingDays(
  addedAtIso: string | null | undefined,
  estimatedQualityDays: number | null | undefined,
  estimatedEndAtIso?: string | null,
): number | null {
  const now = new Date();
  const todayUtc = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());

  if (estimatedEndAtIso) {
    const end = new Date(estimatedEndAtIso);
    if (!Number.isNaN(end.getTime())) {
      const endUtc = Date.UTC(end.getUTCFullYear(), end.getUTCMonth(), end.getUTCDate());
      return Math.max(0, Math.round((endUtc - todayUtc) / 86_400_000));
    }
  }
  if (estimatedQualityDays == null || !addedAtIso) return null;
  const added = new Date(addedAtIso);
  if (Number.isNaN(added.getTime())) return null;
  const end = computeEstimatedEndAt(added, estimatedQualityDays);
  const endUtc = Date.UTC(end.getUTCFullYear(), end.getUTCMonth(), end.getUTCDate());
  return Math.max(0, Math.round((endUtc - todayUtc) / 86_400_000));
}

function qualityPeriodLabel(remaining: number | null): string {
  if (remaining == null) return 'Remaining quality cannot currently be estimated.';
  if (remaining === 0) return '0 days remaining';
  if (remaining === 1) return '1 day remaining';
  return `${remaining} days remaining`;
}

function deriveDisplayStatus(remaining: number | null, freshnessState?: string | null): string {
  const fresh = (freshnessState || 'Fresh').trim();
  if (remaining == null) return fresh;
  if (remaining <= 0) return 'ATTENTION / EXPIRED QUALITY WINDOW';
  if (remaining <= 3) return 'HIGH PRIORITY';
  if (remaining <= 7) return 'MEDIUM PRIORITY';
  return 'LOW PRIORITY';
}

function enrichPantryRow(row: Record<string, any>): Record<string, any> {
  const addedAt = row.addedAt || row.analyzedAt;
  const estDays = row.estimatedQualityDays;
  const remaining = computeRemainingDays(addedAt, estDays, row.estimatedEndAt);
  const foodName = row.foodName || 'Produce';
  const storageType = row.storageType || 'countertop';
  const freshnessState = row.freshnessState || row.status || 'Fresh';

  const sl =
    remaining == null
      ? { status: 'unavailable', reason: 'Remaining quality cannot currently be estimated.' }
      : {
          status: 'available',
          food: foodName,
          storageType,
          remaining: { minDays: remaining, maxDays: remaining },
        };

  const eatRes = evaluateEatFirstPriority(sl, freshnessState);

  return {
    ...row,
    remainingDays: remaining,
    qualityPeriod: qualityPeriodLabel(remaining),
    remainingQualityKnown: remaining != null,
    pantryStatus: deriveDisplayStatus(remaining, freshnessState),
    eatFirstPriority: eatRes.priority || 'UNAVAILABLE',
    eatFirstScore: eatRes.score || 0,
    eatFirstReason: eatRes.reason || '',
    eatFirstUrgency: eatRes.urgency || eatRes.urgencyLabel || '',
    priorityTier: eatRes.tier || eatRes.priority,
  };
}

function formatDbPantryRow(row: Record<string, any>): Record<string, any> {
  let rawMeta: Record<string, any> = {};
  if (row.raw_metadata) {
    try {
      rawMeta = JSON.parse(String(row.raw_metadata));
    } catch {
      rawMeta = {};
    }
  }
  return {
    id: row.id,
    userId: row.user_id,
    analysisId: row.analysis_id ?? null,
    foodName: row.food_name,
    cultivar: row.cultivar || `${row.food_name} • Fresh Harvest`,
    status: row.status,
    statusCategory: row.status_category || 'fresh',
    qualityScore: row.quality_score ?? 80,
    qualityPeriod: row.quality_period,
    remainingDays: row.remaining_days,
    storageEnvironment: row.storage_environment || 'Countertop Ambient',
    storageType: row.storage_type || 'countertop',
    eatFirstPriority: row.eat_first_priority || 'NORMAL',
    eatFirstScore: row.eat_first_score ?? 1,
    eatFirstReason: row.eat_first_reason || '',
    guidance: row.guidance || 'Store in optimal pantry conditions.',
    imageSrc: row.image_src || '',
    analyzedTime: row.created_at,
    analyzedAt: row.created_at,
    addedAt: row.added_at || row.created_at,
    estimatedQualityDays: row.estimated_quality_days ?? row.remaining_days ?? null,
    estimatedEndAt: row.estimated_end_at ?? null,
    freshnessState: row.freshness_state || row.status,
    freshnessConfidence: row.freshness_confidence ?? null,
    updatedAt: row.updated_at || row.created_at,
    foodForm: row.food_form || 'unknown',
    metadata: rawMeta,
  };
}

export function getCurrentPantryState(userId: string): Record<string, any>[] {
  const db = getDb();
  const rows = db
    .prepare('SELECT * FROM pantry_history WHERE user_id = ? ORDER BY created_at DESC')
    .all(userId) as Record<string, any>[];
  const enriched = rows.map((r) => enrichPantryRow(formatDbPantryRow(r)));
  enriched.sort((a, b) => {
    const aNull = a.remainingDays == null ? 1 : 0;
    const bNull = b.remainingDays == null ? 1 : 0;
    if (aNull !== bNull) return aNull - bNull;
    const aRem = a.remainingDays ?? 9999;
    const bRem = b.remainingDays ?? 9999;
    if (aRem !== bRem) return aRem - bRem;
    return String(a.foodName || '').localeCompare(String(b.foodName || ''));
  });
  return enriched;
}

export function addPantryItem(userId: string, item: Record<string, any>): Record<string, any> {
  const db = getDb();
  const itemId = item.id || `pantry_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
  const foodName = item.foodName || item.detectedFood || 'Produce';
  const freshness = typeof item.freshness === 'object' && item.freshness ? item.freshness : {};
  const statusLabel = item.status || freshness.label || 'Fresh';
  const freshnessState = item.freshnessState || freshness.label || statusLabel;

  const lowStatus = String(statusLabel).toLowerCase();
  const statusCategory =
    item.statusCategory ||
    (lowStatus.includes('fresh') ? 'fresh' : lowStatus.includes('slightly') ? 'semi' : 'attention');

  let estQualityDays: number | null = null;
  if (item.estimatedQualityDays != null) {
    estQualityDays = Math.max(0, Math.floor(Number(item.estimatedQualityDays)));
  } else if (item.remainingDays != null) {
    estQualityDays = Math.max(0, Math.floor(Number(item.remainingDays)));
  } else if (item.shelfLife && typeof item.shelfLife === 'object') {
    const rem = item.shelfLife.remaining || {};
    if (rem.maxDays != null) estQualityDays = Math.max(0, Math.floor(Number(rem.maxDays)));
    else if (rem.minDays != null) estQualityDays = Math.max(0, Math.floor(Number(rem.minDays)));
  }

  const addedAtDt = new Date();
  const addedAtIso = item.addedAt || addedAtDt.toISOString();
  const analyzedAtIso = item.analyzedAt || addedAtIso;
  const estimatedEndIso =
    item.estimatedEndAt ??
    (estQualityDays != null ? computeEstimatedEndAt(new Date(addedAtIso), estQualityDays).toISOString() : null);

  const storageEnv = item.storageEnvironment || item.storageSuggestion || 'Countertop Ambient';
  const storageType =
    item.storageType ||
    (storageEnv.toLowerCase().includes('fridge') || storageEnv.toLowerCase().includes('crisper')
      ? 'fridge'
      : 'countertop');

  const qualityScore = item.qualityScore ?? 80;
  const qualityPeriod = item.qualityPeriod || qualityPeriodLabel(estQualityDays);
  const cultivar = item.cultivar || `${foodName} • Fresh Harvest`;
  const ef = item.eatFirstPriority;
  const efPriority = typeof ef === 'object' && ef ? ef.priority : ef || 'NORMAL';
  const efScore = typeof ef === 'object' && ef ? (ef.score ?? 1) : (item.eatFirstScore ?? 1);
  const efReason = typeof ef === 'object' && ef ? (ef.reason ?? '') : (item.eatFirstReason ?? '');
  const guidance = item.guidance || 'Store in optimal pantry conditions.';
  const imageSrc = item.imageSrc || '';
  const freshnessConfidence = item.freshnessConfidence ?? freshness.confidence ?? freshness.score ?? null;
  const foodForm = item.foodForm || inferFoodForm(foodName, freshnessState);

  db.prepare(
    `INSERT OR REPLACE INTO pantry_history (
      id, user_id, food_name, cultivar, status, status_category,
      quality_score, quality_period, remaining_days, storage_environment,
      storage_type, eat_first_priority, eat_first_score, eat_first_reason,
      guidance, image_src, raw_metadata, created_at,
      analysis_id, added_at, estimated_quality_days, estimated_end_at,
      freshness_state, freshness_confidence, updated_at, food_form
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    itemId,
    userId,
    foodName,
    cultivar,
    statusLabel,
    statusCategory,
    qualityScore,
    qualityPeriod,
    estQualityDays,
    storageEnv,
    storageType,
    efPriority,
    efScore,
    efReason,
    guidance,
    imageSrc,
    JSON.stringify(item),
    analyzedAtIso,
    item.analysisId || item.id || null,
    addedAtIso,
    estQualityDays,
    estimatedEndIso,
    freshnessState,
    freshnessConfidence,
    addedAtDt.toISOString(),
    foodForm,
  );

  const enriched = getCurrentPantryState(userId);
  return (
    enriched.find((r) => r.id === itemId) ||
    enrichPantryRow({
      id: itemId,
      userId,
      foodName,
      cultivar,
      status: statusLabel,
      statusCategory,
      qualityScore,
      qualityPeriod,
      remainingDays: estQualityDays,
      storageEnvironment: storageEnv,
      storageType,
      guidance,
      imageSrc,
      analyzedAt: analyzedAtIso,
      addedAt: addedAtIso,
      estimatedQualityDays: estQualityDays,
      estimatedEndAt: estimatedEndIso,
      freshnessState,
      freshnessConfidence,
      foodForm,
    })
  );
}

export function deletePantryItem(itemId: string, userId: string): boolean {
  const db = getDb();
  const res = db.prepare('DELETE FROM pantry_history WHERE id = ? AND user_id = ?').run(itemId, userId);
  return Number(res.changes || 0) > 0;
}

export function clearPantryItems(userId: string): void {
  const db = getDb();
  db.prepare('DELETE FROM pantry_history WHERE user_id = ?').run(userId);
}

// ============================================================================
// PERSISTENT SQLITE CONVERSATIONS & MESSAGES (FRESHO BUDDY)
// ============================================================================
export function createConversation(userId: string, title = 'New Food Discussion'): ConversationRecord {
  const db = getDb();
  const convId = `conv_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
  const now = new Date().toISOString();
  const cleanTitle = (title || 'New Food Discussion').trim().slice(0, 60);

  db.prepare(
    `INSERT INTO conversations (conversation_id, user_id, title, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?)`,
  ).run(convId, userId, cleanTitle, now, now);

  return {
    conversationId: convId,
    userId,
    title: cleanTitle,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function listConversations(userId: string) {
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT conversation_id, user_id, title, created_at, updated_at
       FROM conversations
       WHERE user_id = ?
       ORDER BY updated_at DESC`,
    )
    .all(userId) as Array<{
    conversation_id: string;
    user_id: string;
    title: string;
    created_at: string;
    updated_at: string;
  }>;

  return rows.map((r) => ({
    conversationId: r.conversation_id,
    userId: r.user_id,
    title: r.title,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  }));
}

export function getConversation(conversationId: string, userId: string): ConversationRecord | null {
  const db = getDb();
  const conv = db
    .prepare(
      `SELECT conversation_id, user_id, title, created_at, updated_at
       FROM conversations
       WHERE conversation_id = ? AND user_id = ?`,
    )
    .get(conversationId, userId) as
    | { conversation_id: string; user_id: string; title: string; created_at: string; updated_at: string }
    | undefined;

  if (!conv) return null;

  const msgRows = db
    .prepare(
      `SELECT message_id, conversation_id, role, content, analysis_context, created_at
       FROM messages
       WHERE conversation_id = ?
       ORDER BY created_at ASC`,
    )
    .all(conversationId) as Array<{
    message_id: string;
    conversation_id: string;
    role: string;
    content: string;
    analysis_context: string | null;
    created_at: string;
  }>;

  const messages: ChatMessageRecord[] = msgRows.map((m) => {
    let ctx: Record<string, any> | null = null;
    if (m.analysis_context) {
      try {
        ctx = JSON.parse(m.analysis_context);
      } catch {
        ctx = null;
      }
    }
    return {
      messageId: m.message_id,
      conversationId: m.conversation_id,
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.content,
      analysisContext: ctx,
      createdAt: m.created_at,
    };
  });

  return {
    conversationId: conv.conversation_id,
    userId: conv.user_id,
    title: conv.title,
    createdAt: conv.created_at,
    updatedAt: conv.updated_at,
    messages,
  };
}

export function updateConversationTitle(conversationId: string, userId: string, title: string): boolean {
  const db = getDb();
  const cleanTitle = (title || 'Food Discussion').trim().slice(0, 60);
  const now = new Date().toISOString();
  const res = db
    .prepare(
      `UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ? AND user_id = ?`,
    )
    .run(cleanTitle, now, conversationId, userId);
  return Number(res.changes || 0) > 0;
}

export function addMessage(
  conversationId: string,
  userId: string,
  role: 'user' | 'assistant',
  content: string,
  analysisContext: Record<string, any> | null = null,
): ChatMessageRecord {
  const db = getDb();
  const owner = db
    .prepare('SELECT conversation_id FROM conversations WHERE conversation_id = ? AND user_id = ?')
    .get(conversationId, userId);
  if (!owner) {
    throw new Error('Conversation not found');
  }

  const msgId = `msg_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
  const now = new Date().toISOString();
  const ctxJson = analysisContext ? JSON.stringify(analysisContext) : null;

  db.prepare(
    `INSERT INTO messages (message_id, conversation_id, user_id, role, content, analysis_context, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  ).run(msgId, conversationId, userId, role, content, ctxJson, now);

  db.prepare('UPDATE conversations SET updated_at = ? WHERE conversation_id = ? AND user_id = ?').run(
    now,
    conversationId,
    userId,
  );

  return {
    messageId: msgId,
    conversationId,
    role,
    content,
    analysisContext,
    createdAt: now,
  };
}

export function deleteConversation(conversationId: string, userId: string): boolean {
  const db = getDb();
  const res = db
    .prepare('DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?')
    .run(conversationId, userId);
  return Number(res.changes || 0) > 0;
}

// ============================================================================
// ML FEEDBACK STORE
// ============================================================================
export function createFeedbackRecord(userId: string, payload: Record<string, any>) {
  const db = getDb();
  const feedbackId = payload.feedbackId || `fb_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
  const nowIso = new Date().toISOString();
  const allowedStatuses = new Set(['COLLECTED', 'VALIDATED', 'TRAINING_READY', 'USED_FOR_TRAINING', 'REJECTED']);
  const status = allowedStatuses.has(payload.reviewStatus) ? payload.reviewStatus : 'COLLECTED';

  db.prepare(
    `INSERT INTO ml_feedback (
      feedback_id, user_id, analysis_id, image_path,
      predicted_food, predicted_food_confidence, predicted_food_form,
      predicted_freshness, predicted_freshness_confidence, predicted_shelf_life,
      user_corrected_food, user_corrected_food_form, user_corrected_freshness,
      user_corrected_shelf_life, storage_type, days_stored,
      feedback_source, review_status, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    feedbackId,
    userId,
    payload.analysisId ?? null,
    payload.imagePath ?? null,
    payload.predictedFood ?? null,
    payload.predictedFoodConfidence ?? null,
    payload.predictedFoodForm ?? null,
    payload.predictedFreshness ?? null,
    payload.predictedFreshnessConfidence ?? null,
    payload.predictedShelfLife ?? null,
    payload.userCorrectedFood ?? null,
    payload.userCorrectedFoodForm ?? null,
    payload.userCorrectedFreshness ?? null,
    payload.userCorrectedShelfLife ?? null,
    payload.storageType ?? null,
    payload.daysStored ?? null,
    payload.feedbackSource || 'user',
    status,
    nowIso,
  );

  return {
    feedbackId,
    reviewStatus: status,
    createdAt: nowIso,
    ...payload,
  };
}

export function listFeedbackForUser(userId: string) {
  const db = getDb();
  const rows = db
    .prepare('SELECT * FROM ml_feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT 50')
    .all(userId) as Record<string, any>[];
  return rows.map((r) => ({
    feedbackId: r.feedback_id,
    analysisId: r.analysis_id,
    reviewStatus: r.review_status,
    predictedFood: r.predicted_food,
    createdAt: r.created_at,
  }));
}
