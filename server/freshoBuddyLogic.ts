/**
 * FoodFresh AI - FreshoBuddy Unified Food Intelligence Logic & Groq SDK Service
 *
 * Pure intent detection, pantry triage, 3-tier context isolation (SIMPLE / MEDIUM / STRONG),
 * deterministic application authority, and official Groq SDK inference.
 * Decoupled from Express/SQLite for direct unit testing with node:test.
 */

import Groq from 'groq-sdk';

export type ContextLevel = 'simple' | 'medium' | 'strong';

export enum FreshoIntent {
  PANTRY_PRIORITY = 'PANTRY_PRIORITY',
  PANTRY_LIST = 'PANTRY_LIST',
  EXPIRING_SOON = 'EXPIRING_SOON',
  COOK_TODAY = 'COOK_TODAY',
  PANTRY_ITEM_LOOKUP = 'PANTRY_ITEM_LOOKUP',
  PANTRY_ITEM_STORAGE = 'PANTRY_ITEM_STORAGE',
  STORAGE_ADVICE = 'STORAGE_ADVICE',
  FRESHNESS_EXPLANATION = 'FRESHNESS_EXPLANATION',
  FOOD_WASTE = 'FOOD_WASTE',
  GENERAL_FOOD_QUESTION = 'GENERAL_FOOD_QUESTION',
  ANALYSIS_CONTEXT = 'ANALYSIS_CONTEXT',
  FOLLOW_UP = 'FOLLOW_UP',
}

export const DEFAULT_GROQ_MODEL = 'llama-3.3-70b-versatile';
export const DEFAULT_GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct';
export const DEFAULT_XAI_GROK_MODEL = 'grok-2-latest';
export const DEFAULT_XAI_GROK_VISION_MODEL = 'grok-2-vision-1212';

export const FRESHO_BUDDY_SYSTEM_PROMPT = `You are FreshoBuddy AI 🌱, the unified food intelligence assistant inside FoodFresh AI.

You answer BOTH general food science/storage/shelf-life questions AND questions grounded in the authenticated user's saved pantry records.

CORE APPLICATION AUTHORITY RULES:
1. STRICT CONTEXT BOUNDARIES:
   - If NO saved pantry block is provided (SIMPLE context), answer strictly with general food guidance. Never guess or claim what is in the user's pantry.
   - If a MEDIUM specific-item lookup says the requested food is NOT in the user's pantry, explicitly tell the user that it is not saved in their pantry, then provide helpful general guidance for that food.
   - If a STRONG or MEDIUM saved-pantry block is provided, treat the supplied application data as the single source of truth. Never invent pantry items, dates, remaining days, freshness statuses, or confidence scores.
2. DETERMINISTIC EAT FIRST PRIORITY:
   - The application computes the Eat First priority, urgency tier, and remaining quality days deterministically.
   - Explain the supplied Eat First winner (or ties) accurately without altering or overriding the ranking.
3. FOOD SAFETY DISCLAIMER:
   - Visual freshness analysis and USDA FoodKeeper windows are quality estimates, not laboratory food-safety inspections. Advise users to check smell, texture, firmness, and visible mold before eating questionable food.
4. CLEAR MARKDOWN FORMATTING:
   - Use clean Markdown formatting (**bold** key food names and remaining-day values, bullet lists when helpful). Be concise, practical, and friendly.`;

export class FreshoBuddyGroqError extends Error {
  code: string;
  statusCode: number;

  constructor(message: string, code = 'GROQ_API_ERROR', statusCode = 502) {
    super(message);
    this.name = 'FreshoBuddyGroqError';
    this.code = code;
    this.statusCode = statusCode;
  }
}

export function getGroqConfig(env: NodeJS.ProcessEnv = process.env): {
  apiKey: string;
  provider: 'groq' | 'grok';
  model: string;
  visionModel: string;
  baseUrl: string;
} {
  const apiKey = String(env.GROQ_API_KEY || env.GROK_API_KEY || '').trim();
  const provider: 'groq' | 'grok' = apiKey.startsWith('xai-') ? 'grok' : 'groq';
  const model =
    String(env.GROQ_MODEL || env.GROK_MODEL || '').trim() ||
    (provider === 'grok' ? DEFAULT_XAI_GROK_MODEL : DEFAULT_GROQ_MODEL);
  const visionModel =
    String(env.GROQ_VISION_MODEL || env.GROK_VISION_MODEL || '').trim() ||
    (provider === 'grok' ? DEFAULT_XAI_GROK_VISION_MODEL : DEFAULT_GROQ_VISION_MODEL);
  const baseUrl = String(env.GROQ_BASE_URL || env.GROK_BASE_URL || '').trim();
  return {
    apiKey,
    provider,
    model,
    visionModel,
    baseUrl,
  };
}

export const FOOD_ALIASES: Record<string, string> = {
  'cherry tomatoes': 'cherry tomato',
  'cherry tomato': 'cherry tomato',
  'sweet potatoes': 'sweet potato',
  'sweet potato': 'sweet potato',
  'bell peppers': 'bell pepper',
  'bell pepper': 'bell pepper',
  'green chillies': 'green chilli',
  'green chilli': 'green chilli',
  'green chilis': 'green chilli',
  'green chili': 'green chilli',
  'chili peppers': 'green chilli',
  'chili pepper': 'green chilli',
  'hot peppers': 'green chilli',
  'green beans': 'green beans',
  'green bean': 'green beans',
  strawberries: 'strawberry',
  strawberry: 'strawberry',
  blueberries: 'blueberry',
  blueberry: 'blueberry',
  pomegranates: 'pomegranate',
  pomegranate: 'pomegranate',
  watermelons: 'watermelon',
  watermelon: 'watermelon',
  pineapples: 'pineapple',
  pineapple: 'pineapple',
  cauliflowers: 'cauliflower',
  cauliflower: 'cauliflower',
  cucumbers: 'cucumber',
  cucumber: 'cucumber',
  eggplants: 'eggplant',
  eggplant: 'eggplant',
  aubergines: 'eggplant',
  aubergine: 'eggplant',
  brinjals: 'eggplant',
  brinjal: 'eggplant',
  beetroots: 'beetroot',
  beetroot: 'beetroot',
  beets: 'beetroot',
  beet: 'beetroot',
  avocados: 'avocado',
  avocado: 'avocado',
  tomatoes: 'tomato',
  tomato: 'tomato',
  potatoes: 'potato',
  potato: 'potato',
  mangoes: 'mango',
  mangos: 'mango',
  mango: 'mango',
  bananas: 'banana',
  banana: 'banana',
  apples: 'apple',
  apple: 'apple',
  oranges: 'orange',
  orange: 'orange',
  lemons: 'lemon',
  lemon: 'lemon',
  limes: 'lime',
  lime: 'lime',
  grapes: 'grape',
  grape: 'grape',
  papayas: 'papaya',
  papaya: 'papaya',
  chickoos: 'chickoo',
  chickoo: 'chickoo',
  chiku: 'chickoo',
  sapodilla: 'chickoo',
  pears: 'pear',
  pear: 'pear',
  peaches: 'peach',
  peach: 'peach',
  plums: 'plum',
  plum: 'plum',
  guavas: 'guava',
  guava: 'guava',
  cherries: 'cherry',
  cherry: 'cherry',
  peppers: 'bell pepper',
  pepper: 'bell pepper',
  capsicums: 'bell pepper',
  capsicum: 'bell pepper',
  chillies: 'green chilli',
  chilli: 'green chilli',
  chilis: 'green chilli',
  chili: 'green chilli',
  onions: 'onion',
  onion: 'onion',
  garlic: 'garlic',
  carrots: 'carrot',
  carrot: 'carrot',
  broccoli: 'broccoli',
  cabbage: 'cabbage',
  spinach: 'spinach',
  lettuce: 'lettuce',
  corn: 'corn',
  radishes: 'radish',
  radish: 'radish',
  ginger: 'ginger',
  mushrooms: 'mushroom',
  mushroom: 'mushroom',
  bread: 'bread',
  sourdough: 'bread',
  cakes: 'cake',
  cake: 'cake',
  milk: 'milk',
  cheese: 'cheese',
  yogurt: 'yogurt',
  kiwis: 'kiwi',
  kiwi: 'kiwi',
  dragonfruit: 'dragonfruit',
};

const SORTED_FOOD_KEYS = Object.keys(FOOD_ALIASES).sort((a, b) => b.length - a.length);

const OWNERSHIP_PATTERNS: RegExp[] = [
  /\bmy\s+saved\b/,
  /\bsaved\b/,
  /\bmy\s+logged\b/,
  /\blogged\b/,
  /\bin\s+my\s+pantry\b/,
  /\bfrom\s+my\s+pantry\b/,
  /\bmy\s+pantry\b/,
  /\bdo\s+i\s+have\b/,
  /\bhave\s+any\b/,
  /\bi\s+have\b/,
  /\bi\s+saved\b/,
  /\bi\s+logged\b/,
  /\bfor\s+my\b/,
  /\bof\s+my\b/,
  /\bis\s+my\b/,
  /\bare\s+my\b/,
  /\bcheck\s+my\b/,
  /\bstore\s+my\b/,
];

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function normalizeQueryText(text: string): string {
  return String(text || '')
    .toLowerCase()
    .trim()
    .replace(/[^\w\s?']/g, ' ')
    .replace(/\s+/g, ' ');
}

function hasPantryOwnership(q: string): boolean {
  return OWNERSHIP_PATTERNS.some((pat) => pat.test(q));
}

export function extractFoodNoun(text: string): string | null {
  const q = normalizeQueryText(text);
  if (!q) return null;

  for (const key of SORTED_FOOD_KEYS) {
    if (new RegExp(`\\b${escapeRegex(key)}\\b`).test(q)) {
      return FOOD_ALIASES[key];
    }
  }

  const fallbackPatterns: RegExp[] = [
    /\b(?:my\s+saved|saved|my\s+logged|logged|my)\s+([a-z][a-z\s]{1,24}?)(?:\s+(?:in\s+my\s+pantry|left|remain|remaining|expire|last|yet|still|today)\b|\?|$)/,
    /\bdo\s+i\s+have\s+(?:any\s+)?([a-z][a-z\s]{1,24}?)(?:\s+in\s+my\s+pantry|\?|$)/,
    /\bhow\s+(?:should|to|do)(?:\s+i)?\s+store\s+(?:fresh\s+|ripe\s+|whole\s+)?([a-z][a-z\s]{1,24}?)(?:\?|$)/,
  ];
  const stopTokens = new Set([
    'food',
    'foods',
    'item',
    'items',
    'produce',
    'pantry',
    'it',
    'this',
    'that',
    'them',
    'things',
    'everything',
    'first',
    'one',
  ]);

  for (const pat of fallbackPatterns) {
    const m = q.match(pat);
    if (m?.[1]) {
      let cand = m[1]
        .replace(/\b(?:fresh|ripe|whole|properly|correctly|well|today|any|some)\b/g, '')
        .trim();
      if (cand && !stopTokens.has(cand) && cand.split(/\s+/).length <= 3) {
        if (cand.endsWith('oes') && cand.length > 4) cand = cand.slice(0, -2);
        else if (cand.endsWith('ies') && cand.length > 4) cand = `${cand.slice(0, -3)}y`;
        else if (cand.endsWith('s') && cand.length > 3 && !cand.endsWith('ss')) cand = cand.slice(0, -1);
        return cand;
      }
    }
  }

  return null;
}

export function isFollowUpQuery(text: string): boolean {
  const q = normalizeQueryText(text).replace(/[?.!\s]+$/g, '');
  if (!q) return false;
  const words = q.split(/\s+/);
  const phrases = [
    'why that one',
    'why that',
    'why is it',
    'why is that',
    'that one',
    'how should i store it',
    'how to store it',
    'store it',
    'store them',
    'what about it',
    'what about that',
    'what about them',
    'tell me more about it',
    'tell me more',
    'how long will it last',
    'how long does it last',
    'how many days left for it',
    'explain why',
  ];
  if (words.length <= 9 && phrases.some((p) => new RegExp(`\\b${escapeRegex(p)}\\b`).test(q))) {
    return true;
  }
  return ['why', 'how', 'and storage', 'what else', 'more details'].includes(q);
}

export function detectFreshoIntent(
  query: string,
  hasAnalysisContext = false,
  isShortFollowUp = false,
): { intent: FreshoIntent; contextLevel: ContextLevel; targetFood: string | null } {
  const q = normalizeQueryText(query);
  const ownership = hasPantryOwnership(q);
  const targetFood = extractFoodNoun(q);

  if (isShortFollowUp || isFollowUpQuery(q)) {
    return { intent: FreshoIntent.FOLLOW_UP, contextLevel: 'medium', targetFood };
  }

  if (
    hasAnalysisContext &&
    ['this food', 'tell me about this', 'this item', 'store this', 'about this'].some((p) => q.includes(p))
  ) {
    return { intent: FreshoIntent.ANALYSIS_CONTEXT, contextLevel: 'medium', targetFood };
  }

  if (
    [
      'eat first',
      'use first',
      'consume first',
      'what should i eat first',
      'which food should i eat first',
      'which food should i consume first',
      'which item should i eat first',
      'before it goes bad',
      'most urgent',
      'priority to eat',
    ].some((p) => q.includes(p))
  ) {
    return { intent: FreshoIntent.PANTRY_PRIORITY, contextLevel: 'strong', targetFood: null };
  }

  if (
    [
      'approaching their quality limit',
      'approaching its quality limit',
      'closest to its quality limit',
      'closest to quality limit',
      'shortest remaining quality',
      'expiring soon',
      'expire soon',
      'going bad soon',
      'spoil soon',
      'needs attention',
      'high priority',
      'low priority',
    ].some((p) => q.includes(p))
  ) {
    return { intent: FreshoIntent.EXPIRING_SOON, contextLevel: 'strong', targetFood: null };
  }

  if (
    ['cook today', 'cook tonight', 'what should i cook', 'what can i cook', 'what to make today'].some((p) =>
      q.includes(p),
    )
  ) {
    return { intent: FreshoIntent.COOK_TODAY, contextLevel: 'strong', targetFood: null };
  }

  if (['storage advice for my pantry', 'store differently', 'organize my pantry'].some((p) => q.includes(p))) {
    return { intent: FreshoIntent.PANTRY_LIST, contextLevel: 'strong', targetFood: null };
  }

  if (ownership && targetFood) {
    if (['store', 'storage', 'keep', 'fridge', 'refrigerate', 'countertop'].some((p) => q.includes(p))) {
      return { intent: FreshoIntent.PANTRY_ITEM_STORAGE, contextLevel: 'medium', targetFood };
    }
    return { intent: FreshoIntent.PANTRY_ITEM_LOOKUP, contextLevel: 'medium', targetFood };
  }

  if (
    [
      'what foods are in my pantry',
      'what food is in my pantry',
      'what is in my pantry',
      'what s in my pantry',
      'whats in my pantry',
      'what foods do i have',
      'what food do i have',
      'foods do i have',
      'items in my pantry',
      'show me my pantry',
      'my pantry situation',
      'pantry situation',
      'list my food',
      'my logged food',
      'my saved food',
      'which food is freshest',
      'least fresh',
    ].some((p) => q.includes(p)) ||
    (ownership && (q.includes('pantry') || q.includes('saved food') || q.includes('logged food')))
  ) {
    return { intent: FreshoIntent.PANTRY_LIST, contextLevel: 'strong', targetFood: null };
  }

  const storeMatch = q.match(
    /how\s+(?:should|to|do)(?:\s+i)?\s+store\s+(?:the\s+|this\s+|fresh\s+|ripe\s+)?([a-z\s]+)/,
  );
  if (storeMatch?.[1]) {
    const cleaned = storeMatch[1]
      .replace(/\?/g, '')
      .replace(/\b(?:properly|correctly|well|today|fresh|ripe|whole)\b/g, '')
      .trim();
    return {
      intent: FreshoIntent.STORAGE_ADVICE,
      contextLevel: 'simple',
      targetFood: targetFood || cleaned || null,
    };
  }

  if (['slightly spoiled', 'what is slightly spoiled', 'what does slightly spoiled mean'].some((p) => q.includes(p))) {
    return { intent: FreshoIntent.FRESHNESS_EXPLANATION, contextLevel: 'simple', targetFood };
  }

  if (['brown spot', 'turn brown', 'why do bananas', 'why bananas'].some((p) => q.includes(p))) {
    return { intent: FreshoIntent.FRESHNESS_EXPLANATION, contextLevel: 'simple', targetFood: targetFood || 'banana' };
  }

  if (q.includes('waste') && (q.includes('reduce') || q.includes('food'))) {
    if (q.includes('pantry') && ownership) {
      return { intent: FreshoIntent.FOOD_WASTE, contextLevel: 'strong', targetFood: null };
    }
    return { intent: FreshoIntent.FOOD_WASTE, contextLevel: 'simple', targetFood };
  }

  return { intent: FreshoIntent.GENERAL_FOOD_QUESTION, contextLevel: 'simple', targetFood };
}

export function mapFreshoIntentToLegacyIntent(intent: FreshoIntent): string {
  switch (intent) {
    case FreshoIntent.PANTRY_PRIORITY:
      return 'eat_first';
    case FreshoIntent.PANTRY_LIST:
      return 'pantry_inventory';
    case FreshoIntent.EXPIRING_SOON:
      return 'expiring_soon';
    case FreshoIntent.COOK_TODAY:
      return 'cook_today';
    case FreshoIntent.PANTRY_ITEM_LOOKUP:
      return 'pantry_item_lookup';
    case FreshoIntent.PANTRY_ITEM_STORAGE:
    case FreshoIntent.STORAGE_ADVICE:
      return 'specific_food_storage';
    default:
      return 'general_educational';
  }
}

export function detectQueryIntent(
  query: string,
  hasAnalysisContext = false,
): { intent: string; contextLevel: ContextLevel; targetFood: string | null } {
  const { intent, contextLevel, targetFood } = detectFreshoIntent(query, hasAnalysisContext, false);
  return {
    intent: mapFreshoIntentToLegacyIntent(intent),
    contextLevel,
    targetFood,
  };
}

export function formatRemainingDays(remDays: number | null | undefined): string {
  if (remDays === null || remDays === undefined || Number.isNaN(Number(remDays))) {
    return 'remaining quality unavailable';
  }
  const n = Math.max(0, Math.round(Number(remDays)));
  if (n === 0) return '0 days remaining';
  if (n === 1) return '1 day remaining';
  return `${n} days remaining`;
}

export interface ScoredPantryItem {
  foodName: string;
  status: string;
  remainingDays: number | null;
  remainingQualityKnown: boolean;
  qualityPeriod: string;
  tier: string;
  urgencyScore: number;
  storageEnvironment: string;
  storageType: string;
  qualityScore: number;
  guidance: string;
}

export interface PantryTriageResult {
  hasItems: boolean;
  totalCount: number;
  rankedItems: ScoredPantryItem[];
  topUrgent: ScoredPantryItem | null;
  tiedTop: ScoredPantryItem[];
  isTie: boolean;
  expiringSoon: ScoredPantryItem[];
  freshest: ScoredPantryItem | null;
  leastFresh: ScoredPantryItem | null;
}

export function evaluateItemUrgency(item: Record<string, any>): ScoredPantryItem {
  const foodName = item.foodName || 'Unknown Produce';
  const statusRaw = String(item.status || 'Fresh');
  const status = statusRaw.toLowerCase();
  const remKnown = item.remainingDays !== null && item.remainingDays !== undefined && !Number.isNaN(Number(item.remainingDays));
  const effRem = remKnown ? Math.trunc(Number(item.remainingDays)) : null;

  const storageEnv = String(item.storageEnvironment || '').toLowerCase();
  const isCountertop = storageEnv.includes('countertop') || storageEnv.includes('ambient');

  let baseScore = 15;
  let tier = 'LOW';
  if (effRem === null) {
    baseScore = 5;
    tier = 'UNAVAILABLE';
  } else if (effRem <= 0) {
    baseScore = 100;
    tier = 'VERY_HIGH';
  } else if (effRem === 1) {
    baseScore = 80;
    tier = 'HIGH';
  } else if (effRem <= 3) {
    baseScore = 65;
    tier = 'HIGH';
  } else if (effRem <= 7) {
    baseScore = 35;
    tier = 'MEDIUM';
  }

  let freshnessMod = 0;
  if (status.includes('rotten') || status.includes('spoil') || status.includes('decay')) {
    if (status.includes('slightly')) {
      baseScore = Math.max(baseScore, 75);
      if (tier !== 'VERY_HIGH') tier = 'HIGH';
      freshnessMod = 15;
    } else {
      baseScore = Math.max(baseScore, 120);
      tier = 'VERY_HIGH';
      freshnessMod = 30;
    }
  } else if (status.includes('semi') || status.includes('soft') || status.includes('stale')) {
    baseScore = Math.max(baseScore, 75);
    if (tier !== 'VERY_HIGH') tier = 'HIGH';
    freshnessMod = 15;
  }

  const storageMod = isCountertop && effRem !== null && effRem <= 1 ? 5 : 0;
  const qScore = item.qualityScore !== null && item.qualityScore !== undefined ? Number(item.qualityScore) : 80;

  return {
    foodName,
    status: statusRaw,
    remainingDays: effRem,
    remainingQualityKnown: remKnown,
    qualityPeriod: item.qualityPeriod || formatRemainingDays(effRem),
    tier,
    urgencyScore: baseScore + freshnessMod + storageMod,
    storageEnvironment: item.storageEnvironment || 'Countertop Ambient',
    storageType: item.storageType || 'countertop',
    qualityScore: qScore,
    guidance: item.guidance || '',
  };
}

export function triagePantryItems(items: Record<string, any>[]): PantryTriageResult {
  if (!items || !items.length) {
    return {
      hasItems: false,
      totalCount: 0,
      rankedItems: [],
      topUrgent: null,
      tiedTop: [],
      isTie: false,
      expiringSoon: [],
      freshest: null,
      leastFresh: null,
    };
  }

  const scored = items.map(evaluateItemUrgency);

  scored.sort((a, b) => {
    if (b.urgencyScore !== a.urgencyScore) return b.urgencyScore - a.urgencyScore;
    const aNull = a.remainingDays === null ? 1 : 0;
    const bNull = b.remainingDays === null ? 1 : 0;
    if (aNull !== bNull) return aNull - bNull;
    const aRem = a.remainingDays ?? 9999;
    const bRem = b.remainingDays ?? 9999;
    if (aRem !== bRem) return aRem - bRem;
    return a.foodName.localeCompare(b.foodName);
  });

  const topUrgent = scored[0];
  const tiedTop = scored.filter((it) => it.urgencyScore === topUrgent.urgencyScore);
  const isTie = tiedTop.length > 1;
  const expiringSoon = scored.filter(
    (it) => (it.remainingDays !== null && it.remainingDays <= 3) || it.tier === 'VERY_HIGH' || it.tier === 'HIGH',
  );

  const known = scored.filter((it) => it.remainingDays !== null);
  const freshestPool = known.length ? known : scored;
  const freshest = [...freshestPool].sort((a, b) => {
    const aRem = a.remainingDays ?? -1;
    const bRem = b.remainingDays ?? -1;
    if (bRem !== aRem) return bRem - aRem;
    return b.qualityScore - a.qualityScore;
  })[0];

  return {
    hasItems: true,
    totalCount: scored.length,
    rankedItems: scored,
    topUrgent,
    tiedTop: isTie ? tiedTop : [],
    isTie,
    expiringSoon,
    freshest,
    leastFresh: topUrgent,
  };
}

export function findMatchingPantryItems(
  triage: PantryTriageResult | null,
  targetFood: string | null | undefined,
): ScoredPantryItem[] {
  if (!triage?.hasItems || !targetFood) return [];
  const needle = targetFood.toLowerCase().trim();
  const canonicalNeedle = FOOD_ALIASES[needle] || needle;
  return triage.rankedItems.filter((it) => {
    const nameLow = String(it.foodName || '').toLowerCase();
    const nameCanonical = FOOD_ALIASES[nameLow] || extractFoodNoun(nameLow) || nameLow;
    return (
      nameLow.includes(canonicalNeedle) ||
      canonicalNeedle.includes(nameLow) ||
      nameCanonical === canonicalNeedle
    );
  });
}

export function deriveConversationTitle(
  firstUserMessage: string,
  foodContext?: Record<string, any> | null,
): string {
  const msgLower = (firstUserMessage || '').toLowerCase();
  if (
    msgLower.includes('eat first') ||
    msgLower.includes('which food') ||
    msgLower.includes('use first') ||
    msgLower.includes('consume first')
  ) {
    return 'Pantry Priority';
  }
  if (msgLower.includes('cook today') || msgLower.includes('cook tonight') || msgLower.includes('what to make')) {
    return 'Pantry Meal Ideas';
  }
  if (msgLower.includes('expir') || msgLower.includes('attention') || msgLower.includes('quality limit')) {
    return 'Expiring Foods';
  }
  if (msgLower.includes('reduce') && msgLower.includes('waste')) {
    return 'Reduce Food Waste';
  }
  if (msgLower.includes('brown spot') || (msgLower.includes('banana') && msgLower.includes('brown'))) {
    return 'Banana Browning';
  }
  if (msgLower.includes('pantry') || msgLower.includes('inventory') || msgLower.includes('what food')) {
    return 'Pantry Overview';
  }

  const ctxFood = foodContext?.detectedFood || foodContext?.foodName;
  if (ctxFood) {
    const food = String(ctxFood).trim();
    const titled = food.charAt(0).toUpperCase() + food.slice(1);
    if (msgLower.includes('store') || msgLower.includes('keep') || msgLower.includes('fridge')) {
      return `${titled} Storage Advice`;
    }
    if (msgLower.includes('fresh') || msgLower.includes('spoil') || msgLower.includes('rot')) {
      return `${titled} Freshness`;
    }
    if (msgLower.includes('recipe') || msgLower.includes('cook') || msgLower.includes('eat')) {
      return `${titled} Ideas`;
    }
    return `${titled} Discussion`;
  }

  const storeMatch = msgLower.match(/how (?:should|to|do)(?: i)? store (?:this |the |my |fresh |ripe )?([a-z\s]+)/);
  if (storeMatch?.[1]) {
    const target = storeMatch[1].replace(/\?/g, '').trim();
    if (target) {
      return `${target.charAt(0).toUpperCase() + target.slice(1)} Storage Advice`.slice(0, 40);
    }
  }

  const words = (firstUserMessage || '').replace(/[^\w\s]/g, '').trim().split(/\s+/).filter(Boolean);
  if (!words.length) return 'New Food Discussion';
  return words
    .slice(0, 4)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ')
    .slice(0, 40);
}

export function resolveFollowUpContext(history: Array<{ role?: string; content?: string }>): {
  resolvedFood: string | null;
  wasPantryScoped: boolean;
  priorIntent: FreshoIntent | null;
  priorLevel: ContextLevel;
} {
  for (let i = history.length - 1; i >= 0; i--) {
    const msg = history[i];
    if (msg.role !== 'assistant') continue;
    const text = String(msg.content || '');
    let prevUserText = '';
    for (let j = i - 1; j >= 0; j--) {
      if (history[j].role === 'user') {
        prevUserText = String(history[j].content || '');
        break;
      }
    }

    const prevDetected = detectFreshoIntent(prevUserText, false, false);
    const wasPantryScoped =
      prevDetected.contextLevel === 'strong' ||
      prevDetected.contextLevel === 'medium' ||
      ['saved pantry', 'should be used first', 'pantry produce log', 'days remaining', 'priority:'].some((k) =>
        text.toLowerCase().includes(k),
      );

    let resolvedFood: string | null = null;
    const mFirst = text.match(/\*\*([A-Za-z][A-Za-z\s]{1,30}?)\s+should be used first\*\*/);
    if (mFirst?.[1]) {
      resolvedFood = extractFoodNoun(mFirst[1]) || mFirst[1].trim();
    }
    if (!resolvedFood) {
      const boldMatches = text.matchAll(/\*\*([A-Za-z][A-Za-z\s]{1,30}?)\*\*/g);
      for (const mBold of boldMatches) {
        const candidate = extractFoodNoun(mBold[1]);
        if (candidate) {
          resolvedFood = candidate;
          break;
        }
      }
    }
    if (!resolvedFood) {
      resolvedFood = prevDetected.targetFood || extractFoodNoun(prevUserText) || extractFoodNoun(text);
    }

    return {
      resolvedFood,
      wasPantryScoped,
      priorIntent: prevDetected.intent,
      priorLevel: prevDetected.contextLevel,
    };
  }

  return {
    resolvedFood: null,
    wasPantryScoped: false,
    priorIntent: null,
    priorLevel: 'simple',
  };
}

export function enforceApplicationAuthority(
  replyText: string,
  contextLevel: ContextLevel,
  freshoIntent: FreshoIntent,
  triage: PantryTriageResult | null,
  matchedItems: ScoredPantryItem[],
  targetFood: string | null,
): string {
  let text = (replyText || '').trim();
  if (!text) return text;

  if (contextLevel === 'strong' && freshoIntent === FreshoIntent.PANTRY_PRIORITY) {
    if (triage?.hasItems && triage.topUrgent) {
      const winners = triage.isTie ? triage.tiedTop : [triage.topUrgent];
      const missingWinner = winners.some((w) => !text.toLowerCase().includes(w.foodName.toLowerCase()));
      if (missingWinner) {
        const names = winners.map((w) => `**${w.foodName}**`).join(' and ');
        const remTxt = formatRemainingDays(winners[0].remainingDays);
        text = `**Application Eat First Priority:** ${names} (${remTxt}, ${winners[0].tier} priority).\n\n${text}`;
      }
    }
  } else if (contextLevel === 'medium' && targetFood && matchedItems.length === 0) {
    const low = text.toLowerCase();
    if (!low.includes('not') && !low.includes('pantry')) {
      const titled = targetFood.trim().charAt(0).toUpperCase() + targetFood.trim().slice(1);
      text = `**${titled}** is not currently saved in your pantry log.\n\n${text}`;
    }
  }

  return text;
}

export function buildDeterministicAnswer(
  intent: string,
  triage: PantryTriageResult,
  targetFood: string | null = null,
): string {
  if ((intent === 'pantry_item_lookup' || intent === 'specific_food_storage') && targetFood) {
    const matches = findMatchingPantryItems(triage, targetFood);
    if (matches.length) {
      const m = matches[0];
      return `According to your saved pantry history, you have **${m.foodName}** (${m.status}, **${formatRemainingDays(m.remainingDays)}**, stored in **${m.storageEnvironment}**, Priority: **${m.tier}**).`;
    }
    const titled = targetFood.trim().charAt(0).toUpperCase() + targetFood.trim().slice(1);
    return `**${titled}** is not currently saved in your pantry history.`;
  }

  if (!triage.hasItems) {
    return "Your pantry history is currently empty, so I can't identify a specific food to prioritize yet. Analyze and save a food first, and I'll track its freshness and prioritize it for you 🌱";
  }

  const top = triage.topUrgent!;
  if (intent === 'eat_first') {
    if (triage.isTie) {
      const names = triage.tiedTop.map((it) => `**${it.foodName}**`).join(' and ');
      const bullets = triage.tiedTop
        .map((it) => `- **${it.foodName}** — ${formatRemainingDays(it.remainingDays)} (${it.status}, ${it.tier} priority)`)
        .join('\n');
      return `Based on your saved pantry history, ${names} **should be used first** (${formatRemainingDays(top.remainingDays)}).\n\n**Why:**\n${bullets}`;
    }
    return `Based on your saved pantry history, **${top.foodName} should be used first**.\n\n- **${top.foodName}** — ${formatRemainingDays(top.remainingDays)}\n- **Freshness:** ${top.status}\n- **Priority:** ${top.tier}`;
  }

  if (intent === 'pantry_inventory') {
    const lines = triage.rankedItems
      .map((it) => `- **${it.foodName}**: ${it.status} • ${formatRemainingDays(it.remainingDays)} • Priority: **${it.tier}**`)
      .join('\n');
    return `Here is your current **Pantry Produce Log** (${triage.totalCount} items):\n\n${lines}\n\n**Most urgent:** **${top.foodName}** should be used first.`;
  }

  return `**${top.foodName}** (${formatRemainingDays(top.remainingDays)}) has the highest Eat First priority (${top.tier}).`;
}

export interface ContextBundle {
  freshoIntent: FreshoIntent;
  contextLevel: ContextLevel;
  targetFood: string | null;
  pantryInjected: boolean;
  triage: PantryTriageResult | null;
  matchedItems: ScoredPantryItem[];
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
}

export function buildContextBundle(params: {
  userMessage: string;
  history?: Array<{ role?: string; content?: string }>;
  analysisContext?: Record<string, any> | null;
  getPantryItems?: () => Record<string, any>[];
  pantryItems?: Record<string, any>[];
}): ContextBundle {
  const { userMessage, analysisContext = null } = params;
  const history = params.history || [];
  const shortFollowUp = history.length > 0 && isFollowUpQuery(userMessage);
  let { intent: freshoIntent, contextLevel, targetFood } = detectFreshoIntent(
    userMessage,
    Boolean(analysisContext),
    shortFollowUp,
  );

  if (freshoIntent === FreshoIntent.FOLLOW_UP && history.length > 0) {
    const followCtx = resolveFollowUpContext(history);
    if (!targetFood) targetFood = followCtx.resolvedFood;
    if (!followCtx.wasPantryScoped && !analysisContext) {
      contextLevel = 'simple';
    } else {
      contextLevel = 'medium';
    }
  }

  const pantryIntent = mapFreshoIntentToLegacyIntent(freshoIntent);
  let triage: PantryTriageResult | null = null;
  let matchedItems: ScoredPantryItem[] = [];
  let pantryInjected = false;
  let authorityBlock = '';

  // SIMPLE MUST NEVER query or inject user pantry records into the AI prompt.
  if (contextLevel === 'medium' || contextLevel === 'strong') {
    const shouldLoadPantry =
      params.pantryItems !== undefined ||
      contextLevel === 'strong' ||
      targetFood !== null ||
      freshoIntent === FreshoIntent.FOLLOW_UP;

    const loadedItems = shouldLoadPantry
      ? (params.pantryItems ?? (params.getPantryItems ? params.getPantryItems() : []))
      : [];

    if (contextLevel === 'strong') {
      triage = triagePantryItems(loadedItems);
      pantryInjected = true;
      if (!triage.hasItems) {
        authorityBlock =
          '\n[USER PANTRY AUTHORITY DATA]\n' +
          '- Pantry Status: EMPTY (0 saved food items).\n' +
          '- Instruction: Inform the user clearly that their pantry is currently empty and suggest analyzing and saving food items first. Do NOT invent any food items.\n';
      } else {
        const top = triage.topUrgent!;
        const itemsLines = triage.rankedItems
          .map(
            (it) =>
              `- ${it.foodName}: status=${it.status} | remaining=${formatRemainingDays(it.remainingDays)} | priority=${it.tier} | storage=${it.storageEnvironment}`,
          )
          .join('\n');
        const tiedNames = triage.isTie ? triage.tiedTop.map((it) => it.foodName).join(', ') : 'None';
        authorityBlock =
          `\n[USER SAVED PANTRY LOG — ${triage.totalCount} ITEMS]\n${itemsLines}\n\n` +
          `[APPLICATION DETERMINISTIC EAT-FIRST CALCULATION]\n` +
          `- Intent: ${pantryIntent}\n` +
          `- Calculated Top Priority Food: ${top.foodName} (Tier: ${top.tier}, UrgencyScore: ${top.urgencyScore}, Remaining: ${formatRemainingDays(top.remainingDays)})\n` +
          `- Is Tied At Top: ${triage.isTie} (Tied: ${tiedNames})\n` +
          `- Strict Rule: Ground your answer ONLY in the saved items above. Never override the priority ranking or invent foods not listed above.\n`;
      }
    } else if (contextLevel === 'medium') {
      if (targetFood && shouldLoadPantry) {
        triage = triagePantryItems(loadedItems);
        matchedItems = findMatchingPantryItems(triage, targetFood);
        pantryInjected = true;
        if (matchedItems.length > 0) {
          const lines = matchedItems
            .map(
              (it) =>
                `- ${it.foodName}: status=${it.status} | remaining=${formatRemainingDays(it.remainingDays)} | priority=${it.tier} | storage=${it.storageEnvironment} | guidance=${it.guidance}`,
            )
            .join('\n');
          authorityBlock =
            `\n[MATCHING SAVED PANTRY RECORD FOR '${targetFood.toUpperCase()}']\n${lines}\n` +
            `- Strict Rule: Cite the exact remaining quality days, freshness status, priority tier, and storage above. Do not fabricate additional items.\n`;
        } else {
          const titled = targetFood.trim().charAt(0).toUpperCase() + targetFood.trim().slice(1);
          authorityBlock =
            `\n[PANTRY LOOKUP RESULT FOR '${targetFood.toUpperCase()}']\n` +
            `- Found In User's Saved Pantry: FALSE (0 matching items for '${targetFood}').\n` +
            `- Strict Rule: Explicitly state that ${titled} is NOT currently saved in the user's pantry, and then provide helpful general storage/freshness guidance for ${titled}.\n`;
        }
      } else if (freshoIntent === FreshoIntent.FOLLOW_UP && shouldLoadPantry) {
        triage = triagePantryItems(loadedItems);
        if (triage.hasItems && triage.topUrgent) {
          pantryInjected = true;
          const top = triage.topUrgent;
          authorityBlock =
            `\n[FOLLOW-UP PANTRY CONTEXT]\n` +
            `- Top Ranked Item: ${top.foodName} (status=${top.status}, remaining=${formatRemainingDays(top.remainingDays)}, priority=${top.tier}, storage=${top.storageEnvironment})\n` +
            `- Total Pantry Items: ${triage.totalCount}\n`;
        }
      }
    }
  }

  let scanContextBlock = '';
  const scanFood = analysisContext?.detectedFood || analysisContext?.foodName;
  if (scanFood) {
    const freshness = analysisContext?.freshness;
    const fLabel = typeof freshness === 'object' && freshness ? freshness.label : freshness;
    const fConf = typeof freshness === 'object' && freshness ? freshness.confidence : null;
    const rem = analysisContext?.shelfLife?.remaining;
    const remStr = rem ? `${rem.minDays ?? '?'}–${rem.maxDays ?? '?'} days` : 'Guidance unavailable';
    const storage = analysisContext?.storageType || analysisContext?.storageEnvironment || 'countertop';
    const daysStored = analysisContext?.daysStored ?? 0;
    scanContextBlock =
      `\n[ACTIVE SCANNED FOOD CONTEXT]\n` +
      `- Food Item: ${scanFood}\n` +
      `- Estimated Visible Freshness: ${fLabel || 'Uncertain'} (${fConf ?? 'N/A'}% confidence)\n` +
      `- Estimated Remaining Quality Window: ${remStr}\n` +
      `- Storage Environment: ${storage}\n` +
      `- Days Stored So Far: ${daysStored} days\n`;
  }

  const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: FRESHO_BUDDY_SYSTEM_PROMPT },
  ];

  for (const msg of history.slice(-8)) {
    const content = String(msg.content || '').trim();
    if (!content) continue;
    messages.push({
      role: msg.role === 'user' ? 'user' : 'assistant',
      content,
    });
  }

  const contextPrefix = `${authorityBlock}${scanContextBlock}`.trim();
  const finalUserContent = contextPrefix ? `${contextPrefix}\n\nUser question: ${userMessage}` : userMessage;
  messages.push({ role: 'user', content: finalUserContent });

  return {
    freshoIntent,
    contextLevel,
    targetFood,
    pantryInjected,
    triage,
    matchedItems,
    messages,
  };
}

export async function generateFreshoBuddyGroqReply(params: {
  userMessage: string;
  history?: Array<{ role?: string; content?: string }>;
  analysisContext?: Record<string, any> | null;
  getPantryItems?: () => Record<string, any>[];
  pantryItems?: Record<string, any>[];
  env?: NodeJS.ProcessEnv;
}): Promise<{
  text: string;
  status: 'success';
  model: string;
  contextLevel: ContextLevel;
  intent: FreshoIntent;
  pantryInjected: boolean;
  bundle: ContextBundle;
}> {
  const cfg = getGroqConfig(params.env);
  if (!cfg.apiKey) {
    throw new FreshoBuddyGroqError(
      'FreshoBuddy AI requires a configured GROQ_API_KEY or GROK_API_KEY on the backend.',
      'GROQ_UNAVAILABLE',
      503,
    );
  }

  const bundle = buildContextBundle(params);

  try {
    let rawReply = '';
    if (cfg.provider === 'grok' && !cfg.baseUrl) {
      const resp = await fetch('https://api.x.ai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${cfg.apiKey}`,
        },
        body: JSON.stringify({
          model: cfg.model,
          messages: bundle.messages,
          temperature: 0.4,
          max_tokens: 700,
        }),
        signal: AbortSignal.timeout(25_000),
      });
      if (!resp.ok) {
        const errText = await resp.text().catch(() => '');
        const err: any = new Error(errText || `HTTP ${resp.status}`);
        err.status = resp.status;
        throw err;
      }
      const data: any = await resp.json();
      rawReply = String(data?.choices?.[0]?.message?.content || '').trim();
    } else {
      const client = new Groq({
        apiKey: cfg.apiKey,
        ...(cfg.baseUrl ? { baseURL: cfg.baseUrl } : {}),
        timeout: 25_000,
        maxRetries: 1,
      });
      const callChat = async (modelId: string) => {
        const completion = await client.chat.completions.create({
          model: modelId,
          messages: bundle.messages,
          temperature: 0.4,
          max_tokens: 700,
        });
        return String(completion?.choices?.[0]?.message?.content || '').trim();
      };

      try {
        rawReply = await callChat(cfg.model);
      } catch (primaryErr: any) {
        const pStatus = Number(primaryErr?.status || primaryErr?.statusCode || 0);
        const pMsg = String(primaryErr?.message || '').toLowerCase();
        const isModelMissing =
          !cfg.baseUrl && (pStatus === 404 || pMsg.includes('model_not_found') || pMsg.includes('does not exist'));
        if (!isModelMissing) {
          throw primaryErr;
        }
        const list = await client.models.list().catch(() => ({ data: [] as Array<{ id?: string }> }));
        const availableIds = (list?.data || [])
          .map((m) => String(m?.id || ''))
          .filter(
            (id) =>
              id &&
              id !== cfg.model &&
              !/whisper|tts|guard|orpheus|playai|distil-whisper/i.test(id),
          );
        const preferredOrder = [
          'llama-3.1-8b-instant',
          'meta-llama/llama-4-scout-17b-16e-instruct',
          'meta-llama/llama-4-maverick-17b-128e-instruct',
          'openai/gpt-oss-120b',
          'openai/gpt-oss-20b',
          'qwen/qwen3-32b',
          'llama3-70b-8192',
          'llama3-8b-8192',
        ];
        const orderedCandidates = [
          ...preferredOrder.filter((id) => availableIds.includes(id)),
          ...availableIds.filter((id) => !preferredOrder.includes(id)),
        ];
        if (orderedCandidates.length === 0) {
          throw primaryErr;
        }
        let lastErr: any = primaryErr;
        for (const candidate of orderedCandidates.slice(0, 4)) {
          try {
            rawReply = await callChat(candidate);
            if (rawReply) {
              cfg.model = candidate;
              break;
            }
          } catch (candErr: any) {
            lastErr = candErr;
          }
        }
        if (!rawReply) {
          throw lastErr;
        }
      }
    }

    if (!rawReply) {
      throw new FreshoBuddyGroqError('Groq returned an empty response.', 'GROQ_EMPTY_RESPONSE', 502);
    }

    const finalText = enforceApplicationAuthority(
      rawReply,
      bundle.contextLevel,
      bundle.freshoIntent,
      bundle.triage,
      bundle.matchedItems,
      bundle.targetFood,
    );

    return {
      text: finalText,
      status: 'success',
      model: cfg.model,
      contextLevel: bundle.contextLevel,
      intent: bundle.freshoIntent,
      pantryInjected: bundle.pantryInjected,
      bundle,
    };
  } catch (err: any) {
    if (err instanceof FreshoBuddyGroqError) {
      throw err;
    }
    const status = Number(err?.status || err?.statusCode || 0);
    const msg = String(err?.message || err || 'Unknown Groq error');
    const low = msg.toLowerCase();
    const isAuth =
      status === 401 ||
      status === 403 ||
      low.includes('401') ||
      low.includes('403') ||
      low.includes('invalid api key') ||
      low.includes('authentication') ||
      low.includes('unauthorized');

    throw new FreshoBuddyGroqError(
      isAuth
        ? 'Groq AI authentication failed. Check GROQ_API_KEY configuration.'
        : `Groq AI request failed: ${msg}`,
      isAuth ? 'GROQ_AUTH_ERROR' : 'GROQ_API_ERROR',
      isAuth ? 502 : 503,
    );
  }
}
