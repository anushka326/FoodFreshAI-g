import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  addMessage,
  addPantryItem,
  createConversation,
  deleteConversation,
  getConversation,
  getCurrentPantryState,
  listConversations,
  loginUser,
  registerUser,
  resetDbConnectionForTests,
  resolveSession,
} from './foodServices.ts';
import {
  FreshoBuddyGroqError,
  FreshoIntent,
  buildContextBundle,
  deriveConversationTitle,
  detectFreshoIntent,
  enforceApplicationAuthority,
  generateFreshoBuddyGroqReply,
  getGroqConfig,
  triagePantryItems,
} from './freshoBuddyLogic.ts';

function startStubGroqServer(handler: (body: any, authHeader: string) => { status: number; json: any }): Promise<{
  baseUrl: string;
  captured: Array<{ body: any; authHeader: string }>;
  close: () => Promise<void>;
}> {
  const captured: Array<{ body: any; authHeader: string }> = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf-8');
      let parsed: any = {};
      try {
        parsed = raw ? JSON.parse(raw) : {};
      } catch {
        parsed = {};
      }
      const authHeader = String(req.headers.authorization || '');
      captured.push({ body: parsed, authHeader });
      const result = handler(parsed, authHeader);
      res.writeHead(result.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result.json));
    });
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      resolve({
        baseUrl: `http://127.0.0.1:${port}`,
        captured,
        close: () => new Promise<void>((done) => server.close(() => done())),
      });
    });
  });
}

test('Intent detection & 3-tier context isolation (SIMPLE / MEDIUM / STRONG)', () => {
  // Scenario 1 & 2: General food science & storage stay SIMPLE
  for (const q of [
    'Average shelf life of freshly purchased bananas',
    'How to store fresh potatoes',
    'Why do bananas develop brown spots?',
    'How long do fresh tomatoes last?',
  ]) {
    const res = detectFreshoIntent(q, false, false);
    assert.equal(res.contextLevel, 'simple', `Expected SIMPLE for: ${q}`);
  }

  // Scenario 3 & 4: Saved/pantry specific food stays MEDIUM
  const mangoQ = detectFreshoIntent('How many days remain for my saved mangoes?', false, false);
  assert.equal(mangoQ.contextLevel, 'medium');
  assert.equal(mangoQ.intent, FreshoIntent.PANTRY_ITEM_LOOKUP);
  assert.equal(mangoQ.targetFood, 'mango');

  const potatoQ = detectFreshoIntent('How should I store my saved potatoes?', false, false);
  assert.equal(potatoQ.contextLevel, 'medium');
  assert.equal(potatoQ.intent, FreshoIntent.PANTRY_ITEM_STORAGE);
  assert.equal(potatoQ.targetFood, 'potato');

  const avocadoQ = detectFreshoIntent('Do I have avocados in my pantry?', false, false);
  assert.equal(avocadoQ.contextLevel, 'medium');
  assert.equal(avocadoQ.targetFood, 'avocado');

  // Scenario 5 & 6: Whole-pantry questions trigger STRONG
  for (const q of [
    'Which food should I consume first?',
    'What foods are in my pantry?',
    'Which saved foods are approaching their quality limits?',
  ]) {
    const res = detectFreshoIntent(q, false, false);
    assert.equal(res.contextLevel, 'strong', `Expected STRONG for: ${q}`);
  }
});

test('buildContextBundle enforces pantry isolation, Eat First determinism, and follow-up boundaries', () => {
  const samplePantry = [
    {
      foodName: 'Apple',
      status: 'Fresh',
      remainingDays: 6,
      storageEnvironment: 'Countertop Ambient',
      qualityScore: 92,
      guidance: 'Store cool',
    },
    {
      foodName: 'Mango',
      status: 'Slightly Spoiled',
      remainingDays: 0,
      storageEnvironment: 'Countertop Ambient',
      qualityScore: 0,
      guidance: 'Use immediately',
    },
    {
      foodName: 'Potato',
      status: 'Fresh',
      remainingDays: 7,
      storageEnvironment: 'Countertop Ambient',
      qualityScore: 95,
      guidance: 'Keep in cool dark ventilated basket',
    },
  ];

  // Scenario 2: General question with populated pantry -> ZERO pantry injection
  let pantryQueried = false;
  const generalBundle = buildContextBundle({
    userMessage: 'How to store fresh potatoes',
    getPantryItems: () => {
      pantryQueried = true;
      return samplePantry;
    },
  });
  assert.equal(generalBundle.contextLevel, 'simple');
  assert.equal(generalBundle.pantryInjected, false);
  assert.equal(pantryQueried, false, 'SIMPLE must not query pantry records');
  assert.ok(!JSON.stringify(generalBundle.messages).includes('Slightly Spoiled'));
  assert.ok(!JSON.stringify(generalBundle.messages).includes('Mango'));

  // Scenario 3: Specific saved item -> only Mango record injected
  const savedMangoBundle = buildContextBundle({
    userMessage: 'How many days remain for my saved mangoes?',
    pantryItems: samplePantry,
  });
  assert.equal(savedMangoBundle.contextLevel, 'medium');
  assert.equal(savedMangoBundle.pantryInjected, true);
  assert.equal(savedMangoBundle.matchedItems.length, 1);
  assert.equal(savedMangoBundle.matchedItems[0].foodName, 'Mango');
  const mangoPrompt = savedMangoBundle.messages[savedMangoBundle.messages.length - 1].content;
  assert.ok(mangoPrompt.includes('Mango'));
  assert.ok(mangoPrompt.includes('0 days remaining'));
  assert.ok(!mangoPrompt.includes('Apple'));

  // Scenario 4: Specific absent item -> explicitly states Found In User's Saved Pantry: FALSE
  const absentBundle = buildContextBundle({
    userMessage: 'Do I have avocados in my pantry?',
    pantryItems: samplePantry,
  });
  assert.equal(absentBundle.contextLevel, 'medium');
  assert.equal(absentBundle.matchedItems.length, 0);
  const absentPrompt = absentBundle.messages[absentBundle.messages.length - 1].content;
  assert.ok(absentPrompt.includes('Found In User\'s Saved Pantry: FALSE'));

  // Scenario 5: Eat First deterministic ordering + null remainingDays safety
  const triage = triagePantryItems([
    ...samplePantry,
    { foodName: 'Mystery Spice', status: 'Fresh', remainingDays: null, qualityScore: 50 },
  ]);
  assert.equal(triage.topUrgent?.foodName, 'Mango');
  const mystery = triage.rankedItems.find((i) => i.foodName === 'Mystery Spice');
  assert.equal(mystery?.remainingDays, null);
  assert.equal(mystery?.qualityPeriod, 'remaining quality unavailable');

  // Application authority enforcement prevents model from overriding Eat First winner
  const guarded = enforceApplicationAuthority(
    'You could try eating the Apple today.',
    'strong',
    FreshoIntent.PANTRY_PRIORITY,
    triage,
    [],
    null,
  );
  assert.ok(guarded.startsWith('**Application Eat First Priority:** **Mango**'));

  // Scenario 6: Empty pantry Eat First
  const emptyBundle = buildContextBundle({
    userMessage: 'Which food should I consume first?',
    pantryItems: [],
  });
  assert.equal(emptyBundle.contextLevel, 'strong');
  assert.ok(emptyBundle.messages[emptyBundle.messages.length - 1].content.includes('EMPTY (0 saved food items)'));

  // Scenario 7a: Follow-up after pantry Eat First -> MEDIUM & preserves Mango
  const pantryFollow = buildContextBundle({
    userMessage: 'Why that one?',
    history: [
      { role: 'user', content: 'Which food should I consume first?' },
      { role: 'assistant', content: 'Based on your saved pantry history, **Mango should be used first** (0 days remaining).' },
    ],
    pantryItems: samplePantry,
  });
  assert.equal(pantryFollow.freshoIntent, FreshoIntent.FOLLOW_UP);
  assert.equal(pantryFollow.contextLevel, 'medium');
  assert.equal(pantryFollow.targetFood, 'mango');
  assert.equal(pantryFollow.matchedItems[0]?.foodName, 'Mango');

  // Scenario 7b: Follow-up after SIMPLE educational chat -> stays SIMPLE & never leaks pantry
  const simpleFollow = buildContextBundle({
    userMessage: 'How should I store it?',
    history: [
      { role: 'user', content: 'Average shelf life of freshly purchased bananas' },
      { role: 'assistant', content: 'Fresh **bananas** usually last 2–5 days on the countertop.' },
    ],
    pantryItems: samplePantry,
  });
  assert.equal(simpleFollow.freshoIntent, FreshoIntent.FOLLOW_UP);
  assert.equal(simpleFollow.contextLevel, 'simple');
  assert.equal(simpleFollow.pantryInjected, false);
});

test('Official Groq SDK inference, configurable model, and honest error handling (no mock fallback)', async () => {
  // Configurable model & GROK_API_KEY alias check
  const customCfg = getGroqConfig({ GROQ_API_KEY: 'gsk_k', GROQ_MODEL: 'llama-3.1-8b-instant', GROQ_VISION_MODEL: 'meta-llama/llama-4-maverick-17b-128e-instruct' });
  assert.equal(customCfg.model, 'llama-3.1-8b-instant');
  assert.equal(customCfg.visionModel, 'meta-llama/llama-4-maverick-17b-128e-instruct');
  assert.equal(customCfg.provider, 'groq');
  assert.equal(getGroqConfig({ GROQ_API_KEY: 'gsk_k' }).model, 'llama-3.3-70b-versatile');
  assert.equal(getGroqConfig({ GROQ_API_KEY: 'gsk_k' }).visionModel, 'meta-llama/llama-4-scout-17b-16e-instruct');

  const grokAliasCfg = getGroqConfig({ GROK_API_KEY: 'gsk_alias' });
  assert.equal(grokAliasCfg.apiKey, 'gsk_alias');
  assert.equal(grokAliasCfg.provider, 'groq');

  const xaiCfg = getGroqConfig({ GROK_API_KEY: 'xai-test-key' });
  assert.equal(xaiCfg.provider, 'grok');
  assert.equal(xaiCfg.model, 'grok-2-latest');
  assert.equal(xaiCfg.visionModel, 'grok-2-vision-1212');

  // Scenario 10a: Missing GROQ_API_KEY raises 503 GROQ_UNAVAILABLE with zero mock response
  await assert.rejects(
    () =>
      generateFreshoBuddyGroqReply({
        userMessage: 'Average shelf life of freshly purchased bananas',
        env: { GROQ_API_KEY: '' },
      }),
    (err: any) => err instanceof FreshoBuddyGroqError && err.code === 'GROQ_UNAVAILABLE' && err.statusCode === 503,
  );

  // Scenario 10b: Invalid GROQ_API_KEY (401) raises 502 GROQ_AUTH_ERROR
  const badStub = await startStubGroqServer(() => ({
    status: 401,
    json: { error: { message: 'Invalid API Key', type: 'invalid_request_error' } },
  }));
  try {
    await assert.rejects(
      () =>
        generateFreshoBuddyGroqReply({
          userMessage: 'Average shelf life of freshly purchased bananas',
          env: {
            GROQ_API_KEY: 'gsk_invalid_test_key',
            GROQ_MODEL: 'llama-3.3-70b-versatile',
            GROQ_BASE_URL: badStub.baseUrl,
          },
        }),
      (err: any) => err instanceof FreshoBuddyGroqError && err.code === 'GROQ_AUTH_ERROR' && err.statusCode === 502,
    );
  } finally {
    await badStub.close();
  }

  // Happy path round-trip through official Groq SDK -> verifies configurable model & prompt
  const okStub = await startStubGroqServer((body) => ({
    status: 200,
    json: {
      id: 'chatcmpl_test',
      object: 'chat.completion',
      model: body.model,
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'Fresh **bananas** typically last **2–5 days** at room temperature.',
          },
          finish_reason: 'stop',
        },
      ],
    },
  }));
  try {
    const reply = await generateFreshoBuddyGroqReply({
      userMessage: 'Average shelf life of freshly purchased bananas',
      pantryItems: [{ foodName: 'Secret Mango', status: 'Fresh', remainingDays: 2 }],
      env: {
        GROQ_API_KEY: 'gsk_valid_stub',
        GROQ_MODEL: 'custom-groq-model-test',
        GROQ_BASE_URL: okStub.baseUrl,
      },
    });
    assert.equal(reply.model, 'custom-groq-model-test');
    assert.equal(reply.contextLevel, 'simple');
    assert.equal(reply.pantryInjected, false);
    assert.ok(reply.text.includes('2–5 days'));
    assert.equal(okStub.captured.length, 1);
    assert.equal(okStub.captured[0].body.model, 'custom-groq-model-test');
    assert.equal(okStub.captured[0].authHeader, 'Bearer gsk_valid_stub');
    assert.ok(!JSON.stringify(okStub.captured[0].body.messages).includes('Secret Mango'));
  } finally {
    await okStub.close();
  }
});

test('SQLite persistence across reconnects, qualityScore=0 preservation, and strict 2-user isolation', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'foodfresh-test-'));
  const dbPath = path.join(tmpDir, 'test_fresho.db');
  const prevDbEnv = process.env.FOODFRESH_DB_PATH;
  process.env.FOODFRESH_DB_PATH = dbPath;
  resetDbConnectionForTests();

  try {
    // Register two users
    const userA = registerUser('alice@example.com', 'password123', 'Alice Kitchen');
    const userB = registerUser('bob@example.com', 'password456', 'Bob Kitchen');

    // Login & session token verification
    const loginA = loginUser('alice@example.com', 'password123');
    assert.equal(resolveSession(loginA.token)?.id, userA.user.id);

    // Save items including qualityScore=0 for User A
    const savedMango = addPantryItem(userA.user.id, {
      foodName: 'Mango',
      status: 'Rotten',
      qualityScore: 0,
      remainingDays: 0,
      storageEnvironment: 'Countertop Ambient',
    });
    assert.equal(savedMango.qualityScore, 0, 'qualityScore=0 must not fall back to 80');

    // User A creates a conversation & adds messages
    const convA = createConversation(userA.user.id, deriveConversationTitle('Which food should I eat first?'));
    addMessage(convA.conversationId, userA.user.id, 'user', 'Which food should I eat first?');
    addMessage(convA.conversationId, userA.user.id, 'assistant', '**Mango should be used first** (0 days remaining).');

    // Scenario 9: User B isolation
    assert.equal(getCurrentPantryState(userB.user.id).length, 0);
    assert.equal(listConversations(userB.user.id).length, 0);
    assert.equal(getConversation(convA.conversationId, userB.user.id), null);
    assert.equal(deleteConversation(convA.conversationId, userB.user.id), false);

    // Scenario 8: Simulate server restart / reconnect -> SQLite data persists!
    resetDbConnectionForTests();
    const pantryAfterRestart = getCurrentPantryState(userA.user.id);
    assert.equal(pantryAfterRestart.length, 1);
    assert.equal(pantryAfterRestart[0].foodName, 'Mango');
    assert.equal(pantryAfterRestart[0].qualityScore, 0);

    const convsAfterRestart = listConversations(userA.user.id);
    assert.equal(convsAfterRestart.length, 1);
    assert.equal(convsAfterRestart[0].title, 'Pantry Priority');

    const fullConv = getConversation(convA.conversationId, userA.user.id);
    assert.equal(fullConv?.messages.length, 2);

    // Delete conversation cascades messages cleanly
    assert.equal(deleteConversation(convA.conversationId, userA.user.id), true);
    assert.equal(listConversations(userA.user.id).length, 0);
  } finally {
    resetDbConnectionForTests();
    if (prevDbEnv === undefined) {
      delete process.env.FOODFRESH_DB_PATH;
    } else {
      process.env.FOODFRESH_DB_PATH = prevDbEnv;
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
