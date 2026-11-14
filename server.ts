import 'dotenv/config';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import multer from 'multer';
import Groq from 'groq-sdk';
import {
  addMessage,
  addPantryItem,
  clearPantryItems,
  createConversation,
  createFeedbackRecord,
  deleteConversation,
  deletePantryItem,
  deriveConversationTitle,
  estimateShelfLife,
  evaluateEatFirstPriority,
  evaluateFreshnessPolicy,
  getConversation,
  getCurrentPantryState,
  getDb,
  inferFoodForm,
  listConversations,
  listFeedbackForUser,
  loginUser,
  logoutUser,
  registerUser,
  resolveSession,
  updateConversationTitle,
  type PublicUser,
} from './server/foodServices.ts';
import {
  FreshoBuddyGroqError,
  generateFreshoBuddyGroqReply,
  getGroqConfig,
} from './server/freshoBuddyLogic.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const HOST = '0.0.0.0';

function parseVisionJson(rawText: string): any {
  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/i, '')
    .trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start >= 0 && end > start) {
    return JSON.parse(cleaned.slice(start, end + 1));
  }
  return JSON.parse(cleaned || '{}');
}

async function analyzeFoodImageWithVision(params: {
  mimeType: string;
  base64Data: string;
  prompt: string;
}): Promise<{ parsed: any; model: string; provider: 'groq' | 'grok' }> {
  const cfg = getGroqConfig();
  if (!cfg.apiKey) {
    const err: any = new Error('GROQ_API_KEY or GROK_API_KEY is not configured.');
    err.code = 'VISION_UNCONFIGURED';
    err.status = 503;
    throw err;
  }

  const messages: any[] = [
    {
      role: 'system',
      content: 'You are a food visual analysis engine. Always respond with valid JSON only.',
    },
    {
      role: 'user',
      content: [
        { type: 'text', text: params.prompt },
        {
          type: 'image_url',
          image_url: {
            url: `data:${params.mimeType};base64,${params.base64Data}`,
          },
        },
      ],
    },
  ];

  let rawText = '';
  if (cfg.provider === 'grok' && !cfg.baseUrl) {
    const resp = await fetch('https://api.x.ai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${cfg.apiKey}`,
      },
      body: JSON.stringify({
        model: cfg.visionModel,
        messages,
        response_format: { type: 'json_object' },
        temperature: 0.1,
        max_tokens: 700,
      }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      const err: any = new Error(errText || `Vision HTTP ${resp.status}`);
      err.status = resp.status;
      throw err;
    }
    const data: any = await resp.json();
    rawText = String(data?.choices?.[0]?.message?.content || '{}').trim();
  } else {
    const client = new Groq({
      apiKey: cfg.apiKey,
      ...(cfg.baseUrl ? { baseURL: cfg.baseUrl } : {}),
      timeout: 30_000,
      maxRetries: 1,
    });
    const callVision = async (modelId: string) => {
      const completion = await client.chat.completions.create({
        model: modelId,
        messages,
        response_format: { type: 'json_object' },
        temperature: 0.1,
        max_tokens: 700,
      });
      return String(completion?.choices?.[0]?.message?.content || '{}').trim();
    };

    try {
      rawText = await callVision(cfg.visionModel);
    } catch (primaryErr: any) {
      const pStatus = Number(primaryErr?.status || primaryErr?.statusCode || 0);
      const pMsg = String(primaryErr?.message || '').toLowerCase();
      const isModelMissing =
        !cfg.baseUrl && (pStatus === 404 || pMsg.includes('model_not_found') || pMsg.includes('does not exist'));
      if (!isModelMissing) {
        throw primaryErr;
      }
      const list = await client.models.list().catch(() => ({ data: [] as Array<{ id?: string }> }));
      const availableIds = (list?.data || []).map((m) => String(m?.id || '')).filter(Boolean);
      const visionCandidates = [
        'meta-llama/llama-4-scout-17b-16e-instruct',
        'meta-llama/llama-4-maverick-17b-128e-instruct',
        'llama-3.2-90b-vision-preview',
        'llama-3.2-11b-vision-preview',
        ...availableIds.filter((id) => /scout|maverick|vision|multimodal|qwen3\.8|qwen/i.test(id)),
      ].filter((id, idx, arr) => id !== cfg.visionModel && arr.indexOf(id) === idx && availableIds.includes(id));

      if (visionCandidates.length === 0) {
        const noVisionErr: any = new Error(
          `No vision-capable model is currently enabled on your GROQ_API_KEY (available models: ${availableIds.join(', ') || 'none'}). Enable a vision model such as meta-llama/llama-4-scout-17b-16e-instruct in Groq Console or configure GROQ_VISION_MODEL / GROK_API_KEY.`,
        );
        noVisionErr.status = 404;
        noVisionErr.code = 'NO_VISION_MODEL';
        throw noVisionErr;
      }
      let lastErr: any = primaryErr;
      let succeeded = false;
      for (const candidate of visionCandidates) {
        try {
          rawText = await callVision(candidate);
          cfg.visionModel = candidate;
          succeeded = true;
          break;
        } catch (candErr: any) {
          lastErr = candErr;
        }
      }
      if (!succeeded) {
        const errMsg = String(lastErr?.message || '');
        const noVisionErr: any = new Error(
          `None of the models enabled on your GROQ_API_KEY (${availableIds.join(', ') || 'none'}) supported multimodal vision analysis (${errMsg}). Please enable meta-llama/llama-4-scout-17b-16e-instruct in Groq Console or configure a vision-enabled GROQ_API_KEY / GROK_API_KEY.`,
        );
        noVisionErr.status = Number(lastErr?.status || 502);
        noVisionErr.code = 'NO_VISION_MODEL';
        throw noVisionErr;
      }
    }
  }

  return {
    parsed: parseVisionJson(rawText),
    model: cfg.visionModel,
    provider: cfg.provider,
  };
}

interface AuthedRequest extends Request {
  user?: PublicUser;
}

function requireAuth(req: AuthedRequest, res: Response, next: NextFunction): void {
  const authHeader = req.headers.authorization || '';
  const match = authHeader.match(/^Bearer\s+(.+)$/i);
  const token = match?.[1]?.trim();
  if (!token) {
    res.status(401).json({ detail: 'Authentication required.' });
    return;
  }
  const user = resolveSession(token);
  if (!user) {
    res.status(401).json({ detail: 'Invalid or expired session token.' });
    return;
  }
  req.user = user;
  next();
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
});

async function startServer() {
  getDb();
  const app = express();

  app.set('trust proxy', 1);

  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Vary', 'Origin');
    } else {
      res.setHeader('Access-Control-Allow-Origin', '*');
    }
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    if (req.method === 'OPTIONS') {
      res.status(204).end();
      return;
    }
    next();
  });

  app.use(express.json({ limit: '15mb' }));
  app.use(express.urlencoded({ extended: true, limit: '15mb' }));

  // --------------------------------------------------------------------------
  // HEALTH
  // --------------------------------------------------------------------------
  app.get('/api/health', async (req, res) => {
    const groqCfg = getGroqConfig();
    const groqReady = Boolean(groqCfg.apiKey);
    const visionPipeline = groqCfg.provider === 'grok' ? 'xai_grok_vision' : 'groq_multimodal_vision';
    let availableModels: string[] | undefined;
    if (req.query.probe === '1' && groqReady && groqCfg.provider === 'groq') {
      try {
        const client = new Groq({ apiKey: groqCfg.apiKey, timeout: 10_000 });
        const list = await client.models.list();
        availableModels = (list?.data || []).map((m) => String(m?.id || '')).filter(Boolean);
      } catch {
        availableModels = [];
      }
    }
    res.json({
      status: 'ok',
      service: 'FoodFresh AI backend',
      pipeline: 'Groq Vision Analysis + USDA FoodKeeper Rules + Groq FreshoBuddy',
      ...(availableModels !== undefined ? { availableModels } : {}),
      models: {
        foodRecognition: {
          status: groqReady ? 'ACTIVE' : 'UNCONFIGURED',
          loaded: groqReady,
          pipeline: visionPipeline,
          provider: groqCfg.provider,
          model: groqCfg.visionModel,
        },
        freshness: {
          status: groqReady ? 'ACTIVE' : 'UNCONFIGURED',
          loaded: groqReady,
          version: groqCfg.visionModel,
        },
        shelfLife: {
          status: 'ACTIVE',
          loaded: true,
          source: 'USDA FoodKeeper',
        },
        eatFirstPriority: {
          status: 'ACTIVE',
          loaded: true,
          engine: 'Deterministic Priority Rules',
        },
        freshoBuddy: {
          status: groqReady ? 'ACTIVE' : 'UNCONFIGURED',
          loaded: groqReady,
          provider: groqCfg.provider,
          model: groqCfg.model,
        },
      },
    });
  });

  // --------------------------------------------------------------------------
  // AUTHENTICATION
  // --------------------------------------------------------------------------
  app.post('/api/auth/register', (req, res) => {
    try {
      const { fullName, email, password } = req.body || {};
      const result = registerUser(email, password, fullName);
      res.json({ success: true, token: result.token, user: result.user });
    } catch (err: any) {
      res.status(400).json({ detail: err?.message || 'Registration failed.' });
    }
  });

  app.post('/api/auth/login', (req, res) => {
    try {
      const { email, password } = req.body || {};
      const result = loginUser(email, password);
      res.json({ success: true, token: result.token, user: result.user });
    } catch (err: any) {
      res.status(401).json({ detail: err?.message || 'Invalid email or password.' });
    }
  });

  app.get('/api/auth/me', requireAuth, (req: AuthedRequest, res) => {
    res.json({ success: true, user: req.user });
  });

  app.post('/api/auth/logout', (req, res) => {
    const authHeader = req.headers.authorization || '';
    const match = authHeader.match(/^Bearer\s+(.+)$/i);
    if (match?.[1]) {
      logoutUser(match[1]);
    }
    res.json({ success: true, message: 'Logged out.' });
  });

  // --------------------------------------------------------------------------
  // FOOD RECOGNITION & FRESHNESS + SHELF LIFE + EAT FIRST
  // --------------------------------------------------------------------------
  function hasValidImageMagicBytes(buf: Buffer): boolean {
    if (buf.length < 4) return false;
    // JPEG
    if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return true;
    // PNG
    if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return true;
    // GIF
    if (buf[0] === 0x47 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x38) return true;
    // BMP
    if (buf[0] === 0x42 && buf[1] === 0x4d) return true;
    // WEBP (RIFF....WEBP)
    if (
      buf.length >= 12 &&
      buf.subarray(0, 4).toString('ascii') === 'RIFF' &&
      buf.subarray(8, 12).toString('ascii') === 'WEBP'
    ) {
      return true;
    }
    // ISO BMFF / HEIC / AVIF
    if (buf.length >= 12 && buf.subarray(4, 8).toString('ascii') === 'ftyp') return true;
    return false;
  }

  app.post('/api/food-recognition/predict', upload.single('file'), async (req, res) => {
    const startMs = Date.now();
    const file = req.file;
    if (!file || !file.buffer || file.buffer.length === 0) {
      res.status(400).json({
        success: false,
        status: 'invalid_image',
        message: 'No image file provided. Please upload or capture a food image.',
      });
      return;
    }

    if (!hasValidImageMagicBytes(file.buffer)) {
      res.status(400).json({
        success: false,
        status: 'invalid_image',
        message: 'Please upload a valid food image (JPG, JPEG, PNG, or WEBP).',
      });
      return;
    }

    const mimeType = file.mimetype && file.mimetype.startsWith('image/') ? file.mimetype : 'image/jpeg';
    const storageType = String(req.body?.storage_type || 'countertop');
    const daysStored = Math.max(0, Number(req.body?.days_stored ?? 0) || 0);
    const topK = Math.min(10, Math.max(1, Number(req.query?.top_k ?? 3) || 3));

    const groqCfg = getGroqConfig();
    if (!groqCfg.apiKey) {
      res.status(503).json({
        success: false,
        status: 'model_unavailable',
        detail: 'Food analysis AI model is not configured. Ensure GROQ_API_KEY (or GROK_API_KEY) is available in Secrets.',
        message: 'Food analysis model is currently unavailable.',
      });
      return;
    }

    try {
      const base64Data = file.buffer.toString('base64');
      const prompt = `Analyze this image for food identification and visible freshness assessment. Return ONLY a JSON object with this exact schema:
{
  "isFoodPresent": boolean,
  "detectedFood": string,
  "foodForm": "fresh" | "whole" | "cut" | "sliced" | "diced" | "dried" | "powdered" | "cooked" | "processed" | "unknown",
  "recognitionConfidence": number (0-100),
  "topPredictions": [{ "food": string, "confidence": number (0-100), "category": string }] (up to ${topK} items),
  "detectedObjects": [{ "label": string, "confidence": number (0-100) }],
  "freshnessLabel": "Fresh" | "Slightly Spoiled" | "Rotten",
  "freshnessScore": number (0-100 confidence that freshnessLabel is accurate, e.g. 95 if clearly Fresh OR 95 if clearly Rotten)
}
Identify the primary food item using clear common English names such as Apple, Banana, Tomato, Orange, Bell Pepper, Green Chilli, Potato, Onion, Carrot, Bread, Avocado, Mango, Pomegranate, Strawberry, Grapes, Cucumber, Spinach, Broccoli, Lemon.`;

      const vision = await analyzeFoodImageWithVision({
        mimeType,
        base64Data,
        prompt,
      });
      const parsed = vision.parsed || {};
      const activeVisionModel = vision.model;

      if (parsed.isFoodPresent === false || !parsed.detectedFood) {
        res.json({
          success: true,
          analysisStatus: 'low_confidence',
          detectedFood: null,
          recognitionConfidence: parsed.recognitionConfidence ?? 0,
          topPredictions: [],
          detectedObjects: (parsed.detectedObjects || []).slice(0, 5),
          freshness: {
            status: 'unavailable',
            label: 'Freshness assessment unavailable.',
            confidence: null,
            score: null,
            freshnessUncertain: true,
            source: activeVisionModel,
          },
          shelfLife: {
            status: 'unavailable',
            reason: 'Food was not identified with sufficient certainty for FoodKeeper reference lookup.',
          },
          eatFirstPriority: {
            status: 'unavailable',
            reason: 'Eat First requires an available shelf-life estimate.',
          },
          source: 'ml',
          model: 'FoodFresh Hybrid Vision',
          modelVersion: activeVisionModel,
          status: 'low_confidence',
          latencyMs: Date.now() - startMs,
          message: 'Could not identify food in this photo with sufficient certainty.',
        });
        return;
      }

      const detectedFood = String(parsed.detectedFood).trim();
      const recConf =
        typeof parsed.recognitionConfidence === 'number' && Number.isFinite(parsed.recognitionConfidence)
          ? Math.min(100, Math.max(0, Number(parsed.recognitionConfidence)))
          : 0;
      const isLowConf = recConf < 50;

      const foodForm = parsed.foodForm
        ? inferFoodForm(`${parsed.foodForm} ${detectedFood}`, null)
        : inferFoodForm(detectedFood, null);

      const freshnessResult = evaluateFreshnessPolicy({
        detectedFood,
        foodForm,
        rawFreshnessLabel: parsed.freshnessLabel,
        rawFreshnessScore: parsed.freshnessScore,
        isLowConfidence: isLowConf,
        source: activeVisionModel,
      });

      const topPredictions = (Array.isArray(parsed.topPredictions) && parsed.topPredictions.length
        ? parsed.topPredictions
        : [{ food: detectedFood, confidence: recConf, category: 'Food' }]
      )
        .slice(0, topK)
        .map((p: any) => {
          const conf =
            typeof p.confidence === 'number' && Number.isFinite(p.confidence)
              ? Math.min(100, Math.max(0, Number(p.confidence)))
              : recConf;
          return {
            food: String(p.food || detectedFood),
            confidence: conf,
            percentage: `${conf.toFixed(1)}%`,
            category: String(p.category || 'Food'),
          };
        });

      const shelfLife = isLowConf
        ? {
            status: 'unavailable',
            reason: 'Food was not identified with sufficient certainty for FoodKeeper reference lookup.',
            isEstimate: false,
            isMlRegression: false,
          }
        : estimateShelfLife(
            detectedFood,
            storageType,
            daysStored,
            freshnessResult.status === 'success' ? freshnessResult.label : null,
            foodForm,
          );

      const eatFirstPriority = evaluateEatFirstPriority(
        shelfLife,
        freshnessResult.status === 'success' ? freshnessResult.label : null,
      );

      const rem = shelfLife.remaining || {};
      res.json({
        success: true,
        analysisStatus: isLowConf ? 'low_confidence' : 'complete',
        detectedFood: isLowConf ? null : detectedFood,
        foodForm,
        recognitionConfidence: recConf,
        topPredictions,
        detectedObjects: (parsed.detectedObjects || []).slice(0, 5).map((o: any) => ({
          label: String(o.label || ''),
          confidence: typeof o.confidence === 'number' ? o.confidence : null,
          box: null,
        })),
        foodRecognition: {},
        freshness: freshnessResult,
        shelfLife,
        shelfLifeAvailable: shelfLife.status === 'available',
        shelfLifeSource: shelfLife.source || 'USDA FoodKeeper',
        shelfLifeModelVersion: 'foodkeeper_rules_v1',
        estimatedQualityDays: rem.maxDays ?? rem.minDays ?? null,
        eatFirstPriority,
        modelVersions: {
          detector: activeVisionModel,
          semantic: activeVisionModel,
          freshness: activeVisionModel,
          shelfLife: 'foodkeeper_rules_v1',
          food: activeVisionModel,
        },
        source: 'ml',
        model: 'FoodFresh Hybrid Vision',
        modelVersion: activeVisionModel,
        status: isLowConf ? 'low_confidence' : 'complete',
        latencyMs: Date.now() - startMs,
        message: isLowConf
          ? 'Recognition confidence was below 50%. Please try a clearer angle.'
          : `Recognized as ${detectedFood} with ${recConf.toFixed(1)}% confidence.`,
      });
    } catch (err: any) {
      console.error('Food recognition error:', err);
      const status = Number(err?.status || err?.statusCode || 0);
      const rawSafeMsg = String(err?.message || '').replace(/(gsk_|xai-)[A-Za-z0-9_-]+/g, '[REDACTED]');
      const rawMsg = rawSafeMsg.toLowerCase();
      const isAuth =
        status === 401 ||
        status === 403 ||
        rawMsg.includes('invalid api key') ||
        rawMsg.includes('unauthorized');
      const isQuotaOrRateLimit =
        status === 429 ||
        rawMsg.includes('rate limit') ||
        rawMsg.includes('quota') ||
        rawMsg.includes('resource_exhausted');
      const isNoVisionModel = err?.code === 'NO_VISION_MODEL' || status === 404 || rawMsg.includes('model_not_found');
      const friendlyMessage = isAuth
        ? 'Groq/Grok API authentication failed. Please check GROQ_API_KEY (or GROK_API_KEY) configuration.'
        : isQuotaOrRateLimit
          ? 'Vision analysis API quota or rate limit exceeded. Please wait before retrying.'
          : isNoVisionModel
            ? rawSafeMsg
            : `Food analysis could not be completed: ${rawSafeMsg || 'Please try a clearer image or try again.'}`;
      res.status(isQuotaOrRateLimit ? 429 : 502).json({
        success: false,
        status: isQuotaOrRateLimit ? 'rate_limited' : isAuth ? 'auth_error' : 'pipeline_error',
        detail: friendlyMessage,
        message: friendlyMessage,
      });
    }
  });

  // --------------------------------------------------------------------------
  // PANTRY HISTORY (USER-ISOLATED)
  // --------------------------------------------------------------------------
  app.get('/api/history', requireAuth, (req: AuthedRequest, res) => {
    const items = getCurrentPantryState(req.user!.id);
    res.json({ success: true, items, count: items.length });
  });

  app.get('/api/history/pantry', requireAuth, (req: AuthedRequest, res) => {
    const items = getCurrentPantryState(req.user!.id);
    res.json({ success: true, items, count: items.length });
  });

  app.post('/api/history', requireAuth, (req: AuthedRequest, res) => {
    const item = req.body?.item;
    if (!item || typeof item !== 'object') {
      res.status(400).json({ detail: 'Invalid pantry item payload.' });
      return;
    }
    const saved = addPantryItem(req.user!.id, item);
    res.json({ success: true, item: saved });
  });

  app.delete('/api/history/:itemId', requireAuth, (req: AuthedRequest, res) => {
    const deleted = deletePantryItem(String(req.params.itemId), req.user!.id);
    if (!deleted) {
      res.status(404).json({ detail: 'Pantry item not found or unauthorized.' });
      return;
    }
    res.json({ success: true, message: 'Pantry item removed.' });
  });

  app.delete('/api/history', requireAuth, (req: AuthedRequest, res) => {
    clearPantryItems(req.user!.id);
    res.json({ success: true, message: 'All pantry items cleared.' });
  });

  // --------------------------------------------------------------------------
  // FRESHO BUDDY CONVERSATIONS & GROQ CHAT (USER-ISOLATED)
  // --------------------------------------------------------------------------
  app.get('/api/fresho-buddy/conversations', requireAuth, (req: AuthedRequest, res) => {
    const conversations = listConversations(req.user!.id);
    res.json({ success: true, conversations });
  });

  app.post('/api/fresho-buddy/conversations', requireAuth, (req: AuthedRequest, res) => {
    const title = req.body?.title || 'New Food Discussion';
    const conversation = createConversation(req.user!.id, title);
    res.json({ success: true, conversation });
  });

  app.get('/api/fresho-buddy/conversations/:conversationId', requireAuth, (req: AuthedRequest, res) => {
    const conv = getConversation(String(req.params.conversationId), req.user!.id);
    if (!conv) {
      res.status(404).json({ detail: 'Conversation not found or access denied.' });
      return;
    }
    res.json({ success: true, conversation: conv });
  });

  app.delete('/api/fresho-buddy/conversations/:conversationId', requireAuth, (req: AuthedRequest, res) => {
    const deleted = deleteConversation(String(req.params.conversationId), req.user!.id);
    if (!deleted) {
      res.status(404).json({ detail: 'Conversation not found or could not be deleted.' });
      return;
    }
    res.json({ success: true, message: 'Conversation deleted.' });
  });

  app.post('/api/fresho-buddy/chat', requireAuth, async (req: AuthedRequest, res) => {
    const userId = req.user!.id;
    const message = String(req.body?.message || '').trim();
    const analysisContext = req.body?.analysisContext || null;
    const requestedConvId: string | null = req.body?.conversationId || null;

    if (!message) {
      res.status(400).json({ detail: 'Message cannot be empty.' });
      return;
    }

    const existingConv = requestedConvId ? getConversation(requestedConvId, userId) : null;
    const priorHistory = existingConv ? existingConv.messages : [];

    let groqResult: Awaited<ReturnType<typeof generateFreshoBuddyGroqReply>>;
    try {
      groqResult = await generateFreshoBuddyGroqReply({
        userMessage: message,
        history: priorHistory,
        analysisContext,
        getPantryItems: () => getCurrentPantryState(userId),
      });
    } catch (err: any) {
      if (err instanceof FreshoBuddyGroqError) {
        res.status(err.statusCode).json({
          success: false,
          code: err.code,
          detail: err.message,
        });
        return;
      }
      res.status(503).json({
        success: false,
        code: 'GROQ_API_ERROR',
        detail: err?.message || 'FreshoBuddy Groq inference failed.',
      });
      return;
    }

    let conv = existingConv;
    if (!conv) {
      const title = deriveConversationTitle(message, analysisContext);
      conv = createConversation(userId, title);
    } else if (conv.title === 'New Food Discussion') {
      const newTitle = deriveConversationTitle(message, analysisContext);
      updateConversationTitle(conv.conversationId, userId, newTitle);
      conv.title = newTitle;
    }

    const userMsgRecord = addMessage(conv.conversationId, userId, 'user', message, analysisContext);
    const botMsgRecord = addMessage(conv.conversationId, userId, 'assistant', groqResult.text, analysisContext);

    res.json({
      success: true,
      conversationId: conv.conversationId,
      title: conv.title,
      contextLevel: groqResult.contextLevel,
      intent: groqResult.intent,
      userMessage: userMsgRecord,
      reply: {
        id: botMsgRecord.messageId,
        sender: 'bot',
        text: groqResult.text,
        status: groqResult.status,
        model: groqResult.model,
        contextLevel: groqResult.contextLevel,
        intent: groqResult.intent,
        timestamp: botMsgRecord.createdAt,
      },
    });
  });

  // --------------------------------------------------------------------------
  // ML FEEDBACK
  // --------------------------------------------------------------------------
  app.post('/api/ml-feedback', requireAuth, (req: AuthedRequest, res) => {
    const record = createFeedbackRecord(req.user!.id, req.body || {});
    res.json({ success: true, feedback: record });
  });

  app.get('/api/ml-feedback', requireAuth, (req: AuthedRequest, res) => {
    const items = listFeedbackForUser(req.user!.id);
    res.json({ success: true, items });
  });

  const httpServer = http.createServer(app);

  // --------------------------------------------------------------------------
  // VITE MIDDLEWARE (DEV) OR STATIC DIST (PROD)
  // --------------------------------------------------------------------------
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const hmrDisabled = process.env.DISABLE_HMR === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        allowedHosts: true,
        ...(hmrDisabled ? { ws: false as const } : {}),
        hmr: hmrDisabled ? false : { server: httpServer },
        watch: hmrDisabled ? null : {},
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  httpServer.listen(PORT, HOST, () => {
    console.log(`FoodFresh AI server listening on http://${HOST}:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
