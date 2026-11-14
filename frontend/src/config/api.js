/**
 * FoodFresh AI - Frontend API Configuration
 * Supports environment overrides and falls back to proxy / direct FastAPI backend.
 */

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '0.0.0.0']);

/**
 * @param {unknown} rawValue
 * @param {{ protocol?: string, hostname?: string, port?: string } | undefined} [loc]
 * @returns {string}
 */
export function normalizeApiBaseUrl(rawValue, loc = typeof window !== 'undefined' ? window.location : undefined) {
  if (typeof rawValue !== 'string') return '';
  let trimmed = rawValue.trim().replace(/\/+$/, '');
  if (!trimmed) return '';
  if (trimmed === '/api') return '';
  trimmed = trimmed.replace(/\/api$/i, '');
  if (!trimmed) return '';

  const locHost = (loc?.hostname || '').toLowerCase();
  const isRemoteHost = Boolean(locHost && !LOCAL_HOSTS.has(locHost));

  // Support bare port values (e.g. VITE_API_BASE_URL=3000 or :3000)
  const portMatch = trimmed.match(/^:?(\d{2,5})$/);
  if (portMatch) {
    const port = portMatch[1];
    if (loc) {
      const protocol = loc.protocol || 'http:';
      const hostname = loc.hostname || 'localhost';
      const currentPort = loc.port || '';
      if (isRemoteHost || currentPort === port || (!currentPort && protocol === 'https:')) {
        return '';
      }
      return `${protocol}//${hostname}:${port}`;
    }
    return `http://localhost:${port}`;
  }

  const normalized = !/^https?:\/\//i.test(trimmed) && !trimmed.startsWith('/') ? `http://${trimmed}` : trimmed;

  // If deployed/previewed on a hosted domain (*.run.app), never let localhost override point away from same-origin /api/*
  if (isRemoteHost && /^https?:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$/i.test(normalized)) {
    return '';
  }

  return normalized;
}

const rawApiBaseUrl = normalizeApiBaseUrl(import.meta?.env?.VITE_API_BASE_URL);

export const API_BASE_URL = rawApiBaseUrl;

export const ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/health`,
  AUTH_LOGIN: `${API_BASE_URL}/api/auth/login`,
  AUTH_REGISTER: `${API_BASE_URL}/api/auth/register`,
  AUTH_LOGOUT: `${API_BASE_URL}/api/auth/logout`,
  AUTH_ME: `${API_BASE_URL}/api/auth/me`,
  FOOD_RECOGNITION: `${API_BASE_URL}/api/food-recognition/predict`,
  FRESHO_BUDDY_CHAT: `${API_BASE_URL}/api/fresho-buddy/chat`,
  FRESHO_BUDDY_CONVERSATIONS: `${API_BASE_URL}/api/fresho-buddy/conversations`,
  PANTRY_HISTORY: `${API_BASE_URL}/api/history`,
};
