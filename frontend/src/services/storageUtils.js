/**
 * User storage utility to ensure complete data isolation between user accounts,
 * with safe in-memory fallback when third-party iframe storage is restricted.
 */

const AUTH_STORAGE_KEY = 'foodfresh_ai_session';
const AUTH_TOKEN_KEY = 'foodfresh_ai_token';

const memoryLocal = new Map();
const memorySession = new Map();

export function safeLocalGet(key) {
  try {
    const v = localStorage.getItem(key);
    if (v !== null) return v;
  } catch {
    // ignore storage restriction in embedded iframes
  }
  return memoryLocal.has(key) ? memoryLocal.get(key) : null;
}

export function safeLocalSet(key, value) {
  memoryLocal.set(key, String(value));
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // ignore storage restriction in embedded iframes
  }
}

export function safeLocalRemove(key) {
  memoryLocal.delete(key);
  try {
    localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

export function safeSessionGet(key) {
  try {
    const v = sessionStorage.getItem(key);
    if (v !== null) return v;
  } catch {
    // ignore storage restriction in embedded iframes
  }
  return memorySession.has(key) ? memorySession.get(key) : null;
}

export function safeSessionSet(key, value) {
  memorySession.set(key, String(value));
  try {
    sessionStorage.setItem(key, String(value));
  } catch {
    // ignore storage restriction in embedded iframes
  }
}

export function safeSessionRemove(key) {
  memorySession.delete(key);
  try {
    sessionStorage.removeItem(key);
  } catch {
    // ignore
  }
}

export function getAuthToken() {
  return safeLocalGet(AUTH_TOKEN_KEY) || safeSessionGet(AUTH_TOKEN_KEY);
}

export function setAuthToken(token) {
  if (token) {
    safeLocalSet(AUTH_TOKEN_KEY, token);
    safeSessionSet(AUTH_TOKEN_KEY, token);
  } else {
    safeLocalRemove(AUTH_TOKEN_KEY);
    safeSessionRemove(AUTH_TOKEN_KEY);
  }
}

export function getAuthHeaders(extra = {}) {
  const token = getAuthToken();
  const headers = { ...extra };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

export function getCurrentUserId() {
  try {
    const data = safeLocalGet(AUTH_STORAGE_KEY) || safeSessionGet(AUTH_STORAGE_KEY);
    if (data) {
      const user = JSON.parse(data);
      if (user && user.id) return user.id;
    }
  } catch {
    // ignore
  }
  return null;
}

export function getStoredSessionUser() {
  try {
    const data = safeLocalGet(AUTH_STORAGE_KEY) || safeSessionGet(AUTH_STORAGE_KEY);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
}

export function setStoredSessionUser(user) {
  if (user) {
    const json = JSON.stringify(user);
    safeLocalSet(AUTH_STORAGE_KEY, json);
    safeSessionSet(AUTH_STORAGE_KEY, json);
  } else {
    safeLocalRemove(AUTH_STORAGE_KEY);
    safeSessionRemove(AUTH_STORAGE_KEY);
  }
}

export function clearAuthStorage() {
  setAuthToken(null);
  setStoredSessionUser(null);
  safeSessionRemove('foodfresh_active_food_context');
  try {
    const keysToRemove = [];
    for (let i = 0; i < sessionStorage.length; i += 1) {
      const k = sessionStorage.key(i);
      if (k && k.startsWith('foodfresh_active_conv_')) {
        keysToRemove.push(k);
      }
    }
    keysToRemove.forEach((k) => safeSessionRemove(k));
  } catch {
    // ignore
  }
}
