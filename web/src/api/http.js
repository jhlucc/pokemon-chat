// Minimal fetch wrapper used across the frontend.
// Goals:
// - consistent error shape
// - timeout support
// - works even when backend is offline (callers can catch and degrade)

import { getOfflineMode } from '@/utils/offlineMode';
import { resolveMockResponse } from './mockApi';

const DEFAULT_TIMEOUT_MS = 15000;
const DEFAULT_API_PREFIX = '/api';
const API_PREFIX = (import.meta?.env?.VITE_API_BASE_PATH || DEFAULT_API_PREFIX).replace(/\/+$/, '') || DEFAULT_API_PREFIX;

export class ApiError extends Error {
  constructor(
    message,
    {
      status = null,
      data = null,
      isNetworkError = false,
      isCancelled = false,
      requestId = null,
      url = null,
      method = null,
    } = {}
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
    this.isNetworkError = isNetworkError;
    this.isCancelled = isCancelled;
    this.requestId = requestId;
    this.url = url;
    this.method = method;
  }
}

function normalizePath(path) {
  if (!path) return API_PREFIX;
  if (/^https?:\/\//i.test(path)) return path;
  const p = path.startsWith('/') ? path : `/${path}`;
  if (p === API_PREFIX || p.startsWith(`${API_PREFIX}/`)) return p;
  return `${API_PREFIX}${p}`;
}

function buildUrl(path, query) {
  const url = new URL(normalizePath(path), window.location.origin);
  if (query && typeof query === 'object') {
    Object.entries(query).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      url.searchParams.set(k, String(v));
    });
  }
  return url.toString();
}

function buildHeaders(body, headers) {
  const hasJsonBody =
    body !== undefined &&
    body !== null &&
    typeof body === 'object' &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer) &&
    !(body instanceof URLSearchParams);

  return {
    ...(hasJsonBody ? { 'Content-Type': 'application/json' } : {}),
    ...(headers || {}),
  };
}

function buildBody(body) {
  const hasJsonBody =
    body !== undefined &&
    body !== null &&
    typeof body === 'object' &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer) &&
    !(body instanceof URLSearchParams);
  if (!hasJsonBody) return body;
  return JSON.stringify(body);
}

function anyAbortSignal(signals) {
  const controller = new AbortController();

  const onAbort = () => controller.abort();
  for (const s of signals) {
    if (!s) continue;
    if (s.aborted) {
      controller.abort();
      break;
    }
    s.addEventListener('abort', onAbort, { once: true });
  }

  return controller.signal;
}

export async function apiRequest(path, { method = 'GET', query, body, headers, timeoutMs, signal } = {}) {
  const offlineMode = getOfflineMode();
  if (offlineMode === 'on') {
    const mocked = resolveMockResponse(path, { method, query, body });
    if (mocked) return mocked;
    throw new ApiError(`Mock handler not found for ${method} ${path}`, { isNetworkError: true, method, url: buildUrl(path, query) });
  }

  const controller = new AbortController();
  let timedOut = false;
  const t = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs || DEFAULT_TIMEOUT_MS);

  try {
    const combinedSignal = signal ? anyAbortSignal([signal, controller.signal]) : controller.signal;
    const res = await fetch(buildUrl(path, query), {
      method,
      headers: buildHeaders(body, headers),
      body: buildBody(body),
      signal: combinedSignal,
    });

    if (!res.ok) {
      const requestId = res.headers.get('x-request-id');
      const contentType = res.headers.get('content-type') || '';
      const data = contentType.includes('application/json') ? await res.json().catch(() => null) : await res.text().catch(() => null);
      const msg = (data && (data.detail || data.message)) || `HTTP ${res.status}`;
      const err = new ApiError(msg, { status: res.status, data, requestId, url: res.url, method });

      // In "auto" mode, treat 5xx as backend-unavailable and fall back to mocks where possible.
      if (offlineMode === 'auto' && res.status >= 500) {
        const mocked = resolveMockResponse(path, { method, query, body });
        if (mocked) return mocked;
      }

      throw err;
    }

    return res;
  } catch (e) {
    if (e?.name === 'AbortError') {
      if (timedOut) {
        const err = new ApiError('Request timeout', { isNetworkError: true, method, url: buildUrl(path, query) });
        if (offlineMode === 'auto') {
          const mocked = resolveMockResponse(path, { method, query, body });
          if (mocked) return mocked;
        }
        throw err;
      }
      // User-initiated cancel should not fall back to mocks.
      throw new ApiError('Request cancelled', { isCancelled: true, method, url: buildUrl(path, query) });
    }
    if (e instanceof ApiError) throw e;
    const err = new ApiError(e?.message || 'Network error', { isNetworkError: true, method, url: buildUrl(path, query) });
    if (offlineMode === 'auto') {
      const mocked = resolveMockResponse(path, { method, query, body });
      if (mocked) return mocked;
    }
    throw err;
  } finally {
    clearTimeout(t);
  }
}

export async function apiFetch(path, { method = 'GET', query, body, headers, timeoutMs, signal } = {}) {
  try {
    const res = await apiRequest(path, { method, query, body, headers, timeoutMs, signal });

    const contentType = res.headers.get('content-type') || '';
    const data = contentType.includes('application/json') ? await res.json().catch(() => null) : await res.text().catch(() => null);

    return data;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(e?.message || 'Network error', { isNetworkError: true, method, url: buildUrl(path, query) });
  } finally {
    // no-op: timeout handled in apiRequest
  }
}
