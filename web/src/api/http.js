// Minimal fetch wrapper used across the frontend.
// Goals:
// - consistent error shape
// - timeout support
// - works even when backend is offline (callers can catch and degrade)

const DEFAULT_TIMEOUT_MS = 15000;

export class ApiError extends Error {
  constructor(message, { status = null, data = null, isNetworkError = false } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
    this.isNetworkError = isNetworkError;
  }
}

function buildUrl(path, query) {
  const url = new URL(path, window.location.origin);
  if (query && typeof query === 'object') {
    Object.entries(query).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      url.searchParams.set(k, String(v));
    });
  }
  return url.toString();
}

export async function apiFetch(path, { method = 'GET', query, body, headers, timeoutMs } = {}) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs || DEFAULT_TIMEOUT_MS);

  try {
    const res = await fetch(buildUrl(path, query), {
      method,
      headers: {
        ...(body ? { 'Content-Type': 'application/json' } : {}),
        ...(headers || {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    const contentType = res.headers.get('content-type') || '';
    const data = contentType.includes('application/json') ? await res.json().catch(() => null) : await res.text().catch(() => null);

    if (!res.ok) {
      const msg = (data && (data.detail || data.message)) || `HTTP ${res.status}`;
      throw new ApiError(msg, { status: res.status, data });
    }

    return data;
  } catch (e) {
    if (e?.name === 'AbortError') {
      throw new ApiError('Request timeout', { isNetworkError: true });
    }
    if (e instanceof ApiError) throw e;
    throw new ApiError(e?.message || 'Network error', { isNetworkError: true });
  } finally {
    clearTimeout(t);
  }
}

