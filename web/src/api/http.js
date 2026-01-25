// Minimal fetch wrapper used across the frontend.
// Goals:
// - consistent error shape
// - timeout support
// - request id propagation

import { createRequestId } from '@/utils/id'

const DEFAULT_TIMEOUT_MS = 15000
const DEFAULT_API_PREFIX = '/api'
const API_PREFIX =
  (import.meta?.env?.VITE_API_BASE_PATH || DEFAULT_API_PREFIX).replace(/\/+$/, '') ||
  DEFAULT_API_PREFIX

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
      method = null
    } = {}
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
    this.isNetworkError = isNetworkError
    this.isCancelled = isCancelled
    this.requestId = requestId
    this.url = url
    this.method = method
  }
}

function normalizePath(path) {
  if (!path) return API_PREFIX
  if (/^https?:\/\//i.test(path)) return path
  const p = path.startsWith('/') ? path : `/${path}`
  if (p === API_PREFIX || p.startsWith(`${API_PREFIX}/`)) return p
  return `${API_PREFIX}${p}`
}

function buildUrl(path, query) {
  const url = new URL(normalizePath(path), window.location.origin)
  if (query && typeof query === 'object') {
    Object.entries(query).forEach(([k, v]) => {
      if (v === undefined || v === null) return
      url.searchParams.set(k, String(v))
    })
  }
  return url.toString()
}

function buildHeaders(body, headers) {
  const hasJsonBody =
    body !== undefined &&
    body !== null &&
    typeof body === 'object' &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer) &&
    !(body instanceof URLSearchParams)

  return {
    ...(hasJsonBody ? { 'Content-Type': 'application/json' } : {}),
    ...(headers || {})
  }
}

function buildBody(body) {
  const hasJsonBody =
    body !== undefined &&
    body !== null &&
    typeof body === 'object' &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer) &&
    !(body instanceof URLSearchParams)
  if (!hasJsonBody) return body
  return JSON.stringify(body)
}

function anyAbortSignal(signals) {
  const controller = new AbortController()

  const onAbort = () => controller.abort()
  for (const s of signals) {
    if (!s) continue
    if (s.aborted) {
      controller.abort()
      break
    }
    s.addEventListener('abort', onAbort, { once: true })
  }

  return controller.signal
}

export async function apiRequest(
  path,
  { method = 'GET', query, body, headers, timeoutMs, signal } = {}
) {
  const reqId = headers?.['X-Request-ID'] || headers?.['x-request-id'] || createRequestId()
  const finalHeaders = { ...buildHeaders(body, headers), 'X-Request-ID': reqId }

  const controller = new AbortController()
  let timedOut = false
  const t = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs || DEFAULT_TIMEOUT_MS)

  try {
    const combinedSignal = signal ? anyAbortSignal([signal, controller.signal]) : controller.signal
    const res = await fetch(buildUrl(path, query), {
      method,
      headers: finalHeaders,
      body: buildBody(body),
      signal: combinedSignal
    })

    if (!res.ok) {
      const requestId = res.headers.get('x-request-id') || reqId
      const contentType = res.headers.get('content-type') || ''
      const data = contentType.includes('application/json')
        ? await res.json().catch(() => null)
        : await res.text().catch(() => null)
      const msg = (data && (data.detail || data.message)) || `HTTP ${res.status}`
      throw new ApiError(msg, { status: res.status, data, requestId, url: res.url, method })
    }

    return res
  } catch (e) {
    if (e?.name === 'AbortError') {
      if (timedOut) {
        throw new ApiError('Request timeout', {
          isNetworkError: true,
          method,
          url: buildUrl(path, query),
          requestId: reqId
        })
      }
      throw new ApiError('Request cancelled', {
        isCancelled: true,
        method,
        url: buildUrl(path, query),
        requestId: reqId
      })
    }
    if (e instanceof ApiError) throw e
    throw new ApiError(e?.message || 'Network error', {
      isNetworkError: true,
      method,
      url: buildUrl(path, query),
      requestId: reqId
    })
  } finally {
    clearTimeout(t)
  }
}

export async function apiFetch(
  path,
  { method = 'GET', query, body, headers, timeoutMs, signal } = {}
) {
  try {
    const res = await apiRequest(path, { method, query, body, headers, timeoutMs, signal })

    const contentType = res.headers.get('content-type') || ''
    const data = contentType.includes('application/json')
      ? await res.json().catch(() => null)
      : await res.text().catch(() => null)

    return data
  } catch (e) {
    if (e instanceof ApiError) throw e
    throw new ApiError(e?.message || 'Network error', {
      isNetworkError: true,
      method,
      url: buildUrl(path, query)
    })
  }
}

