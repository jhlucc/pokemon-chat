// Minimal fetch wrapper used across the frontend.
// Goals:
// - consistent error shape
// - timeout support
// - request id propagation

import { createRequestId } from '@/utils/id'

const DEFAULT_TIMEOUT_MS = 15000
const DEFAULT_API_PREFIX = '/api'
const RAW_API_ORIGIN = (import.meta.env.VITE_API_URL || '').trim()
const API_ORIGIN = RAW_API_ORIGIN.replace(/\/+$/, '')
const IS_DEV = Boolean(import.meta.env.DEV)

const BASE_ORIGIN = !IS_DEV && API_ORIGIN ? API_ORIGIN : window.location.origin
const API_PREFIX =
  (import.meta.env.VITE_API_BASE_PATH || DEFAULT_API_PREFIX).replace(/\/+$/, '') ||
  DEFAULT_API_PREFIX

export interface ApiRequestOptions {
  method?: string
  query?: Record<string, unknown> | URLSearchParams
  body?: unknown
  headers?: Record<string, string>
  timeoutMs?: number
  signal?: AbortSignal
}

export interface ApiErrorOptions {
  status?: number | null
  data?: unknown
  isNetworkError?: boolean
  isCancelled?: boolean
  requestId?: string | null
  url?: string | null
  method?: string | null
}

export class ApiError extends Error {
  status: number | null
  data: unknown
  isNetworkError: boolean
  isCancelled: boolean
  requestId: string | null
  url: string | null
  method: string | null

  constructor(
    message: string,
    {
      status = null,
      data = null,
      isNetworkError = false,
      isCancelled = false,
      requestId = null,
      url = null,
      method = null
    }: ApiErrorOptions = {}
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

function normalizePath(path: string): string {
  if (!path) return API_PREFIX
  if (/^https?:\/\//i.test(path)) return path
  const p = path.startsWith('/') ? path : `/${path}`
  if (p === API_PREFIX || p.startsWith(`${API_PREFIX}/`)) return p
  return `${API_PREFIX}${p}`
}

function buildUrl(path: string, query?: Record<string, unknown> | URLSearchParams): string {
  const url = new URL(normalizePath(path), BASE_ORIGIN)
  if (query) {
    if (query instanceof URLSearchParams) {
      query.forEach((value, key) => url.searchParams.append(key, value))
    } else if (typeof query === 'object') {
      Object.entries(query).forEach(([k, v]) => {
        if (v === undefined || v === null) return
        url.searchParams.set(k, String(v))
      })
    }
  }
  return url.toString()
}

function buildHeaders(body: unknown, headers?: Record<string, string>): Record<string, string> {
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

function buildBody(body: unknown): BodyInit | null | undefined {
  const hasJsonBody =
    body !== undefined &&
    body !== null &&
    typeof body === 'object' &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof ArrayBuffer) &&
    !(body instanceof URLSearchParams)
  
  if (!hasJsonBody) return body as BodyInit | null | undefined
  return JSON.stringify(body)
}

function anyAbortSignal(signals: (AbortSignal | undefined | null)[]): AbortSignal {
  const controller = new AbortController()

  const onAbort = () => controller.abort()
  
  // Check if already aborted
  for (const s of signals) {
    if (!s) continue
    if (s.aborted) {
      controller.abort()
      return controller.signal
    }
  }

  for (const s of signals) {
    if (!s) continue
    s.addEventListener('abort', onAbort, { once: true })
  }

  return controller.signal
}

/**
 * Low-level request function.
 * Returns the raw `Response` so callers can handle streaming bodies.
 */
export async function apiRequest(
  path: string,
  { method = 'GET', query, body, headers, timeoutMs, signal }: ApiRequestOptions = {}
): Promise<Response> {
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
      
      let data: any = null
      if (contentType.includes('application/json')) {
        data = await res.json().catch(() => null)
      } else {
        data = await res.text().catch(() => null)
      }
      
      const msg = (data && (data.detail || data.message)) || `HTTP ${res.status}`
      throw new ApiError(msg, { status: res.status, data, requestId, url: res.url, method })
    }

    return res
  } catch (e: any) {
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

/**
 * Convenience helper for non-streaming requests.
 */
export async function apiFetch<T = any>(
  path: string, 
  { method = 'GET', query, body, headers, timeoutMs, signal }: ApiRequestOptions = {}
): Promise<T> {
  try {
    const res = await apiRequest(path, { method, query, body, headers, timeoutMs, signal })

    const contentType = res.headers.get('content-type') || ''
    if (!contentType.includes('application/json')) {
      const text = await res.text().catch(() => null)
      const requestId = res.headers.get('x-request-id') || null
      throw new ApiError('Unexpected response (expected JSON)', {
        status: res.status,
        data: text,
        requestId,
        url: res.url,
        method
      })
    }

    const data = await res.json().catch(() => null)
    if (data === null) {
      const requestId = res.headers.get('x-request-id') || null
      throw new ApiError('Invalid JSON response', {
        status: res.status,
        data: null,
        requestId,
        url: res.url,
        method
      })
    }

    return data as T
  } catch (e: any) {
    if (e instanceof ApiError) throw e
    throw new ApiError(e?.message || 'Network error', {
      isNetworkError: true,
      method,
      url: buildUrl(path, query)
    })
  }
}
