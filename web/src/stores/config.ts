import { ref } from 'vue'
import { defineStore } from 'pinia'
import { DEFAULT_CONFIG, LOCAL_CONFIG_KEY } from '@/config/defaultConfig'
import { apiFetch } from '@/api/http'

// Define types for Config based on usage
export interface BackendState {
  online: boolean
  ready?: boolean
  last_error?: string | null
  checks?: any | null
}

export interface UIState {
  show_knowledge_base: boolean
  show_knowledge_graph: boolean
  show_web_search: boolean
  show_mcp: boolean
  show_map: boolean
}

export interface ConfigState {
  backend?: BackendState
  ui?: UIState
  model_provider?: string
  model_name?: string
  enable_knowledge_base?: boolean
  enable_knowledge_graph?: boolean
  enable_web_search?: boolean
  enable_mcp?: boolean
  enable_reranker?: boolean
  enable_asr?: boolean
  enable_ner_bert?: boolean
  embed_model?: string
  reranker?: string
  model_names?: any
  custom_models?: any[]
  [key: string]: any // allow dynamic keys
}

// 缓存 TTL（毫秒）
const CACHE_TTL = 30000
// 请求去重标志
let pendingRefresh: Promise<ConfigState> | null = null
let lastRefreshTime = 0

export const useConfigStore = defineStore('config', () => {
  const config = ref<ConfigState>(normalizeLocalConfig(loadLocalConfig()))

  function saveLocalConfig() {
    try {
      // Do not persist large static catalogs; keep storage forward-compatible.
      const toStore = { ...(config.value || {}) }
      delete toStore.model_names
      localStorage.setItem(LOCAL_CONFIG_KEY, JSON.stringify(toStore))
    } catch {
      // ignore quota / private mode errors
    }
  }

  function setConfig(newConfig: ConfigState) {
    config.value = { ...structuredClone(DEFAULT_CONFIG), ...(newConfig || {}) }
    // Always restore static model catalog (avoid stale localStorage payloads).
    config.value.model_names = DEFAULT_CONFIG.model_names
    saveLocalConfig()
  }

  function patchLocal(patch: Partial<ConfigState>) {
    config.value = {
      ...config.value,
      ...(patch || {}),
      backend: {
        ...(config.value.backend || DEFAULT_CONFIG.backend),
        ...(patch?.backend || {})
      }
    }
    saveLocalConfig()
  }

  async function refreshConfig(options: { force?: boolean } = {}) {
    const { force = false } = options
    const now = Date.now()

    // 缓存检查：如果最近刚刷新过且不是强制刷新，跳过
    if (!force && lastRefreshTime && now - lastRefreshTime < CACHE_TTL) {
      return config.value
    }

    // 请求去重：如果有正在进行的请求，复用它
    if (pendingRefresh) {
      return pendingRefresh
    }

    // Keep UI usable even if backend is down.
    pendingRefresh = (async () => {
      try {
        const remote = await apiFetch<ConfigState>('/config', { method: 'GET' })
        patchLocal({
          ...remote,
          backend: { online: true, last_error: null, ...(remote?.backend || {}) }
        })

        // Best-effort readiness probe (keeps UI informative, not required for rendering).
        try {
          const ready = await apiFetch<{status: string, checks: any}>('/readyz', { method: 'GET', timeoutMs: 5000 })
          patchLocal({
            backend: {
              ready: ready?.status === 'ok',
              checks: ready?.checks || null,
              online: true,
              last_error: null
            }
          })
        } catch {
          patchLocal({ backend: { ready: false, checks: null, online: true, last_error: null } })
        }

        lastRefreshTime = Date.now()
        return config.value
      } catch (e: any) {
        patchLocal({ backend: { online: false, last_error: e?.message || 'offline', ready: false, checks: null } })
        return config.value
      } finally {
        pendingRefresh = null
      }
    })()

    return pendingRefresh
  }

  async function setConfigValue(key: string, value: any, { syncRemote = true } = {}) {
    patchLocal({ [key]: value })

    // Optional sync to backend (best-effort)
    if (!syncRemote) return
    try {
      const remote = await apiFetch<ConfigState>('/config', { method: 'PATCH', body: { [key]: value } })
      patchLocal({
        ...remote,
        backend: { online: true, last_error: null, ...(remote?.backend || {}) }
      })
    } catch (e: any) {
      // Keep local; surface offline status via backend.last_error.
      patchLocal({ backend: { online: false, last_error: e?.message || 'offline', ready: false, checks: null } })
    }
  }

  async function setConfigValues(patch: Partial<ConfigState>, { syncRemote = true } = {}) {
    patchLocal(patch)
    if (!syncRemote) return
    try {
      const remote = await apiFetch<ConfigState>('/config', { method: 'PATCH', body: patch })
      patchLocal({
        ...remote,
        backend: { online: true, last_error: null, ...(remote?.backend || {}) }
      })
    } catch (e: any) {
      patchLocal({ backend: { online: false, last_error: e?.message || 'offline', ready: false, checks: null } })
    }
  }

  return { config, setConfig, setConfigValue, setConfigValues, refreshConfig, patchLocal }
})

function loadLocalConfig(): ConfigState | null {
  try {
    const raw = localStorage.getItem(LOCAL_CONFIG_KEY)
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function normalizeLocalConfig(raw: ConfigState | null): ConfigState {
  // Merge with defaults so older localStorage snapshots don't break new UI fields.
  const merged = { ...structuredClone(DEFAULT_CONFIG), ...(raw || {}) }

  // Static catalog should always come from the build, not localStorage.
  merged.model_names = DEFAULT_CONFIG.model_names

  // Ensure nested objects exist (older snapshots may miss them).
  merged.backend = { ...(structuredClone(DEFAULT_CONFIG.backend) || {}), ...(merged.backend || {}) }
  merged.ui = { ...(structuredClone(DEFAULT_CONFIG.ui) || {}), ...(merged.ui || {}) }

  return merged
}
