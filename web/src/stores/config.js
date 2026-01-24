import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import { DEFAULT_CONFIG, LOCAL_CONFIG_KEY } from '@/config/defaultConfig'
import { apiFetch } from '@/api/http'

export const useCounterStore = defineStore('counter', () => {
  const count = ref(0)
  const doubleCount = computed(() => count.value * 2)
  function increment() {
    count.value++
  }

  return { count, doubleCount, increment }
})


export const useConfigStore = defineStore('config', () => {
  const config = ref(loadLocalConfig() || structuredClone(DEFAULT_CONFIG))

  function saveLocalConfig() {
    try {
      localStorage.setItem(LOCAL_CONFIG_KEY, JSON.stringify(config.value))
    } catch {
      // ignore quota / private mode errors
    }
  }

  function setConfig(newConfig) {
    config.value = { ...structuredClone(DEFAULT_CONFIG), ...(newConfig || {}) }
    saveLocalConfig()
  }

  function patchLocal(patch) {
    config.value = {
      ...config.value,
      ...(patch || {}),
      backend: {
        ...(config.value.backend || {}),
        ...(patch?.backend || {}),
      },
    }
    saveLocalConfig()
  }

  async function refreshConfig() {
    // Keep UI usable even if backend is down.
    try {
      const remote = await apiFetch('/config', { method: 'GET' })
      patchLocal({
        ...remote,
        backend: { online: true, last_error: null, ...(remote?.backend || {}) },
      })

      // Best-effort readiness probe (keeps UI informative, not required for rendering).
      try {
        const ready = await apiFetch('/readyz', { method: 'GET', timeoutMs: 5000 })
        patchLocal({
          backend: {
            ready: ready?.status === 'ok',
            checks: ready?.checks || null,
          },
        })
      } catch {
        patchLocal({ backend: { ready: false, checks: null } })
      }
    } catch (e) {
      patchLocal({ backend: { online: false, last_error: e?.message || 'offline' } })
    }
  }

  async function setConfigValue(key, value, { syncRemote = true } = {}) {
    patchLocal({ [key]: value })

    // Optional sync to backend (best-effort)
    if (!syncRemote) return
    try {
      const remote = await apiFetch('/config', { method: 'PATCH', body: { [key]: value } })
      patchLocal({
        ...remote,
        backend: { online: true, last_error: null, ...(remote?.backend || {}) },
      })
    } catch (e) {
      // Keep local; surface offline status via backend.last_error.
      patchLocal({ backend: { online: false, last_error: e?.message || 'offline' } })
    }
  }

  async function setConfigValues(patch, { syncRemote = true } = {}) {
    patchLocal(patch)
    if (!syncRemote) return
    try {
      const remote = await apiFetch('/config', { method: 'PATCH', body: patch })
      patchLocal({
        ...remote,
        backend: { online: true, last_error: null, ...(remote?.backend || {}) },
      })
    } catch (e) {
      patchLocal({ backend: { online: false, last_error: e?.message || 'offline' } })
    }
  }

  return { config, setConfig, setConfigValue, setConfigValues, refreshConfig, patchLocal }
})

function loadLocalConfig() {
  try {
    const raw = localStorage.getItem(LOCAL_CONFIG_KEY)
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}
