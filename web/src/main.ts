import { createApp } from 'vue'
import { createPinia } from 'pinia'

import Antd, { message, notification } from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'

import App from './App.vue'
import router from './router'

import './assets/main.css'
import { initTheme } from './assets/theme'
import { initUiDensity } from './utils/uiDensity'
import { initThemePreset } from './utils/themePreset'
import { ApiError } from './api/http'
import { trackError } from './utils/telemetry'

message.config({ maxCount: 3, duration: 2 })
notification.config({ placement: 'bottomRight' })

function formatApiError(err: ApiError): string {
  const parts: string[] = []
  parts.push(err.message || 'Request failed')
  if (err.status) parts.push(`HTTP ${err.status}`)
  if (err.requestId) parts.push(`RID: ${err.requestId}`)
  return parts.join(' · ')
}

let lastToastKey = ''
let lastToastAt = 0
function toastOnce(key: string, fn: () => void): void {
  const now = Date.now()
  if (key === lastToastKey && now - lastToastAt < 1500) return
  lastToastKey = key
  lastToastAt = now
  fn()
}

function notifyUnhandledError(err: unknown): void {
  if (err instanceof ApiError) {
    toastOnce(`api:${err.method}:${err.url}:${err.status}:${err.message}`, () => {
      message.error(formatApiError(err))
    })
    return
  }

  const msg =
    err instanceof Error ? err.message : typeof err === 'string' ? err : 'Unexpected error'
  toastOnce(`err:${msg}`, () => message.error(msg))
}

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(Antd)

// Vue render errors
app.config.errorHandler = (err, instance, info) => {
  trackError(err, { info, component: instance?.$?.type?.name || 'anonymous' })
  notifyUnhandledError(err)
}

// Promise / global errors
window.addEventListener('unhandledrejection', (event) => {
  trackError(event.reason, { type: 'unhandledrejection' })
  notifyUnhandledError(event.reason)
})
window.addEventListener('error', (event) => {
  trackError(event.error || event.message, { type: 'error' })
  notifyUnhandledError(event.error || event.message)
})

// Apply theme ASAP to avoid flash (mode is persisted in localStorage).
initTheme()
// Apply UI density ASAP to avoid layout flash.
initUiDensity()
// Apply theme preset ASAP (affects CSS vars + antd token).
initThemePreset()
app.mount('#app')
