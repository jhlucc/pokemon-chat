import { message } from 'ant-design-vue'
import { ApiError } from '@/api/http'

type NotifyOpts = {
  context?: string
  fallback?: string
}

export function notifyApiError(err: unknown, opts: NotifyOpts = {}): void {
  const ctx = opts.context ? `${opts.context}：` : ''

  if (err instanceof ApiError) {
    const rid = err.requestId ? `（RID: ${err.requestId}）` : ''
    message.error(`${ctx}${err.message}${rid}`)
    return
  }

  const msg =
    err instanceof Error ? err.message : typeof err === 'string' ? err : opts.fallback || '操作失败'
  message.error(`${ctx}${msg}`)
}
