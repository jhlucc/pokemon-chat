/**
 * 在线状态检测 Composable
 * 监测网络连接状态并提供离线提示
 */
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

export interface OnlineStatusOptions {
  /** 检查间隔（毫秒） */
  pingInterval?: number
  /** 检查 URL */
  pingUrl?: string
  /** 超时时间（毫秒） */
  pingTimeout?: number
  /** 离线后自动重试间隔（毫秒） */
  retryInterval?: number
}

const DEFAULT_OPTIONS: Required<OnlineStatusOptions> = {
  pingInterval: 30000,
  pingUrl: '/api/health',
  pingTimeout: 5000,
  retryInterval: 5000
}

/**
 * 在线状态检测
 */
export function useOnlineStatus(options: OnlineStatusOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options }

  // 浏览器在线状态
  const browserOnline = ref(navigator.onLine)
  // 服务器在线状态
  const serverOnline = ref(true)
  // 上次检查时间
  const lastCheckTime = ref<Date | null>(null)
  // 是否正在检查
  const isChecking = ref(false)
  // 最后一次错误
  const lastError = ref<string | null>(null)

  // 综合在线状态
  const isOnline = computed(() => browserOnline.value && serverOnline.value)

  // 离线原因
  const offlineReason = computed(() => {
    if (!browserOnline.value) {
      return 'network'
    }
    if (!serverOnline.value) {
      return 'server'
    }
    return null
  })

  // 离线提示消息
  const offlineMessage = computed(() => {
    if (!browserOnline.value) {
      return '网络连接已断开，请检查您的网络设置'
    }
    if (!serverOnline.value) {
      return '服务器连接失败，正在尝试重新连接...'
    }
    return null
  })

  let pingTimer: number | null = null
  let retryTimer: number | null = null

  /**
   * 检查服务器状态
   */
  async function checkServer(): Promise<boolean> {
    if (isChecking.value) return serverOnline.value

    isChecking.value = true

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), opts.pingTimeout)

      const response = await fetch(opts.pingUrl, {
        method: 'HEAD',
        signal: controller.signal,
        cache: 'no-store'
      })

      clearTimeout(timeoutId)

      serverOnline.value = response.ok
      lastError.value = null
      lastCheckTime.value = new Date()

      return response.ok
    } catch (error) {
      serverOnline.value = false
      lastError.value = error instanceof Error ? error.message : 'Unknown error'
      lastCheckTime.value = new Date()

      return false
    } finally {
      isChecking.value = false
    }
  }

  /**
   * 处理浏览器在线状态变化
   */
  function handleOnline() {
    browserOnline.value = true
    // 恢复在线后立即检查服务器
    checkServer()
  }

  function handleOffline() {
    browserOnline.value = false
    serverOnline.value = false
  }

  /**
   * 启动定期检查
   */
  function startPing() {
    stopPing()
    pingTimer = window.setInterval(() => {
      if (browserOnline.value) {
        checkServer()
      }
    }, opts.pingInterval)
  }

  /**
   * 停止定期检查
   */
  function stopPing() {
    if (pingTimer !== null) {
      clearInterval(pingTimer)
      pingTimer = null
    }
  }

  /**
   * 启动离线重试
   */
  function startRetry() {
    stopRetry()
    retryTimer = window.setInterval(() => {
      if (!isOnline.value && browserOnline.value) {
        checkServer()
      }
    }, opts.retryInterval)
  }

  /**
   * 停止离线重试
   */
  function stopRetry() {
    if (retryTimer !== null) {
      clearInterval(retryTimer)
      retryTimer = null
    }
  }

  /**
   * 手动刷新状态
   */
  async function refresh(): Promise<boolean> {
    browserOnline.value = navigator.onLine
    if (browserOnline.value) {
      return checkServer()
    }
    return false
  }

  // 监听在线状态变化，自动管理重试
  watch(isOnline, (online) => {
    if (online) {
      stopRetry()
    } else {
      startRetry()
    }
  })

  // 生命周期管理
  onMounted(() => {
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    // 初始检查
    checkServer()
    startPing()
  })

  onUnmounted(() => {
    window.removeEventListener('online', handleOnline)
    window.removeEventListener('offline', handleOffline)

    stopPing()
    stopRetry()
  })

  return {
    // 状态
    isOnline,
    browserOnline,
    serverOnline,
    isChecking,
    lastCheckTime,
    lastError,
    offlineReason,
    offlineMessage,
    // 方法
    checkServer,
    refresh,
    startPing,
    stopPing
  }
}

/**
 * 离线提示组件的 props 类型
 */
export interface OfflineAlertProps {
  show: boolean
  reason: 'network' | 'server' | null
  message: string | null
  onRetry?: () => void
}
