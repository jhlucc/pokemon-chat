/**
 * 重试机制 Composable
 * 提供指数退避重试功能
 */
import { ref, computed, type Ref } from 'vue'

export interface RetryOptions {
  /** 最大重试次数 */
  maxRetries?: number
  /** 初始延迟时间（毫秒） */
  initialDelay?: number
  /** 最大延迟时间（毫秒） */
  maxDelay?: number
  /** 退避倍数 */
  backoffMultiplier?: number
  /** 是否添加抖动 */
  jitter?: boolean
  /** 可重试的错误判断函数 */
  shouldRetry?: (error: unknown) => boolean
}

export interface RetryState {
  /** 当前重试次数 */
  attempts: Ref<number>
  /** 是否正在重试中 */
  isRetrying: Ref<boolean>
  /** 最后一次错误 */
  lastError: Ref<Error | null>
  /** 下次重试延迟（毫秒） */
  nextRetryDelay: Ref<number>
  /** 是否还可以重试 */
  canRetry: Ref<boolean>
}

const DEFAULT_OPTIONS: Required<RetryOptions> = {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  jitter: true,
  shouldRetry: () => true
}

/**
 * 计算带抖动的延迟时间
 */
function calculateDelay(
  attempt: number,
  initialDelay: number,
  maxDelay: number,
  multiplier: number,
  jitter: boolean
): number {
  // 指数退避
  let delay = initialDelay * Math.pow(multiplier, attempt)

  // 添加随机抖动（±25%）
  if (jitter) {
    const jitterFactor = 0.75 + Math.random() * 0.5
    delay *= jitterFactor
  }

  // 限制最大延迟
  return Math.min(delay, maxDelay)
}

/**
 * 等待指定时间
 */
function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * 重试机制
 */
export function useRetry(options: RetryOptions = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options }

  const attempts = ref(0)
  const isRetrying = ref(false)
  const lastError = ref<Error | null>(null)
  const nextRetryDelay = ref(0)

  const canRetry = computed(() => attempts.value < opts.maxRetries)

  /**
   * 重置重试状态
   */
  function reset() {
    attempts.value = 0
    isRetrying.value = false
    lastError.value = null
    nextRetryDelay.value = 0
  }

  /**
   * 执行带重试的异步操作
   */
  async function execute<T>(
    fn: () => Promise<T>,
    customOptions?: Partial<RetryOptions>
  ): Promise<T> {
    const mergedOpts = { ...opts, ...customOptions }

    reset()

    while (true) {
      try {
        isRetrying.value = attempts.value > 0
        const result = await fn()
        reset()
        return result
      } catch (error) {
        lastError.value = error instanceof Error ? error : new Error(String(error))

        // 检查是否应该重试
        if (!mergedOpts.shouldRetry(error)) {
          throw error
        }

        // 检查是否超过最大重试次数
        if (attempts.value >= mergedOpts.maxRetries) {
          throw error
        }

        // 计算下次重试延迟
        nextRetryDelay.value = calculateDelay(
          attempts.value,
          mergedOpts.initialDelay,
          mergedOpts.maxDelay,
          mergedOpts.backoffMultiplier,
          mergedOpts.jitter
        )

        attempts.value++

        // 等待后重试
        await wait(nextRetryDelay.value)
      }
    }
  }

  /**
   * 手动触发重试（用于 UI 按钮）
   */
  async function retry<T>(fn: () => Promise<T>): Promise<T | null> {
    if (!canRetry.value) {
      return null
    }

    try {
      isRetrying.value = true
      const result = await fn()
      reset()
      return result
    } catch (error) {
      lastError.value = error instanceof Error ? error : new Error(String(error))
      attempts.value++
      throw error
    } finally {
      isRetrying.value = false
    }
  }

  return {
    // 状态
    attempts,
    isRetrying,
    lastError,
    nextRetryDelay,
    canRetry,
    // 方法
    execute,
    retry,
    reset
  }
}

/**
 * 判断是否为网络错误（可重试）
 */
export function isNetworkError(error: unknown): boolean {
  if (error instanceof TypeError && error.message.includes('fetch')) {
    return true
  }
  if (error instanceof Error) {
    const message = error.message.toLowerCase()
    return (
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('abort') ||
      message.includes('econnrefused') ||
      message.includes('enotfound')
    )
  }
  return false
}

/**
 * 判断是否为服务器错误（可重试）
 */
export function isServerError(error: unknown): boolean {
  if (error && typeof error === 'object' && 'status' in error) {
    const status = (error as { status: number }).status
    return status >= 500 && status < 600
  }
  return false
}

/**
 * 默认的可重试错误判断
 */
export function isRetryableError(error: unknown): boolean {
  return isNetworkError(error) || isServerError(error)
}
