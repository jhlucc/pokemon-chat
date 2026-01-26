/**
 * 聊天滚动管理 Composable
 * 处理聊天区域的滚动行为和自动滚动
 */
import { ref, computed, onMounted, onUnmounted, type Ref } from 'vue'

export interface UseChatScrollOptions {
  /** 距离底部多少像素内视为"在底部" */
  threshold?: number
  /** 滚动延迟（毫秒） */
  scrollDelay?: number
}

/**
 * 聊天滚动管理
 */
export function useChatScroll(
  containerRef: Ref<HTMLElement | null>,
  options: UseChatScrollOptions = {}
) {
  const { threshold = 20, scrollDelay = 10 } = options

  // 用户是否正在手动滚动（不在底部）
  const userIsScrolling = ref(false)
  // 是否应该自动滚动到底部
  const shouldAutoScroll = ref(true)

  /**
   * 计算是否需要显示"滚动到底部"按钮
   */
  const showScrollToBottom = computed(() => !shouldAutoScroll.value)

  /**
   * 检查是否接近底部
   */
  function isNearBottom(): boolean {
    const container = containerRef.value
    if (!container) return true

    const { scrollHeight, scrollTop, clientHeight } = container
    return scrollHeight - scrollTop - clientHeight < threshold
  }

  /**
   * 处理用户滚动事件
   */
  function handleUserScroll() {
    const nearBottom = isNearBottom()
    userIsScrolling.value = !nearBottom
    shouldAutoScroll.value = nearBottom
  }

  /**
   * 滚动到底部（如果启用自动滚动）
   */
  function scrollToBottom() {
    if (!shouldAutoScroll.value) return

    const container = containerRef.value
    if (!container) return

    setTimeout(() => {
      container.scrollTop = container.scrollHeight - container.clientHeight
    }, scrollDelay)
  }

  /**
   * 强制滚动到底部（无视自动滚动设置）
   */
  function forceScrollToBottom() {
    shouldAutoScroll.value = true
    userIsScrolling.value = false

    const container = containerRef.value
    if (!container) return

    setTimeout(() => {
      container.scrollTop = container.scrollHeight - container.clientHeight
    }, scrollDelay)
  }

  /**
   * 跳转到底部（用户主动点击按钮）
   */
  function jumpToBottom() {
    shouldAutoScroll.value = true
    userIsScrolling.value = false
    forceScrollToBottom()
  }

  /**
   * 滚动到指定元素
   */
  function scrollToElement(element: HTMLElement, behavior: ScrollBehavior = 'smooth') {
    element.scrollIntoView({ behavior, block: 'center' })
  }

  // 生命周期管理
  onMounted(() => {
    const container = containerRef.value
    if (container) {
      container.addEventListener('scroll', handleUserScroll)
      // 初始滚动到底部
      scrollToBottom()
    }
  })

  onUnmounted(() => {
    const container = containerRef.value
    if (container) {
      container.removeEventListener('scroll', handleUserScroll)
    }
  })

  return {
    userIsScrolling,
    shouldAutoScroll,
    showScrollToBottom,
    isNearBottom,
    scrollToBottom,
    forceScrollToBottom,
    jumpToBottom,
    scrollToElement,
    handleUserScroll
  }
}
