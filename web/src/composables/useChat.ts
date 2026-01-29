/**
 * 聊天核心逻辑 Composable
 * 处理消息发送、接收和流式响应
 */
import { ref, type Ref } from 'vue'
import type { ChatMeta, Conversation, Database, StreamResponse } from '@/types/chat'
import { apiRequest } from '@/api/http'
import { readNdjsonStream } from '@/utils/ndjsonStream'
import { useMessageUpdate } from './useMessageUpdate'

export interface UseChatOptions {
  /** 请求超时时间（毫秒） */
  timeout?: number
  /** 流式响应解析错误回调 */
  onParseError?: (error: unknown, line: string) => void
}

/**
 * 聊天核心逻辑
 */
export function useChat(conversation: Ref<Conversation>, options: UseChatOptions = {}) {
  const { timeout = 300000, onParseError } = options

  // 消息更新工具
  const messageUtils = useMessageUpdate(conversation)

  // 流状态
  const isStreaming = ref(false)
  const activeStreamId = ref(0)
  const abortController = ref<AbortController | null>(null)

  /**
   * 取消当前请求
   */
  function abortCurrentRequest() {
    abortController.value?.abort()
    abortController.value = null
  }

  /**
   * 发送聊天请求
   */
  async function sendChatRequest(
    userInput: string,
    meta: ChatMeta,
    databases: Database[] = [],
    callbacks?: {
      onScroll?: () => void
      onTitleRename?: (title: string) => void
    }
  ): Promise<boolean> {
    // 验证输入
    if (!userInput.trim()) {
      return false
    }

    // 如果正在处理中，不允许新请求
    if (isStreaming.value) {
      return false
    }

    // 开始流处理
    isStreaming.value = true
    const streamId = ++activeStreamId.value

    // 添加用户消息
    messageUtils.appendUserMessage(userInput)

    // 添加 AI 消息占位
    const aiMessage = messageUtils.appendAssistantMessage()
    const curResId = aiMessage.id

    // 触发滚动
    callbacks?.onScroll?.()

    // 准备数据库 ID
    const dbId =
      databases.length > 0 && meta.selectedKB != null
        ? databases[meta.selectedKB]?.db_id
        : null

    // 创建请求控制器
    const controller = new AbortController()
    abortController.value = controller

    // 判断是否为 Agent 模式
    const isAgentMode = Boolean(meta.use_agent)

    // 构建请求参数
    const history = messageUtils.getHistory(meta.history_round).slice(0, -1)
    const requestMeta = {
      ...meta,
      db_id: dbId,
      mcp_id: meta.use_mcp ? 'default' : null
    }
    if (isAgentMode) {
      const allowedWorkers = ['rag_worker', 'stats_worker']
      if (meta.agent_allow_web !== false) allowedWorkers.push('web_worker')
      if (meta.agent_allow_graph !== false) allowedWorkers.push('graph_worker')
      if (meta.agent_allow_mcp !== false) allowedWorkers.push('mcp_worker')
      requestMeta.agent_constraints = { allowed_workers: allowedWorkers }
    }

    const params = isAgentMode
      ? {
          query: userInput,
          history,
          meta: requestMeta,
          cfg: {
            thread_id: conversation.value?.id,
            msg_id: curResId,
            request_id: curResId
          }
        }
      : {
          query: userInput,
          history,
          meta: requestMeta,
          cur_res_id: curResId
        }

    // 缓冲区（非流式输出时使用）
    let bufferedText = ''
    let bufferedReasoning = ''
    let bufferedRefs: StreamResponse['refs']
    let bufferedMeta: StreamResponse['meta']

    try {
      const endpoint = isAgentMode ? '/chat/agent/supervisor_agent' : '/chat/'
      const response = await apiRequest(endpoint, {
        method: 'POST',
        body: params,
        signal: controller.signal,
        timeoutMs: timeout
      })

      await readNdjsonStream(
        response,
        async (data: StreamResponse | null) => {
          if (!data) return

          if (meta.stream) {
            // 流式输出：实时更新
            messageUtils.updateMessage({
              id: curResId,
              content: data.response,
              reasoning_content: data.reasoning_content,
              status: data.status,
              meta: data.meta,
              refs: data.refs,
              message: data.message || data.error
            })
          } else {
            // 非流式：缓冲内容
            if (data.response) bufferedText += data.response
            if (data.reasoning_content) bufferedReasoning = data.reasoning_content
            if (data.refs) bufferedRefs = data.refs
            if (data.meta) bufferedMeta = data.meta

            // 更新状态但不输出内容
            messageUtils.updateMessage({
              id: curResId,
              status: data.status,
              meta: data.meta,
              refs: data.refs,
              message: data.message || data.error
            })
          }

          // 后端可能返回重建的历史
          if (data.history && conversation.value.messages.length === 0) {
            // 这种情况通常不会发生，但保留兼容性
          }

          callbacks?.onScroll?.()
        },
        {
          onParseError: onParseError || ((e, line) => console.debug('JSON 解析错误:', e, line))
        }
      )

      // 非流式输出：一次性渲染缓冲内容
      if (!meta.stream && bufferedText) {
        messageUtils.updateMessage({
          id: curResId,
          content: bufferedText,
          reasoning_content: bufferedReasoning,
          refs: bufferedRefs,
          meta: bufferedMeta,
          status: 'finished'
        })
      }

      // 处理引用分组
      messageUtils.groupMessageRefs(curResId)

      // 完成消息
      messageUtils.finishMessage(curResId)

      // 首次对话：生成标题
      if (conversation.value.messages.length === 2 && callbacks?.onTitleRename) {
        const firstUserMsg = conversation.value.messages[0]?.content || ''
        callbacks.onTitleRename(firstUserMsg)
      }

      return true
    } catch (error: any) {
      // 用户取消：保留部分回答
      if (!error?.isCancelled) {
        console.error('Chat request error:', error)
        messageUtils.setMessageError(curResId, error?.message || '请求失败')
      }
      return false
    } finally {
      // 只有当前流 ID 匹配时才清理状态
      if (activeStreamId.value === streamId) {
        abortController.value = null
        isStreaming.value = false
      }
    }
  }

  /**
   * 停止当前生成
   */
  function stopGeneration(): boolean {
    if (!isStreaming.value) return false

    abortCurrentRequest()
    isStreaming.value = false

    const lastMessage = messageUtils.getLastMessage()
    if (lastMessage) {
      messageUtils.markAsStopped(lastMessage.id)
    }

    return true
  }

  /**
   * 重试消息
   */
  function prepareRetry(messageId: string): { userInput: string; success: boolean } {
    const { userMessage } = messageUtils.deleteForRetry(messageId)

    if (!userMessage) {
      return { userInput: '', success: false }
    }

    return { userInput: userMessage.content, success: true }
  }

  /**
   * 重试被停止的消息（不自动发送）
   */
  function prepareRetryStoppedMessage(messageId: string): { userInput: string; success: boolean } {
    const index = messageUtils.findMessageIndex(messageId)
    if (index <= 0) {
      return { userInput: '', success: false }
    }

    const userMessage = conversation.value.messages[index - 1]
    if (!userMessage || (userMessage.role !== 'sent' && userMessage.role !== 'user')) {
      return { userInput: '', success: false }
    }

    // 删除被停止的消息及之前的用户消息
    conversation.value.messages = conversation.value.messages.slice(0, index - 1)

    return { userInput: userMessage.content, success: true }
  }

  /**
   * 清理资源
   */
  function cleanup() {
    activeStreamId.value++
    abortCurrentRequest()
  }

  return {
    // 状态
    isStreaming,
    // 方法
    sendChatRequest,
    stopGeneration,
    prepareRetry,
    prepareRetryStoppedMessage,
    abortCurrentRequest,
    cleanup,
    // 消息工具（暴露以便直接使用）
    ...messageUtils
  }
}
