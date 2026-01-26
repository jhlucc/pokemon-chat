/**
 * 消息更新管理 Composable
 * 处理消息的创建、更新和状态管理
 */
import { type Ref } from 'vue'
import type {
  Message,
  Conversation,
  MessageStatus,
  MessageRefs,
  MessageMeta,
  MessageUpdateInfo,
  KnowledgeBaseResult
} from '@/types/chat'
import { randomId } from '@/utils/id'

/**
 * 消息更新管理
 */
export function useMessageUpdate(conversation: Ref<Conversation>) {
  /**
   * 创建用户消息
   */
  function createUserMessage(content: string): Message {
    return {
      id: randomId(16),
      role: 'user',
      content
    }
  }

  /**
   * 创建 AI 消息（初始状态）
   */
  function createAssistantMessage(): Message {
    return {
      id: randomId(16),
      role: 'assistant',
      content: '',
      reasoning_content: '',
      refs: null,
      groupedResults: {},
      status: 'init',
      meta: {},
      showThinking: 'show'
    }
  }

  /**
   * 添加用户消息
   */
  function appendUserMessage(content: string): Message {
    const message = createUserMessage(content)
    conversation.value.messages.push(message)
    conversation.value.updatedAt = Date.now()
    return message
  }

  /**
   * 添加 AI 消息
   */
  function appendAssistantMessage(): Message {
    const message = createAssistantMessage()
    conversation.value.messages.push(message)
    conversation.value.updatedAt = Date.now()
    return message
  }

  /**
   * 根据 ID 查找消息
   */
  function findMessage(id: string): Message | undefined {
    return conversation.value.messages.find((msg) => msg.id === id)
  }

  /**
   * 根据 ID 查找消息索引
   */
  function findMessageIndex(id: string): number {
    return conversation.value.messages.findIndex((msg) => msg.id === id)
  }

  /**
   * 更新消息
   */
  function updateMessage(info: MessageUpdateInfo): boolean {
    const msg = findMessage(info.id)
    if (!msg) {
      console.error('Message not found:', info.id)
      return false
    }

    try {
      // 更新内容（追加模式）
      if (info.content != null && info.content !== '') {
        msg.content += info.content
      }

      // 更新推理内容（替换模式）
      if (info.reasoning_content != null && info.reasoning_content !== '') {
        msg.reasoning_content = info.reasoning_content
      }

      // 更新引用
      if (info.refs !== undefined) {
        msg.refs = info.refs
      }

      // 更新元数据
      if (info.meta != null) {
        msg.meta = info.meta
      }

      // 更新状态
      if (info.status != null && info.status !== '') {
        msg.status = info.status
      }

      // 更新错误消息
      if (info.message != null) {
        msg.message = info.message
      }

      // 更新模型名称
      if (info.model_name != null && info.model_name !== '') {
        msg.model_name = info.model_name
      }

      // 更新思考显示状态
      if (info.showThinking != null) {
        msg.showThinking = info.showThinking
      }

      return true
    } catch (error) {
      console.error('Error updating message:', error)
      msg.status = 'error'
      msg.content = '消息更新失败'
      return false
    }
  }

  /**
   * 设置消息状态
   */
  function setMessageStatus(id: string, status: MessageStatus): boolean {
    const msg = findMessage(id)
    if (!msg) return false
    msg.status = status
    return true
  }

  /**
   * 设置消息错误
   */
  function setMessageError(id: string, errorMessage: string): boolean {
    const msg = findMessage(id)
    if (!msg) return false
    msg.status = 'error'
    msg.message = errorMessage
    return true
  }

  /**
   * 标记消息为用户停止
   */
  function markAsStopped(id: string): boolean {
    const msg = findMessage(id)
    if (!msg) return false
    msg.isStoppedByUser = true
    msg.status = 'stopped'
    return true
  }

  /**
   * 完成消息
   */
  function finishMessage(id: string): boolean {
    const msg = findMessage(id)
    if (!msg) return false
    msg.status = 'finished'
    msg.showThinking = 'no'
    return true
  }

  /**
   * 对消息的引用进行分组
   */
  function groupMessageRefs(id: string): void {
    const msg = findMessage(id)
    if (!msg) return

    const results = msg.refs?.knowledge_base?.results
    if (!Array.isArray(results) || results.length === 0) {
      msg.groupedResults = {}
      return
    }

    msg.groupedResults = results
      .filter((result): result is KnowledgeBaseResult & { file: { filename: string } } =>
        Boolean(result?.file?.filename)
      )
      .reduce(
        (acc, result) => {
          const filename = result.file.filename
          if (!acc[filename]) acc[filename] = []
          acc[filename].push(result)
          return acc
        },
        {} as Record<string, KnowledgeBaseResult[]>
      )
  }

  /**
   * 删除消息及其后的所有消息
   */
  function deleteMessageAndAfter(id: string): Message[] {
    const index = findMessageIndex(id)
    if (index === -1) return []

    const deleted = conversation.value.messages.splice(index)
    conversation.value.updatedAt = Date.now()
    return deleted
  }

  /**
   * 删除消息及其前一条消息（用于重试）
   */
  function deleteForRetry(id: string): { userMessage: Message | null; index: number } {
    const index = findMessageIndex(id)
    if (index <= 0) {
      return { userMessage: null, index: -1 }
    }

    const userMessage = conversation.value.messages[index - 1]
    conversation.value.messages = conversation.value.messages.slice(0, index - 1)
    conversation.value.updatedAt = Date.now()

    return { userMessage, index: index - 1 }
  }

  /**
   * 获取最后一条消息
   */
  function getLastMessage(): Message | undefined {
    const messages = conversation.value.messages
    return messages.length > 0 ? messages[messages.length - 1] : undefined
  }

  /**
   * 获取历史记录（用于 API 请求）
   */
  function getHistory(maxRounds: number): Array<{ role: 'user' | 'assistant'; content: string }> {
    return conversation.value.messages
      .filter((msg) => msg.content)
      .map((msg) => ({
        role: (msg.role === 'sent' || msg.role === 'user' ? 'user' : 'assistant') as
          | 'user'
          | 'assistant',
        content: msg.content
      }))
      .slice(-maxRounds)
  }

  /**
   * 检查是否有空内容的 AI 消息（错误状态）
   */
  function checkEmptyMessages(): void {
    conversation.value.messages.forEach((msg) => {
      if (msg.role === 'received' && (!msg.content || msg.content.trim() === '')) {
        msg.status = 'error'
        msg.message = '内容加载失败'
      }
    })
  }

  return {
    // 创建消息
    createUserMessage,
    createAssistantMessage,
    appendUserMessage,
    appendAssistantMessage,
    // 查找消息
    findMessage,
    findMessageIndex,
    getLastMessage,
    // 更新消息
    updateMessage,
    setMessageStatus,
    setMessageError,
    markAsStopped,
    finishMessage,
    groupMessageRefs,
    // 删除消息
    deleteMessageAndAfter,
    deleteForRetry,
    // 工具方法
    getHistory,
    checkEmptyMessages
  }
}
