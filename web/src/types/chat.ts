/**
 * Pokemon-Chat 聊天模块类型定义
 */

// ============= 消息相关类型 =============

/**
 * 消息角色
 */
export type MessageRole = 'user' | 'sent' | 'assistant' | 'received'

/**
 * 消息状态
 */
export type MessageStatus =
  | 'init'
  | 'searching'
  | 'generating'
  | 'reasoning'
  | 'finished'
  | 'error'
  | 'stopped'

/**
 * 知识库检索结果
 */
export interface KnowledgeBaseResult {
  file?: {
    filename: string
    [key: string]: unknown
  }
  content?: string
  score?: number
  [key: string]: unknown
}

/**
 * 消息引用来源
 */
export interface MessageRefs {
  knowledge_base?: {
    results: KnowledgeBaseResult[]
  }
  web_search?: {
    results: unknown[]
    [key: string]: unknown
  }
  knowledge_graph?: {
    nodes?: unknown[]
    edges?: unknown[]
    [key: string]: unknown
  }
  [key: string]: unknown
}

/**
 * 消息元数据
 */
export interface MessageMeta {
  active_step?: string
  server_model_name?: string
  response_time?: number
  [key: string]: unknown
}

/**
 * 工具调用
 */
export interface ToolCall {
  id: string
  name: string
  arguments?: Record<string, unknown>
  result?: unknown
  status?: 'pending' | 'running' | 'completed' | 'error'
}

/**
 * 聊天消息
 */
export interface Message {
  id: string
  role: MessageRole
  content: string
  reasoning_content?: string
  refs?: MessageRefs | null
  groupedResults?: Record<string, KnowledgeBaseResult[]>
  status?: MessageStatus
  meta?: MessageMeta
  message?: string // 错误消息
  showThinking?: 'show' | 'no'
  model_name?: string
  toolCalls?: Record<string, ToolCall>
  isStoppedByUser?: boolean
}

// ============= 会话相关类型 =============

/**
 * 会话对象
 */
export interface Conversation {
  id: string
  title: string
  messages: Message[]
  inputText: string
  createdAt: number
  updatedAt: number
}

/**
 * 会话列表项（带索引）
 */
export interface ConversationItem {
  conv: Conversation
  index: number
}

/**
 * 会话分组
 */
export interface ConversationGroup {
  key: string
  label: string
  items: ConversationItem[]
}

// ============= 聊天配置类型 =============

/**
 * 字体大小选项
 */
export type FontSize = 'smaller' | 'default' | 'larger'

/**
 * 聊天元数据/配置
 */
export interface ChatMeta {
  use_agent: boolean
  use_graph: boolean
  use_web: boolean
  use_mcp: boolean
  /**
   * Agent-mode constraints (used when `use_agent=true`):
   * - null/undefined: allow (subject to backend feature flags)
   * - false: explicitly disallow
   */
  agent_allow_web?: boolean | null
  agent_allow_graph?: boolean | null
  agent_allow_mcp?: boolean | null
  /**
   * Computed constraints sent to backend (derived from allow flags).
   * The backend reads this to restrict supervisor routing.
   */
  agent_constraints?: {
    allowed_workers: string[]
  }
  graph_name: string
  selectedKB: number | null
  mcp_id: string | null
  stream: boolean
  summary_title: boolean
  history_round: number
  db_id: string | null
  fontSize: FontSize
  wideScreen: boolean
  model_provider?: string
  model_name?: string
}

/**
 * 聊天状态
 */
export interface ChatState {
  isSidebarOpen: boolean
}

/**
 * 聊天选项面板状态
 */
export interface ChatOptions {
  showPanel: boolean
  showModelCard: boolean
  openDetail: boolean
  databases: Database[]
  mcps: McpServer[]
}

// ============= 后端数据类型 =============

/**
 * 知识库
 */
export interface Database {
  db_id: string
  name: string
  embed_model: string
  description?: string
  document_count?: number
}

/**
 * MCP 服务器
 */
export interface McpServer {
  id: string
  name: string
  description?: string
  status?: 'online' | 'offline'
}

/**
 * 模型提供商配置
 */
export interface ModelProvider {
  name: string
  models: string[]
}

/**
 * 自定义模型
 */
export interface CustomModel {
  custom_id: string
  name?: string
  endpoint?: string
}

// ============= API 请求/响应类型 =============

/**
 * 聊天请求参数
 */
export interface ChatRequestParams {
  query: string
  history: Array<{ role: 'user' | 'assistant'; content: string }>
  meta: ChatMeta
  cur_res_id?: string
  cfg?: {
    thread_id?: string
    msg_id?: string
    request_id?: string
  }
}

/**
 * 流式响应数据
 */
export interface StreamResponse {
  response?: string
  reasoning_content?: string
  refs?: MessageRefs
  meta?: MessageMeta
  status?: MessageStatus
  message?: string
  error?: string
  history?: Array<{ role: string; content: string }>
}

// ============= 组件 Props 类型 =============

/**
 * MessageComponent Props
 */
export interface MessageComponentProps {
  message: Message
  isProcessing?: boolean
  customClasses?: Record<string, string>
  showRefs?: string[] | boolean
  debugMode?: boolean
  showKnowledgeBase?: boolean
  showKnowledgeGraph?: boolean
  showWebSearch?: boolean
  showMcp?: boolean
}

/**
 * ChatComponent Props
 */
export interface ChatComponentProps {
  conv: Conversation
  state: ChatState
}

/**
 * ConversationList Props
 */
export interface ConversationListProps {
  conversations: Conversation[]
  activeIndex: number
  isOpen: boolean
}

/**
 * ChatCapabilityBar Props
 */
export interface ChatCapabilityBarProps {
  backendOnline: boolean
  useAgent: boolean
  canAgent: boolean
  useWeb: boolean
  canWebSearch: boolean
  useGraph: boolean
  canGraph: boolean
  useMcp: boolean
  canMcp: boolean
  selectedKbIndex: number | null
  canKb: boolean
  databases: Database[]
  showWebSearch?: boolean
  showKnowledgeGraph?: boolean
  showMcp?: boolean
  showKnowledgeBase?: boolean
}

// ============= 工具函数类型 =============

/**
 * 默认 ChatMeta 工厂函数类型
 */
export type DefaultMetaFactory = () => ChatMeta

/**
 * 消息更新信息
 */
export interface MessageUpdateInfo {
  id: string
  content?: string
  reasoning_content?: string
  refs?: MessageRefs | null
  meta?: MessageMeta
  status?: MessageStatus
  message?: string
  showThinking?: 'show' | 'no'
  model_name?: string
}
