/**
 * 聊天元数据管理 Composable
 * 处理聊天配置的持久化和响应式状态
 */
import { reactive, watch } from 'vue'
import { useDebounceFn } from '@vueuse/core'
import type { ChatMeta, FontSize } from '@/types/chat'
import { readJson, writeJson } from '@/utils/storage'

const LEGACY_META_STORAGE_KEY = 'meta'
const META_STORAGE_KEY = 'pokemon-chat-meta-v1'

/**
 * 创建默认的聊天元数据
 */
export function createDefaultMeta(): ChatMeta {
  return {
    use_agent: false,
    use_graph: false,
    use_web: false,
    use_mcp: false,
    graph_name: 'neo4j',
    selectedKB: null,
    mcp_id: null,
    stream: true,
    summary_title: false,
    history_round: 20,
    db_id: null,
    fontSize: 'default',
    wideScreen: false
  }
}

/**
 * 聊天元数据管理
 */
export function useChatMeta() {
  // 从存储加载元数据，优先使用新 key，兼容旧 key
  const storedMeta = readJson<Partial<ChatMeta>>(
    META_STORAGE_KEY,
    readJson<Partial<ChatMeta>>(LEGACY_META_STORAGE_KEY, {})
  )

  // 合并默认值和存储值
  const meta = reactive<ChatMeta>({
    ...createDefaultMeta(),
    ...storedMeta
  })

  // 防抖持久化
  const persistMeta = useDebounceFn(
    () => {
      // 同时写入新旧 key，保持向后兼容
      writeJson(META_STORAGE_KEY, meta)
      writeJson(LEGACY_META_STORAGE_KEY, meta)
    },
    600,
    { maxWait: 2000 }
  )

  // 监听变化并持久化
  watch(meta, () => persistMeta(), { deep: true })

  // ============= Agent 模式控制 =============

  /**
   * 切换 Agent 模式
   * Agent 模式下会清除手动检索设置
   */
  function toggleAgent(): boolean {
    const next = !meta.use_agent
    meta.use_agent = next

    // Agent 模式下清除手动配置
    if (next) {
      meta.use_web = false
      meta.use_graph = false
      meta.use_mcp = false
      meta.mcp_id = null
      meta.selectedKB = null
      meta.db_id = null
    }

    return next
  }

  /**
   * 切换联网搜索
   */
  function toggleWebSearch(): boolean {
    meta.use_web = !meta.use_web
    return meta.use_web
  }

  /**
   * 切换知识图谱
   */
  function toggleGraph(): boolean {
    meta.use_graph = !meta.use_graph
    return meta.use_graph
  }

  /**
   * 切换 MCP
   */
  function toggleMcp(): boolean {
    meta.use_mcp = !meta.use_mcp
    meta.mcp_id = meta.use_mcp ? 'default' : null
    return meta.use_mcp
  }

  /**
   * 选择知识库
   */
  function selectKnowledgeBase(index: number | null, dbId: string | null = null) {
    meta.selectedKB = index
    meta.db_id = dbId
  }

  // ============= UI 设置 =============

  /**
   * 设置字体大小
   */
  function setFontSize(size: FontSize) {
    meta.fontSize = size
  }

  /**
   * 切换宽屏模式
   */
  function toggleWideScreen(): boolean {
    meta.wideScreen = !meta.wideScreen
    return meta.wideScreen
  }

  /**
   * 切换流式输出
   */
  function toggleStream(): boolean {
    meta.stream = !meta.stream
    return meta.stream
  }

  /**
   * 切换标题总结
   */
  function toggleSummaryTitle(): boolean {
    meta.summary_title = !meta.summary_title
    return meta.summary_title
  }

  /**
   * 设置历史轮数
   */
  function setHistoryRound(round: number) {
    meta.history_round = Math.max(1, Math.min(50, round))
  }

  /**
   * 设置模型
   */
  function setModel(provider: string, name: string) {
    meta.model_provider = provider
    meta.model_name = name
  }

  /**
   * 重置为默认设置
   */
  function resetToDefault() {
    const defaults = createDefaultMeta()
    Object.assign(meta, defaults)
  }

  return {
    meta,
    // Agent 模式
    toggleAgent,
    toggleWebSearch,
    toggleGraph,
    toggleMcp,
    selectKnowledgeBase,
    // UI 设置
    setFontSize,
    toggleWideScreen,
    toggleStream,
    toggleSummaryTitle,
    setHistoryRound,
    setModel,
    resetToDefault
  }
}
