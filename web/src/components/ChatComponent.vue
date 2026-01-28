<template>
  <div class="chat" ref="chatContainer">
    <!-- Global Background Decoration -->
    <div class="chat-bg-decor">
      <img src="/welcome-bg.png" alt="" class="bg-image" loading="lazy" decoding="async" />
    </div>

    <!--     顶部左侧是打开侧边栏、新建会话、切换模型，右侧是打开选项设置面板。>-->
    <div class="chat-header">
      <!--      聊天界面顶部导航栏（新建对话、切换模型、选项面板）-->
      <div class="header__left">
        <div
          v-if="!state.isSidebarOpen"
          class="close nav-btn"
          @click="state.isSidebarOpen = true"
          role="button"
          tabindex="0"
          aria-label="打开对话侧栏"
          @keydown.enter.prevent="state.isSidebarOpen = true"
          @keydown.space.prevent="state.isSidebarOpen = true"
        >
          <img
            src="@/assets/icons/sidebar_left.svg"
            class="iconfont icon-20 sidebar-icon"
            alt="侧栏"
          />
        </div>

      </div>
      <div class="header__right">
        <a-dropdown>
          <a-tooltip
            placement="bottom"
            :title="`${configStore.config?.model_provider || '-'} / ${configStore.config?.model_name || '-'}`"
          >
            <div
              class="model-select"
              @click.prevent
              role="button"
              tabindex="0"
              aria-label="选择模型"
              @keydown.enter.prevent="$event.currentTarget?.click?.()"
              @keydown.space.prevent="$event.currentTarget?.click?.()"
            >
              <img
                class="model-select-icon"
                :src="getProviderIcon(configStore.config?.model_provider)"
                :alt="configStore.config?.model_provider || 'provider'"
              />
                <span class="text">{{ getModelDisplayName(configStore.config?.model_name) }}</span>
                <DownOutlined class="model-select-caret" />
              </div>
          </a-tooltip>
          <template #overlay>
            <a-menu class="scrollable-menu">
              <a-menu-item-group
                v-for="(item, key) in modelKeys"
                :key="key"
                :title="modelNames[item]?.name"
              >
                <a-menu-item
                  v-for="(model, idx) in modelNames[item]?.models"
                  :key="`${item}-${idx}`"
                  @click="selectModel(item, model)"
                >
                  <span class="model-menu-item">
                    <img class="model-menu-icon" :src="getProviderIcon(item)" :alt="item" />
                    <span class="model-menu-text">{{ model }}</span>
                  </span>
                </a-menu-item>
              </a-menu-item-group>
              <a-menu-item-group v-if="customModels.length > 0" title="自定义模型">
                <a-menu-item
                  v-for="(model, idx) in customModels"
                  :key="`custom-${idx}`"
                  @click="selectModel('custom', model.custom_id)"
                >
                  custom/{{ model.custom_id }}
                </a-menu-item>
              </a-menu-item-group>
            </a-menu>
          </template>
        </a-dropdown>
        <ThemeToggle />
        <div
          class="nav-btn text"
          @click="opts.showPanel = !opts.showPanel"
          role="button"
          tabindex="0"
          aria-label="聊天选项"
          @keydown.enter.prevent="opts.showPanel = !opts.showPanel"
          @keydown.space.prevent="opts.showPanel = !opts.showPanel"
        >
          <component :is="opts.showPanel ? FolderOpenOutlined : FolderOutlined" />
          <span class="text">选项</span>
        </div>
        <div
          v-if="opts.showPanel"
          class="options-panel options-panel--right swing-in-top-fwd"
          ref="panel"
        >
          <div class="options-row">
            字体大小
            <a-select v-model:value="meta.fontSize" style="width: 100px" placeholder="选择字体大小">
              <a-select-option value="smaller">更小</a-select-option>
              <a-select-option value="default">默认</a-select-option>
              <a-select-option value="larger">更大</a-select-option>
            </a-select>
          </div>
          <div class="options-row" @click="meta.wideScreen = !meta.wideScreen">
            宽屏模式
            <div @click.stop><a-switch v-model:checked="meta.wideScreen" /></div>
          </div>
          <div class="options-row" @click="meta.summary_title = !meta.summary_title">
            自动命名对话
            <div @click.stop><a-switch v-model:checked="meta.summary_title" /></div>
          </div>
          <div class="options-row">
            历史轮数
            <a-input-number
              id="inputNumber"
              v-model:value="meta.history_round"
              :min="1"
              :max="50"
            />
          </div>
          <div class="options-row" @click="meta.stream = !meta.stream">
            流式输出
            <div @click.stop><a-switch v-model:checked="meta.stream" /></div>
          </div>
        </div>
      </div>
    </div>
    <div
      class="chat-box"
      role="log"
      aria-live="polite"
      aria-label="聊天消息"
      :class="{
        'wide-screen': meta.wideScreen,
        'font-smaller': meta.fontSize === 'smaller',
        'font-larger': meta.fontSize === 'larger'
      }"
    >
      <WelcomeHero
        v-if="conv.messages.length === 0"
        @select="(prompt) => { conv.inputText = prompt }"
      />
      <MessageComponent
        v-for="message in conv.messages"
        :message="message"
        :key="message.id"
        :is-processing="isStreaming"
        :show-refs="true"
        :show-knowledge-base="configStore.config?.ui?.show_knowledge_base !== false"
        :show-knowledge-graph="configStore.config?.ui?.show_knowledge_graph !== false"
        :show-web-search="configStore.config?.ui?.show_web_search !== false"
        :show-mcp="configStore.config?.ui?.show_mcp === true"
        @retry="retryMessage(message.id)"
        @retryStoppedMessage="retryStoppedMessage(message.id)"
      >
      </MessageComponent>
    </div>

    <a-float-button
      v-if="showScrollToBottom"
      class="scroll-to-bottom"
      :style="{ right: '20px', bottom: '96px' }"
      @click="jumpToBottom"
      tooltip="回到底部"
    >
      <template #icon>
        <DownOutlined />
      </template>
    </a-float-button>

    <div class="bottom">
      <div class="message-input-wrapper" :class="{ 'wide-screen': meta.wideScreen }">
        <MessageInputComponent
          v-model="conv.inputText"
          :is-loading="isStreaming"
          :send-button-disabled="!conv.inputText && !isStreaming"
          :auto-size="{ minRows: 2, maxRows: 10 }"
          @send="handleSendOrStop"
          @keydown="handleKeyDown"
        >
          <template #options-left>
            <ChatCapabilityBar
              :backend-online="backendOnline"
              :use-agent="meta.use_agent"
              :can-agent="canAgent"
              :use-web="meta.use_web"
              :can-web-search="canWebSearch"
              :use-graph="meta.use_graph"
              :can-graph="canGraph"
              :use-mcp="meta.use_mcp"
              :can-mcp="canMcp"
              :selected-kb-index="meta.selectedKB"
              :can-kb="canKb"
              :databases="opts.databases"
              :show-web-search="configStore.config?.ui?.show_web_search"
              :show-knowledge-graph="configStore.config?.ui?.show_knowledge_graph"
              :show-mcp="configStore.config?.ui?.show_mcp"
              :show-knowledge-base="configStore.config?.ui?.show_knowledge_base"
              @toggle-agent="toggleAgent"
              @toggle-web="toggleWebSearch"
              @toggle-graph="toggleGraph"
              @toggle-mcp="toggleMcp"
              @select-kb="useDatabase"
            />
          </template>
        </MessageInputComponent>
        <p class="note">
          请注意辨别内容的可靠性 By {{ configStore.config?.model_provider }}:
          {{ configStore.config?.model_name }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted, toRefs, nextTick, onUnmounted, watch, computed } from 'vue'
import {
  FolderOutlined,
  FolderOpenOutlined,
  DownOutlined
} from '@ant-design/icons-vue'
import { onClickOutside, useDebounceFn } from '@vueuse/core'
import { useConfigStore } from '@/stores/config'
import { message } from 'ant-design-vue'
import ChatCapabilityBar from '@/components/chat/ChatCapabilityBar.vue'
import WelcomeHero from '@/components/chat/WelcomeHero.vue'
import ThemeToggle from '@/components/common/ThemeToggle.vue'
import MessageInputComponent from '@/components/MessageInputComponent.vue'
import MessageComponent from '@/components/MessageComponent.vue'
import { readNdjsonStream } from '@/utils/ndjsonStream'
import { apiFetch, apiRequest } from '@/api/http'
import { randomId } from '@/utils/id'
import { readJson, writeJson } from '@/utils/storage'
import { getProviderIcon, getModelDisplayName } from '@/utils/providerIcon'

const props = defineProps({
  conv: Object,
  state: Object
})

const emit = defineEmits(['rename-title', 'newconv'])
const configStore = useConfigStore()

const { conv, state } = toRefs(props)
const chatContainer = ref(null)

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const canAgent = computed(() => backendOnline.value)
const canWebSearch = computed(
  () => backendOnline.value && Boolean(configStore.config.enable_web_search)
)
const canGraph = computed(
  () => backendOnline.value && Boolean(configStore.config.enable_knowledge_graph)
)
const canMcp = computed(() => backendOnline.value && Boolean(configStore.config.enable_mcp))
const canKb = computed(
  () => backendOnline.value && Boolean(configStore.config.enable_knowledge_base)
)

const toggleWebSearch = () => {
  if (!canWebSearch.value) {
    message.info(backendOnline.value ? '后端未启用联网搜索' : '后端离线/不可用')
    return
  }
  meta.use_web = !meta.use_web
}

const toggleGraph = () => {
  if (!canGraph.value) {
    message.info(backendOnline.value ? '后端未启用知识图谱' : '后端离线/不可用')
    return
  }
  meta.use_graph = !meta.use_graph
}

const toggleMcp = () => {
  if (!canMcp.value) {
    message.info(backendOnline.value ? '后端未启用 MCP' : '后端离线/不可用')
    return
  }
  meta.use_mcp = !meta.use_mcp
  meta.mcp_id = meta.use_mcp ? 'default' : null
}

const toggleAgent = () => {
  if (!canAgent.value) {
    message.info('后端离线/不可用')
    return
  }
  const next = !meta.use_agent
  meta.use_agent = next

  // Agent mode is handled server-side (supervisor_agent routes to workers).
  // Keep client meta deterministic by clearing manual retrieval toggles when entering Agent mode.
  if (next) {
    meta.use_web = false
    meta.use_graph = false
    meta.use_mcp = false
    meta.mcp_id = null
    meta.selectedKB = null
    meta.db_id = null
  }
}

const isStreaming = ref(false)
const activeChatAbortController = ref(null)
const activeStreamId = ref(0)
const userIsScrolling = ref(false)
const shouldAutoScroll = ref(true)

const panel = ref(null)

const opts = reactive({
  showPanel: false,
  showModelCard: false,
  openDetail: false,
  databases: [],
  mcps: []
})

const showScrollToBottom = computed(
  () => !shouldAutoScroll.value && (conv.value?.messages?.length || 0) > 0
)
const jumpToBottom = () => {
  shouldAutoScroll.value = true
  userIsScrolling.value = false
  forceScrollToBottom()
}

const LEGACY_META_STORAGE_KEY = 'meta'
const META_STORAGE_KEY = 'pokemon-chat-meta-v1'

function defaultMeta() {
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

const meta = reactive(readJson(META_STORAGE_KEY, readJson(LEGACY_META_STORAGE_KEY, defaultMeta())))

const persistMeta = useDebounceFn(
  () => {
    // Keep legacy key for backward compatibility with existing users.
    writeJson(META_STORAGE_KEY, meta)
    writeJson(LEGACY_META_STORAGE_KEY, meta)
  },
  600,
  { maxWait: 2000 }
)

onClickOutside(panel, () => setTimeout(() => (opts.showPanel = false), 30))

// 从 message 中获取 history 信息，每个消息都是 {role, content} 的格式
const getHistory = () => {
  const history = conv.value.messages
    .map((msg) => {
      if (msg.content) {
        return {
          role: msg.role === 'sent' || msg.role === 'user' ? 'user' : 'assistant',
          content: msg.content
        }
      }
    })
    .reduce((acc, cur) => {
      if (cur) {
        acc.push(cur)
      }
      return acc
    }, [])
  return history.slice(-meta.history_round)
}

const useDatabase = (index) => {
  if (!canKb.value) {
    message.info(backendOnline.value ? '后端未启用知识库' : '后端离线/不可用')
    return
  }
  const selected = opts.databases[index]
  if (index != null && configStore.config.embed_model != selected.embed_model) {
    message.error(
      `所选知识库的向量模型（${selected.embed_model}）与当前向量模型（${configStore.config.embed_model}) 不匹配，请重新选择`
    )
  } else {
    meta.selectedKB = index
  }
}

const handleKeyDown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  } else if (e.key === 'Enter' && e.shiftKey) {
    // Insert a newline character at the current cursor position
    const textarea = e.target
    const start = textarea.selectionStart
    const end = textarea.selectionEnd
    const cur = conv.value.inputText || ''
    conv.value.inputText = cur.substring(0, start) + '\n' + cur.substring(end)
    nextTick(() => {
      textarea.setSelectionRange(start + 1, start + 1)
    })
  }
}

const renameTitle = () => {
  if (meta.summary_title) {
    const prompt = '请用一个很短的句子关于下面的对话内容的主题起一个名字，不要带标点符号：'
    const firstUserMessage = conv.value.messages[0].content
    const firstAiMessage = conv.value.messages[1].content
    const context = `${prompt}\n\n问题: ${firstUserMessage}\n\n回复: ${firstAiMessage}，主题是（一句话）：`
    simpleCall(context).then((data) => {
      const response = data.response
        .split('：')[0]
        .replace(/^["'"']/g, '')
        .replace(/["'"']$/g, '')
      emit('rename-title', response)
    })
  } else {
    emit('rename-title', conv.value.messages[0].content)
  }
}

const handleUserScroll = () => {
  // 计算我们是否接近底部（100像素以内）
  const isNearBottom =
    chatContainer.value.scrollHeight -
      chatContainer.value.scrollTop -
      chatContainer.value.clientHeight <
    20

  // 如果用户不在底部，则仅将其标记为用户滚动
  userIsScrolling.value = !isNearBottom

  // 如果用户再次滚动到底部，请恢复自动滚动
  shouldAutoScroll.value = isNearBottom
}

const scrollToBottom = () => {
  if (shouldAutoScroll.value) {
    setTimeout(() => {
      chatContainer.value.scrollTop =
        chatContainer.value.scrollHeight - chatContainer.value.clientHeight
    }, 10)
  }
}

const appendUserMessage = (msg) => {
  const data = {
    id: randomId(16),
    role: 'user',
    content: msg
  }
  conv.value.messages.push(data)
  conv.value.updatedAt = Date.now()
  scrollToBottom()
}

const appendAiMessage = (content, refs = null) => {
  conv.value.messages.push({
    id: randomId(16),
    role: 'assistant',
    content: content,
    reasoning_content: '',
    refs,
    groupedResults: {},
    status: 'init',
    meta: {},
    showThinking: 'show'
  })
  conv.value.updatedAt = Date.now()
  scrollToBottom()
}

const updateMessage = (info) => {
  const msg = conv.value.messages.find((msg) => msg.id === info.id)
  if (msg) {
    try {
      // 只有在 text 不为空时更新
      if (info.content !== null && info.content !== undefined && info.content !== '') {
        msg.content += info.content
      }

      if (
        info.reasoning_content !== null &&
        info.reasoning_content !== undefined &&
        info.reasoning_content !== ''
      ) {
        msg.reasoning_content = info.reasoning_content
      }

      // 只有在 refs 不为空时更新
      if (info.refs !== null && info.refs !== undefined) {
        msg.refs = info.refs
      }

      if (info.model_name !== null && info.model_name !== undefined && info.model_name !== '') {
        msg.model_name = info.model_name
      }

      // 只有在 status 不为空时更新
      if (info.status !== null && info.status !== undefined && info.status !== '') {
        msg.status = info.status
      }

      if (info.meta !== null && info.meta !== undefined) {
        msg.meta = info.meta
      }

      if (info.message !== null && info.message !== undefined) {
        msg.message = info.message
      }

      if (info.showThinking !== null && info.showThinking !== undefined) {
        msg.showThinking = info.showThinking
      }

      scrollToBottom()
    } catch (error) {
      console.error('Error updating message:', error)
      msg.status = 'error'
      msg.content = '消息更新失败'
    }
  } else {
    console.error('Message not found:', info.id)
  }
}

const groupRefs = (id) => {
  const msg = conv.value.messages.find((msg) => msg.id === id)
  if (!msg) return
  const results = msg.refs?.knowledge_base?.results
  if (!Array.isArray(results) || results.length === 0) {
    msg.groupedResults = {}
    return
  }

  msg.groupedResults = results
    .filter((result) => result?.file?.filename)
    .reduce((acc, result) => {
      const filename = result.file.filename
      if (!acc[filename]) acc[filename] = []
      acc[filename].push(result)
      return acc
    }, {})
  scrollToBottom()
}

const simpleCall = (msg) => {
  return apiFetch('/chat/call', {
    method: 'POST',
    body: {
      query: msg,
      meta: {
        model_provider: configStore.config?.model_provider,
        model_name: configStore.config?.model_name
      }
    },
    timeoutMs: 30000
  })
}

const loadDatabases = () => {
  apiFetch('/data/', { method: 'GET', timeoutMs: 5000 })
    .then((data) => {
      opts.databases = data?.databases || []
    })
    .catch(() => {
      opts.databases = []
    })
}

const fetchChatResponse = async (user_input, cur_res_id) => {
  const streamId = activeStreamId.value + 1
  activeStreamId.value = streamId

  const controller = new AbortController()
  activeChatAbortController.value = controller

  const isAgentMode = Boolean(meta.use_agent)
  const params = isAgentMode
    ? {
        query: user_input,
        history: getHistory().slice(0, -1), // 去掉最后一条刚添加的用户消息
        meta,
        cfg: {
          // Keep thread stable per conversation so the backend agent can checkpoint.
          thread_id: conv.value?.id || undefined,
          // Align stream message id with the placeholder assistant message.
          msg_id: cur_res_id,
          request_id: cur_res_id
        }
      }
    : {
        query: user_input,
        history: getHistory().slice(0, -1), // 去掉最后一条刚添加的用户消息
        meta,
        cur_res_id
      }

  // If the user disables streaming output, we still call the streaming endpoint
  // (so refs/metadata remain available) but buffer content and render it at the end.
  const streamOutput = Boolean(meta.stream)
  let bufferedText = ''
  let bufferedReasoning = ''
  let bufferedRefs = null
  let bufferedMeta = null

  try {
    const endpoint = isAgentMode ? '/chat/agent/supervisor_agent' : '/chat/'
    const response = await apiRequest(endpoint, {
      method: 'POST',
      body: params,
      signal: controller.signal,
      timeoutMs: 300000
    })

    await readNdjsonStream(
      response,
      async (data) => {
        if (!data) return

        if (streamOutput) {
          updateMessage({
            ...data,
            id: cur_res_id,
            content: data.response,
            reasoning_content: data.reasoning_content,
            status: data.status,
            meta: data.meta,
            message: data.message || data.error
          })
        } else {
          if (data.response) bufferedText += data.response
          if (data.reasoning_content) bufferedReasoning = data.reasoning_content
          if (data.refs) bufferedRefs = data.refs
          if (data.meta) bufferedMeta = data.meta

          // Update status/meta in-place without streaming tokens to the UI.
          updateMessage({
            ...data,
            id: cur_res_id,
            status: data.status,
            meta: data.meta,
            refs: data.refs,
            message: data.message || data.error,
            content: '' // never stream partial content
          })
        }

        // Backward-compatible: backend may return reconstructed history
        if (data.history && conv.value.messages.length === 0) {
          conv.value.messages = data.history.map((msg) => ({
            id: randomId(8),
            role: msg.role,
            content: msg.content
          }))
        }
      },
      {
        onParseError: (e, line) => console.debug('JSON 解析错误:', e?.message || e, line)
      }
    )

    if (!streamOutput && bufferedText) {
      // Render buffered output once at the end.
      updateMessage({
        id: cur_res_id,
        content: bufferedText,
        reasoning_content: bufferedReasoning,
        refs: bufferedRefs,
        meta: bufferedMeta,
        status: 'finished'
      })
    }

    groupRefs(cur_res_id)
    updateMessage({ showThinking: 'no', id: cur_res_id })
    if (conv.value.messages.length === 2) renameTitle()
  } catch (error) {
    // user cancelled: keep partial answer and allow retry
    if (!error?.isCancelled) {
      console.error(error)
      updateMessage({
        id: cur_res_id,
        status: 'error',
        message: error?.message || '请求失败'
      })
    }
  } finally {
    if (activeStreamId.value === streamId) {
      activeChatAbortController.value = null
      isStreaming.value = false
    }
  }
}

// 更新后的 sendMessage 函数
const sendMessage = () => {
  const user_input = conv.value.inputText.trim()
  const dbID = opts.databases.length > 0 ? opts.databases[meta.selectedKB]?.db_id : null
  if (isStreaming.value) {
    message.error('请等待上一条消息处理完成')
    return
  }
  if (user_input) {
    isStreaming.value = true
    appendUserMessage(user_input)
    appendAiMessage('', null)
    forceScrollToBottom()

    const cur_res_id = conv.value.messages[conv.value.messages.length - 1].id
    conv.value.inputText = ''
    meta.db_id = dbID
    meta.mcp_id = meta.use_mcp ? 'default' : null
    meta.model_provider = configStore.config?.model_provider
    meta.model_name = configStore.config?.model_name
    fetchChatResponse(user_input, cur_res_id)
  } else {
    message.warning('请输入消息')
  }
}

const retryMessage = (id) => {
  // 找到 id 对应的 message，然后删除包含 message 在内以及后面所有的 message
  const index = conv.value.messages.findIndex((msg) => msg.id === id)
  const pastMessage = conv.value.messages[index - 1]
  conv.value.inputText = pastMessage.content
  if (index !== -1) {
    conv.value.messages = conv.value.messages.slice(0, index - 1)
  }
  sendMessage()
}

// 从本地存储加载数据
onMounted(() => {
  scrollToBottom()
  loadDatabases()

  if (chatContainer.value) chatContainer.value.addEventListener('scroll', handleUserScroll)

  // 检查现有消息中是否有内容为空的情况
  if (conv.value.messages && conv.value.messages.length > 0) {
    conv.value.messages.forEach((msg) => {
      if (msg.role === 'received' && (!msg.content || msg.content.trim() === '')) {
        msg.status = 'error'
        msg.message = '内容加载失败'
      }
    })
  }
})

onUnmounted(() => {
  activeStreamId.value += 1
  activeChatAbortController.value?.abort?.()
  if (chatContainer.value) {
    chatContainer.value.removeEventListener('scroll', handleUserScroll)
  }
})

// 添加新函数来处理特定的滚动行为
const forceScrollToBottom = () => {
  shouldAutoScroll.value = true
  setTimeout(() => {
    chatContainer.value.scrollTop =
      chatContainer.value.scrollHeight - chatContainer.value.clientHeight
  }, 10)
}

// 监听 meta 对象的变化，并保存到本地存储
watch(meta, () => persistMeta(), { deep: true })
// 处理发送或停止
const handleSendOrStop = () => {
  if (isStreaming.value) {
    // 停止生成
    isStreaming.value = false
    activeChatAbortController.value?.abort?.()
    const lastMessage = conv.value.messages[conv.value.messages.length - 1]
    if (lastMessage) {
      lastMessage.isStoppedByUser = true
      lastMessage.status = 'stopped'
    }
  } else {
    // 发送消息
    sendMessage()
  }
}

// 重试被停止的消息
const retryStoppedMessage = (id) => {
  // 找到用户的原始问题
  const messageIndex = conv.value.messages.findIndex((msg) => msg.id === id)
  if (messageIndex > 0) {
    const userMessage = conv.value.messages[messageIndex - 1]
    if (userMessage && (userMessage.role === 'sent' || userMessage.role === 'user')) {
      conv.value.inputText = userMessage.content
      // 删除被停止的消息，以及所有后面的消息
      conv.value.messages = conv.value.messages.slice(0, messageIndex - 1)
      // sendMessage();
    }
  }
}

const modelNames = computed(() => configStore.config?.model_names || {})
const customModels = computed(() => configStore.config?.custom_models || [])

// 筛选 modelStatus 中为真的key
const modelKeys = computed(() => {
  return Object.keys(modelNames.value || {}).filter((k) => k !== 'custom')
})

// 选择模型的方法
const selectModel = (provider, name) => {
  configStore.setConfigValues({ model_provider: provider, model_name: name })
  message.success(`已切换到模型: ${provider}/${name}`)
}
</script>

<style lang="less" scoped>
.chat {
  position: relative;
  width: 100%;
  max-height: 100vh;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;

  /* --- Rotom-Dex Screen Background --- */
  background-color: var(--background-color);

  box-sizing: border-box;
  flex: 5 5 200px;
  overflow-y: scroll;

  .chat-header {
    user-select: none;
    position: sticky;
    top: 0;
    z-index: 10;

    /* Glassmorphism Header */
    background: var(--glass-bg);
    backdrop-filter: blur(var(--blur-md));
    border-bottom: 1px solid var(--border-color);

    height: var(--header-height);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-3) var(--space-6);

    .header__left,
    .header__right {
      display: flex;
      align-items: center;
      gap: var(--space-3);
    }

    .header__left {
      .close {
        margin-right: 0;
      }
    }
  }

  .nav-btn {
    height: var(--button-height-lg);
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: var(--radius-sm);
    color: var(--text-color);
    cursor: pointer;
    width: auto;
    transition: all var(--duration-slow);
    padding: var(--space-2) var(--space-3);

    .text {
      margin-left: var(--space-2);
      font-weight: 500;
    }

    &:hover {
      background-color: var(--hover-bg);
      color: var(--primary-color);
    }
  }

  .model-select {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: 6px var(--space-4);
    border-radius: var(--radius-lg);
    /* Tech Pill Style */
    border: 1px solid var(--border-color);
    background: color-mix(in srgb, var(--glass-bg) 70%, transparent);
    backdrop-filter: blur(var(--blur-sm));

    color: var(--text-color);
    cursor: pointer;
    transition: all var(--duration-base) ease;
    max-width: 300px;
    box-shadow: var(--shadow-xs);

    &:hover {
      background: var(--surface-color);
      border-color: var(--primary-color);
      box-shadow: var(--shadow-sm);
      transform: translateY(-1px);
    }

    .text {
      font-weight: 500;
      font-size: var(--font-size-sm);
    }

    .model-select-icon {
      width: 18px;
      height: 18px;
      border-radius: var(--radius-xs);
      object-fit: contain;
    }

    .model-select-caret {
      font-size: 10px;
      color: var(--gray-500);
    }
  }
}

/* Global Background Decoration - Full Page Coverage */
.chat-bg-decor {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;

  .bg-image {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    object-fit: cover;
    object-position: center top;
    opacity: 0.25;
    /* Gradient mask: visible at top, fading towards bottom */
    mask-image: linear-gradient(
      to bottom,
      rgba(0, 0, 0, 0.8) 0%,
      rgba(0, 0, 0, 0.5) 30%,
      rgba(0, 0, 0, 0.2) 60%,
      rgba(0, 0, 0, 0) 100%
    );
    -webkit-mask-image: linear-gradient(
      to bottom,
      rgba(0, 0, 0, 0.8) 0%,
      rgba(0, 0, 0, 0.5) 30%,
      rgba(0, 0, 0, 0.2) 60%,
      rgba(0, 0, 0, 0) 100%
    );
  }
}

:root[data-theme='dark'] .chat-bg-decor .bg-image {
  opacity: 0.15;
}

.scroll-to-bottom {
  z-index: 12;
}

.model-menu-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: 100%;
}

.model-menu-icon {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  background: var(--surface-color-2);
  object-fit: contain;
  flex: 0 0 auto;
}

.model-menu-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.metas {
  display: flex;
  gap: 8px;
}

.options-panel {
  position: absolute;
  margin-top: var(--space-2);
  background-color: color-mix(in srgb, var(--surface-color) 95%, transparent);
  backdrop-filter: blur(var(--blur-md));
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-md);
  border-radius: var(--radius-md);
  padding: var(--space-4);
  z-index: 11;
  width: 300px;
  transition: transform var(--duration-base) ease, opacity var(--duration-base) ease;

  .options-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background-color var(--duration-base);
    font-weight: 500;
    color: var(--gray-700);

    &:hover {
      background-color: var(--hover-bg);
      color: var(--primary-color);
    }

    .anticon { margin-right: var(--space-2); }
  }
}

.options-panel--right {
  top: 100%;
  right: 0;
}

.options-panel--left {
  top: 100%;
  left: 0;
}

.chat-empty {
  flex: 1 1 auto;
  min-height: 55vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: var(--space-8);
  text-align: center;
  padding: var(--space-8) 0;
  z-index: 1;

  .chat-empty__hero {
    max-width: 720px;
    h1 {
      margin: 0 0 var(--space-3);
      font-size: var(--font-size-3xl);
      font-weight: 800;
      color: var(--gray-900);
      letter-spacing: -0.5px;
      background: var(--gradient-hero);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p {
      margin: 0;
      color: var(--gray-600);
      font-size: var(--font-size-lg);
    }
  }

  .chat-empty__prompts {
    width: 100%;
    max-width: 680px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
  }

  .prompt-card {
    position: relative;
    text-align: left;
    /* Rich Tech Data Chip Style */
    background: var(--glass-bg);
    backdrop-filter: blur(var(--blur-md));
    border: 1px solid var(--border-color);
    color: var(--gray-800);
    border-radius: var(--radius-md);
    padding: var(--space-4) var(--space-5);
    cursor: pointer;
    box-shadow: var(--shadow-xs);
    transition: all var(--duration-slow) var(--ease-default);
    font-weight: 500;
    overflow: hidden;

    /* Accent Strip */
    &::before {
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: var(--pokedex-red);
      opacity: 0.6;
      transition: all var(--duration-slow) ease;
    }

    &:hover {
      background: var(--surface-color);
      border-color: var(--pokedex-red);
      box-shadow: var(--shadow-sm);
      transform: translateY(-2px);

      &::before {
        opacity: 1;
        width: 6px;
        box-shadow: 0 0 8px var(--pokedex-red);
      }
    }

    &:active {
      transform: translateY(0);
      box-shadow: var(--shadow-xs);
    }
  }
}

.chat-box {
  width: 100%;
  max-width: 820px;
  margin: 0 auto;
  flex-grow: 1;
  padding: 1rem 2rem;
  display: flex;
  flex-direction: column;
  transition: max-width 0.3s ease;
  z-index: 1;

  &.wide-screen {
    max-width: 960px;
  }

  &.font-smaller {
    font-size: 14px;
    .message-box { font-size: 14px; }
  }

  &.font-larger {
    font-size: 16px;
    .message-box { font-size: 16px; }
  }
}

.bottom {
  position: sticky;
  bottom: 0;
  width: 100%;
  margin: 0 auto;
  padding: var(--space-3) 2rem var(--space-6) 2rem;
  /* Subtle gradient fade - more transparent to let global background show through */
  background: linear-gradient(to top, color-mix(in srgb, var(--background-color) 85%, transparent) 40%, transparent);
  z-index: 20;

  .message-input-wrapper {
    width: 100%;
    max-width: 820px;
    margin: 0 auto;

    /* Glassmorphism Style - Cockpit feel */
    background-color: color-mix(in srgb, var(--surface-color) 90%, transparent);
    backdrop-filter: blur(var(--blur-lg));
    -webkit-backdrop-filter: blur(var(--blur-lg));
    border-radius: var(--radius-xl);
    box-shadow:
      0 4px 20px rgba(0, 0, 0, 0.06),
      0 8px 40px rgba(255, 125, 0, 0.08);
    border: 1px solid color-mix(in srgb, var(--border-color) 70%, transparent);

    padding: var(--space-1);
    animation: width var(--duration-slow) ease-in-out;
    transition: all var(--duration-slow) ease;

    &:focus-within {
      box-shadow:
        0 4px 20px rgba(0, 0, 0, 0.08),
        0 12px 48px rgba(255, 125, 0, 0.12);
      border-color: color-mix(in srgb, var(--primary-color) 25%, transparent);
      transform: translateY(-2px);
    }

    &.wide-screen {
      max-width: 960px;
    }

    .note {
      width: 100%;
      font-size: 11px;
      text-align: center;
      padding: 0;
      color: var(--gray-400);
      margin-top: var(--space-2);
      margin-bottom: 0;
      user-select: none;
      opacity: 0.8;
    }
  }
}

.ant-dropdown-link {
  color: var(--gray-900);
  cursor: pointer;
}

.action-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: var(--space-2) var(--space-4);
  font-size: var(--font-size-base);
  font-weight: 500;
  color: var(--message-user-text);
  /* Primary Action Button */
  background: var(--gradient-primary);
  border: none;
  border-radius: var(--radius-lg);
  cursor: pointer;
  box-shadow: var(--message-user-shadow);
  transition: all var(--duration-base) ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
    background: var(--gradient-primary-hover);
  }

  .icon {
    font-size: var(--font-size-lg);
  }

  .text {
    white-space: nowrap;
  }
}

/* Scrollbar Styling */
.chat::-webkit-scrollbar {
  width: 6px;
}
.chat::-webkit-scrollbar-track {
  background: transparent;
}
.chat::-webkit-scrollbar-thumb {
  background: rgba(0,0,0,0.1);
  border-radius: 3px;
}
.chat::-webkit-scrollbar-thumb:hover {
  background: rgba(0,0,0,0.2);
}

/* Animations */
.slide-out-left {
  animation: slide-out-left 0.5s cubic-bezier(0.55, 0.085, 0.68, 0.53) both;
}
.swing-in-top-fwd {
  animation: swing-in-top-fwd 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
}

@keyframes swing-in-top-fwd {
  0% {
    transform: rotateX(-100deg);
    transform-origin: top;
    opacity: 0;
  }
  100% {
    transform: rotateX(0deg);
    transform-origin: top;
    opacity: 1;
  }
}

@media (max-width: 520px) {
  .chat {
    height: calc(100vh - 60px);
  }

  .chat-container .chat .chat-header {
    background: rgba(255,255,255,0.9);
    padding: 8px 16px;

    .header__left,
    .header__right {
      gap: 16px;
    }
  }

  .bottom {
    padding: 0 1rem 1rem 1rem;
    
    .message-input-wrapper {
        border-radius: 16px;
    }

    .note {
      display: none;
    }
  }

  .chat-box {
    padding: 1rem;
  }

  .chat-empty {
    min-height: 45vh;
    padding: 24px 0;
    .chat-empty__prompts {
      grid-template-columns: 1fr;
    }
  }

  .scroll-to-bottom {
    bottom: 84px !important;
  }
}

.controls {
  display: flex;
  align-items: center;
  gap: 8px;
  .search-switch {
    margin-right: 8px;
  }
}

.scrollable-menu {
  max-height: 300px;
  overflow-y: auto;
  &::-webkit-scrollbar { width: 6px; }
  &::-webkit-scrollbar-track { background: transparent; }
  &::-webkit-scrollbar-thumb { background: var(--gray-400); border-radius: 3px; }
}

</style>

<style lang="less">
// 添加全局样式以确保滚动功能在dropdown内正常工作
.ant-dropdown-menu {
  &.scrollable-menu {
    max-height: 300px;
    overflow-y: auto;
  }
}
</style>
