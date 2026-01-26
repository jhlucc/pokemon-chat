<template>
  <div class="chat-container">
    <!-- Sidebar -->
    <ConversationList
      :conversations="convs"
      :active-index="curConvId"
      :is-open="state.isSidebarOpen"
      @select="goToConversation"
      @delete="delConv"
      @new-conversation="addNewConv"
      @close="state.isSidebarOpen = false"
      @export="exportConversations"
      @import="triggerImportConversations"
      @clear-all="clearAllConversations"
    />

    <input
      ref="importInput"
      type="file"
      accept="application/json"
      style="display: none"
      @change="onImportConversationsFileChange"
    />

    <!-- Sidebar Mask (Mobile) -->
    <div
      v-if="state.isSidebarOpen"
      class="sidebar-mask"
      @click="state.isSidebarOpen = false"
      aria-hidden="true"
    />

    <!-- Chat Component -->
    <ChatComponent
      ref="chatRef"
      :conv="convs[curConvId]"
      :state="state"
      @rename-title="renameTitle"
      @newconv="addNewConv"
    />

    <!-- Keyboard Shortcuts Help -->
    <KeyboardShortcutsHelp
      ref="shortcutsHelpRef"
      :shortcuts="keyboardShortcuts"
    />
  </div>
</template>

<script setup>
import { reactive, ref, watch, onMounted, onUnmounted } from 'vue'

import ChatComponent from '@/components/ChatComponent.vue'
import ConversationList from '@/components/chat/ConversationList.vue'
import KeyboardShortcutsHelp from '@/components/common/KeyboardShortcutsHelp.vue'
import { Modal, message } from 'ant-design-vue'
import { useDebounceFn } from '@vueuse/core'
import { randomId } from '@/utils/id'
import { readJson, writeJson } from '@/utils/storage'
import { downloadJson } from '@/utils/download'
// 从 localStorage 里读取历史对话记录，如果没有就用一个初始默认对话。
const CONVS_STORAGE_KEY = 'chat-convs'
const SIDEBAR_STORAGE_KEY = 'chat-sidebar-open'
const MAX_CONVS_TO_STORE = 30
const MAX_MESSAGES_PER_CONV = 200

function makeEmptyConv() {
  const now = Date.now()
  return {
    id: randomId(8),
    title: '新对话',
    history: [],
    messages: [],
    inputText: '',
    createdAt: now,
    updatedAt: now
  }
}

function pruneConvsForStorage(list) {
  const safeList = Array.isArray(list) ? list : []
  return safeList.slice(0, MAX_CONVS_TO_STORE).map((c) => ({
    ...c,
    messages: Array.isArray(c?.messages) ? c.messages.slice(-MAX_MESSAGES_PER_CONV) : []
  }))
}

function normalizeImportedConvs(payload) {
  const rawList = Array.isArray(payload)
    ? payload
    : payload?.convs || payload?.chat_convs || payload?.data || []
  if (!Array.isArray(rawList)) return [makeEmptyConv()]

  const list = rawList
    .filter((c) => c && typeof c === 'object')
    .map((c) => {
      const convId = typeof c.id === 'string' && c.id ? c.id : randomId(8)
      const title = typeof c.title === 'string' && c.title ? c.title : '导入对话'
      const messages = Array.isArray(c.messages) ? c.messages : []

      const normalizedMessages = messages
        .filter((m) => m && typeof m === 'object')
        .map((m) => ({
          ...m,
          id: typeof m.id === 'string' && m.id ? m.id : randomId(16),
          role: typeof m.role === 'string' ? m.role : 'assistant',
          content: typeof m.content === 'string' ? m.content : '',
          groupedResults:
            m?.groupedResults &&
            typeof m.groupedResults === 'object' &&
            !Array.isArray(m.groupedResults)
              ? m.groupedResults
              : {}
        }))

      return {
        id: convId,
        title,
        history: Array.isArray(c.history) ? c.history : [],
        messages: normalizedMessages,
        inputText: typeof c.inputText === 'string' ? c.inputText : ''
      }
    })

  return list.length > 0 ? list : [makeEmptyConv()]
}

const storedConvs = readJson(CONVS_STORAGE_KEY, null)
const convs = reactive(
  Array.isArray(storedConvs) && storedConvs.length > 0 ? storedConvs : [makeEmptyConv()]
)

const state = reactive({
  isSidebarOpen: Boolean(readJson(SIDEBAR_STORAGE_KEY, true))
})

// Watch isSidebarOpen and save to localStorage
watch(
  () => state.isSidebarOpen,
  (newValue) => {
    writeJson(SIDEBAR_STORAGE_KEY, newValue)
  }
)
const curConvId = ref(0)
const importInput = ref(null)

const renameTitle = (newTitle) => {
  convs[curConvId.value].title = newTitle
  convs[curConvId.value].updatedAt = Date.now()
}

const goToConversation = (index) => {
  curConvId.value = index
}

const addNewConv = () => {
  curConvId.value = 0
  if (convs.length > 0 && convs[0].messages.length === 0) {
    return
  }
  convs.unshift(makeEmptyConv())
}

const delConv = (index) => {
  convs.splice(index, 1)

  if (index < curConvId.value) {
    curConvId.value -= 1
  } else if (index === curConvId.value) {
    curConvId.value = 0
  }

  if (convs.length === 0) {
    addNewConv()
  }
}

const exportConversations = () => {
  const exportedAt = new Date().toISOString()
  const safeTs = exportedAt.replace(/[:.]/g, '-')
  const payload = {
    version: 1,
    exported_at: exportedAt,
    convs: pruneConvsForStorage(convs)
  }
  const ok = downloadJson(`pokemon-chat-convs-${safeTs}.json`, payload)
  if (ok) message.success('已导出对话')
  else message.error('导出失败')
}

const triggerImportConversations = () => {
  importInput.value?.click?.()
}

const onImportConversationsFileChange = async (e) => {
  try {
    const file = e?.target?.files?.[0]
    if (!file) return
    const text = await file.text()
    const parsed = JSON.parse(text)
    const normalized = normalizeImportedConvs(parsed)

    convs.splice(0, convs.length, ...normalized)
    curConvId.value = 0
    writeJson(CONVS_STORAGE_KEY, pruneConvsForStorage(convs))
    message.success('已导入对话')
  } catch (err) {
    console.error('导入对话失败:', err)
    message.error(err?.message ? `导入失败：${err.message}` : '导入失败')
  } finally {
    // reset so the same file can be chosen again
    try {
      if (e?.target) e.target.value = ''
    } catch {
      // ignore
    }
  }
}

const clearAllConversations = () => {
  Modal.confirm({
    title: '清空所有对话',
    content: '这将删除本地保存的所有对话记录（仅影响本机浏览器）。确定继续吗？',
    okText: '清空',
    okType: 'danger',
    cancelText: '取消',
    onOk: () => {
      convs.splice(0, convs.length, makeEmptyConv())
      curConvId.value = 0
      writeJson(CONVS_STORAGE_KEY, pruneConvsForStorage(convs))
      message.success('已清空对话')
    }
  })
}

// Watch convs and save to localStorage
const persistConvs = useDebounceFn(
  () => {
    writeJson(CONVS_STORAGE_KEY, pruneConvsForStorage(convs))
  },
  600,
  { maxWait: 2000 }
)

watch(convs, () => persistConvs(), { deep: true })

// Keyboard shortcuts
const chatRef = ref(null)
const shortcutsHelpRef = ref(null)

const keyboardShortcuts = [
  { key: 'n', ctrl: true, description: '新建对话' },
  { key: 'b', ctrl: true, description: '切换侧边栏' },
  { key: '/', ctrl: false, description: '聚焦输入框' },
  { key: 'Escape', ctrl: false, description: '停止生成' }
]

const handleGlobalKeyDown = (e) => {
  const target = e.target
  const isInput =
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.isContentEditable

  // Ctrl/Cmd + N: New conversation
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'n') {
    e.preventDefault()
    addNewConv()
    return
  }

  // Ctrl/Cmd + B: Toggle sidebar
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
    e.preventDefault()
    state.isSidebarOpen = !state.isSidebarOpen
    return
  }

  // "/" to focus input (only when not in input)
  if (!isInput && e.key === '/') {
    e.preventDefault()
    // Focus the input in ChatComponent
    const inputEl = document.querySelector('.user-input textarea')
    if (inputEl) {
      inputEl.focus()
    }
    return
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleGlobalKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown)
})
</script>

<style lang="less" scoped>
.chat-container {
  display: flex;
  width: 100%;
  height: 100%;
  position: relative;
}

.sidebar-mask {
  display: none;
}

@media (max-width: 640px) {
  .sidebar-mask {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: 999;
    backdrop-filter: blur(2px);
  }
}
</style>
