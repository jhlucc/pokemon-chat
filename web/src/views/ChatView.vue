<template>
  <div class="chat-container">
    <!--    左边是侧边栏（对话列表 conversations）-->
    <div class="conversations" :class="{ 'is-open': state.isSidebarOpen }">
      <div class="actions">
        <!-- <div class="action new" @click="addNewConv"><FormOutlined /></div> -->
        <div class="actions-left">
          <div
            class="action new"
            @click="addNewConv"
            title="新建对话"
            role="button"
            tabindex="0"
            aria-label="新建对话"
            @keydown.enter.prevent="addNewConv"
            @keydown.space.prevent="addNewConv"
          >
            <!-- 使用 PlusCircleOutlined 比较直观 -->
            <PlusCircleOutlined />
          </div>
        </div>
        <span class="header-title">对话历史</span>
        <div class="actions-right">
          <a-dropdown placement="bottomRight" trigger="click">
            <div
              class="action more"
              title="更多操作"
              role="button"
              tabindex="0"
              aria-label="更多操作"
              @click.prevent
              @keydown.enter.prevent="$event.currentTarget?.click?.()"
              @keydown.space.prevent="$event.currentTarget?.click?.()"
            >
              <MoreOutlined />
            </div>
            <template #overlay>
              <a-menu>
                <a-menu-item @click="exportConversations">
                  <DownloadOutlined /> 导出对话
                </a-menu-item>
                <a-menu-item @click="triggerImportConversations">
                  <UploadOutlined /> 导入对话
                </a-menu-item>
                <a-menu-divider />
                <a-menu-item danger @click="clearAllConversations">
                  <ClearOutlined /> 清空所有对话
                </a-menu-item>
              </a-menu>
            </template>
          </a-dropdown>

          <div
            class="action close"
            @click="state.isSidebarOpen = false"
            role="button"
            tabindex="0"
            aria-label="关闭侧栏"
            @keydown.enter.prevent="state.isSidebarOpen = false"
            @keydown.space.prevent="state.isSidebarOpen = false"
          >
            <img
              src="@/assets/icons/sidebar_left.svg"
              class="iconfont icon-20 sidebar-icon"
              alt="侧栏"
            />
          </div>
        </div>
      </div>
      <input
        ref="importInput"
        type="file"
        accept="application/json"
        style="display: none"
        @change="onImportConversationsFileChange"
      />
      <div class="conversation-search">
        <a-input v-model:value="convSearch" allow-clear size="small" placeholder="搜索对话" />
      </div>
      <div class="conversation-list">
        <template v-if="filteredConvs.length > 0">
          <div
            v-for="item in filteredConvs"
            :key="item.conv.id || item.index"
            class="conversation"
            :class="{ active: curConvId === item.index }"
            @click="goToConversation(item.index)"
            role="button"
            tabindex="0"
            :aria-current="curConvId === item.index ? 'page' : undefined"
            @keydown.enter.prevent="goToConversation(item.index)"
            @keydown.space.prevent="goToConversation(item.index)"
          >
            <div class="conversation__title"><CommentOutlined /> &nbsp;{{ item.conv.title }}</div>
            <a-popconfirm
              title="确定删除该对话吗？"
              ok-text="删除"
              cancel-text="取消"
              @confirm="delConv(item.index)"
              @click.stop
            >
              <div class="conversation__delete" @click.stop><DeleteOutlined /></div>
            </a-popconfirm>
          </div>
        </template>
        <a-empty v-else class="conversation-empty" description="暂无对话">
          <a-space>
            <a-button size="small" type="primary" @click="addNewConv">新建对话</a-button>
            <a-button size="small" @click="convSearch = ''">清除搜索</a-button>
          </a-space>
        </a-empty>
      </div>
    </div>
    <!--    聊天组件（ChatComponent） 渲染右边聊天内容区域。  把当前选中的对话 (convs[curConvId]) 作为 prop 传给 ChatComponent,传递状态对象 state-->
    <div
      v-if="state.isSidebarOpen"
      class="sidebar-mask"
      @click="state.isSidebarOpen = false"
      aria-hidden="true"
    />
    <ChatComponent
      :conv="convs[curConvId]"
      :state="state"
      @rename-title="renameTitle"
      @newconv="addNewConv"
    />
    <!--  重命名对话&新建对话-->
  </div>
</template>

<script setup>
import { reactive, ref, watch, computed } from 'vue'

import ChatComponent from '@/components/ChatComponent.vue'
import { Modal, message } from 'ant-design-vue'
import {
  DeleteOutlined,
  CommentOutlined,
  PlusCircleOutlined,
  MoreOutlined,
  DownloadOutlined,
  UploadOutlined,
  ClearOutlined
} from '@ant-design/icons-vue'
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
  return {
    id: randomId(8),
    title: '新对话',
    history: [],
    messages: [],
    inputText: ''
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

const convSearch = ref('')
const filteredConvs = computed(() => {
  const q = (convSearch.value || '').trim().toLowerCase()
  const list = Array.from(convs).map((conv, index) => ({ conv, index }))
  if (!q) return list
  return list.filter((item) => (item.conv?.title || '').toLowerCase().includes(q))
})

const renameTitle = (newTitle) => {
  convs[curConvId.value].title = newTitle
}

const goToConversation = (index) => {
  curConvId.value = index
}

const addNewConv = () => {
  curConvId.value = 0
  if (convs.length > 0 && convs[0].messages.length === 0) {
    return
  }
  convs.unshift({
    id: randomId(8),
    title: `新对话`,
    history: [],
    messages: [],
    inputText: ''
  })
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

.conversations {
  width: 260px;
  max-width: 260px;
  background-color: var(--bg-sider);
  border-right: 1px solid var(--border-color);
  /* Use adaptive text color */
  color: var(--text-color);
  
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 100%;

  &.is-open {
    width: 260px;
  }

  &:not(.is-open) {
    width: 0;
    padding: 0;
    overflow: hidden;
  }
  
  /* --- Top Actions --- */
  & .actions {
    height: var(--header-height);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    z-index: 9;
    border-bottom: 1px solid var(--border-color);

    .header-title {
      font-weight: 600;
      color: var(--text-color);
      letter-spacing: 0.5px;
      user-select: none;
    }

    .action {
      font-size: 1.2rem;
      width: 2.2rem;
      height: 2.2rem;
      display: flex;
      justify-content: center;
      align-items: center;
      border-radius: 50%;
      color: var(--gray-600);
      cursor: pointer;
      transition: all 0.2s;

      &:hover {
        background-color: var(--hover-bg);
        color: var(--text-color);
      }
      
      /* New Chat Button: Prominent Red Circle */
      &.new {
        background-color: var(--pokedex-red);
        color: #fff;
        box-shadow: 0 4px 12px rgba(255, 83, 80, 0.4);
        
        &:hover {
          transform: scale(1.1) rotate(90deg);
          box-shadow: 0 6px 16px rgba(255, 83, 80, 0.6);
        }
      }
    }
  }

  /* --- Search Bar --- */
  .conversation-search {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    
    :deep(.ant-input-affix-wrapper) {
      background-color: var(--input-background-color);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      padding-left: 12px;
      
       input {
          background-color: transparent !important;
       }
    }
  }

  /* --- List Items --- */
  .conversation-list {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    flex: 1 1 auto;
    padding-top: 8px;
  }

  .conversation-list .conversation {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    margin: 4px 10px;
    cursor: pointer;
    user-select: none;
    border-radius: 12px;
    border: 1px solid transparent;
    transition: all 0.2s ease-in-out;
    position: relative;
    color: var(--text-color);

    &__title {
      color: var(--gray-700);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 8px;
      
      .anticon {
        font-size: 1.1em;
        opacity: 0.7;
      }
    }

    &__delete {
      display: none;
      color: var(--gray-500);
      
      &:hover {
        color: var(--danger-500);
      }
    }

    /* Hover State */
    &:not(.active):hover {
      background-color: var(--hover-bg);
      
      & .conversation__delete {
        display: block;
      }
    }

    /* Active State: Glowing Data Card */
    &.active {
      background: linear-gradient(90deg, rgba(255, 83, 80, 0.15) 0%, rgba(255, 83, 80, 0.05) 100%);
      border: 1px solid rgba(255, 83, 80, 0.3);
      
      & .conversation__title {
        color: var(--pokedex-red);
        font-weight: 600;
        
        .anticon {
           color: var(--pokedex-red);
           opacity: 1;
        }
      }
      
      /* Active Indicator Bar */
      &::before {
        content: '';
        position: absolute;
        left: 0;
        top: 15%;
        bottom: 15%;
        width: 3px;
        background-color: var(--pokedex-red);
        border-radius: 0 4px 4px 0;
      }
    }
  }

  .conversation-empty {
    padding: 28px 12px;
    :deep(.ant-empty-description) {
      color: var(--gray-500);
    }
  }
}

.conversation-list::-webkit-scrollbar {
  position: absolute;
  width: 4px;
}

.conversation-list::-webkit-scrollbar-track {
  background: transparent;
  border-radius: 4px;
}

.conversation-list::-webkit-scrollbar-thumb {
  background: var(--gray-400);
  border-radius: 4px;
}

.conversation-list::-webkit-scrollbar-thumb:hover {
  background: rgb(100, 100, 100);
  border-radius: 4px;
}

.conversation-list::-webkit-scrollbar-thumb:active {
  background: rgb(68, 68, 68);
  border-radius: 4px;
}

@media (max-width: 520px) {
  .sidebar-mask {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.28);
    z-index: 100;
  }

  .conversations {
    position: absolute;
    z-index: 101;
    width: 300px;
    height: 100%;
    border-radius: 0 16px 16px 0;
    box-shadow: 0 0 10px 1px rgba(0, 0, 0, 0.05);

    &:not(.is-open) {
      width: 0;
      padding: 0;
      overflow: hidden;
    }
  }
}
</style>
