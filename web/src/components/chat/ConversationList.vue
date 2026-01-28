<template>
  <div class="conversation-sidebar" :class="{ 'is-open': isOpen }">
    <!-- Header -->
    <div class="sidebar-header">
      <div class="header-left">
        <button
          class="action-btn action-btn--primary"
          @click="emit('new-conversation')"
          title="新建对话"
          aria-label="新建对话"
        >
          <PlusOutlined />
        </button>
      </div>
      <span class="header-title">对话历史</span>
      <div class="header-right">
        <a-dropdown placement="bottomRight" trigger="click">
          <button
            class="action-btn"
            title="更多操作"
            aria-label="更多操作"
            @click.prevent
          >
            <MoreOutlined />
          </button>
          <template #overlay>
            <a-menu class="sidebar-menu">
              <a-menu-item @click="emit('export')">
                <DownloadOutlined /> 导出对话
              </a-menu-item>
              <a-menu-item @click="emit('import')">
                <UploadOutlined /> 导入对话
              </a-menu-item>
              <a-menu-divider />
              <a-menu-item danger @click="emit('clear-all')">
                <DeleteOutlined /> 清空所有对话
              </a-menu-item>
            </a-menu>
          </template>
        </a-dropdown>

        <button
          class="action-btn"
          @click="emit('close')"
          title="关闭侧栏"
          aria-label="关闭侧栏"
        >
          <LeftOutlined />
        </button>
      </div>
    </div>

    <!-- Search -->
    <div class="sidebar-search">
      <a-input
        v-model:value="searchQuery"
        allow-clear
        size="small"
        placeholder="搜索对话..."
      >
        <template #prefix>
          <SearchOutlined class="search-icon" />
        </template>
      </a-input>
    </div>

    <!-- Conversation List -->
    <div
      class="conversation-list"
      ref="listRef"
      role="listbox"
      aria-label="对话列表"
    >
      <template v-if="groupedConversations.length > 0">
        <div
          v-for="group in groupedConversations"
          :key="group.label"
          class="conversation-group"
        >
          <div class="group-header">
            <span class="group-label">{{ group.label }}</span>
            <span class="group-count">{{ group.items.length }}</span>
          </div>

          <div
            v-for="item in group.items"
            :key="item.conv.id || item.index"
            class="conversation-item"
            :class="{ active: activeIndex === item.index }"
            @click="emit('select', item.index)"
            role="option"
            tabindex="0"
            :aria-selected="activeIndex === item.index"
            :aria-label="item.conv.title || '未命名对话'"
            @keydown.enter.prevent="emit('select', item.index)"
            @keydown.space.prevent="emit('select', item.index)"
          >
            <div class="item-content">
              <div class="item-title" v-html="highlightText(item.conv.title, searchQuery)"></div>
              <div v-if="getLastMessage(item.conv)" class="item-preview">
                {{ getLastMessage(item.conv) }}
              </div>
              <div class="item-meta">
                <span class="item-time">{{ formatTime(item.conv.updatedAt || item.conv.createdAt) }}</span>
                <span v-if="item.conv.messages?.length" class="item-count">
                  {{ item.conv.messages.length }} 条消息
                </span>
              </div>
            </div>

            <div class="item-actions">
              <a-popconfirm
                title="确定删除该对话吗？"
                ok-text="删除"
                cancel-text="取消"
                placement="right"
                @confirm="emit('delete', item.index)"
                @click.stop
              >
                <button class="item-action-btn" @click.stop title="删除对话">
                  <DeleteOutlined />
                </button>
              </a-popconfirm>
            </div>
          </div>
        </div>
      </template>

      <a-empty v-else class="empty-state">
        <template #image>
          <img src="/empty-chat.png" alt="暂无对话" class="empty-image" />
        </template>
        <template #description>
          <span class="empty-text">暂无对话</span>
        </template>
        <a-space direction="vertical" :size="8">
          <a-button type="primary" @click="emit('new-conversation')">
            <PlusOutlined /> 新建对话
          </a-button>
          <a-button v-if="searchQuery" @click="searchQuery = ''">
            清除搜索
          </a-button>
        </a-space>
      </a-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import {
  PlusOutlined,
  MoreOutlined,
  DownloadOutlined,
  UploadOutlined,
  DeleteOutlined,
  LeftOutlined,
  SearchOutlined
} from '@ant-design/icons-vue'

const props = defineProps({
  conversations: {
    type: Array,
    default: () => []
  },
  activeIndex: {
    type: Number,
    default: 0
  },
  isOpen: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits([
  'select',
  'delete',
  'new-conversation',
  'close',
  'export',
  'import',
  'clear-all'
])

const searchQuery = ref('')
const listRef = ref(null)

// Date grouping helper
const getDateGroup = (timestamp) => {
  if (!timestamp) return 'older'

  const now = new Date()
  const date = new Date(timestamp)
  const diffMs = now - date
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) {
    // Check if it's actually today
    if (date.toDateString() === now.toDateString()) {
      return 'today'
    }
    return 'yesterday'
  } else if (diffDays === 1) {
    return 'yesterday'
  } else if (diffDays < 7) {
    return 'thisWeek'
  } else if (diffDays < 30) {
    return 'thisMonth'
  } else {
    return 'older'
  }
}

const groupLabels = {
  today: '今天',
  yesterday: '昨天',
  thisWeek: '本周',
  thisMonth: '本月',
  older: '更早'
}

const groupOrder = ['today', 'yesterday', 'thisWeek', 'thisMonth', 'older']

// Filter and group conversations
const groupedConversations = computed(() => {
  const q = (searchQuery.value || '').trim().toLowerCase()

  // Create indexed items
  let items = props.conversations.map((conv, index) => ({ conv, index }))

  // Filter by search query
  if (q) {
    items = items.filter((item) => {
      const titleMatch = (item.conv?.title || '').toLowerCase().includes(q)
      const contentMatch = item.conv?.messages?.some((msg) =>
        (msg.content || '').toLowerCase().includes(q)
      )
      return titleMatch || contentMatch
    })
  }

  // Group by date
  const groups = {}
  items.forEach((item) => {
    const groupKey = getDateGroup(item.conv?.updatedAt || item.conv?.createdAt)
    if (!groups[groupKey]) {
      groups[groupKey] = []
    }
    groups[groupKey].push(item)
  })

  // Convert to array and sort by order
  return groupOrder
    .filter((key) => groups[key]?.length > 0)
    .map((key) => ({
      key,
      label: groupLabels[key],
      items: groups[key]
    }))
})

// Get last message preview
const getLastMessage = (conv) => {
  if (!conv?.messages?.length) return null
  const lastMsg = conv.messages[conv.messages.length - 1]
  if (!lastMsg?.content) return null
  const text = lastMsg.content.slice(0, 50)
  return text.length < lastMsg.content.length ? text + '...' : text
}

// Format time
const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  const now = new Date()

  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }

  const yesterday = new Date(now)
  yesterday.setDate(yesterday.getDate() - 1)
  if (date.toDateString() === yesterday.toDateString()) {
    return '昨天 ' + date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }

  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

// Highlight search text
const highlightText = (text, query) => {
  if (!query || !text) return text
  const q = query.trim().toLowerCase()
  if (!q) return text

  const regex = new RegExp(`(${escapeRegExp(q)})`, 'gi')
  return text.replace(regex, '<mark class="highlight">$1</mark>')
}

const escapeRegExp = (string) => {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
</script>

<style scoped lang="less">
.conversation-sidebar {
  width: 280px;
  max-width: 280px;
  background: var(--bg-sider);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  height: 100%;
  transition: width var(--duration-base) var(--ease-default);
  overflow: hidden;

  &:not(.is-open) {
    width: 0;
    max-width: 0;
  }
}

/* Header */
.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: var(--header-height);
  padding: var(--space-8) var(--space-4) var(--space-3);
  border-bottom: 1px solid var(--border-color);
  flex-shrink: 0;

  .header-title {
    font-weight: 600;
    font-size: var(--font-size-base);
    color: var(--text-color);
    letter-spacing: 0.025em;
    line-height: 32px;
  }

  .header-left,
  .header-right {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: var(--gray-600);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  flex-shrink: 0;

  &:hover {
    background: var(--hover-bg);
    color: var(--text-color);
  }

  &--primary {
    background: var(--primary-color);
    color: white;
    box-shadow: 0 2px 8px rgba(255, 125, 0, 0.25);

    &:hover {
      background: var(--primary-light-color);
      transform: scale(1.05);
      box-shadow: 0 3px 12px rgba(255, 125, 0, 0.35);
    }
  }
}

/* Search */
.sidebar-search {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--border-color);
  flex-shrink: 0;

  :deep(.ant-input-affix-wrapper) {
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 6px 12px;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04);

    &:hover {
      border-color: var(--gray-300);
      background: var(--gray-0);
    }

    &:focus-within {
      border-color: var(--primary-color);
      background: var(--gray-0);
      box-shadow: 0 0 0 2px rgba(255, 83, 80, 0.1);
    }

    input {
      background: transparent !important;
    }
  }

  .search-icon {
    color: var(--gray-400);
  }
}

/* Conversation List */
.conversation-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-2) 0;

  &::-webkit-scrollbar {
    width: 4px;
  }

  &::-webkit-scrollbar-track {
    background: transparent;
  }

  &::-webkit-scrollbar-thumb {
    background: var(--gray-300);
    border-radius: 2px;
  }
}

/* Group */
.conversation-group {
  margin-bottom: var(--space-2);
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-4);
  font-size: var(--font-size-xs);
  color: var(--gray-500);
  user-select: none;

  .group-label {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .group-count {
    background: var(--surface-color-2);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-size: var(--font-size-xs);
    color: var(--gray-500);
  }
}

/* Conversation Item */
.conversation-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: var(--space-3) var(--space-4);
  margin: 3px var(--space-3);
  border-radius: 12px;
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  position: relative;
  border: 1px solid transparent;

  &:hover {
    background: var(--hover-bg);

    .item-actions {
      opacity: 1;
    }
  }

  /* 选中态：柔和的胶囊形晨光效果 */
  &.active {
    background: color-mix(in srgb, var(--primary-color) 10%, var(--surface-color));
    border-color: color-mix(in srgb, var(--primary-color) 20%, transparent);
    box-shadow: 0 2px 8px rgba(255, 125, 0, 0.08);

    .item-title {
      color: var(--primary-color);
      font-weight: 600;
    }

    .item-time {
      color: var(--primary-color);
      opacity: 0.7;
    }
  }
}

.item-content {
  flex: 1;
  min-width: 0;
  overflow: hidden;
}

.item-title {
  font-size: var(--font-size-sm);
  font-weight: 500;
  color: var(--text-color);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 2px;

  :deep(.highlight) {
    background: var(--warning-color);
    color: white;
    padding: 0 2px;
    border-radius: 2px;
  }
}

.item-preview {
  font-size: var(--font-size-xs);
  color: var(--gray-500);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 4px;
}

.item-meta {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--font-size-xs);
  color: var(--gray-400);

  .item-time {
    font-family: var(--font-family-mono);
  }

  .item-count {
    &::before {
      content: '·';
      margin-right: var(--space-2);
    }
  }
}

.item-actions {
  display: flex;
  align-items: center;
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease-default);
  flex-shrink: 0;
  margin-left: var(--space-2);
}

.item-action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--gray-400);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: color-mix(in srgb, var(--error-color) 10%, transparent);
    color: var(--error-color);
  }
}

/* Empty State */
.empty-state {
  padding: var(--space-8) var(--space-4);

  .empty-image {
    width: 120px;
    height: 120px;
    object-fit: contain;
    margin-bottom: var(--space-4);
    opacity: 0.9;
  }

  .empty-text {
    color: var(--gray-500);
    font-size: var(--font-size-sm);
  }

  :deep(.ant-empty-description) {
    color: var(--gray-500);
  }
}

/* Sidebar Menu */
.sidebar-menu {
  min-width: 160px;

  :deep(.ant-dropdown-menu-item) {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
  }
}

/* Responsive */
@media (max-width: 640px) {
  .conversation-sidebar {
    position: fixed;
    z-index: 1000;
    width: 300px;
    max-width: 85vw;
    box-shadow: var(--shadow-lg);
    border-radius: 0 var(--radius-lg) var(--radius-lg) 0;

    &:not(.is-open) {
      transform: translateX(-100%);
    }
  }
}
</style>
