<template>
  <div
    class="message-wrapper hover-reveal-trigger"
    :class="{ 'from-user': isUser, 'from-ai': !isUser }"
    role="article"
    :aria-label="isUser ? '用户消息' : '助手消息'"
  >
    <img class="avatar" :src="`/${avatar}`" :alt="isUser ? '用户头像' : '助手头像'" />

    <!-- 内容包装器：包含气泡 + 工具栏 -->
    <div class="message-content-wrapper">
      <div class="message-box" :class="message.role">
        <!-- 用户消息 -->
        <template v-if="isUser">
          {{ message.content }}
        </template>

        <!-- 助手消息 -->
        <template v-else-if="message.role === 'assistant'">
          <p v-if="debugMode" class="debug-status">{{ message.status }}</p>

          <!-- 推理过程 -->
          <div v-if="message.reasoning_content" class="reasoning-box">
            <div class="reasoning-header" @click="reasoningOpen = !reasoningOpen">
              <div class="reasoning-indicator">
                <ThunderboltOutlined class="reasoning-icon" :class="{ active: message.status === 'reasoning' }" />
                <span class="reasoning-title">
                  {{ message.status === 'reasoning' ? '正在思考...' : '推理过程' }}
                </span>
              </div>
              <CaretRightOutlined class="reasoning-caret" :class="{ open: reasoningOpen }" />
            </div>
            <transition name="collapse">
              <div v-if="reasoningOpen" class="reasoning-body">
                <p class="reasoning-content">{{ message.reasoning_content }}</p>
              </div>
            </transition>
          </div>

          <!-- 加载状态 -->
          <div v-if="isEmptyAndLoading" class="loading-state">
            <div class="typing-indicator">
              <div class="typing-indicator__dot"></div>
              <div class="typing-indicator__dot"></div>
              <div class="typing-indicator__dot"></div>
            </div>
          </div>

          <!-- 检索状态 (详细进度) -->
          <RetrievalStatus
            v-else-if="showRetrievalStatus"
            :status="message.status"
            :refs="message.refs"
            :active-step="message.meta?.active_step"
            :show-knowledge-base="showKnowledgeBase"
            :show-knowledge-graph="showKnowledgeGraph"
            :show-web-search="showWebSearch"
            :show-mcp="showMcp"
          />

          <!-- 错误状态 -->
          <div v-else-if="message.status === 'error'" class="error-msg" @click="$emit('retry')">
            <img src="/error-state.png" alt="" class="error-image" />
            <div class="error-content">
              <span class="error-text">请求出错，点击重试</span>
              <span v-if="message.message" class="error-detail">{{ message.message }}</span>
            </div>
            <ReloadOutlined class="error-retry" />
          </div>

          <!-- Markdown 内容 -->
          <MdPreview
            v-else-if="message.content"
            ref="editorRef"
            editorId="preview-only"
            previewTheme="github"
            :showCodeRowNumber="false"
            :modelValue="message.content"
            :key="message.id"
            class="message-md"
          />
          <div v-else-if="message.reasoning_content" class="empty-block"></div>

          <!-- 工具调用 -->
          <slot
            v-else-if="message.toolCalls && Object.keys(message.toolCalls).length > 0"
            name="tool-calls"
          ></slot>

          <!-- 回退错误 -->
          <div v-else class="error-msg" @click="$emit('retry')">
            <ExclamationCircleOutlined class="error-icon" />
            <div class="error-content">
              <span class="error-text">请求出错，点击重试</span>
              <span v-if="message.message" class="error-detail">{{ message.message }}</span>
            </div>
            <ReloadOutlined class="error-retry" />
          </div>

          <!-- 停止生成提示 -->
          <div v-if="message.isStoppedByUser" class="stopped-hint">
            <span class="stopped-text">已停止生成</span>
            <a class="stopped-link" @click="emit('retryStoppedMessage', message.id)">重新编辑问题</a>
          </div>
        </template>

        <!-- 自定义内容 -->
        <slot></slot>
      </div>

      <!-- 工具栏移到气泡外面 -->
      <div v-if="!isUser && message.status === 'finished'" class="message-footer">
        <!-- 消息工具栏 (hover 显示) -->
        <div class="message-toolbar hover-reveal">
          <!-- 模型名做成小标签 -->
          <span v-if="message.meta?.server_model_name" class="toolbar-model-badge">
            {{ shortenModelName(message.meta.server_model_name) }}
          </span>

          <a-tooltip title="复制">
            <button class="toolbar-btn" @click="copyContent">
              <CopyOutlined />
            </button>
          </a-tooltip>
          <a-tooltip title="重新生成">
            <button class="toolbar-btn" @click="emit('retry')">
              <ReloadOutlined />
            </button>
          </a-tooltip>
          <span class="toolbar-divider"></span>
          <a-tooltip title="有帮助">
            <button
              class="toolbar-btn"
              :class="{ active: feedback === 'positive' }"
              @click="setFeedback('positive')"
            >
              <LikeOutlined />
            </button>
          </a-tooltip>
          <a-tooltip title="没帮助">
            <button
              class="toolbar-btn"
              :class="{ active: feedback === 'negative' }"
              @click="setFeedback('negative')"
            >
              <DislikeOutlined />
            </button>
          </a-tooltip>
        </div>

        <!-- 引用来源 -->
        <RefsComponent
          v-if="showRefs"
          :message="message"
          :show-refs="filteredRefs"
          @retry="emit('retry')"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, defineAsyncComponent, ref } from 'vue'
import { useClipboard } from '@vueuse/core'
import { message as antdMessage } from 'ant-design-vue'
import {
  CaretRightOutlined,
  ThunderboltOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  CopyOutlined,
  LikeOutlined,
  DislikeOutlined
} from '@ant-design/icons-vue'
import RefsComponent from '@/components/RefsComponent.vue'
import RetrievalStatus from '@/components/chat/RetrievalStatus.vue'

// Lazy-load markdown preview to keep the initial chat bundle smaller.
const MdPreview = defineAsyncComponent({
  loader: async () => {
    const mod = await import('md-editor-v3')
    await import('md-editor-v3/lib/preview.css')
    return mod.MdPreview
  },
  delay: 120,
  timeout: 20000
})

const props = defineProps({
  message: {
    type: Object,
    required: true
  },
  isProcessing: {
    type: Boolean,
    default: false
  },
  customClasses: {
    type: Object,
    default: () => ({})
  },
  showRefs: {
    type: [Array, Boolean],
    default: () => false
  },
  debugMode: {
    type: Boolean,
    default: false
  },
  showKnowledgeBase: {
    type: Boolean,
    default: true
  },
  showKnowledgeGraph: {
    type: Boolean,
    default: true
  },
  showWebSearch: {
    type: Boolean,
    default: true
  },
  showMcp: {
    type: Boolean,
    default: false
  }
})
const isUser = computed(() => props.message.role === 'user' || props.message.role === 'sent')
const avatar = computed(() => (isUser.value ? 'avatar.jpg' : 'user.png'))
const editorRef = ref()

const emit = defineEmits(['retry', 'retryStoppedMessage'])

// 推理面板展开状态
const reasoningOpen = ref(true)

// 内容为空且正在加载
const isEmptyAndLoading = computed(() => {
  const isEmpty = !props.message.content || props.message.content.length === 0
  const isLoading = props.message.status === 'init' && props.isProcessing
  return isEmpty && isLoading
})

// 显示检索状态组件
const showRetrievalStatus = computed(() => {
  const status = props.message.status
  const isEmpty = !props.message.content || props.message.content.length === 0
  const isSearching = status === 'searching' && props.isProcessing
  const isGenerating = status === 'generating' && props.isProcessing && isEmpty
  return isSearching || isGenerating
})

// 反馈状态
const feedback = ref(null)
const setFeedback = (type) => {
  feedback.value = feedback.value === type ? null : type
}

// 复制功能
const { copy, isSupported } = useClipboard()
const copyContent = async () => {
  if (!isSupported.value) {
    antdMessage.error('当前浏览器不支持复制')
    return
  }
  try {
    await copy(props.message.content || '')
    antdMessage.success('已复制到剪贴板')
  } catch {
    antdMessage.error('复制失败')
  }
}

// 过滤掉工具栏中已有的操作 (copy, regenerate 已移到 toolbar)
const filteredRefs = computed(() => {
  if (props.showRefs === true) {
    return ['subGraph', 'webSearch']
  }
  if (Array.isArray(props.showRefs)) {
    return props.showRefs.filter(k => k !== 'copy' && k !== 'regenerate')
  }
  return props.showRefs
})

// 缩短模型名称显示
const shortenModelName = (name) => {
  if (!name) return ''
  // 移除提供商前缀 (如 "Qwen/")
  const shortName = name.split('/').pop()
  // 如果仍然太长，截断
  if (shortName.length > 20) {
    return shortName.slice(0, 18) + '...'
  }
  return shortName
}
</script>

<!-- =============== scoped styles =============== -->
<style lang="less" scoped>
/* ===== wrapper layout (CSS Grid Transformation) ===== */
.message-wrapper {
  display: grid;
  grid-template-columns: auto 1fr; /* Avatar | Content */
  grid-template-rows: auto auto;   /* Bubble | Footer */
  column-gap: var(--space-3);      /* Gap between avatar and bubble */
  row-gap: 4px;                    /* Gap between bubble and footer */
  margin-bottom: var(--space-6);
  animation: fadeInUp var(--duration-slow) var(--ease-out);

  /* AI Message Layout (Default) */
  .avatar {
    grid-row: 1;
    grid-column: 1;
    align-self: center; /* Vertically center relative to the BUBBLE only */
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
  }

  &.from-ai .avatar {
    border: 2px solid var(--surface-color);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
  }

  /* User Message Layout (Reversed) */
  &.from-user {
    grid-template-columns: 1fr auto; /* Content | Avatar */

    .avatar {
      grid-column: 2;
      box-shadow: 0 2px 8px rgba(255, 125, 0, 0.2);
    }
  }
}

/* Flatten the content wrapper so children participate in the grid directly */
.message-content-wrapper {
  display: contents;
}

/* ===== message box ===== */
.message-box {
  grid-row: 1; /* Always in the first row */
  display: inline-block; /* Standard behavior for chat bubbles */
  padding: 12px 16px;     /* Balanced padding */
  user-select: text;
  word-break: normal;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  text-align: left;       /* Standard left alignment */
  font-size: var(--font-size-md);
  line-height: 1.6;
  position: relative;
  width: fit-content;     /* Crucial: Shrink to fit content */
  max-width: min(640px, 85%);

  /* AI Box Position */
  .from-ai & {
    grid-column: 2;
    justify-self: start;
    border-radius: 18px 18px 18px 4px;
    background: color-mix(in srgb, var(--surface-color) 95%, transparent);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--text-color);
    border: 1px solid var(--gray-200);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  }

  /* User Box Position */
  .from-user & {
    grid-column: 1;
    justify-self: end;
    border-radius: 18px 18px 4px 18px;
    background: linear-gradient(135deg, #FFA940 0%, #FF7D00 100%);
    color: var(--message-user-text);
    border: none;
    box-shadow: 0 4px 12px rgba(255, 125, 0, 0.25);
    :deep(a) { color: var(--message-user-text); text-decoration: underline; }
  }

  /* --- Deep Markdown Fixes (The ultimate shrink-wrap) --- */
  :deep(.message-md),
  :deep(.md-editor-preview-wrapper),
  :deep(.md-editor-preview) {
    display: inline-block !important;
    width: auto !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    vertical-align: top;
  }

  :deep(.md-editor-preview p) {
    margin: 0 !important;
    line-height: inherit !important;
  }

  :deep(.md-editor-preview p:not(:last-child)) {
    margin-bottom: 12px !important;
  }
}

/* ===== message footer (Toolbars) ===== */
.message-footer {
  grid-row: 2; /* Always in the second row */
  /* Remove old margins */
  margin: 0; 
  padding-left: 2px; /* Slight offset alignment */

  /* AI Footer Position */
  .from-ai & {
    grid-column: 2;
    justify-self: start;
  }

  /* User Footer Position */
  .from-user & {
    grid-column: 1;
    justify-self: end;
  }
}

/* Cleanup old styles that are no longer needed */
.message-wrapper.from-user,
.message-wrapper.from-ai {
  flex-direction: initial; /* Reset flex direction since we use grid */
}

/* ===== message toolbar (hover 显示) ===== */
.message-toolbar {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease-default);
}

/* 父容器 hover 时显示工具栏 */
.message-wrapper:hover .message-toolbar {
  opacity: 1;
}

/* 模型名做成小标签 */
.toolbar-model-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 6px;
  font-size: 10px;
  font-family: var(--font-family-mono);
  background: var(--gray-100);
  color: var(--gray-500);
  border-radius: 4px;
  margin-right: var(--space-1);
}

.toolbar-btn {
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
  font-size: 13px;

  &:hover {
    background: var(--hover-bg);
    color: var(--primary-color);
  }

  &.active {
    color: var(--primary-color);
  }
}

.toolbar-divider {
  width: 1px;
  height: 14px;
  background: var(--gray-200);
  margin: 0 var(--space-1);
}

/* ===== tool call capsules ===== */
:deep(.tool-calls-container) {
  display: inline-flex !important;
  flex-wrap: wrap;
  gap: var(--space-2);
  width: auto !important;
  margin-top: var(--space-3);
  background: transparent !important;
  border: none !important;
}

:deep(.tool-call-container) {
  display: inline-block !important;
  width: auto !important;
  max-width: max-content !important;
  background: transparent !important;
  padding: 0 !important;
}

:deep(.tool-call-display) {
  display: inline-flex !important;
  flex: 0 0 auto !important;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  width: auto !important;
  max-width: max-content !important;
  background: var(--surface-color-2);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);

  .tool-header {
    background: transparent;
    border: none;
    padding: 0;
    margin: 0;
    gap: 6px;
  }

  .anticon {
    color: var(--gray-600);
    cursor: pointer;
    &:hover {
      color: var(--gray-800);
    }
  }
}

:deep(.tool-call-display > .tool-content) {
  display: inline-flex !important;
  width: auto !important;
  max-width: max-content !important;
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
}
</style>

<!-- =============== global styles: markdown / font =============== -->
<style lang="less">
/* font size scaling */
.chat-box.font-smaller #preview-only-preview {
  font-size: var(--font-size-base);
  h1, h2 { font-size: 1.1rem; }
  h3, h4 { font-size: 1rem; }
}
.chat-box.font-larger #preview-only-preview {
  font-size: var(--font-size-lg);
  h1, h2 { font-size: 1.3rem; }
  h3, h4 { font-size: 1.2rem; }
  h5, h6 { font-size: 1.1rem; }
  code { font-size: var(--font-size-base); }
}
</style>
