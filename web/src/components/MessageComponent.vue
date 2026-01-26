<template>
  <div
    class="message-wrapper hover-reveal-trigger"
    :class="{ 'from-user': isUser, 'from-ai': !isUser }"
    role="article"
    :aria-label="isUser ? '用户消息' : '助手消息'"
  >
    <img class="avatar" :src="`/${avatar}`" :alt="isUser ? '用户头像' : '助手头像'" />
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

        <!-- 引用来源与工具栏 -->
        <div v-if="message.status === 'finished'" class="message-footer">
          <!-- 消息工具栏 (hover 显示) -->
          <div class="message-toolbar hover-reveal">
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

            <!-- 模型名和响应时间 -->
            <span v-if="message.meta?.server_model_name" class="toolbar-meta">
              {{ message.meta.server_model_name }}
            </span>
          </div>

          <!-- 引用来源 -->
          <RefsComponent
            v-if="showRefs"
            :message="message"
            :show-refs="filteredRefs"
            @retry="emit('retry')"
          />
        </div>
      </template>

      <!-- 自定义内容 -->
      <slot></slot>
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
  LoadingOutlined,
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
</script>

<!-- =============== scoped styles =============== -->
<style lang="less" scoped>
/* ===== wrapper layout ===== */
.message-wrapper {
  display: flex;
  align-items: flex-start;
  margin-bottom: var(--space-6);
  animation: fadeInUp var(--duration-slow) var(--ease-out);

  &.from-user {
    flex-direction: row-reverse;
    .message-box {
      background: var(--gradient-primary);
      color: var(--message-user-text);
      border: none;
      box-shadow: var(--message-user-shadow);
      border-radius: var(--message-user-radius);
      max-width: var(--message-max-width);

      :deep(a) { color: var(--message-user-text); text-decoration: underline; }
    }

    .avatar {
      margin-left: var(--space-3);
      margin-right: 0;
      box-shadow: var(--message-user-shadow);
    }
  }

  &.from-ai {
    flex-direction: row;
    .message-box {
      background: var(--surface-color);
      color: var(--chat-assistant-text);
      border: 1px solid var(--border-color);
      box-shadow: var(--shadow-xs);
      border-radius: var(--radius-xs) var(--radius-md) var(--radius-md) var(--radius-md);
      max-width: 960px;
    }

    .avatar {
      margin-right: var(--space-3);
      margin-left: 0;
      border: 2px solid var(--surface-color);
      box-shadow: var(--shadow-xs);
    }
  }

  .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
  }
}

/* ===== message box ===== */
.message-box {
  display: inline-block;
  padding: var(--space-3) var(--space-5);
  user-select: text;
  word-break: break-word;
  font-size: var(--font-size-md);
  line-height: var(--line-height-relaxed);
  position: relative;

  &.assistant,
  &.received {
    display: block;
    width: 100%;
  }
}

.debug-status {
  font-size: var(--font-size-xs);
  color: var(--gray-500);
  margin: 0 0 var(--space-2);
  font-family: var(--font-family-mono);
}

/* ===== reasoning box ===== */
.reasoning-box {
  margin: var(--space-2) 0 var(--space-3);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--surface-color-2) 80%, transparent);
  overflow: hidden;
}

.reasoning-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  cursor: pointer;
  transition: background var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--hover-bg);
  }
}

.reasoning-indicator {
  display: flex;
  align-items: center;
  gap: var(--space-2);

  .reasoning-icon {
    font-size: 14px;
    color: var(--gray-500);

    &.active {
      color: var(--warning-color);
      animation: pulse 1.5s infinite;
    }
  }

  .reasoning-title {
    font-size: var(--font-size-sm);
    font-weight: 500;
    color: var(--gray-600);
  }
}

.reasoning-caret {
  font-size: 12px;
  color: var(--gray-400);
  transition: transform var(--duration-fast) var(--ease-default);

  &.open {
    transform: rotate(90deg);
  }
}

.reasoning-body {
  padding: 0 var(--space-3) var(--space-3);
}

.reasoning-content {
  font-size: var(--font-size-sm);
  color: var(--gray-600);
  white-space: pre-wrap;
  margin: 0;
  line-height: var(--line-height-relaxed);
}

.collapse-enter-active,
.collapse-leave-active {
  transition: all var(--duration-base) var(--ease-default);
  max-height: 500px;
  overflow: hidden;
}

.collapse-enter-from,
.collapse-leave-to {
  max-height: 0;
  opacity: 0;
}

/* ===== loading & status states ===== */
.loading-state {
  padding: var(--space-2) 0;
}

.status-msg {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--gray-500);
  font-size: var(--font-size-sm);
  padding: var(--space-2) 0;
  animation: pulse 1.5s infinite;

  .status-icon {
    font-size: 14px;
    color: var(--primary-color);
  }
}

/* ===== error state ===== */
.error-msg {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--error-color) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--error-color) 25%, transparent);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: color-mix(in srgb, var(--error-color) 12%, transparent);
    border-color: color-mix(in srgb, var(--error-color) 40%, transparent);
  }

  .error-image {
    width: 48px;
    height: 48px;
    object-fit: contain;
    flex-shrink: 0;
  }

  .error-icon {
    color: var(--error-color);
    font-size: 16px;
    flex-shrink: 0;
  }

  .error-content {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
  }

  .error-text {
    color: var(--text-color);
    font-size: var(--font-size-sm);
    font-weight: 500;
  }

  .error-detail {
    color: var(--gray-600);
    font-size: var(--font-size-xs);
    margin-top: 2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .error-retry {
    color: var(--primary-color);
    font-size: 14px;
    flex-shrink: 0;
  }
}

/* ===== stopped hint ===== */
.stopped-hint {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-top: var(--space-3);
  padding: var(--space-2) var(--space-3);
  background: var(--surface-color-2);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);

  .stopped-text {
    color: var(--gray-500);
  }

  .stopped-link {
    color: var(--primary-color);
    cursor: pointer;
    font-weight: 500;

    &:hover {
      text-decoration: underline;
    }
  }
}

/* ===== message footer (toolbar + refs) ===== */
.message-footer {
  margin-top: var(--space-3);
  padding-top: var(--space-2);
}

/* ===== message toolbar (hover reveal) ===== */
.message-toolbar {
  display: flex;
  align-items: center;
  gap: var(--toolbar-gap);
  margin-bottom: var(--space-2);
  transition: opacity var(--duration-fast) var(--ease-default);
}

.toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: var(--toolbar-btn-size);
  height: var(--toolbar-btn-size);
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--gray-500);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  font-size: var(--font-size-base);

  &:hover {
    background: var(--hover-bg);
    color: var(--gray-700);
  }

  &.active {
    color: var(--primary-color);
    background: color-mix(in srgb, var(--primary-color) 5%, transparent);
  }
}

.toolbar-divider {
  width: 1px;
  height: 16px;
  background: var(--gray-200);
  margin: 0 var(--space-1);
}

.toolbar-meta {
  margin-left: auto;
  font-size: var(--font-size-xs);
  color: var(--gray-400);
  font-family: var(--font-family-mono);
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
.message-md.md-editor {
  height: auto;
  background-color: transparent;
  border: none;
  --md-color: var(--text-color);
  --md-hover-color: var(--text-color);
  --md-bk-color: transparent;
  --md-bk-color-outstand: var(--surface-color-2);
  --md-bk-hover-color: var(--hover-bg);
  --md-border-color: var(--border-color);
  --md-border-hover-color: var(--border-color);
  --md-border-active-color: var(--primary-color);
  --md-scrollbar-bg-color: transparent;
  --md-scrollbar-thumb-color: var(--gray-400);
  --md-scrollbar-thumb-hover-color: var(--gray-500);
  --md-scrollbar-thumb-active-color: var(--gray-600);
}

.message-md .md-editor-preview-wrapper {
  color: var(--text-color);
  max-width: 100%;
  padding: 0;
  font-family: var(--font-family-base);
  #preview-only-preview {
    font-size: var(--font-size-md);
  }
  h1,
  h2 {
    font-size: 1.2rem;
  }
  h3,
  h4 {
    font-size: 1.1rem;
  }
  h5,
  h6 {
    font-size: 1rem;
  }
  a {
    color: var(--primary-color);
    text-decoration: none;
  }
  a:hover {
    color: var(--primary-light-color);
    text-decoration: underline;
  }
  code {
    font-size: var(--font-size-sm);
    font-family: var(--font-family-mono);
    line-height: var(--line-height-base);
    letter-spacing: 0.025em;
    tab-size: 4;
    -moz-tab-size: 4;
    background: var(--surface-color-2);
    color: var(--text-color);
  }

  pre {
    background: var(--surface-color-2);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: var(--space-3);
    overflow: auto;
  }

  pre code {
    background: transparent;
  }
}

.model-name {
  display: inline;
  font-weight: 600;
  margin-right: 0.5em;
}

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
