<template>
  <div class="message-wrapper" :class="{ 'from-user': isUser, 'from-ai': !isUser }">
    <img class="avatar" :src="`/${avatar}`" alt="avatar" />
    <div class="message-box" :class="message.role">
      <!-- 用户消息 -->
      <template v-if="isUser">
        {{ message.content }}
      </template>

      <!-- 助手消息 -->
      <template v-else-if="message.role === 'assistant'">
        <p v-if="debugMode">{{ message.status }}</p>

        <!-- 推理过程 -->
        <div v-if="message.reasoning_content" class="reasoning-box">
          <a-collapse v-model:activeKey="reasoningActiveKey" :bordered="false">
            <template #expandIcon="{ isActive }">
              <caret-right-outlined :rotate="isActive ? 90 : 0" />
            </template>
            <a-collapse-panel
              key="show"
              :header="message.status == 'reasoning' ? '正在思考...' : '推理过程'"
              class="reasoning-header"
            >
              <p class="reasoning-content">{{ message.reasoning_content }}</p>
            </a-collapse-panel>
          </a-collapse>
        </div>

        <div v-if="isEmptyAndLoading" class="loading-dots">
          <div></div>
          <div></div>
          <div></div>
        </div>
        <div v-else-if="message.status === 'searching' && isProcessing" class="searching-msg">
          <i>正在检索……</i>
        </div>
        <div v-else-if="message.status === 'generating' && isProcessing" class="searching-msg">
          <i>正在生成……</i>
        </div>
        <div v-else-if="message.status === 'error'" class="err-msg" @click="$emit('retry')">
          请求错误，请重试。{{ message.message }}
        </div>

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

        <slot
          v-else-if="message.toolCalls && Object.keys(message.toolCalls).length > 0"
          name="tool-calls"
        ></slot>
        <div v-else class="err-msg" @click="$emit('retry')">
          请求错误，请重试。{{ message.message }}
        </div>

        <div v-if="message.isStoppedByUser" class="retry-hint">
          你停止生成了本次回答
          <span class="retry-link" @click="emit('retryStoppedMessage', message.id)"
            >重新编辑问题</span
          >
        </div>

        <div v-if="message.status === 'finished' && showRefs">
          <RefsComponent :message="message" :show-refs="showRefs" @retry="emit('retry')" />
        </div>
      </template>

      <!-- 自定义内容 -->
      <slot></slot>
    </div>
  </div>
</template>

<script setup>
import { computed, defineAsyncComponent, ref } from 'vue'
import { CaretRightOutlined } from '@ant-design/icons-vue'
import RefsComponent from '@/components/RefsComponent.vue'

// Lazy-load markdown preview to keep the initial chat bundle smaller.
// This creates a separate chunk for md-editor-v3 (and its CSS) and loads it only when needed.
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
  // 消息角色：'user'|'assistant'|'sent'|'received'
  message: {
    type: Object,
    required: true
  },
  // 是否正在处理中
  isProcessing: {
    type: Boolean,
    default: false
  },
  // 自定义类
  customClasses: {
    type: Object,
    default: () => ({})
  },
  // 是否显示推理过程
  showRefs: {
    type: [Array, Boolean],
    default: () => false
  },
  debugMode: {
    type: Boolean,
    default: false
  }
})
const isUser = computed(() => props.message.role === 'user' || props.message.role === 'sent')
// ⚠️ 头像文件放在 public/images 下，或改成你的实际路径
const avatar = computed(() => (isUser.value ? 'avatar.jpg' : 'user.png'))
const editorRef = ref()

const emit = defineEmits(['retry', 'retryStoppedMessage'])

// 推理面板展开状态
const reasoningActiveKey = ref(['show'])

// 计算属性：内容为空且正在加载
const isEmptyAndLoading = computed(() => {
  const isEmpty = !props.message.content || props.message.content.length === 0
  const isLoading = props.message.status === 'init' && props.isProcessing
  return isEmpty && isLoading
})
</script>

<!-- =============== style scoped：气泡 & deepseek 胶囊 =============== -->
<style lang="less" scoped>
/* ===== wrapper 布局 + 头像 + 气泡背景 ===== */
.message-wrapper {
  display: flex;
  align-items: flex-start;
  margin-bottom: 24px;
  animation: fadeInUp 0.3s ease-out;

  &.from-user {
    flex-direction: row-reverse;
    .message-box {
      /* Pokedex User Bubble: Red Gradient */
      background: linear-gradient(135deg, var(--pokedex-red), #ff706d);
      color: #ffffff;
      border: 1px solid rgba(0,0,0,0.05);
      box-shadow: 0 4px 12px rgba(255, 83, 80, 0.25);
      border-radius: 20px 20px 4px 20px; /* Speech bubble shape */
      max-width: min(600px, 85%);
      
      /* Make sure links in user bubble are white */
      :deep(a) { color: #fff; text-decoration: underline; }
    }
    
    .avatar {
      margin-left: 12px;
      margin-right: 0;
      box-shadow: 0 4px 12px rgba(255, 83, 80, 0.2);
    }
  }

  &.from-ai {
    flex-direction: row;
    .message-box {
      /* AI Bubble: Glass Data Display */
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(8px);
      color: var(--chat-assistant-text);
      border: 1px solid rgba(255, 255, 255, 0.6);
      box-shadow: 0 4px 16px rgba(0,0,0,0.04);
      border-radius: 4px 20px 20px 20px; /* Speech bubble shape */
      max-width: 960px;
    }
    
    .avatar {
      margin-right: 12px;
      margin-left: 0;
      border: 2px solid #fff;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
  }

  .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
  }
}

/* ===== 公共文字 / loading / 提示 ===== */
.retry-hint {
  margin-top: 8px;
  padding: 8px 16px;
  color: var(--chat-muted-text);
  font-size: 14px;
  text-align: left;
}
.retry-link {
  color: var(--main-600);
  cursor: pointer;
  margin-left: 4px;
  &:hover {
    text-decoration: underline;
  }
}
.ant-btn-icon-only:has(.anticon-stop) {
  background: var(--danger-600) !important;
  &:hover {
    background: var(--danger-500) !important;
  }
}
.loading-dots {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  div {
    width: 8px;
    height: 8px;
    margin: 0 4px;
    background: var(--chat-muted-text);
    border-radius: 50%;
    opacity: 0.3;
    animation: pulse 0.5s infinite both;
    &:nth-child(1) {
      animation-delay: -0.32s;
    }
    &:nth-child(2) {
      animation-delay: -0.16s;
    }
  }
}
@keyframes pulse {
  0%,
  80%,
  100% {
    transform: scale(0.8);
    opacity: 0.3;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* ===== message-box 内排版 ===== */
.message-box {
  display: inline-block;
  /* Default styling overridden by specific classes above */
  padding: 12px 20px;
  user-select: text;
  word-break: break-word;
  font-size: 15px;
  line-height: 1.6;
  position: relative;
  
  &.assistant,
  &.received {
     /* Legacy classes support */
    display: block;
    width: 100%;
  }

  .err-msg {
    color: var(--text-color);
    border: 1px solid color-mix(in srgb, var(--danger-500) 45%, transparent);
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--danger-500) 10%, transparent);
    margin-bottom: 10px;
    cursor: pointer;
  }
  .searching-msg {
    color: var(--chat-muted-text);
    animation: opacityPulse 1s infinite ease-in-out;
  }
  .reasoning-box {
    margin: 10px 0 15px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: rgba(0,0,0,0.02);
    
    .reasoning-content {
      font-size: 13px;
      color: var(--chat-muted-text);
      white-space: pre-wrap;
      margin: 0;
      padding: 8px;
    }
  }
  :deep(.tool-calls-container) {
    display: inline-flex !important;
    flex-wrap: wrap;
    gap: 8px;
    width: auto !important;
    margin-top: 10px;
    background: transparent !important;
    border: none !important;
  }
}
@keyframes opacityPulse {
  0% {
    opacity: 0.7;
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 0.7;
  }
}

/* ============ 关键修改：deepseek 胶囊自适应宽度 ============ */
/* 1. container 变 inline-block */
:deep(.tool-call-container) {
  display: inline-block !important;
  width: auto !important;
  max-width: max-content !important;
  background: transparent !important;
  padding: 0 !important;
}

/* 2. 胶囊本体 inline-flex */
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
  border-radius: 8px;

  /* 标题区域去掉 block 背景 & 边框 */
  .tool-header {
    background: transparent;
    border: none;
    padding: 0;
    margin: 0;
    gap: 6px;
  }

  /* 小图标配色 */
  .anticon {
    color: var(--gray-600);
    cursor: pointer;
    &:hover {
      color: var(--gray-800);
    }
  }
}

/* D. 最里层 .tool-content 保险起见也设成 inline-flex */
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

<!-- =============== style (全局)：markdown / 字体 等 =============== -->
<style lang="less">
.message-md.md-editor {
  height: auto;
  background-color: transparent;
  border: none;
  /* Align md-editor internal tokens with our app theme tokens. */
  --md-color: var(--text-color);
  --md-hover-color: var(--text-color);
  --md-bk-color: transparent;
  --md-bk-color-outstand: var(--surface-color-2);
  --md-bk-hover-color: var(--hover-bg);
  --md-border-color: var(--border-color);
  --md-border-hover-color: var(--border-color);
  --md-border-active-color: var(--main-500);
  --md-scrollbar-bg-color: transparent;
  --md-scrollbar-thumb-color: var(--gray-400);
  --md-scrollbar-thumb-hover-color: var(--gray-500);
  --md-scrollbar-thumb-active-color: var(--gray-600);
}

.message-md .md-editor-preview-wrapper {
  color: var(--text-color);
  max-width: 100%;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei',
    'Hiragino Sans GB', 'Courier New', monospace;
  #preview-only-preview {
    font-size: 15px;
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
    color: var(--main-400);
    text-decoration: underline;
  }
  code {
    font-size: 13px;
    font-family: 'Menlo', 'Monaco', 'Consolas', 'Courier New', monospace;
    line-height: 1.5;
    letter-spacing: 0.025em;
    tab-size: 4;
    -moz-tab-size: 4;
    background: var(--surface-color-2);
    color: var(--text-color);
  }

  pre {
    background: var(--surface-color-2);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 12px;
    overflow: auto;
  }

  pre code {
    background: transparent;
  }
}

/* deepseek-chat 模型名单独行 → 改成 inline */
.model-name {
  display: inline;
  font-weight: 600;
  margin-right: 0.5em;
}

/* 聊天字体缩放 */
.chat-box.font-smaller #preview-only-preview {
  font-size: 14px;
  h1,
  h2 {
    font-size: 1.1rem;
  }
  h3,
  h4 {
    font-size: 1rem;
  }
}
.chat-box.font-larger #preview-only-preview {
  font-size: 16px;
  h1,
  h2 {
    font-size: 1.3rem;
  }
  h3,
  h4 {
    font-size: 1.2rem;
  }
  h5,
  h6 {
    font-size: 1.1rem;
  }
  code {
    font-size: 14px;
  }
}
</style>
