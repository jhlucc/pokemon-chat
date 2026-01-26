<template>
  <div class="message-toolbar hover-reveal">
    <a-tooltip title="复制">
      <button class="toolbar-btn" @click="handleCopy">
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
        @click="toggleFeedback('positive')"
      >
        <LikeOutlined />
      </button>
    </a-tooltip>
    <a-tooltip title="没帮助">
      <button
        class="toolbar-btn"
        :class="{ active: feedback === 'negative' }"
        @click="toggleFeedback('negative')"
      >
        <DislikeOutlined />
      </button>
    </a-tooltip>

    <!-- 模型名和响应时间 -->
    <span v-if="modelName" class="toolbar-meta">
      {{ modelName }}
    </span>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useClipboard } from '@vueuse/core'
import { message as antdMessage } from 'ant-design-vue'
import {
  CopyOutlined,
  ReloadOutlined,
  LikeOutlined,
  DislikeOutlined
} from '@ant-design/icons-vue'

type FeedbackType = 'positive' | 'negative' | null

const props = defineProps<{
  content: string
  modelName?: string
}>()

const emit = defineEmits<{
  (e: 'retry'): void
  (e: 'feedback', type: FeedbackType): void
}>()

const feedback = ref<FeedbackType>(null)
const { copy, isSupported } = useClipboard()

function toggleFeedback(type: 'positive' | 'negative') {
  feedback.value = feedback.value === type ? null : type
  emit('feedback', feedback.value)
}

async function handleCopy() {
  if (!isSupported.value) {
    antdMessage.error('当前浏览器不支持复制')
    return
  }
  try {
    await copy(props.content || '')
    antdMessage.success('已复制到剪贴板')
  } catch {
    antdMessage.error('复制失败')
  }
}
</script>

<style scoped lang="less">
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
</style>
