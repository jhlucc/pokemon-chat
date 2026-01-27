<template>
  <div class="input-box" :class="[customClasses, { 'is-focused': isFocused }]">
    <div class="input-area">
      <a-textarea
        ref="textareaRef"
        class="user-input"
        v-model:value="inputValue"
        @keydown="handleKeyPress"
        @focus="isFocused = true"
        @blur="isFocused = false"
        :placeholder="placeholder"
        :disabled="disabled"
        :auto-size="autoSize"
      />
    </div>
    <div class="input-options">
      <div class="options__left">
        <slot name="options-left"></slot>
      </div>
      <div class="options__right">
        <!-- 快捷键提示 -->
        <div class="input-hints">
          <kbd>Shift + Enter</kbd>
          <span>换行</span>
        </div>

        <!-- 字符计数 -->
        <div
          v-if="showCharCount"
          class="char-count"
          :class="{ 'near-limit': charPercent > 90, 'at-limit': charPercent >= 100 }"
        >
          {{ charCount }} / {{ maxLength }}
        </div>

        <a-tooltip :title="isRecording ? '点击停止录音' : '点击开始语音输入'">
          <a-button
            type="text"
            class="icon-btn"
            @click="toggleRecording"
            :class="{ recording: isRecording }"
          >
            <template #icon>
              <component :is="isRecording ? LoadingOutlined : AudioOutlined" />
            </template>
          </a-button>
        </a-tooltip>
        <a-tooltip :title="isLoading ? '停止生成' : '发送消息'">
          <a-button
            @click="handleSendOrStop"
            :disabled="sendButtonDisabled && !isLoading"
            type="primary"
            class="send-btn"
            :class="{ 'is-loading': isLoading }"
          >
            <template #icon>
              <component :is="getIcon" />
            </template>
          </a-button>
        </a-tooltip>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import {
  AudioOutlined,
  SendOutlined,
  ArrowUpOutlined,
  LoadingOutlined,
  PauseOutlined
} from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import { apiFetch } from '@/api/http'

const isRecording = ref(false)
const isRecordingLocked = ref(false)
const isFocused = ref(false)
const textareaRef = ref(null)

let mediaRecorder = null
let audioChunks = []

const props = defineProps({
  modelValue: {
    type: String,
    default: ''
  },
  placeholder: {
    type: String,
    default: '输入问题...'
  },
  isLoading: {
    type: Boolean,
    default: false
  },
  disabled: {
    type: Boolean,
    default: false
  },
  sendButtonDisabled: {
    type: Boolean,
    default: false
  },
  autoSize: {
    type: Object,
    default: () => ({ minRows: 2, maxRows: 6 })
  },
  sendIcon: {
    type: String,
    default: 'ArrowUpOutlined'
  },
  customClasses: {
    type: Object,
    default: () => ({})
  },
  maxLength: {
    type: Number,
    default: 4000
  },
  showCharCount: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['update:modelValue', 'send', 'keydown'])

// 图标映射
const iconComponents = {
  SendOutlined: SendOutlined,
  ArrowUpOutlined: ArrowUpOutlined,
  PauseOutlined: PauseOutlined
}

// 根据传入的图标名动态获取组件
const getIcon = computed(() => {
  if (props.isLoading) {
    return PauseOutlined
  }
  return iconComponents[props.sendIcon] || ArrowUpOutlined
})

const toggleRecording = async () => {
  if (isRecordingLocked.value) return

  if (!isRecording.value) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      mediaRecorder = new MediaRecorder(stream)
      audioChunks = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data)
        }
      }

      mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })
          const formData = new FormData()
          formData.append('file', audioBlob, 'recording.wav')

          const result = await apiFetch('/chat/asr/', {
            method: 'POST',
            body: formData,
            timeoutMs: 120000
          })
          inputValue.value += result.text || ''
        } catch (e) {
          console.error('上传识别失败：', e)
          message.error(e?.message ? `语音识别失败：${e.message}` : '语音识别失败')
        } finally {
          // ✅ 释放麦克风
          const tracks = mediaRecorder?.stream?.getTracks?.()
          if (tracks && Array.isArray(tracks)) {
            tracks.forEach((track) => track.stop())
          }
          mediaRecorder = null
          isRecordingLocked.value = false
          isRecording.value = false
        }
      }

      mediaRecorder.start()
      isRecording.value = true
      isRecordingLocked.value = false // ✅ 注意：开始不加锁，允许点击停止
    } catch (err) {
      console.error('无法开始录音：', err)
      message.error(err?.message ? `无法开始录音：${err.message}` : '无法开始录音')
      isRecording.value = false
    }
  } else {
    try {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop()
        isRecordingLocked.value = true // ✅ 锁定，等待 onstop 完成
      } else {
        isRecording.value = false
      }
    } catch (e) {
      console.error('停止录音失败：', e)
      isRecording.value = false
      isRecordingLocked.value = false
    }
  }
}

// 创建本地引用以进行双向绑定
const inputValue = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})

// 字符计数
const charCount = computed(() => inputValue.value?.length || 0)
const charPercent = computed(() => (charCount.value / props.maxLength) * 100)

// 处理键盘事件
const handleKeyPress = (e) => {
  emit('keydown', e)
}

// 处理发送按钮点击
const handleSendOrStop = () => {
  emit('send')
}
</script>

<style lang="less" scoped>
.input-box {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: auto;
  margin: 0 auto;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  background: var(--surface-color);
  box-shadow: var(--shadow-xs);
  transition: all var(--duration-base) var(--ease-default);

  &.is-focused {
    border-color: color-mix(in srgb, var(--primary-color) 40%, var(--border-color));
    box-shadow: var(--focus-ring), var(--shadow-sm);
  }

  .input-area {
    display: flex;
    align-items: flex-end;
    gap: 8px;
  }

  .user-input {
    flex: 1;
    min-height: 44px;
    padding: var(--space-2) 0;
    background-color: transparent;
    border: none;
    margin: 0;
    color: var(--text-color);
    font-size: var(--font-size-base);
    font-family: var(--font-family-base);
    outline: none;
    resize: none;
    line-height: var(--line-height-relaxed);

    &:focus {
      outline: none;
      box-shadow: none;
    }

    &::placeholder {
      color: var(--gray-500);
    }
  }

  .input-options {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: var(--space-2);
    margin-top: var(--space-2);
    border-top: 1px solid color-mix(in srgb, var(--border-color) 50%, transparent);

    .options__left,
    .options__right {
      display: flex;
      align-items: center;
      gap: var(--space-3);
    }

    .options__right {
      flex-shrink: 0;
    }

    .options__left {
      flex: 1;
      overflow-x: auto;

      &::-webkit-scrollbar {
        height: 0;
      }

      :deep(.opt-item) {
        display: inline-flex;
        align-items: center;
        border-radius: var(--radius-sm);
        border: none;
        padding: 6px 10px;
        min-height: 28px;
        cursor: pointer;
        font-size: var(--font-size-xs);
        color: var(--gray-500);
        background: transparent;
        transition: all var(--duration-fast) var(--ease-default);
        user-select: none;
        white-space: nowrap;

        &:hover {
          background-color: var(--gray-100);
          color: var(--gray-700);
        }

        &:focus-visible {
          outline: none;
          box-shadow: var(--focus-ring);
        }

        &.active {
          color: var(--primary-color);
          background-color: color-mix(in srgb, var(--primary-color) 8%, transparent);
        }

        &.disabled {
          opacity: 0.5;
          cursor: not-allowed;
          pointer-events: none;
        }
      }
    }
  }
}

/* 快捷键提示 - 默认隐藏，聚焦时显示 */
.input-hints {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: var(--font-size-xs);
  color: var(--gray-300);
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease-default);

  .input-box.is-focused & {
    opacity: 1;
  }

  kbd {
    display: inline-flex;
    align-items: center;
    padding: 2px 5px;
    font-size: 10px;
    font-family: var(--font-family-mono);
    background: transparent;
    border: 1px solid var(--gray-200);
    border-radius: 3px;
    color: var(--gray-400);
  }

  span {
    margin-left: 2px;
  }
}

/* 字符计数 - 默认隐藏，聚焦时显示 */
.char-count {
  font-size: var(--font-size-xs);
  color: var(--gray-300);
  font-family: var(--font-family-mono);
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease-default), color var(--duration-fast) var(--ease-default);

  .input-box.is-focused & {
    opacity: 1;
  }

  &.near-limit {
    opacity: 1;
    color: var(--warning-color);
  }

  &.at-limit {
    opacity: 1;
    color: var(--error-color);
    font-weight: 600;
  }
}

/* 图标按钮 - 轻量化设计 */
.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  color: var(--gray-400);
  background: transparent;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--gray-100);
    color: var(--gray-600);
  }

  &.recording {
    color: var(--error-color);
    background: color-mix(in srgb, var(--error-color) 10%, transparent);
    animation: pulse 1.5s infinite;
  }
}

/* 发送按钮 - 缩小尺寸 */
.send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--primary-color);
  border: none;
  color: white;
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  box-shadow: 0 1px 4px rgba(255, 125, 0, 0.25);

  &:hover:not(:disabled) {
    background: var(--main-600);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(255, 125, 0, 0.35);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  &:disabled {
    background: var(--gray-300);
    cursor: not-allowed;
    box-shadow: none;
  }

  &.is-loading {
    background: var(--error-color);

    &:hover:not(:disabled) {
      background: var(--danger-600);
    }
  }
}

/* 响应式 */
@media (max-width: 640px) {
  .input-box {
    padding: var(--space-2) var(--space-3);
  }

  .input-hints {
    display: none;
  }

  .char-count {
    display: none;
  }
}

@media (max-width: 520px) {
  .input-box {
    border-radius: var(--radius-md);
  }
}
</style>
