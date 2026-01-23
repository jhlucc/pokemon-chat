<template>
  <div class="input-box" :class="customClasses">
    <div class="input-area">
      <a-textarea
        class="user-input"
        v-model:value="inputValue"
        @keydown="handleKeyPress"
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
      <a-tooltip :title="isRecording ? '点击停止录音' : '点击开始语音输入'">
        <a-button
          type="link"
          @click="toggleRecording"
          :style="{ color: isRecording ? 'red' : '' }"
        >
          <template #icon>
            <component :is="isRecording ? LoadingOutlined : AudioOutlined" />
          </template>
        </a-button>
      </a-tooltip>
        <a-tooltip :title="isLoading ? '停止回答' : ''">
          <a-button
            @click="handleSendOrStop"
            :disabled="sendButtonDisabled"
            type="link"
          >
            <template #icon>
              <component :is="getIcon" class="send-btn" />
            </template>
          </a-button>
        </a-tooltip>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, toRefs } from 'vue';
import {
  AudioOutlined,
  SendOutlined,
  ArrowUpOutlined,
  LoadingOutlined,
  PauseOutlined
} from '@ant-design/icons-vue';

const isRecording = ref(false);
const isRecordingLocked = ref(false); // ✅ 新增锁

let mediaRecorder = null;
let audioChunks = [];

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
  }
});

const emit = defineEmits(['update:modelValue', 'send', 'keydown']);

// 图标映射
const iconComponents = {
  'SendOutlined': SendOutlined,
  'ArrowUpOutlined': ArrowUpOutlined,
  'PauseOutlined': PauseOutlined
};

// 根据传入的图标名动态获取组件
const getIcon = computed(() => {
  if (props.isLoading) {
    return PauseOutlined;
  }
  return iconComponents[props.sendIcon] || ArrowUpOutlined;
});

const toggleRecording = async () => {
  console.log('[录音按钮] 点击了！当前状态 isRecording =', isRecording.value);
  if (isRecordingLocked.value) return;

  if (!isRecording.value) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('[录音] 获取麦克风成功', stream);

      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

mediaRecorder.onstop = async () => {
  console.log('[录音] 已停止，开始识别上传');
  try {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    const res = await fetch('/api/chat/asr/', {
      method: 'POST',
      body: formData
    });

    console.log('[上传] Whisper 返回结果状态：', res.status);
    const result = await res.json();
    inputValue.value += result.text || '';
  } catch (e) {
    console.error('上传识别失败：', e);
  } finally {
    // ✅ 释放麦克风
  const tracks = mediaRecorder?.stream?.getTracks?.();
if (tracks && Array.isArray(tracks)) {
  tracks.forEach(track => track.stop());
}
    mediaRecorder = null;
    isRecordingLocked.value = false;
    isRecording.value = false;
  }
};

      mediaRecorder.start();
      console.log('[录音] 已开始');
      isRecording.value = true;
      isRecordingLocked.value = false; // ✅ 注意：开始不加锁，允许点击停止
    } catch (err) {
      console.error('无法开始录音：', err);
      isRecording.value = false;
    }
  } else {
    try {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        console.log('[录音] 正在停止...');
        mediaRecorder.stop();
        isRecordingLocked.value = true; // ✅ 锁定，等待 onstop 完成
      } else {
        console.warn('[录音] stop 被调用但无效：', mediaRecorder);
        isRecording.value = false;
      }
    } catch (e) {
      console.error('停止录音失败：', e);
      isRecording.value = false;
      isRecordingLocked.value = false;
    }
  }
};



// 创建本地引用以进行双向绑定
const inputValue = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
});

// 处理键盘事件
const handleKeyPress = (e) => {
  emit('keydown', e);
};

// 处理发送按钮点击
const handleSendOrStop = () => {
  emit('send');
};
</script>

<style lang="less" scoped>
.input-box {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: auto;
  margin: 0 auto;
  /* Terminal Window Style */
  background: var(--surface-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  padding: 12px 16px;
  transition: all 0.3s ease;

  &:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px var(--primary-bg-light), var(--shadow-lg);
  }

  .input-area {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 4px;
    
    /* Terminal Prompt Symbol */
    &:before {
        content: '$';
        font-family: var(--font-mono);
        color: var(--primary-color);
        font-weight: bold;
        margin-top: 10px; /* Align with first line of text */
        font-size: 15px;
    }
  }

  .user-input {
    flex: 1;
    min-height: 24px;
    padding: 0.5rem 0;
    /* Transparent bg for terminal feel */
    background-color: transparent;
    border: none;
    margin: 0;
    color: var(--text-color);
    font-family: var(--font-mono); /* Monospace input */
    font-size: 15px;
    outline: none;
    resize: none;
    line-height: 1.6;

    &::placeholder {
      color: var(--subtext-color);
      opacity: 0.6;
    }
    
    &:disabled {
        color: var(--subtext-color);
        background-color: transparent;
        cursor: not-allowed;
    }
  }

  .input-options {
    display: flex;
    padding: 8px 0 0;
    margin-top: 6px;
    border-top: 1px solid var(--border-color);

    .options__left,
    .options__right {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .options__right {
      width: fit-content;
    }

    .options__left {
      flex: 1;

      :deep(.opt-item) {
        border-radius: var(--radius-sm); /* Squarer tech look */
        border: 1px solid transparent;
        padding: 4px 8px;
        cursor: pointer;
        font-size: 12px;
        font-family: var(--font-mono);
        color: var(--subtext-color);
        transition: all 0.2s ease;
        background: var(--surface-secondary);

        &:hover {
          background-color: var(--gray-200); /* Slightly darker */
          color: var(--text-color);
        }

        &.active {
          color: var(--primary-color);
          background-color: var(--primary-bg-light);
          border-color: var(--primary-light-color);
        }
      }
    }
  }
}

button.ant-btn-icon-only {
  /* Clean tech button */
  height: 32px;
  width: 32px;
  background-color: transparent;
  color: var(--subtext-color);
  border-radius: var(--radius-sm);
  border: 1px solid transparent;
  box-shadow: none;
  
  &:hover {
    color: var(--primary-color);
    background-color: var(--primary-bg-light);
  }
  
  &.send-btn {
      color: var(--primary-color);
  }

  &:disabled {
    color: var(--gray-300);
    background: transparent;
  }
}

@media (max-width: 520px) {
  .input-box {
    border-radius: var(--radius-lg);
    padding: 10px 12px;
  }
}
</style>
