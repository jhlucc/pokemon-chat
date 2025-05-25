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
  padding: 0.4rem 0.75rem;
  border: 2px solid var(--gray-200);
  border-radius: 0.8rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;

  &:focus-within {
    border-color: var(--main-500);
    background: white;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  }

  .input-area {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    margin-bottom: 4px;
  }

  .user-input {
    flex: 1;
    min-height: 44px;
    padding: 0.5rem 0;
    background-color: transparent;
    border: none;
    margin: 0;
    color: #222222;
    font-size: 14px;
    outline: none;
    resize: none;
    line-height: 1.6;

    &:focus {
      outline: none;
      box-shadow: none;
    }

    &:active {
      outline: none;
    }

    &::placeholder {
      color: #888888;
    }
  }

  .input-options {
    display: flex;
    padding: 8px 0 0;
    margin-top: 6px;
    border-top: 1px solid var(--gray-100);

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
        border-radius: 12px;
        border: 1px solid var(--gray-300);
        padding: 5px 10px;
        cursor: pointer;
        font-size: 12px;
        color: var(--gray-700);
        transition: all 0.2s ease;

        &:hover {
          background-color: var(--main-10);
          color: var(--main-600);
        }

        &.active {
          color: var(--main-600);
          border: 1px solid var(--main-500);
          background-color: var(--main-10);
        }
      }
    }
  }
}

button.ant-btn-icon-only {
  height: 32px;
  width: 32px;
  cursor: pointer;
  background-color: var(--main-500);
  border-radius: 50%;
  border: none;
  transition: all 0.2s ease;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  color: white;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;

  &:hover {
    background-color: var(--main-600);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    color: white;
  }

  &:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  &:disabled {
    background-color: var(--gray-400);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
}

@media (max-width: 520px) {
  .input-box {
    border-radius: 15px;
    padding: 0.625rem 0.875rem;
  }
}
</style>
