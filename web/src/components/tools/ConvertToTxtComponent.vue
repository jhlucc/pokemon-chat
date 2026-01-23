<template>
  <div class="pdf2txt-container window-card">
    <div class="sidebar">
      <div class="additional-params">
        <h4>相关参数</h4>
        <div class="empty-state">
           <span class="subtext">暂无相关参数</span>
        </div>
      </div>
    </div>
    <div class="result-container">
      <div class="input-container">
        <div class="upload-wrapper">
          <a-upload-dragger
            class="upload-dragger"
            v-model:fileList="fileList"
            name="file"
            :max-count="1"
            :disabled="state.uploading"
            action="/api/data/upload"
            @change="handleFileUpload"
            @drop="handleDrop"
          >
            <p class="ant-upload-drag-icon">
              <inbox-outlined />
            </p>
            <p class="ant-upload-text">点击或者把PDF文件拖拽到这里上传</p>
            <p class="ant-upload-hint">
              仅支持上传PDF文件。同名文件无法重复添加
            </p>
          </a-upload-dragger>
        </div>
        <a-button type="primary" size="large" @click="convertPdfToText" :loading="state.loading" class="convert-btn">
          Start Conversion
        </a-button>
      </div>
      <div class="output-container">
        <div class="textarea-wrapper">
          <textarea v-model="convertedText" placeholder="Converted text will appear here..." readonly></textarea>
        </div>
        <div class="infos">
          <a-tag color="blue">字符数: {{ charCount }}</a-tag>
          <a-tag color="cyan">Token 数: {{ estimatedTokenCount }}</a-tag>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref, computed } from 'vue';
import { message } from 'ant-design-vue';
import { InboxOutlined } from '@ant-design/icons-vue';
import axios from 'axios';

const state = reactive({
  loading: false,
  uploading: false,
});

const fileList = ref([]);
const convertedText = ref('');

const charCount = computed(() => convertedText.value.length);
const estimatedTokenCount = computed(() => {
  const chars = convertedText.value.split('');
  let tokenCount = 0;
  for (let char of chars) {
    if (/[\u4e00-\u9fff]/.test(char)) {
      tokenCount += 1;
    } else if (/[a-zA-Z]/.test(char)) {
      tokenCount += 0.25;
    } else {
      tokenCount += 0.5;
    }
  }
  return Math.ceil(tokenCount);
});

const handleFileUpload = (info) => {
  const { status } = info.file;
  if (status !== 'uploading') {
    // console.log(info.file, info.fileList);
  }
  if (status === 'done') {
    message.success(`${info.file.name} file uploaded successfully.`);
  } else if (status === 'error') {
    message.error(`${info.file.name} file upload failed.`);
  }
};

const handleDrop = (e) => {
  // console.log(e);
};

const convertPdfToText = async () => {
  if (fileList.value.length === 0) {
    message.warning("请先上传PDF文件");
    return;
  }

  // Ensure we have a valid response from the upload
  const fileResponse = fileList.value[0].response;
  if (!fileResponse || !fileResponse.file_path) {
     message.error("文件上传尚未完成或失败");
     return;
  }
  
  const file = fileResponse.file_path;

  try {
    state.loading = true;
    const response = await axios.post('/api/tools/pdf2txt', { file: file.toString() });
    
    if (response.data) {
       convertedText.value = response.data.text;
    } else {
       throw new Error("No data received");
    }
  } catch (error) {
    console.error('Error converting PDF to text:', error);
    message.error('PDF转换失败，请重试');
  } finally {
    state.loading = false;
  }
};
</script>

<style lang="less" scoped>
.pdf2txt-container {
  display: flex;
  height: 100%;
  overflow: hidden;
  background-color: var(--surface-card); /* Standardized */
  border: 1px solid var(--border-color);
  
  /* Inherits .window-card styles if class added, but scoped here for safety */
  
  .sidebar {
    width: 280px;
    background-color: var(--surface-secondary); /* Standardized */
    border-right: 1px solid var(--border-color); /* Standardized */
    padding: 24px;
    display: flex;
    flex-direction: column;

    .additional-params {
      h4 {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      
      .subtext {
        color: var(--subtext-color);
        font-size: 13px;
      }
    }
  }

  .result-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 24px;
    gap: 24px;
    overflow-y: auto;

    .input-container {
      display: flex;
      flex-direction: column;
      gap: 16px;

      .upload-wrapper {
        border-radius: var(--radius-lg);
        overflow: hidden;
        border: 1px dashed var(--border-color);
        transition: border-color 0.3s;
        
         &:hover {
            border-color: var(--primary-color);
         }
         
         .ant-upload-text {
            color: var(--text-color);
            font-weight: 500;
         }
         
         .ant-upload-hint {
            color: var(--subtext-color);
         }
      }
      
      .convert-btn {
         align-self: flex-start;
         min-width: 120px;
      }
    }

    .output-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0; 

      .textarea-wrapper {
        flex: 1;
        display: flex;
        flex-direction: column;
        
        textarea {
            flex: 1;
            width: 100%;
            padding: 16px;
            border: 1px solid var(--border-color); /* Standardized */
            border-radius: var(--radius-lg);
            font-size: 14px;
            line-height: 1.6;
            font-family: var(--font-mono);
            background-color: var(--input-background-color); /* Standardized */
            color: var(--text-color);
            resize: none;
            transition: all 0.2s;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    
            &:focus {
              border-color: var(--primary-color); /* Standardized */
              outline: none;
              box-shadow: 0 0 0 2px var(--primary-bg-light);
            }
            
            &::placeholder {
              color: var(--subtext-color);
            }
          }
      }

      .infos {
        display: flex;
        gap: 12px;
        align-items: center;
      }
    }
  }
}
</style>
