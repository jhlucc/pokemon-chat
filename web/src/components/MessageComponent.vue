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
          <a-collapse-panel key="show" :header="message.status=='reasoning' ? '正在思考...' : '推理过程'" class="reasoning-header">
            <p class="reasoning-content">{{ message.reasoning_content }}</p>
          </a-collapse-panel>
        </a-collapse>
      </div>

      <div v-if="isEmptyAndLoading" class="loading-dots">
        <div></div><div></div><div></div>
      </div>
      <div v-else-if="message.status === 'searching' && isProcessing" class="searching-msg"><i>正在检索……</i></div>
      <div v-else-if="message.status === 'generating' && isProcessing" class="searching-msg"><i>正在生成……</i></div>
      <div v-else-if="message.status === 'error'" class="err-msg" @click="$emit('retry')">请求错误，请重试。{{ message.message }}</div>
      
      <div v-else-if="message.content" class="content-block">
          <template v-for="(part, idx) in contentParts" :key="idx">
             <MdPreview v-if="part.type === 'text'"
                editorId="preview-only"
                previewTheme="github"
                :showCodeRowNumber="false"
                :modelValue="part.content"
                class="message-md"
             />
             <GraphRenderer v-else-if="part.type === 'graph'" :data="part.data" />
          </template>
      </div>
      <div v-else-if="message.reasoning_content" class="empty-block"></div>

      <slot v-else-if="message.toolCalls && Object.keys(message.toolCalls).length > 0" name="tool-calls"></slot>
      <div v-else class="err-msg" @click="$emit('retry')">请求错误，请重试。{{ message.message }}</div>

      <div v-if="message.isStoppedByUser" class="retry-hint">
        你停止生成了本次回答
        <span class="retry-link" @click="emit('retryStoppedMessage', message.id)">重新编辑问题</span>
      </div>

      <div v-if="message.status==='finished' && showRefs">
        <RefsComponent :message="message" :show-refs="showRefs" @retry="emit('retry')" />
      </div>
    </template>

    <!-- 自定义内容 -->
    <slot></slot>
  </div>
  </div>
</template>


<script setup>
import { computed, ref } from 'vue';
import { CaretRightOutlined } from '@ant-design/icons-vue';
import RefsComponent from '@/components/RefsComponent.vue'
import GraphRenderer from '@/components/GraphRenderer.vue'


import { MdPreview } from 'md-editor-v3'
import 'md-editor-v3/lib/preview.css';

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
  },
});
const isUser = computed(() => props.message.role === 'user' || props.message.role === 'sent')
// ⚠️ 头像文件放在 public/images 下，或改成你的实际路径
const avatar = computed(() =>
  isUser.value ? 'avatar.jpg' : 'user.png'
)
const editorRef = ref()
const statusDefination = {
  init: '初始化',
  loading: '加载中',
  reasoning: '推理中',
  generating: '生成中',
  error: '错误'
}

const emit = defineEmits(['retry', 'retryStoppedMessage']);

// 推理面板展开状态
const reasoningActiveKey = ref(['show']);


// 计算属性：内容为空且正在加载
const isEmptyAndLoading = computed(() => {
  const isEmpty = !props.message.content || props.message.content.length === 0;
  const isLoading = props.message.status === 'init' && props.isProcessing
  return isEmpty && isLoading;
});

const contentParts = computed(() => {
    const text = props.message.content || "";
    const parts = [];
    const regex = /```json-graph\n([\s\S]*?)\n```/g;
    
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
        // Text before match
        if (match.index > lastIndex) {
            parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
        }
        
        // Graph data
        try {
            const json = JSON.parse(match[1]);
            parts.push({ type: 'graph', data: json });
        } catch (e) {
            console.error("Failed to parse graph json", e);
            parts.push({ type: 'text', content: match[0] }); // Fallback to text
        }
        
        lastIndex = regex.lastIndex;
    }
    
    // Remaining text
    if (lastIndex < text.length) {
        parts.push({ type: 'text', content: text.substring(lastIndex) });
    }
    
    return parts;
});
</script>

<!-- =============== style scoped：气泡 & deepseek 胶囊 =============== -->
<style lang="less" scoped>
/* ===== wrapper 布局 + 头像 + 气泡背景 ===== */
.message-wrapper {
  display: flex;
  align-items: flex-start;
  margin-bottom: 24px;

  &.from-user {
    flex-direction: row-reverse;
    .message-box { 
        /* User: Clean block, faint background or just text */
        background: var(--surface-card);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg) 0 var(--radius-lg) var(--radius-lg);
        box-shadow: var(--shadow-sm);
    }
  }
  &.from-ai   {
    flex-direction: row;
    .message-box { 
        /* AI: Transparent or minimal */
        background: transparent;
        color: var(--text-color);
        border: none;
        padding-left: 0;
    }
  }
  .avatar{
    width: 32px;
    height: 32px;
    border-radius: 4px; /* Tech square */
    margin: 0 16px;
    object-fit: cover;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
  }
}

/* ===== 公共文字 / loading / 提示 ===== */
.retry-hint{margin-top:8px;padding:8px 16px;color:var(--subtext-color);font-size:13px;text-align:left;}
.retry-link{color:var(--primary-color);cursor:pointer;margin-left:4px;font-weight: 500;&:hover{text-decoration:underline;}}
.ant-btn-icon-only:has(.anticon-stop){background: var(--error-color) !important;&:hover{background: #d32f2f !important;}}

.loading-dots{display:inline-flex;align-items:center;justify-content:center;
  div{width:4px;height:4px;margin:0 4px;background:var(--primary-color);border-radius:0; /* Square dots */ opacity:.5;animation:pulse .5s infinite both;
    &:nth-child(1){animation-delay:-.32s} &:nth-child(2){animation-delay:-.16s}}
}
@keyframes pulse{0%,80%,100%{transform:scale(.8);opacity:.5}40%{transform:scale(1);opacity:1}}

/* ===== message-box 内排版 ===== */
.message-box{
  display:inline-block;
  padding: 12px 18px;
  user-select: text;
  word-break: break-word;
  font-size: 15px;
  line-height: 1.6;
  max-width: 100%;
  position: relative;
  
  &.assistant,&.received{
      width:100%;
      text-align:left;
      margin:0;
      padding-top: 4px;
  }
  
  .err-msg{color: var(--error-color); border:1px solid var(--error-color); padding:.5rem 1rem; border-radius:4px; background: rgba(239, 68, 68, 0.05); margin-bottom:10px; cursor:pointer;}
  .searching-msg{color:var(--subtext-color); font-size: 13px; font-family: var(--font-mono);}
  
  .reasoning-box{
      margin: 8px 0 16px;
      border-left: 2px solid var(--primary-light-color);
      background-color: var(--surface-secondary); /* Contrast */
      border-radius: 0 4px 4px 0; 
      padding: 8px 16px;
    
    .reasoning-content{font-size:13px;color:var(--subtext-color); font-family: var(--font-mono); white-space:pre-wrap;margin:0}
    .reasoning-header { font-size: 12px; font-weight: 600; color: var(--text-color); margin-bottom: 4px; text-transform:uppercase; letter-spacing:0.05em;}
  }
  
  :deep(.tool-calls-container){
      display:inline-flex !important;
      flex-wrap: wrap;
      gap:8px;
      width:auto !important;
      margin-top:10px;
      background:transparent !important;
      border:none !important;
  }
}

/* ============ Tool Call Pills ============ */
:deep(.tool-call-container){
  display:inline-block!important;
  width:auto!important;
  max-width:max-content!important;
  background:transparent!important;
  padding:0!important;
}

:deep(.tool-call-display){
  display:inline-flex!important;
  flex: 0 0 auto !important;
  align-items:center;
  gap:6px;
  padding:4px 10px;
  width:auto!important;
  max-width:max-content!important;
  background:var(--surface-card);
  border:1px solid var(--border-color);
  border-radius:4px; /* Tech pill */
  box-shadow: var(--shadow-sm);
  font-family: var(--font-mono);
  font-size: 12px;

  .tool-header{background:transparent;border:none;padding:0;margin:0;gap:6px;}
  .anticon{color:var(--primary-color);}
}

:deep(.tool-call-display>.tool-content){
  display:inline-flex!important;
  width:auto!important;
  max-width:max-content!important;
  padding:0!important;
  margin:0!important;
  background:transparent!important;
  border:none!important;
}
</style>

<!-- =============== style (全局)：markdown / 字体 等 =============== -->
<style lang="less">
.message-md .md-editor-preview-wrapper{
  color: var(--text-color);
  max-width:100%;
  padding:0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  
  #preview-only-preview{font-size:15px; line-height: 1.7;}
  h1,h2{font-size:1.4rem; font-weight: 600; margin-top: 1.5em; margin-bottom: 0.8em; color: var(--text-color);}
  h3,h4{font-size:1.2rem; font-weight: 600; margin-top: 1.2em; margin-bottom: 0.6em;}
  h5,h6{font-size:1rem; font-weight: 600;}
  
  p { margin-bottom: 1em; }
  
  a{color:var(--primary-color); text-decoration: none; &:hover{text-decoration: underline;}}
  
  code{
    font-size:13px;
    font-family:'Menlo','Monaco','Consolas','Courier New',monospace;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--gray-100);
    color: var(--primary-color);
  }
  
  pre code {
      background: transparent;
      padding: 0;
      color: inherit;
  }
  
  blockquote {
      border-left: 4px solid var(--primary-light-color);
      background: var(--gray-50);
      padding: 12px 16px;
      margin: 1em 0;
      border-radius: 4px;
      color: var(--subtext-color);
  }
}

.model-name{display:inline;font-weight:600;margin-right:.5em; color: var(--subtext-color);}

.chat-box.font-smaller #preview-only-preview{font-size:14px;}
.chat-box.font-larger  #preview-only-preview{font-size:16px;}
</style>
