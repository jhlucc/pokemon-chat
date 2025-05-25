<template>
  <div class="message-wrapper" :class="{ 'from-user': isUser, 'from-ai': !isUser }">
  <img class="avatar" :src="avatar" alt="avatar" />
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

      <MdPreview v-else-if="message.content"
        ref="editorRef"
        editorId="preview-only"
        previewTheme="github"
        :showCodeRowNumber="false"
        :modelValue="message.content"
        :key="message.id"
        class="message-md"
      />
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
</script>

<!-- =============== style scoped：气泡 & deepseek 胶囊 =============== -->
<style lang="less" scoped>
/* ===== wrapper 布局 + 头像 + 气泡背景 ===== */
.message-wrapper {
  display: flex;
  align-items: flex-start;
  margin-bottom: 1rem;

  &.from-user {
    flex-direction: row-reverse;
    .message-box { background:#e6f4ff; color:#222; }
  }
  &.from-ai   {
    flex-direction: row;
    .message-box { background:#f8f8f8; color:#000; }
  }
  .avatar{
    width:36px;height:36px;border-radius:50%;
    margin:0 12px;object-fit:cover;
  }
}

/* ===== 公共文字 / loading / 提示 ===== */
.retry-hint{margin-top:8px;padding:8px 16px;color:#666;font-size:14px;text-align:left;}
.retry-link{color:#1890ff;cursor:pointer;margin-left:4px;&:hover{text-decoration:underline;}}
.ant-btn-icon-only:has(.anticon-stop){background: #bd0707 !important;&:hover{background:#ff7875!important;}}
.loading-dots{display:inline-flex;align-items:center;justify-content:center;
  div{width:8px;height:8px;margin:0 4px;background:#666;border-radius:50%;opacity:.3;animation:pulse .5s infinite both;
    &:nth-child(1){animation-delay:-.32s} &:nth-child(2){animation-delay:-.16s}}
}
@keyframes pulse{0%,80%,100%{transform:scale(.8);opacity:.3}40%{transform:scale(1);opacity:1}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* ===== message-box 内排版 ===== */
.message-box{
  display:inline-block;border-radius:1.5rem;margin:.8rem 0;padding:.625rem 1.25rem;
  user-select:text;word-break:break-word;font-size:15px;line-height:24px;max-width:100%;position:relative;
  &.assistant,&.received{color:initial;width:100%;text-align:left;margin:0 0 16px;padding:0;background:transparent}
  .err-msg{color: #100f0f;border:1px solid #f19999;padding:.5rem 1rem;border-radius:8px;background:#fffbfb;margin-bottom:10px;cursor:pointer}
  .searching-msg{color:#666;animation:colorPulse 1s infinite ease-in-out}
  .reasoning-box{margin:10px 0 15px;border:1px solid var(--main-light-3);border-radius:8px;
    .reasoning-content{font-size:13px;color:#444;white-space:pre-wrap;margin:0}}
  :deep(.tool-calls-container){
  display:inline-flex !important;     /* 横向摆放多枚胶囊 */
  flex-wrap: wrap;
  gap:8px;                             /* 胶囊之间间距 */
  width:auto !important;
  margin-top:10px;                     /* 原来的外边距保留 */
  background:transparent !important;
  border:none !important;
}
}
@keyframes colorPulse{0%{color:#666}50%{color:#ccc}100%{color:#666}}

/* ============ 关键修改：deepseek 胶囊自适应宽度 ============ */
/* 1. container 变 inline-block */
:deep(.tool-call-container){
  display:inline-block!important;
  width:auto!important;
  max-width:max-content!important;
  background:transparent!important;
  padding:0!important;
}

/* 2. 胶囊本体 inline-flex */
:deep(.tool-call-display){
  display:inline-flex!important;
   flex: 0 0 auto !important;
  align-items:center;
  gap:6px;
  padding:4px 8px;
  width:auto!important;
  max-width:max-content!important;
  background:var(--gray-50);
  border:1px solid var(--gray-200);
  border-radius:8px;

  /* 标题区域去掉 block 背景 & 边框 */
  .tool-header{background:transparent;border:none;padding:0;margin:0;gap:6px;}

  /* 小图标配色 */
  .anticon{color:#999;cursor:pointer;&:hover{color:#555}}
}

/* D. 最里层 .tool-content 保险起见也设成 inline-flex */
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
  color:var(--gray-900);
  max-width:100%;
  padding:0;
  font-family:-apple-system,BlinkMacSystemFont,'Noto Sans SC','PingFang SC','Microsoft YaHei','Hiragino Sans GB','Courier New',monospace;
  #preview-only-preview{font-size:15px;}
  h1,h2{font-size:1.2rem;} h3,h4{font-size:1.1rem;} h5,h6{font-size:1rem;}
  a{color:var(--main-700);}
  code{
    font-size:13px;
    font-family:'Menlo','Monaco','Consolas','Courier New',monospace;
    line-height:1.5;letter-spacing:.025em;tab-size:4;-moz-tab-size:4;
    background:var(--gray-100);
  }
}

/* deepseek-chat 模型名单独行 → 改成 inline */
.model-name{display:inline;font-weight:600;margin-right:.5em;}

/* 聊天字体缩放 */
.chat-box.font-smaller #preview-only-preview{font-size:14px;h1,h2{font-size:1.1rem;}h3,h4{font-size:1rem;}}
.chat-box.font-larger  #preview-only-preview{font-size:16px;h1,h2{font-size:1.3rem;}h3,h4{font-size:1.2rem;}
  h5,h6{font-size:1.1rem;}code{font-size:14px;}}
</style>
