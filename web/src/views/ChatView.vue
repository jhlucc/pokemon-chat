<template>
  <div class="chat-container">
<!--    左边是侧边栏（对话列表 conversations）-->
    <div class="conversations" :class="{ 'is-open': state.isSidebarOpen }">
      <div class="actions">
        <!-- <div class="action new" @click="addNewConv"><FormOutlined /></div> -->
            <div class="action new" @click="addNewConv" title="新建对话">
      <!-- 使用 PlusCircleOutlined 比较直观 -->
        <PlusCircleOutlined />
            </div>
         <span class="header-title">对话历史</span>
        <div class="action close" @click="state.isSidebarOpen = false">
          <img src="@/assets/icons/sidebar_left.svg" class="iconfont icon-20" alt="设置" />
        </div>
      </div>
      <div class="conversation-list">
        <div class="conversation"
          v-for="(state, index) in convs"
          :key="index"
          :class="{ active: curConvId === index }"
          @click="goToConversation(index)">
          <div class="conversation__title"><CommentOutlined /> &nbsp;{{ state.title }}</div>
          <div class="conversation__delete" @click.stop="delConv(index)"><DeleteOutlined /></div>
        </div>
      </div>
    </div>
<!--    聊天组件（ChatComponent） 渲染右边聊天内容区域。  把当前选中的对话 (convs[curConvId]) 作为 prop 传给 ChatComponent,传递状态对象 state-->
    <ChatComponent
      :conv="convs[curConvId]"
      :state="state"
      @rename-title="renameTitle"
      @newconv="addNewConv"/>
<!--  重命名对话&新建对话-->
  </div>
</template>

<script setup>
import { reactive, ref, watch, onMounted } from 'vue'

import ChatComponent from '@/components/ChatComponent.vue'
import { DeleteOutlined, CommentOutlined, PlusCircleOutlined } from '@ant-design/icons-vue'
// 从 localStorage 里读取历史对话记录，如果没有就用一个初始默认对话。
const convs = reactive(JSON.parse(localStorage.getItem('chat-convs')) || [
  {
    id: 0,
    title: '新对话',
    history: [],
    messages: [],
    inputText: ''
  },
])

const state = reactive({
  isSidebarOpen: JSON.parse(localStorage.getItem('chat-sidebar-open') || 'true'),
})

// Watch isSidebarOpen and save to localStorage
watch(
  () => state.isSidebarOpen,
  (newValue) => {
    localStorage.setItem('chat-sidebar-open', JSON.stringify(newValue))
  }
)
const curConvId = ref(0)

const generateRandomHash = (length) => {
    let chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let hash = '';
    for (let i = 0; i < length; i++) {
        hash += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return hash;
}

const renameTitle = (newTitle) => {
  convs[curConvId.value].title = newTitle
}

const goToConversation = (index) => {
  curConvId.value = index
  console.log(convs[curConvId.value])
}

const addNewConv = () => {
  curConvId.value = 0
  if (convs.length > 0 && convs[0].messages.length === 0) {
    return
  }
  convs.unshift({
    id: generateRandomHash(8),
    title: `新对话`,
    history: [],
    messages: [],
    inputText: ''
  })
}

const delConv = (index) => {
  convs.splice(index, 1)

  if (index < curConvId.value) {
    curConvId.value -= 1
  } else if (index === curConvId.value) {
    curConvId.value = 0
  }

  if (convs.length === 0) {
    addNewConv()
  }
}

// Watch convs and save to localStorage
watch(
  () => convs,
  (newStates) => {
    localStorage.setItem('chat-convs', JSON.stringify(newStates))
  },
  { deep: true }
)

// Load convs from localStorage on mount
onMounted(() => {
  const savedSonvs = JSON.parse(localStorage.getItem('chat-convs'))
  if (savedSonvs) {
    for (let i = 0; i < savedSonvs.length; i++) {
      convs[i] = savedSonvs[i]
    }
  }
})
</script>

<style lang="less" scoped>
@import '@/assets/main.css';

.chat-container {
  display: flex;
  width: 100%;
  height: 100%;
  position: relative;
  background-color: var(--background-color);
}

.conversations {
  width: 260px; /* Slightly wider for better readability */
  max-width: 280px;
  border-right: 1px solid var(--border-color);
  background-color: var(--sidebar-background-color);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap;
  overflow: hidden;
  display: flex;
  flex-direction: column;

  &.is-open {
    width: 260px;
    opacity: 1;
  }

  &:not(.is-open) {
    width: 0;
    padding: 0;
    opacity: 0;
    overflow: hidden;
  }

  /* Header Actions */
  .actions {
    height: 64px; /* Match header height */
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 16px;
    z-index: 9;
    /* No border bottom for cleaner look, or very subtle */
    border-bottom: 1px solid transparent; 

    .header-title {
      font-weight: 600;
      font-size: 16px;
      color: var(--text-color);
      user-select: none;
    }

    .action {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      color: var(--subtext-color);
      cursor: pointer;
      transition: all 0.2s ease;

      &:hover {
        background-color: var(--gray-100);
        color: var(--text-color);
      }

      &.new {
        color: var(--primary-color);
        background-color: var(--primary-bg-light);
        
        &:hover {
            background-color: var(--primary-light-color);
            color: #FFF;
        }
      }
    }
  }

  .conversation-list {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 12px;
    gap: 4px; /* Spacing between items */
  }

  .conversation-list .conversation {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    cursor: pointer;
    width: 100%;
    user-select: none;
    border-radius: 8px; /* Rounded items */
    transition: all 0.2s ease;
    border: 1px solid transparent;

    &__title {
      color: var(--subtext-color);
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    &__delete {
      display: none;
      color: var(--subtext-color);
      font-size: 12px;
      padding: 4px;
      border-radius: 4px;
      transition: all 0.2s ease;

      &:hover {
        color: var(--error-color);
        background-color: rgba(239, 68, 68, 0.1);
      }
    }

    &.active {
      background-color: var(--primary-bg-light);
      border-color: rgba(79, 70, 229, 0.1);

      & .conversation__title {
        color: var(--primary-color);
        font-weight: 500;
      }
    }

    &:not(.active):hover {
      background-color: var(--gray-100);
      
      & .conversation__delete {
        display: block;
      }
    }
  }
}

/* Scrollbar Styling */
.conversation-list::-webkit-scrollbar {
  width: 4px;
}

.conversation-list::-webkit-scrollbar-track {
  background: transparent;
}

.conversation-list::-webkit-scrollbar-thumb {
  background: var(--gray-300);
  border-radius: 4px;
}

.conversation-list::-webkit-scrollbar-thumb:hover {
  background: var(--gray-400);
}

@media (max-width: 520px) {
  .conversations {
    position: absolute;
    z-index: 101;
    width: 280px;
    height: 100%;
    border-radius: 0;
    box-shadow: var(--shadow-xl); /* Stronger shadow for floating sidebar */
    background-color: var(--background-color);

    &.is-open {
        transform: translateX(0);
    }

    &:not(.is-open) {
      width: 280px; /* Keep width but transform off-screen */
      transform: translateX(-100%);
      padding: 0;
    }
  }
}
</style>
