<template>
  <div class="chat-container">
<!--    左边是侧边栏（对话列表 conversations）-->
    <div class="conversations" :class="{ 'is-open': state.isSidebarOpen }">
      <div class="sidebar-header">
        <button class="new-chat-btn" @click="addNewConv">
          <PlusOutlined />
          <span>新对话</span>
        </button>
        <div class="action close" @click="state.isSidebarOpen = false" title="收起列表">
             <MenuFoldOutlined />
        </div>
      </div>
      
      <div class="conversation-list">
        <div class="list-title" v-if="convs.length > 0">近期记录</div>
        <div class="conversation"
          v-for="(state, index) in convs"
          :key="index"
          :class="{ active: curConvId === index }"
          @click="goToConversation(index)">
          <div class="conversation__icon">
              <MessageOutlined v-if="curConvId !== index"/>
              <MessageFilled v-else />
          </div>
          <div class="conversation__content">
              <div class="conversation__title">{{ state.title }}</div>
          </div>
          <div class="conversation__delete" @click.stop="delConv(index)"><DeleteOutlined /></div>
        </div>
      </div>
    </div>
    
    <!-- Sidebar Toggle (Visible when closed) -->
     <div v-if="!state.isSidebarOpen" class="sidebar-toggle" @click="state.isSidebarOpen = true">
      <MenuUnfoldOutlined />
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
import { 
    DeleteOutlined, 
    CommentOutlined, 
    PlusOutlined, 
    MessageOutlined, 
    MessageFilled,
    MenuFoldOutlined,
    MenuUnfoldOutlined
} from '@ant-design/icons-vue'

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
  width: 280px; 
  height: 100%;
  border-right: 1px solid var(--border-color);
  background-color: var(--sidebar-background-color);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;

  &.is-open {
    width: 280px;
    opacity: 1;
    transform: translateX(0);
  }

  &:not(.is-open) {
    width: 0;
    opacity: 0;
    padding: 0;
    overflow: hidden;
    transform: translateX(-20px);
  }

  /* Header Actions */
  .sidebar-header {
    height: auto;
    padding: 20px 16px 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    
    .new-chat-btn {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        height: 44px;
        background-color: var(--surface-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        color: var(--text-color);
        font-weight: 500;
        font-size: 14px;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
        
        &:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
            background-color: var(--surface-card);
            box-shadow: var(--shadow-md);
        }
    }
    
    .action.close {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--subtext-color);
        cursor: pointer;
        border-radius: var(--radius-md);
        transition: all 0.2s ease;
        
        &:hover {
            background-color: var(--gray-200);
            color: var(--text-color);
        }
    }
  }

  .conversation-list {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 8px 12px 20px 12px;
    gap: 4px;
    
    .list-title {
        font-size: 11px;
        font-weight: 600;
        color: var(--subtext-color);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 12px 0 8px 8px;
    }
  }

  .conversation-list .conversation {
    display: flex;
    align-items: center;
    padding: 10px 12px;
    cursor: pointer;
    width: 100%;
    user-select: none;
    border-radius: var(--radius-md); 
    transition: all 0.2s ease;
    border: 1px solid transparent;
    color: var(--text-color);
    position: relative;
    height: 44px;

    &__icon {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        color: var(--subtext-color);
        margin-right: 12px;
        opacity: 0.7;
    }

    &__content {
        flex: 1;
        overflow: hidden;
    }

    &__title {
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-weight: 400;
    }

    &__delete {
      display: none;
      position: absolute;
      right: 8px;
      color: var(--subtext-color);
      font-size: 14px;
      padding: 4px;
      border-radius: 4px;
      z-index: 2;
      background: linear-gradient(to right, transparent, var(--sidebar-background-color) 20%);
      padding-left: 12px;

      &:hover {
        color: var(--error-color);
      }
    }

    /* Active State */
    &.active {
      background-color: var(--surface-card);
      box-shadow: var(--shadow-sm);
      border-color: var(--border-color);

      .conversation__title {
        color: var(--primary-color);
        font-weight: 500;
      }
      
      .conversation__icon {
          color: var(--primary-color);
          opacity: 1;
      }
    }
    
    /* Hover State */
    &:hover:not(.active) {
      background-color: var(--gray-100);
      
      & .conversation__delete {
        display: block;
      }
    }
  }
}

.sidebar-toggle {
    position: absolute;
    top: 20px;
    left: 20px;
    z-index: 10;
    width: 36px;
    height: 36px;
    background-color: var(--surface-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: var(--shadow-md);
    color: var(--subtext-color);
    transition: all 0.2s ease;
    
    &:hover {
        color: var(--primary-color);
        transform: scale(1.05);
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
  background: transparent;
  border-radius: 4px;
}

.conversation-list:hover::-webkit-scrollbar-thumb {
    background: var(--gray-300);
}

@media (max-width: 520px) {
  .conversations {
    position: absolute;
    z-index: 101;
    width: 80%;
    height: 100%;
    border-radius: 0;
    box-shadow: var(--shadow-xl); 
    background-color: var(--sidebar-background-color);

    &.is-open {
        transform: translateX(0);
    }

    &:not(.is-open) {
      width: 80%; 
      transform: translateX(-100%);
      padding: 0;
    }
  }
  
  .sidebar-toggle {
      left: 16px;
      top: 12px;
  }
}
</style>
