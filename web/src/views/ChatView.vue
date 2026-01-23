<template>
  <ChatLayout 
    :show-artifacts="state.showArtifacts" 
    @close-artifacts="closeArtifact"
  >
      <template #chat>
        <div class="chat-container">
            <!-- Sidebar (Conversations) -->
            <div class="conversations" :class="{ 'is-open': state.isSidebarOpen }">
              <div class="sidebar-header">
                <button class="new-chat-btn" @click="addNewConv">
                  <PlusOutlined />
                  <span>New Chat</span>
                </button>
                <div class="action close" @click="toggleSidebar" title="Collapse Sidebar">
                     <MenuFoldOutlined />
                </div>
              </div>
              
              <div class="conversation-list">
                <div class="list-title" v-if="convs.length > 0">Recent Chats</div>
                <div class="conversation"
                  v-for="(conv, index) in convs"
                  :key="conv.id"
                  :class="{ active: curConvId === index }"
                  @click="goToConversation(index)">
                  <div class="conversation__icon">
                      <MessageOutlined v-if="curConvId !== index"/>
                      <MessageFilled v-else />
                  </div>
                  <div class="conversation__content">
                      <div class="conversation__title">{{ conv.title }}</div>
                  </div>
                  <div class="conversation__delete" @click.stop="delConv(index)"><DeleteOutlined /></div>
                </div>
              </div>
            </div>
            
            <!-- Sidebar Toggle (Visible when closed) -->
            <div v-if="!state.isSidebarOpen" class="sidebar-toggle" @click="toggleSidebar">
              <MenuUnfoldOutlined />
            </div>

            <!-- Main Chat Area -->
            <div class="main-chat-area">
                <ChatComponent
                  ref="chatRef"
                  :conv="convs[curConvId]"
                  :state="state"
                  @rename-title="renameTitle"
                  @newconv="addNewConv"
                  @open-artifact="openArtifact"
                />
            </div>
          </div>
      </template>

      <template #artifacts>
          <ArtifactsView :artifact="state.currentArtifact" />
      </template>
  </ChatLayout>
</template>

<script setup>
import { reactive, ref, watch, onMounted } from 'vue'
import { 
    DeleteOutlined, 
    PlusOutlined, 
    MessageOutlined, 
    MessageFilled,
    MenuFoldOutlined,
    MenuUnfoldOutlined
} from '@ant-design/icons-vue'

import ChatComponent from '@/components/ChatComponent.vue'
import ChatLayout from '@/layouts/ChatLayout.vue'
import ArtifactsView from '@/components/Artifacts/ArtifactsView.vue'

// Data Persistence
const convs = reactive(JSON.parse(localStorage.getItem('chat-convs')) || [
  {
    id: 0,
    title: 'New Conversation',
    history: [],
    messages: [],
    inputText: ''
  },
])

const state = reactive({
  isSidebarOpen: JSON.parse(localStorage.getItem('chat-sidebar-open') || 'true'),
  showArtifacts: false,
  currentArtifact: null
})

const curConvId = ref(0)
const chatRef = ref(null)

// Methods
const generateRandomHash = (length) => {
    let chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let hash = '';
    for (let i = 0; i < length; i++) {
        hash += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return hash;
}

const renameTitle = (newTitle) => {
  if(convs[curConvId.value]) convs[curConvId.value].title = newTitle
}

const goToConversation = (index) => {
  curConvId.value = index
}

const addNewConv = () => {
  // If current empty, don't add new
  if (convs.length > 0 && convs[0].messages.length === 0) {
      curConvId.value = 0
      return
  }
  
  convs.unshift({
    id: generateRandomHash(8),
    title: `New Conversation`,
    history: [],
    messages: [],
    inputText: ''
  })
  curConvId.value = 0
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

const toggleSidebar = () => {
    state.isSidebarOpen = !state.isSidebarOpen
}

// Artifacts Logic
const openArtifact = (artifact) => {
    state.currentArtifact = artifact
    state.showArtifacts = true
}

const closeArtifact = () => {
    state.showArtifacts = false
}

// Watchers
watch(
  () => state.isSidebarOpen,
  (newValue) => {
    localStorage.setItem('chat-sidebar-open', JSON.stringify(newValue))
  }
)

watch(
  () => convs,
  (newStates) => {
    localStorage.setItem('chat-convs', JSON.stringify(newStates))
  },
  { deep: true }
)

onMounted(() => {
    // Basic persistence check
   const savedSonvs = JSON.parse(localStorage.getItem('chat-convs'))
   if (savedSonvs && savedSonvs.length > 0) {
       // Assuming reactive replacement handled by Vue 3 nicely or modify in place
       // Ideally we just trust the initial state reactive call, but let's be safe
   }
   
   // TEST: Trigger artifact for demo (optional, maybe remove before shipping)
   /*
   setTimeout(() => {
       openArtifact({
           type: 'code',
           language: 'javascript',
           content: 'console.log("Hello Artifacts!");'
       })
   }, 1000)
   */
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
  overflow: hidden;
}

.conversations {
  width: 280px; 
  height: 100%;
  border-right: 1px solid var(--border-color);
  background-color: var(--sidebar-background-color);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  display: flex;
  flex-direction: column;
  flex-shrink: 0; 
  z-index: 10;

  &.is-open {
    width: 280px;
    transform: translateX(0);
  }

  &:not(.is-open) {
    width: 0;
    padding: 0;
    overflow: hidden;
    transform: translateX(-100%);
    border-right: none;
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
        cursor: pointer;
        
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
    z-index: 20;
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

.main-chat-area {
    flex: 1;
    height: 100%;
    position: relative;
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
