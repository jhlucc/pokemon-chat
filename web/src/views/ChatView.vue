<template>
  <ChatLayout 
    :show-artifacts="state.showArtifacts" 
    @close-artifacts="closeArtifact"
  >
      <template #chat>
        <div class="chat-container">
            <!-- Sidebar (Conversations) -->
            <ConversationSidebar 
                :is-open="state.isSidebarOpen"
                :conversations="convs"
                :current-id="curConvId"
                @toggle="toggleSidebar"
                @new-chat="addNewConv"
                @select="goToConversation"
                @delete="delConv"
            />
            
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
import { MenuUnfoldOutlined } from '@ant-design/icons-vue'

import ChatComponent from '@/components/ChatComponent.vue'
import ChatLayout from '@/layouts/ChatLayout.vue'
import ArtifactsView from '@/components/Artifacts/ArtifactsView.vue'
import ConversationSidebar from '@/components/ConversationSidebar.vue'

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
   const savedSonvs = JSON.parse(localStorage.getItem('chat-convs'))
   if (savedSonvs && savedSonvs.length > 0) {
       // logic
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
  overflow: hidden;
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

@media (max-width: 520px) {
  .sidebar-toggle {
      left: 16px;
      top: 12px;
  }
}
</style>
