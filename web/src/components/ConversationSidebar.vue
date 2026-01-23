<template>
  <div class="conversations" :class="{ 'is-open': isOpen }">
    <div class="sidebar-header">
      <button class="new-chat-btn" @click="$emit('new-chat')">
        <PlusOutlined />
        <span>New Chat</span>
      </button>
      <div class="action close" @click="$emit('toggle')" title="Collapse Sidebar">
           <MenuFoldOutlined />
      </div>
    </div>
    
    <div class="conversation-list">
      <div class="list-title" v-if="conversations.length > 0">Recent Chats</div>
      <div class="conversation"
        v-for="(conv, index) in conversations"
        :key="conv.id"
        :class="{ active: currentId === index }"
        @click="$emit('select', index)">
        <div class="conversation__icon">
            <MessageOutlined v-if="currentId !== index"/>
            <MessageFilled v-else />
        </div>
        <div class="conversation__content">
            <div class="conversation__title">{{ conv.title }}</div>
        </div>
        <div class="conversation__delete" @click.stop="$emit('delete', index)"><DeleteOutlined /></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { 
    DeleteOutlined, 
    PlusOutlined, 
    MessageOutlined, 
    MessageFilled,
    MenuFoldOutlined 
} from '@ant-design/icons-vue'

defineProps({
    isOpen: Boolean,
    conversations: {
        type: Array,
        default: () => []
    },
    currentId: {
        type: Number,
        default: 0
    }
})

defineEmits(['toggle', 'new-chat', 'select', 'delete'])
</script>

<style lang="less" scoped>
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
        height: 40px;
        background-color: var(--primary-color);
        border: 1px solid transparent;
        border-radius: var(--radius-md);
        color: #FFFFFF;
        font-weight: 500;
        font-size: 14px;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
        cursor: pointer;
        
        &:hover {
            background-color: var(--primary-hover-color);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        
        &:active {
            transform: translateY(0);
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
            background-color: var(--surface-secondary);
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
      color: var(--text-color);
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
        background-color: var(--surface-secondary);
      }
    }

    /* Active State */
    &.active {
      background-color: var(--surface-secondary);
      color: var(--primary-color);

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
      background-color: var(--surface-secondary);
      
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
  background: transparent;
  border-radius: 4px;
}

.conversation-list:hover::-webkit-scrollbar-thumb {
    background: var(--slate-300);
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
}
</style>
