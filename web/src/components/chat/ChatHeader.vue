<template>
  <div class="chat-header">
    <div class="header__left">
      <div
        v-if="!sidebarOpen"
        class="nav-btn"
        @click="emit('toggle-sidebar')"
        role="button"
        tabindex="0"
        aria-label="打开对话侧栏"
        @keydown.enter.prevent="emit('toggle-sidebar')"
        @keydown.space.prevent="emit('toggle-sidebar')"
      >
        <img
          src="@/assets/icons/sidebar_left.svg"
          class="iconfont icon-20 sidebar-icon"
          alt="侧栏"
        />
      </div>

      <div
        class="action-button"
        @click="emit('new-conversation')"
        role="button"
        tabindex="0"
        aria-label="新建会话"
        @keydown.enter.prevent="emit('new-conversation')"
        @keydown.space.prevent="emit('new-conversation')"
      >
        <PlusCircleOutlined class="icon" />
        <span class="text">新建会话</span>
      </div>
    </div>

    <div class="header__right">
      <ModelSelector
        :model-provider="modelProvider"
        :model-name="modelName"
        :model-names="modelNames"
        :custom-models="customModels"
        @select-model="(provider, name) => emit('select-model', provider, name)"
      />
      <ThemeToggle />
      <div
        class="nav-btn text"
        @click="emit('toggle-options')"
        role="button"
        tabindex="0"
        aria-label="聊天选项"
        @keydown.enter.prevent="emit('toggle-options')"
        @keydown.space.prevent="emit('toggle-options')"
      >
        <component :is="optionsOpen ? FolderOpenOutlined : FolderOutlined" />
        <span class="text">选项</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  PlusCircleOutlined,
  FolderOutlined,
  FolderOpenOutlined
} from '@ant-design/icons-vue'
import ThemeToggle from '@/components/common/ThemeToggle.vue'
import ModelSelector from './ModelSelector.vue'
import type { ModelProvider, CustomModel } from '@/types/chat'

defineProps<{
  sidebarOpen: boolean
  optionsOpen: boolean
  modelProvider?: string
  modelName?: string
  modelNames: Record<string, ModelProvider>
  customModels: CustomModel[]
}>()

const emit = defineEmits<{
  (e: 'toggle-sidebar'): void
  (e: 'new-conversation'): void
  (e: 'toggle-options'): void
  (e: 'select-model', provider: string, name: string): void
}>()
</script>

<style scoped lang="less">
.chat-header {
  user-select: none;
  position: sticky;
  top: 0;
  z-index: 10;

  /* Glassmorphism Header */
  background: var(--glass-bg);
  backdrop-filter: blur(var(--blur-md));
  border-bottom: 1px solid var(--border-color);

  height: var(--header-height);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-3) var(--space-6);

  .header__left,
  .header__right {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
}

.nav-btn {
  height: var(--button-height-lg);
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: var(--radius-sm);
  color: var(--text-color);
  cursor: pointer;
  width: auto;
  transition: all var(--duration-slow);
  padding: var(--space-2) var(--space-3);

  .text {
    margin-left: var(--space-2);
    font-weight: 500;
  }

  &:hover {
    background-color: var(--hover-bg);
    color: var(--primary-color);
  }
}

.action-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: var(--space-2) var(--space-4);
  font-size: var(--font-size-base);
  font-weight: 500;
  color: var(--message-user-text);
  background: var(--gradient-primary);
  border: none;
  border-radius: var(--radius-lg);
  cursor: pointer;
  box-shadow: var(--message-user-shadow);
  transition: all var(--duration-base) ease;

  &:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
    background: var(--gradient-primary-hover);
  }

  .icon {
    font-size: var(--font-size-lg);
  }

  .text {
    white-space: nowrap;
  }
}
</style>
