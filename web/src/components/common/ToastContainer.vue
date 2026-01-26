<template>
  <transition-group name="toast-list" tag="div" class="toast-container">
    <div
      v-for="toast in toasts"
      :key="toast.id"
      class="toast-item"
      :class="toast.type"
      @click="removeToast(toast.id)"
    >
      <component :is="getIcon(toast.type)" class="toast-icon" />
      <div class="toast-content">
        <span v-if="toast.title" class="toast-title">{{ toast.title }}</span>
        <span class="toast-message">{{ toast.message }}</span>
      </div>
      <button class="toast-close" @click.stop="removeToast(toast.id)">
        <CloseOutlined />
      </button>
    </div>
  </transition-group>
</template>

<script setup lang="ts">
import { ref, markRaw } from 'vue'
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CloseCircleOutlined,
  CloseOutlined
} from '@ant-design/icons-vue'

interface Toast {
  id: number
  type: 'success' | 'error' | 'warning' | 'info'
  title?: string
  message: string
  duration: number
}

const toasts = ref<Toast[]>([])
let toastId = 0

const iconMap = {
  success: markRaw(CheckCircleOutlined),
  error: markRaw(CloseCircleOutlined),
  warning: markRaw(ExclamationCircleOutlined),
  info: markRaw(InfoCircleOutlined)
}

const getIcon = (type: Toast['type']) => iconMap[type]

const addToast = (options: Omit<Toast, 'id'>) => {
  const id = ++toastId
  const toast: Toast = {
    id,
    ...options,
    duration: options.duration || 3000
  }
  toasts.value.push(toast)

  if (toast.duration > 0) {
    setTimeout(() => {
      removeToast(id)
    }, toast.duration)
  }

  return id
}

const removeToast = (id: number) => {
  const index = toasts.value.findIndex((t) => t.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

const success = (message: string, title?: string) =>
  addToast({ type: 'success', message, title, duration: 3000 })

const error = (message: string, title?: string) =>
  addToast({ type: 'error', message, title, duration: 5000 })

const warning = (message: string, title?: string) =>
  addToast({ type: 'warning', message, title, duration: 4000 })

const info = (message: string, title?: string) =>
  addToast({ type: 'info', message, title, duration: 3000 })

defineExpose({
  success,
  error,
  warning,
  info,
  addToast,
  removeToast
})
</script>

<style scoped lang="less">
.toast-container {
  position: fixed;
  top: var(--space-4);
  right: var(--space-4);
  z-index: var(--z-toast);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  max-width: 360px;
  pointer-events: none;
}

.toast-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);
  cursor: pointer;
  pointer-events: auto;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    box-shadow: var(--shadow-lg);
    transform: translateX(-4px);
  }

  &.success {
    border-left: 4px solid var(--success-color);
    .toast-icon { color: var(--success-color); }
  }

  &.error {
    border-left: 4px solid var(--error-color);
    .toast-icon { color: var(--error-color); }
  }

  &.warning {
    border-left: 4px solid var(--warning-color);
    .toast-icon { color: var(--warning-color); }
  }

  &.info {
    border-left: 4px solid var(--holo-accent);
    .toast-icon { color: var(--holo-accent); }
  }
}

.toast-icon {
  font-size: 18px;
  flex-shrink: 0;
  margin-top: 2px;
}

.toast-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.toast-title {
  font-weight: 600;
  font-size: var(--font-size-sm);
  color: var(--text-color);
}

.toast-message {
  font-size: var(--font-size-sm);
  color: var(--gray-600);
  word-break: break-word;
}

.toast-close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--gray-400);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  flex-shrink: 0;

  &:hover {
    background: var(--hover-bg);
    color: var(--gray-600);
  }
}

/* Animation */
.toast-list-enter-active {
  animation: slideInRight var(--duration-base) var(--ease-out);
}

.toast-list-leave-active {
  animation: slideOutRight var(--duration-fast) var(--ease-in);
  position: absolute;
  right: 0;
}

.toast-list-move {
  transition: transform var(--duration-base) var(--ease-default);
}

@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slideOutRight {
  from {
    opacity: 1;
    transform: translateX(0);
  }
  to {
    opacity: 0;
    transform: translateX(100%);
  }
}

/* Mobile */
@media (max-width: 640px) {
  .toast-container {
    top: auto;
    bottom: var(--space-4);
    left: var(--space-4);
    right: var(--space-4);
    max-width: none;
  }

  .toast-item {
    width: 100%;
  }
}
</style>
