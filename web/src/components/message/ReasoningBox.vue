<template>
  <div v-if="content" class="reasoning-box">
    <div class="reasoning-header" @click="isOpen = !isOpen">
      <div class="reasoning-indicator">
        <ThunderboltOutlined
          class="reasoning-icon"
          :class="{ active: isReasoning }"
        />
        <span class="reasoning-title">
          {{ isReasoning ? '正在思考...' : '推理过程' }}
        </span>
      </div>
      <CaretRightOutlined class="reasoning-caret" :class="{ open: isOpen }" />
    </div>
    <transition name="collapse">
      <div v-if="isOpen" class="reasoning-body">
        <p class="reasoning-content">{{ content }}</p>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ThunderboltOutlined, CaretRightOutlined } from '@ant-design/icons-vue'

defineProps<{
  content?: string
  isReasoning?: boolean
}>()

const isOpen = ref(true)
</script>

<style scoped lang="less">
.reasoning-box {
  margin: var(--space-2) 0 var(--space-3);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--surface-color-2) 80%, transparent);
  overflow: hidden;
}

.reasoning-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  cursor: pointer;
  transition: background var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--hover-bg);
  }
}

.reasoning-indicator {
  display: flex;
  align-items: center;
  gap: var(--space-2);

  .reasoning-icon {
    font-size: 14px;
    color: var(--gray-500);

    &.active {
      color: var(--warning-color);
      animation: pulse 1.5s infinite;
    }
  }

  .reasoning-title {
    font-size: var(--font-size-sm);
    font-weight: 500;
    color: var(--gray-600);
  }
}

.reasoning-caret {
  font-size: 12px;
  color: var(--gray-400);
  transition: transform var(--duration-fast) var(--ease-default);

  &.open {
    transform: rotate(90deg);
  }
}

.reasoning-body {
  padding: 0 var(--space-3) var(--space-3);
}

.reasoning-content {
  font-size: var(--font-size-sm);
  color: var(--gray-600);
  white-space: pre-wrap;
  margin: 0;
  line-height: var(--line-height-relaxed);
}

/* Collapse transition */
.collapse-enter-active,
.collapse-leave-active {
  transition: all var(--duration-base) var(--ease-default);
  max-height: 500px;
  overflow: hidden;
}

.collapse-enter-from,
.collapse-leave-to {
  max-height: 0;
  opacity: 0;
}

/* Pulse animation */
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
