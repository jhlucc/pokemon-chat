<template>
  <a-modal
    v-model:open="visible"
    title="键盘快捷键"
    :footer="null"
    :width="400"
    class="shortcuts-modal"
    centered
  >
    <div class="shortcuts-list">
      <div
        v-for="(shortcut, index) in shortcuts"
        :key="index"
        class="shortcut-item"
      >
        <span class="shortcut-description">{{ shortcut.description }}</span>
        <div class="shortcut-keys">
          <kbd v-if="shortcut.ctrl" class="key">{{ isMac ? '⌘' : 'Ctrl' }}</kbd>
          <span v-if="shortcut.ctrl" class="key-separator">+</span>
          <kbd v-if="shortcut.shift" class="key">Shift</kbd>
          <span v-if="shortcut.shift" class="key-separator">+</span>
          <kbd v-if="shortcut.alt" class="key">{{ isMac ? '⌥' : 'Alt' }}</kbd>
          <span v-if="shortcut.alt" class="key-separator">+</span>
          <kbd class="key">{{ formatKey(shortcut.key) }}</kbd>
        </div>
      </div>
    </div>

    <div class="shortcuts-footer">
      <span class="hint">按 <kbd>?</kbd> 显示此帮助</span>
    </div>
  </a-modal>
</template>

<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue'

defineProps({
  shortcuts: {
    type: Array,
    default: () => []
  }
})

const visible = ref(false)

const isMac = computed(() => {
  if (typeof navigator === 'undefined') return false
  return /Mac|iPod|iPhone|iPad/.test(navigator.platform)
})

const formatKey = (key) => {
  const keyMap = {
    'Enter': '↵',
    'Escape': 'Esc',
    'ArrowUp': '↑',
    'ArrowDown': '↓',
    'ArrowLeft': '←',
    'ArrowRight': '→',
    ' ': 'Space',
    '/': '/'
  }
  return keyMap[key] || key.toUpperCase()
}

const handleKeyDown = (e) => {
  // Show help on "?"
  if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
    const target = e.target
    const isInput =
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable
    if (!isInput) {
      e.preventDefault()
      visible.value = !visible.value
    }
  }

  // Close on Escape
  if (e.key === 'Escape' && visible.value) {
    visible.value = false
  }
}

const show = () => {
  visible.value = true
}

const hide = () => {
  visible.value = false
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})

defineExpose({ show, hide, visible })
</script>

<style scoped lang="less">
.shortcuts-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.shortcut-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-3);
  background: var(--surface-color-2);
  border-radius: var(--radius-sm);
  transition: background var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--hover-bg);
  }
}

.shortcut-description {
  font-size: var(--font-size-sm);
  color: var(--text-color);
}

.shortcut-keys {
  display: flex;
  align-items: center;
  gap: 4px;
}

.key {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 24px;
  padding: 0 8px;
  background: var(--surface-color);
  border: 1px solid var(--gray-300);
  border-radius: 4px;
  font-family: var(--font-family-mono);
  font-size: var(--font-size-xs);
  color: var(--gray-700);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.key-separator {
  font-size: var(--font-size-xs);
  color: var(--gray-400);
}

.shortcuts-footer {
  margin-top: var(--space-4);
  padding-top: var(--space-3);
  border-top: 1px solid var(--border-color);
  text-align: center;

  .hint {
    font-size: var(--font-size-xs);
    color: var(--gray-500);

    kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 20px;
      height: 20px;
      padding: 0 6px;
      margin: 0 4px;
      background: var(--surface-color);
      border: 1px solid var(--gray-300);
      border-radius: 3px;
      font-family: var(--font-family-mono);
      font-size: 11px;
    }
  }
}

:deep(.ant-modal-header) {
  border-bottom: 1px solid var(--border-color);
}

:deep(.ant-modal-title) {
  font-weight: 600;
}
</style>
