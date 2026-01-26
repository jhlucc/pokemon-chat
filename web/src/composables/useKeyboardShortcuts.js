import { onMounted, onUnmounted, ref } from 'vue'

/**
 * Composable for handling keyboard shortcuts
 *
 * Usage:
 * const { registerShortcut, unregisterAll } = useKeyboardShortcuts()
 *
 * registerShortcut({
 *   key: 'n',
 *   ctrl: true,
 *   handler: () => createNewConversation()
 * })
 */
export function useKeyboardShortcuts() {
  const shortcuts = ref([])

  const handleKeyDown = (event) => {
    // Ignore if user is typing in an input field
    const target = event.target
    const isInput =
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable

    for (const shortcut of shortcuts.value) {
      const ctrlMatch = shortcut.ctrl ? (event.ctrlKey || event.metaKey) : true
      const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey
      const altMatch = shortcut.alt ? event.altKey : !event.altKey
      const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase()

      // Skip input-sensitive shortcuts when typing
      if (isInput && !shortcut.allowInInput) {
        continue
      }

      if (ctrlMatch && shiftMatch && altMatch && keyMatch) {
        event.preventDefault()
        shortcut.handler(event)
        return
      }
    }
  }

  const registerShortcut = (config) => {
    const shortcut = {
      key: config.key,
      ctrl: config.ctrl || false,
      shift: config.shift || false,
      alt: config.alt || false,
      handler: config.handler,
      allowInInput: config.allowInInput || false,
      description: config.description || ''
    }
    shortcuts.value.push(shortcut)
    return shortcut
  }

  const unregisterShortcut = (shortcut) => {
    const index = shortcuts.value.indexOf(shortcut)
    if (index > -1) {
      shortcuts.value.splice(index, 1)
    }
  }

  const unregisterAll = () => {
    shortcuts.value = []
  }

  const getShortcuts = () => {
    return shortcuts.value.map((s) => ({
      key: s.key,
      ctrl: s.ctrl,
      shift: s.shift,
      alt: s.alt,
      description: s.description
    }))
  }

  const formatShortcut = (shortcut) => {
    const parts = []
    if (shortcut.ctrl) parts.push('⌘/Ctrl')
    if (shortcut.shift) parts.push('Shift')
    if (shortcut.alt) parts.push('Alt')
    parts.push(shortcut.key.toUpperCase())
    return parts.join(' + ')
  }

  onMounted(() => {
    window.addEventListener('keydown', handleKeyDown)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeyDown)
  })

  return {
    registerShortcut,
    unregisterShortcut,
    unregisterAll,
    getShortcuts,
    formatShortcut
  }
}

/**
 * Common keyboard shortcuts for chat application
 */
export const CHAT_SHORTCUTS = {
  NEW_CONVERSATION: { key: 'n', ctrl: true, description: '新建对话' },
  TOGGLE_SIDEBAR: { key: 'b', ctrl: true, description: '切换侧边栏' },
  FOCUS_INPUT: { key: '/', ctrl: false, description: '聚焦输入框' },
  SEND_MESSAGE: { key: 'Enter', ctrl: false, description: '发送消息' },
  CANCEL_GENERATION: { key: 'Escape', ctrl: false, description: '停止生成', allowInInput: true }
}
