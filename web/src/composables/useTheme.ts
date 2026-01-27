import { onMounted, ref } from 'vue'

const THEME_STORAGE_KEY = 'pokemon-chat-theme'

export type Theme = 'light' | 'dark' | 'system'

const theme = ref<Theme>('system')
const resolvedTheme = ref<'light' | 'dark'>('light')

/**
 * Apply theme to document root
 */
function applyTheme(newTheme: 'light' | 'dark') {
  resolvedTheme.value = newTheme
  if (newTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark')
  } else {
    document.documentElement.removeAttribute('data-theme')
  }
}

/**
 * Get system preference
 */
function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

/**
 * Resolve theme based on setting
 */
function resolveTheme(setting: Theme): 'light' | 'dark' {
  if (setting === 'system') {
    return getSystemTheme()
  }
  return setting
}

/**
 * Theme composable
 */
export function useTheme() {
  const setTheme = (newTheme: Theme) => {
    theme.value = newTheme
    localStorage.setItem(THEME_STORAGE_KEY, newTheme)
    applyTheme(resolveTheme(newTheme))
  }

  const toggleTheme = () => {
    const current = resolvedTheme.value
    setTheme(current === 'light' ? 'dark' : 'light')
  }

  const initTheme = () => {
    // Read from localStorage
    const stored = localStorage.getItem(THEME_STORAGE_KEY) as Theme | null
    if (stored && ['light', 'dark', 'system'].includes(stored)) {
      theme.value = stored
    }

    // Apply theme
    applyTheme(resolveTheme(theme.value))

    // Watch for system changes
    if (typeof window !== 'undefined') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      mediaQuery.addEventListener('change', (e) => {
        if (theme.value === 'system') {
          applyTheme(e.matches ? 'dark' : 'light')
        }
      })
    }
  }

  onMounted(() => {
    initTheme()
  })

  return {
    theme,
    resolvedTheme,
    setTheme,
    toggleTheme,
    initTheme
  }
}

/**
 * Theme options for UI
 */
export const themeOptions: Array<{ value: Theme; label: string; icon: string }> = [
  { value: 'light', label: '浅色模式', icon: '☀️' },
  { value: 'dark', label: '深色模式', icon: '🌙' },
  { value: 'system', label: '跟随系统', icon: '💻' }
]
