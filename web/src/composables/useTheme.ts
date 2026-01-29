import { onMounted, ref } from 'vue'
import {
  getSavedThemeMode,
  setThemeMode,
  getAppliedTheme,
  normalizeThemeMode,
  THEME_MODE_CHANGED_EVENT
} from '@/assets/theme'

export type Theme = 'system' | 'dark'

const theme = ref<Theme>(getSavedThemeMode() as Theme)
const resolvedTheme = ref<'light' | 'dark'>(getAppliedTheme(theme.value))

/**
 * Theme composable — delegates to theme.js as single source of truth
 */
export function useTheme() {
  const setTheme = (newTheme: Theme) => {
    const normalized = normalizeThemeMode(newTheme)
    theme.value = normalized as Theme
    setThemeMode(normalized)
    resolvedTheme.value = getAppliedTheme(normalized)
  }

  const toggleTheme = () => {
    setTheme(theme.value === 'dark' ? 'system' : 'dark')
  }

  const syncFromSource = () => {
    theme.value = getSavedThemeMode() as Theme
    resolvedTheme.value = getAppliedTheme(theme.value)
  }

  onMounted(() => {
    syncFromSource()

    // Listen for external theme changes (e.g. sidebar toggle)
    window.addEventListener(THEME_MODE_CHANGED_EVENT, syncFromSource)

    // Watch for system preference changes
    if (typeof window !== 'undefined') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      mediaQuery.addEventListener('change', () => {
        if (theme.value === 'system') {
          resolvedTheme.value = getAppliedTheme('system')
        }
      })
    }
  })

  return {
    theme,
    resolvedTheme,
    setTheme,
    toggleTheme
  }
}

/**
 * Theme options for UI
 */
export const themeOptions: Array<{ value: Theme; label: string; icon: string }> = [
  { value: 'system', label: '跟随系统', icon: '💻' },
  { value: 'dark', label: '深色模式', icon: '🌙' }
]
