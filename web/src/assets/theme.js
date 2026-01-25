export const themeConfig = {
  token: {
    colorPrimary: '#2C86A8',
    colorInfo: '#191919',
    borderRadius: 12,
    fontFamily:
      "'HarmonyOS Sans SC', Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif"
  }
}

export const THEME_MODE_STORAGE_KEY = 'theme'
export const THEME_MODE_CHANGED_EVENT = 'theme-mode-changed'

export function normalizeThemeMode(mode) {
  // Migration note: older UI allowed "light". We now support only "system" | "dark".
  return mode === 'dark' ? 'dark' : 'system'
}

export function getSavedThemeMode() {
  try {
    return normalizeThemeMode(localStorage.getItem(THEME_MODE_STORAGE_KEY))
  } catch {
    return 'system'
  }
}

export function getAppliedTheme(mode = getSavedThemeMode()) {
  const m = normalizeThemeMode(mode)
  if (m === 'dark') return 'dark'
  try {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    return prefersDark ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export function applyTheme(applied /* 'light' | 'dark' */) {
  document.documentElement.setAttribute('data-theme', applied === 'dark' ? 'dark' : 'light')
}

export function setThemeMode(mode) {
  const m = normalizeThemeMode(mode)
  try {
    localStorage.setItem(THEME_MODE_STORAGE_KEY, m)
  } catch {
    // ignore quota / private mode errors
  }
  applyTheme(getAppliedTheme(m))
  try {
    window.dispatchEvent(new Event(THEME_MODE_CHANGED_EVENT))
  } catch {
    // ignore in non-browser env
  }
  return m
}

export function initTheme() {
  return setThemeMode(getSavedThemeMode())
}
