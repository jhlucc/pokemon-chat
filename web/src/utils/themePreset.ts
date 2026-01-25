export const THEME_PRESETS = {
  ocean: { label: 'Ocean', primary: '#2C86A8', primaryLight: '#8CC6E1' },
  blue: { label: 'Blue', primary: '#1677FF', primaryLight: '#69B1FF' },
  violet: { label: 'Violet', primary: '#722ED1', primaryLight: '#B37FEB' },
  green: { label: 'Green', primary: '#2F9E44', primaryLight: '#69DB7C' }
} as const

export type ThemePresetKey = keyof typeof THEME_PRESETS

const THEME_PRESET_STORAGE_KEY = 'pokemon_chat_theme_preset_v1'
const THEME_PRESET_EVENT = 'ui-theme-preset-changed'

export function normalizeThemePreset(value: unknown): ThemePresetKey {
  if (value && typeof value === 'string' && value in THEME_PRESETS) {
    return value as ThemePresetKey
  }
  return 'ocean'
}

export function getThemePreset(): ThemePresetKey {
  try {
    return normalizeThemePreset(localStorage.getItem(THEME_PRESET_STORAGE_KEY))
  } catch {
    return 'ocean'
  }
}

export function applyThemePreset(preset: ThemePresetKey): void {
  const p = THEME_PRESETS[preset]
  document.documentElement.style.setProperty('--primary-color', p.primary)
  document.documentElement.style.setProperty('--primary-light-color', p.primaryLight)
}

export function setThemePreset(preset: ThemePresetKey): ThemePresetKey {
  const key = normalizeThemePreset(preset)
  try {
    localStorage.setItem(THEME_PRESET_STORAGE_KEY, key)
  } catch {
    // ignore quota / private mode errors
  }
  applyThemePreset(key)
  try {
    window.dispatchEvent(new Event(THEME_PRESET_EVENT))
  } catch {
    // ignore in non-browser env
  }
  return key
}

export function initThemePreset(): ThemePresetKey {
  const key = getThemePreset()
  applyThemePreset(key)
  return key
}

export function getThemePresetToken(preset: ThemePresetKey): { colorPrimary: string } {
  return { colorPrimary: THEME_PRESETS[preset].primary }
}

