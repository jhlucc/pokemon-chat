export type UiDensity = 'comfortable' | 'compact'

const UI_DENSITY_STORAGE_KEY = 'pokemon_chat_ui_density_v1'
const UI_DENSITY_EVENT = 'ui-density-changed'

export function normalizeUiDensity(value: unknown): UiDensity {
  return value === 'compact' ? 'compact' : 'comfortable'
}

export function getUiDensity(): UiDensity {
  try {
    return normalizeUiDensity(localStorage.getItem(UI_DENSITY_STORAGE_KEY))
  } catch {
    return 'comfortable'
  }
}

export function applyUiDensity(density: UiDensity): void {
  document.documentElement.setAttribute('data-density', density)
}

export function setUiDensity(density: UiDensity): UiDensity {
  const d = normalizeUiDensity(density)
  try {
    localStorage.setItem(UI_DENSITY_STORAGE_KEY, d)
  } catch {
    // ignore quota / private mode errors
  }
  applyUiDensity(d)
  try {
    window.dispatchEvent(new Event(UI_DENSITY_EVENT))
  } catch {
    // ignore in non-browser env
  }
  return d
}

export function initUiDensity(): UiDensity {
  const d = getUiDensity()
  applyUiDensity(d)
  return d
}

