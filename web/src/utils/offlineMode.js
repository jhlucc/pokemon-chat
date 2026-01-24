export const OFFLINE_MODE_STORAGE_KEY = 'pokemon_chat_offline_mode';

// Supported values:
// - 'auto': use real backend; if request fails (network), fall back to mocks where possible
// - 'on'  : always use mocks (frontend demo mode)
// - 'off' : always use real backend (do not fall back)
export function normalizeOfflineMode(mode) {
  if (mode === 'on' || mode === 'off') return mode;
  return 'auto';
}

export function getOfflineMode() {
  try {
    return normalizeOfflineMode(localStorage.getItem(OFFLINE_MODE_STORAGE_KEY));
  } catch {
    return 'auto';
  }
}

export function setOfflineMode(mode) {
  const m = normalizeOfflineMode(mode);
  try {
    localStorage.setItem(OFFLINE_MODE_STORAGE_KEY, m);
    // Allow UI to react without polling.
    try {
      window.dispatchEvent(new CustomEvent('offline-mode-changed', { detail: { mode: m } }));
    } catch {
      // ignore
    }
  } catch {
    // ignore quota / private mode errors
  }
  return m;
}
