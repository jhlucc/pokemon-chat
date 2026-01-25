export function safeJsonParse(raw, fallback = null) {
  if (raw === null || raw === undefined || raw === '') return fallback
  try {
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

export function readJson(key, fallback = null) {
  try {
    return safeJsonParse(localStorage.getItem(key), fallback)
  } catch {
    return fallback
  }
}

export function writeJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
    return true
  } catch {
    return false
  }
}

export function removeKey(key) {
  try {
    localStorage.removeItem(key)
    return true
  } catch {
    return false
  }
}
