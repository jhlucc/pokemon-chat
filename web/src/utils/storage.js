/**
 * Safe JSON.parse with fallback.
 * @template T
 * @param {string | null | undefined} raw
 * @param {T} [fallback]
 * @returns {T}
 */
export function safeJsonParse(raw, fallback = null) {
  if (raw === null || raw === undefined || raw === '') return fallback
  try {
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

/**
 * Read and parse a localStorage value.
 * @template T
 * @param {string} key
 * @param {T} [fallback]
 * @returns {T}
 */
export function readJson(key, fallback = null) {
  try {
    return safeJsonParse(localStorage.getItem(key), fallback)
  } catch {
    return fallback
  }
}

/**
 * Write a value to localStorage as JSON.
 * @param {string} key
 * @param {unknown} value
 * @returns {boolean}
 */
export function writeJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
    return true
  } catch {
    return false
  }
}

/**
 * Remove a key from localStorage.
 * @param {string} key
 * @returns {boolean}
 */
export function removeKey(key) {
  try {
    localStorage.removeItem(key)
    return true
  } catch {
    return false
  }
}
