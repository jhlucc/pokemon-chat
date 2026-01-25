export function randomId(length = 16) {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
  let id = ''
  for (let i = 0; i < length; i += 1) {
    id += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return id
}

export function createRequestId() {
  try {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  } catch {
    // ignore
  }
  // Fallback: good enough for correlating logs in a single deployment.
  return randomId(24)
}
