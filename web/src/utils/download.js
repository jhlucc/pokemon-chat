function safeFilename(name) {
  const s = String(name || '').trim() || 'download'
  // Replace characters that are problematic on common filesystems.
  return s.replace(/[\\/:*?"<>|]+/g, '-')
}

export function downloadBlob(filename, blob) {
  try {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = safeFilename(filename)
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    return true
  } catch {
    return false
  }
}

export function downloadText(filename, text, { mime = 'text/plain;charset=utf-8' } = {}) {
  const blob = new Blob([String(text ?? '')], { type: mime })
  return downloadBlob(filename, blob)
}

export function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data ?? null, null, 2)], {
    type: 'application/json;charset=utf-8'
  })
  return downloadBlob(filename, blob)
}
