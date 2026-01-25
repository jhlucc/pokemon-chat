export function chunkPlainText(text, { chunkSize = 1000, chunkOverlap = 200 } = {}) {
  const t = String(text || '')
  const size = Math.max(1, Number(chunkSize) || 1000)
  const overlap = Math.max(0, Math.min(size - 1, Number(chunkOverlap) || 0))

  const chunks = []
  let start = 0
  while (start < t.length) {
    const end = Math.min(t.length, start + size)
    chunks.push({ text: t.slice(start, end), meta: { start, end } })
    if (end >= t.length) break
    start = end - overlap
  }
  return chunks
}
