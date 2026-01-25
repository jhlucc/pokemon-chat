import { randomId } from '@/utils/id'
import { chunkPlainText } from '@/utils/chunking'
import { getMockState, setMockState } from './state'

function nowSeconds() {
  return Math.floor(Date.now() / 1000)
}

function guessFileType(filename = '') {
  const lower = String(filename).toLowerCase()
  if (lower.endsWith('.pdf')) return 'pdf'
  if (lower.endsWith('.md')) return 'md'
  if (lower.endsWith('.doc') || lower.endsWith('.docx')) return 'docx'
  if (lower.endsWith('.txt')) return 'txt'
  return 'txt'
}

// chunkPlainText comes from utils/chunking

export function mockListDatabases() {
  const state = getMockState()
  const list = (state.databases || []).map((db) => ({
    db_id: db.db_id,
    name: db.name,
    description: db.description,
    embed_model: db.embed_model,
    dimension: db.dimension,
    files: db.files || {}
  }))
  return { databases: list }
}

export function mockGetDatabaseInfo(db_id) {
  const state = getMockState()
  const db = (state.databases || []).find((d) => String(d.db_id) === String(db_id))
  if (!db) return { status: 'failed', message: 'Database not found' }
  return {
    db_id: db.db_id,
    name: db.name,
    description: db.description,
    embed_model: db.embed_model,
    dimension: db.dimension,
    files: db.files || {}
  }
}

export function mockGetDocument(db_id, file_id) {
  const db = mockGetDatabaseInfo(db_id)
  if (db?.status === 'failed') return db
  const file = db.files?.[file_id]
  if (!file) return { status: 'failed', message: 'File not found', lines: [] }
  return { lines: Array.isArray(file.lines) ? file.lines : [] }
}

export function mockCreateDatabase({ database_name, description = '', dimension = null } = {}) {
  const name = String(database_name || '').trim()
  if (!name) return { status: 'failed', message: '数据库名称不能为空' }

  const db_id = `local_${randomId(8)}`
  const next = getMockState()
  next.databases = Array.isArray(next.databases) ? next.databases : []
  next.databases.unshift({
    db_id,
    name,
    description,
    embed_model: 'BAAI/bge-m3',
    dimension: dimension || 1024,
    created_at: nowSeconds(),
    files: {}
  })
  setMockState(next)
  return { status: 'ok', db_id, message: '已创建（离线）' }
}

export function mockDeleteDatabase(db_id) {
  const next = getMockState()
  next.databases = (next.databases || []).filter((d) => String(d.db_id) !== String(db_id))
  setMockState(next)
  return { status: 'ok', message: '已删除（离线）' }
}

export function mockDeleteFile({ db_id, file_id } = {}) {
  const next = getMockState()
  const db = (next.databases || []).find((d) => String(d.db_id) === String(db_id))
  if (!db) return { status: 'failed', message: 'Database not found' }
  db.files = db.files || {}
  delete db.files[file_id]
  setMockState(next)
  return { status: 'ok', message: '已删除（离线）' }
}

export function mockFileToChunk({ file, chunk_size = 1000, chunk_overlap = 200 } = {}) {
  // In mock mode we don't have real filesystem access; return chunks from placeholder text.
  const filename = String(file || 'mock.txt')
    .split('/')
    .pop()
  const placeholder = `离线分块预览：${filename}\n\n你可以在设置页开启后端后体验真实分块。`
  const chunks = chunkPlainText(placeholder, { chunkSize: chunk_size, chunkOverlap: chunk_overlap })
  return { chunks }
}

export function mockAddByChunks({ db_id, file_chunks } = {}) {
  const next = getMockState()
  const db = (next.databases || []).find((d) => String(d.db_id) === String(db_id))
  if (!db) return { status: 'failed', message: 'Database not found' }

  db.files = db.files || {}
  const fc = file_chunks && typeof file_chunks === 'object' ? file_chunks : {}

  Object.values(fc).forEach((file) => {
    const fileId = file?.file_id || file?.filename || `file_${randomId(8)}.txt`
    const filename = file?.filename || fileId
    const nodes = Array.isArray(file?.nodes) ? file.nodes : []
    db.files[fileId] = {
      file_id: fileId,
      filename,
      type: guessFileType(filename),
      status: 'done',
      created_at: nowSeconds(),
      lines: nodes.map((n) => ({ id: randomId(8), text: n?.text || '' }))
    }
  })

  setMockState(next)
  return { status: 'ok', message: '已写入（离线）' }
}
