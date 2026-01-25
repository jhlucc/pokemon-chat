import { getOfflineMode } from '@/utils/offlineMode'
import { readJson, writeJson } from '@/utils/storage'
import { randomId } from '@/utils/id'
import { chunkPlainText } from '@/utils/chunking'

import { MOCK_TOOLS } from '@/mock/tools'
import { mockAgentListResponse } from '@/mock/agents'
import { mockConfigResponse, mockReadyzResponse } from '@/mock/config'
import {
  mockAddByChunks,
  mockCreateDatabase,
  mockDeleteDatabase,
  mockDeleteFile,
  mockFileToChunk,
  mockGetDatabaseInfo,
  mockGetDocument,
  mockListDatabases
} from '@/mock/database'
import { mockCoordsResponse } from '@/mock/coords'
import { mockLogResponse } from '@/mock/log'
import { mockCreateToken, mockDeleteToken, mockListTokens, mockVerifyToken } from '@/mock/tokens'
import { mockGraphInfo, mockGraphSampleNodes, mockGraphSearch } from '@/mock/graph'
import { mockRefsForRequest } from '@/mock/refs'

const MOCK_CONFIG_STORAGE_KEY = 'pokemon_chat_mock_config_v1'

function normalizeMethod(method) {
  return String(method || 'GET').toUpperCase()
}

function normalizePath(path) {
  if (!path) return '/'
  // apiFetch/apiRequest pass in "/xxx" (without /api); keep it.
  return String(path).replace(/\?.*$/, '')
}

function jsonResponse(data, { status = 200, headers = {} } = {}) {
  const body = JSON.stringify(data ?? null)
  return new Response(body, {
    status,
    headers: {
      'content-type': 'application/json',
      ...headers
    }
  })
}

function ndjsonResponse(objs, { status = 200 } = {}) {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      try {
        for (const obj of objs) {
          controller.enqueue(encoder.encode(`${JSON.stringify(obj)}\n`))
        }
      } finally {
        controller.close()
      }
    }
  })

  return new Response(stream, {
    status,
    headers: {
      'content-type': 'application/x-ndjson; charset=utf-8',
      'cache-control': 'no-cache'
    }
  })
}

function getMockConfig() {
  const stored = readJson(MOCK_CONFIG_STORAGE_KEY, null)
  if (stored && typeof stored === 'object') return stored
  const init = mockConfigResponse()
  writeJson(MOCK_CONFIG_STORAGE_KEY, init)
  return init
}

function patchMockConfig(patch) {
  const cur = getMockConfig()
  const next = { ...cur, ...(patch || {}) }
  writeJson(MOCK_CONFIG_STORAGE_KEY, next)
  return next
}

// chunkPlainText comes from utils/chunking

function mockChatStream(body = {}) {
  const q = String(body?.query || '').trim()
  const meta = body?.meta && typeof body.meta === 'object' ? body.meta : {}
  const reply = q
    ? `离线模式：你问的是「${q}」。当前未连接后端，所以这里只返回占位回答。`
    : '离线模式：未收到问题。'

  const refs = mockRefsForRequest({ meta, query: q })
  const chunks = []
  const step = 18
  for (let i = 0; i < reply.length; i += step) chunks.push(reply.slice(i, i + step))

  const hasRefs = refs && typeof refs === 'object' && Object.keys(refs).length > 0
  return ndjsonResponse([
    { status: 'init', response: '' },
    ...chunks.map((part) => ({ status: 'generating', response: part })),
    {
      status: 'finished',
      response: '',
      ...(hasRefs ? { refs } : {}),
      meta: { server_model_name: 'mock/offline' }
    }
  ])
}

function mockAgentStream({ agent_name, query, meta } = {}) {
  const q = String(query || '').trim()
  const agent = String(agent_name || 'demo')
  const requestId = randomId(12)
  const msgId = randomId(16)

  const content = `离线模式（${agent}）：收到问题「${q || '（空）'}」，这里是 Mock 回答。`

  // Shape matches backend agent stream:
  // { request_id, response, status, meta, msg: { id, type: 'assistant' } }
  const refs = mockRefsForRequest({ meta, query: q })
  const chunks = []
  const step = 20
  for (let i = 0; i < content.length; i += step) chunks.push(content.slice(i, i + step))
  const hasRefs = refs && typeof refs === 'object' && Object.keys(refs).length > 0

  return ndjsonResponse([
    {
      request_id: requestId,
      response: '',
      status: 'init',
      meta: { server_model_name: 'mock/offline' },
      msg: { id: msgId, type: 'assistant' }
    },
    ...chunks.map((part) => ({
      request_id: requestId,
      response: part,
      status: 'loading',
      meta: { server_model_name: 'mock/offline' },
      msg: { id: msgId, type: 'assistant' }
    })),
    {
      request_id: requestId,
      response: '',
      status: 'finished',
      ...(hasRefs ? { refs } : {}),
      meta: { server_model_name: 'mock/offline' },
      msg: { id: msgId, type: 'assistant' }
    }
  ])
}

export function getMockMode() {
  return getOfflineMode()
}

/**
 * Try to resolve a mock Response for this request.
 * Returns null when no mock handler matches.
 */
export function resolveMockResponse(path, { method = 'GET', query, body } = {}) {
  const m = normalizeMethod(method)
  const p = normalizePath(path)

  // --- config / health ---
  if (m === 'GET' && p === '/config') return jsonResponse(getMockConfig())
  if (m === 'PATCH' && p === '/config') return jsonResponse(patchMockConfig(body))
  if (m === 'GET' && p === '/readyz') return jsonResponse(mockReadyzResponse())
  if (m === 'POST' && p === '/restart') return jsonResponse({ status: 'ok' })

  // --- tools ---
  if (m === 'GET' && p === '/tools/') return jsonResponse({ tools: MOCK_TOOLS })
  if (m === 'GET' && p === '/tools') return jsonResponse({ tools: MOCK_TOOLS })
  if (m === 'POST' && p === '/tools/file-chunking') {
    // Accept both backend-like payload and existing frontend payload shapes.
    const chunkSize = body?.chunk_size ?? body?.params?.chunk_size ?? body?.params?.chunkSize ?? 500
    const chunkOverlap =
      body?.chunk_overlap ?? body?.params?.chunk_overlap ?? body?.params?.chunkOverlap ?? 20
    const input = body?.text ?? body?.file ?? ''
    const nodes = chunkPlainText(input, { chunkSize, chunkOverlap }).map((c) => ({
      text: c.text,
      meta: c.meta
    }))
    // Be tolerant: return both "nodes" and "chunks".
    return jsonResponse({ nodes, chunks: nodes })
  }
  if (m === 'POST' && p === '/tools/pdf2txt') {
    return jsonResponse({
      text: '离线演示：PDF 转文本需要后端能力。你可以在设置页关闭离线模式并启动后端后再试。'
    })
  }

  // --- chat ---
  if (m === 'POST' && p === '/chat/') return mockChatStream(body)
  if (m === 'POST' && p === '/chat') return mockChatStream(body)
  if (m === 'POST' && (p === '/chat/asr' || p === '/chat/asr/')) {
    return jsonResponse({ text: '（离线演示）语音识别暂不可用，这里返回占位文本。' })
  }
  if (m === 'POST' && p === '/chat/call') {
    const q = String(body?.query || '').trim()
    return jsonResponse({ response: q ? `（离线）${q.slice(0, 12)}` : '（离线）新对话' })
  }
  if (m === 'POST' && p === '/chat/call_lite') {
    const q = String(body?.query || '').trim()
    return jsonResponse({ response: q ? `离线 Lite：${q}` : '离线 Lite：OK' })
  }

  // --- agents ---
  if (m === 'GET' && p === '/chat/agent') return jsonResponse(mockAgentListResponse())
  if (m === 'POST' && p.startsWith('/chat/agent/')) {
    const agentName = p.split('/').pop()
    return mockAgentStream({ agent_name: agentName, query: body?.query, meta: body?.meta })
  }

  // --- knowledge base / data ---
  if (m === 'GET' && p === '/data/') return jsonResponse(mockListDatabases())
  if (m === 'GET' && p === '/data') return jsonResponse(mockListDatabases())
  if (m === 'POST' && p === '/data/') return jsonResponse(mockCreateDatabase(body))
  if (m === 'POST' && p === '/data') return jsonResponse(mockCreateDatabase(body))
  if (m === 'DELETE' && (p === '/data/' || p === '/data')) {
    const dbId = query?.db_id || body?.db_id
    return jsonResponse(mockDeleteDatabase(dbId))
  }
  if (m === 'GET' && p === '/data/info') {
    const dbId = query?.db_id
    return jsonResponse(mockGetDatabaseInfo(dbId))
  }
  if (m === 'GET' && p === '/data/document') {
    const dbId = query?.db_id
    const fileId = query?.file_id
    return jsonResponse(mockGetDocument(dbId, fileId))
  }
  if (m === 'DELETE' && p === '/data/document') return jsonResponse(mockDeleteFile(body))
  if (m === 'POST' && p === '/data/file-to-chunk') return jsonResponse(mockFileToChunk(body))
  if (m === 'POST' && p === '/data/add-by-chunks') return jsonResponse(mockAddByChunks(body))

  // --- knowledge graph ---
  if (m === 'GET' && p === '/data/graph') return jsonResponse(mockGraphInfo())
  if (m === 'GET' && p === '/data/graph/nodes')
    return jsonResponse(mockGraphSampleNodes({ num: query?.num }))
  if (m === 'GET' && p === '/data/graph/node')
    return jsonResponse(mockGraphSearch({ entity_name: query?.entity_name }))

  // --- coords ---
  if (m === 'GET' && p === '/mcp/coords') return jsonResponse(mockCoordsResponse(query?.place))

  // --- admin tokens ---
  if (p === '/admin/tokens' && m === 'GET') return jsonResponse(mockListTokens(query?.agent_id))
  if (p === '/admin/tokens' && m === 'POST') return jsonResponse(mockCreateToken(body))
  if (p.startsWith('/admin/tokens/') && m === 'DELETE') {
    const tokenId = p.split('/').pop()
    return jsonResponse(mockDeleteToken({ tokenId }))
  }
  if (p === '/admin/verify_token' && m === 'POST') {
    const r = mockVerifyToken(body)
    if (r.ok) return jsonResponse({ ok: true })
    return jsonResponse({ detail: r.detail || 'invalid token' }, { status: 401 })
  }

  // --- debug log ---
  if (p === '/log' && m === 'GET') return jsonResponse(mockLogResponse(query))

  return null
}
