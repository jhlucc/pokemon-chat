import demoGraph from '@/assets/mock/graph.sample.json'

function cloneNodes(nodes) {
  return (nodes || []).map((n) => ({ id: String(n.id), name: String(n.name) }))
}

function cloneEdges(edges) {
  return (edges || []).map((e) => ({
    source_id: String(e.source_id),
    target_id: String(e.target_id),
    type: String(e.type)
  }))
}

export function mockGraphInfo() {
  return {
    status: 'ok',
    graph_name: 'demo',
    entity_count: (demoGraph?.nodes || []).length,
    relationship_count: (demoGraph?.edges || []).length
  }
}

export function mockGraphSampleNodes({ num } = {}) {
  const n = Math.max(1, Math.min(Number(num) || 100, (demoGraph?.nodes || []).length || 1))
  const nodes = cloneNodes((demoGraph?.nodes || []).slice(0, n))
  const allowed = new Set(nodes.map((x) => x.id))
  const edges = cloneEdges(
    (demoGraph?.edges || []).filter(
      (e) => allowed.has(String(e.source_id)) && allowed.has(String(e.target_id))
    )
  )
  return { result: { nodes, edges } }
}

export function mockGraphSearch({ entity_name } = {}) {
  const q = String(entity_name || '').trim()
  if (!q) return { result: { nodes: [], edges: [] } }

  const matched = (demoGraph?.nodes || []).filter((n) => String(n.name || '').includes(q))
  if (matched.length === 0) return { result: { nodes: [], edges: [] } }

  const matchIds = new Set(matched.map((n) => String(n.id)))
  const neighborIds = new Set(matchIds)
  ;(demoGraph?.edges || []).forEach((e) => {
    const s = String(e.source_id)
    const t = String(e.target_id)
    if (matchIds.has(s)) neighborIds.add(t)
    if (matchIds.has(t)) neighborIds.add(s)
  })

  const nodes = cloneNodes((demoGraph?.nodes || []).filter((n) => neighborIds.has(String(n.id))))
  const edges = cloneEdges(
    (demoGraph?.edges || []).filter(
      (e) => neighborIds.has(String(e.source_id)) && neighborIds.has(String(e.target_id))
    )
  )
  return { result: { nodes, edges } }
}
