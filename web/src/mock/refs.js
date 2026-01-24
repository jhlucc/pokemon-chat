import demoGraph from '@/assets/mock/graph.sample.json';
import { randomId } from '@/utils/id';
import { getMockState } from './state';

function cloneGraph() {
  return {
    nodes: (demoGraph?.nodes || []).map((n) => ({ id: String(n.id), name: String(n.name) })),
    edges: (demoGraph?.edges || []).map((e) => ({
      source_id: String(e.source_id),
      target_id: String(e.target_id),
      type: String(e.type),
    })),
  };
}

function buildKbResults(dbId, query) {
  const q = String(query || '').trim();
  const state = getMockState();
  const db = (state?.databases || []).find((d) => String(d.db_id) === String(dbId)) || (state?.databases || [])[0];
  const files = db?.files ? Object.values(db.files) : [];
  if (files.length === 0) return [];

  const fileScore = (f) => {
    if (!q) return 0;
    let s = 0;
    const filename = String(f?.filename || '');
    if (filename.includes(q)) s += 3;
    const lines = Array.isArray(f?.lines) ? f.lines : [];
    if (lines.some((l) => String(l?.text || '').includes(q))) s += 5;
    // Heuristic: prefer smaller file names (more "specific" in demo data)
    s += Math.max(0, 2 - Math.floor(filename.length / 20));
    return s;
  };

  const ordered = files
    .map((f) => ({ f, score: fileScore(f) }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.f);

  const pickedFiles = ordered.slice(0, Math.min(2, ordered.length));
  const baseScore = q ? 0.86 : 0.72;
  const results = [];

  pickedFiles.forEach((file, fileIdx) => {
    const lines = Array.isArray(file?.lines) ? file.lines : [];
    const matchLines = q ? lines.filter((l) => String(l?.text || '').includes(q)) : [];
    const source = matchLines.length > 0 ? matchLines : lines;
    source.slice(0, 2).forEach((l, idx) => {
      const scoreDecay = fileIdx * 0.08 + idx * 0.07;
      results.push({
        id: randomId(8),
        distance: Math.max(0, Math.min(1, baseScore - scoreDecay)),
        rerank_score: Math.max(0, Math.min(1, baseScore - scoreDecay * 0.8)),
        entity: { text: String(l?.text || '').slice(0, 400) },
        file: {
          filename: String(file?.filename || 'demo.txt'),
          type: String(file?.type || 'txt'),
          created_at: Number(file?.created_at) || Math.floor(Date.now() / 1000),
        },
      });
    });
  });

  return results;
}

function buildWebResults(query) {
  const q = String(query || '').trim() || '宝可梦';
  return [
    {
      url: 'https://example.com/pokemon',
      title: `（离线演示）${q} - 结果 1`,
      content: `这是离线模式下的示例网页搜索摘要，用于演示 UI。关键词：${q}`,
      score: 0.82,
    },
    {
      url: 'https://example.com/wiki',
      title: `（离线演示）${q} - 结果 2`,
      content: `你可以在设置页关闭离线模式并启动后端，获得真实联网搜索结果。`,
      score: 0.71,
    },
  ];
}

function buildGraphResults(query) {
  const q = String(query || '').trim();
  const g = cloneGraph();
  if (!q) return g;

  // If query matches a node name, return its 1-hop neighborhood; else return full demo graph.
  const matched = g.nodes.filter((n) => n.name.includes(q));
  if (matched.length === 0) return g;
  const matchIds = new Set(matched.map((n) => n.id));
  const neighborIds = new Set(matchIds);
  g.edges.forEach((e) => {
    if (matchIds.has(e.source_id)) neighborIds.add(e.target_id);
    if (matchIds.has(e.target_id)) neighborIds.add(e.source_id);
  });
  return {
    nodes: g.nodes.filter((n) => neighborIds.has(n.id)),
    edges: g.edges.filter((e) => neighborIds.has(e.source_id) && neighborIds.has(e.target_id)),
  };
}

export function mockRefsForRequest({ meta, query } = {}) {
  const m = meta && typeof meta === 'object' ? meta : {};
  const refs = {};

  // Knowledge base: use db_id presence as signal.
  if (m.db_id) {
    const results = buildKbResults(m.db_id, query);
    if (Array.isArray(results) && results.length > 0) {
      refs.knowledge_base = { results };
    }
  }

  // Web search: ChatComponent uses meta.use_web
  if (m.use_web) {
    const results = buildWebResults(query);
    if (Array.isArray(results) && results.length > 0) {
      refs.web_search = { results };
    }
  }

  // Graph: meta.use_graph
  if (m.use_graph) {
    const results = buildGraphResults(query);
    if (results?.nodes?.length > 0) {
      refs.graph_base = { results };
    }
  }

  return refs;
}
