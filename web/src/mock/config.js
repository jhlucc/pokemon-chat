export function mockConfigResponse() {
  return {
    backend: { mock: true },
    // "capabilities" - in mock mode we enable everything so UI is fully visible.
    enable_knowledge_base: true,
    enable_knowledge_graph: true,
    enable_web_search: true,
    enable_mcp: true,
    enable_reranker: true,

    embed_model: 'BAAI/bge-m3',
    reranker: 'BAAI/bge-reranker-v2-m3',

    // Models (frontend sends provider/name to backend; in mock mode this is display-only)
    model_provider: 'mock',
    model_name: 'offline',
    model_names: {
      mock: {
        name: 'Mock (Offline)',
        default: 'offline',
        models: ['offline'],
      },
    },
    custom_models: [],
  };
}

export function mockReadyzResponse() {
  return {
    status: 'ok',
    checks: [{ name: 'mock', status: 'ok' }],
  };
}
