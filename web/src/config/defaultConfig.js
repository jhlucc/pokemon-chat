import { MODEL_CATALOG } from './models'

const DEFAULT_PROVIDER = 'siliconflow'

export const DEFAULT_CONFIG = {
  // --- backend status (derived) ---
  backend: {
    online: false,
    ready: false,
    last_error: null,
    checks: null
  },

  // --- UI feature visibility (frontend-controlled) ---
  // These control what the frontend *shows*. Backend capability flags below only control
  // whether actions call the real backend or are disabled / mocked.
  ui: {
    show_knowledge_base: true,
    show_knowledge_graph: true,
    show_web_search: true,
    show_mcp: true,
    show_tools: true,
    show_agents: true,
    show_map: true
  },

  // --- model selection (frontend-controlled; sent with requests) ---
  model_provider: DEFAULT_PROVIDER,
  model_name: MODEL_CATALOG[DEFAULT_PROVIDER]?.default || 'Qwen/Qwen2.5-7B-Instruct',

  // --- feature flags (backend capability; UI uses them to enable/disable buttons) ---
  enable_knowledge_base: false,
  enable_knowledge_graph: false,
  enable_web_search: false,
  enable_mcp: false,
  enable_reranker: true,

  // --- embedding / reranker (display only; backend owns the real config) ---
  embed_model: 'BAAI/bge-m3',
  reranker: 'BAAI/bge-reranker-v2-m3',

  // --- UI helpers (frontend only) ---
  model_names: MODEL_CATALOG,
  custom_models: [] // [{ custom_id, name, api_base, api_key? }]
}

export const LOCAL_CONFIG_KEY = 'pokemon_chat_config_v1'
