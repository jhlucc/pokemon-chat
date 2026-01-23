import { MODEL_CATALOG } from './models';

const DEFAULT_PROVIDER = 'siliconflow';

export const DEFAULT_CONFIG = {
  // --- backend status (derived) ---
  backend: {
    online: false,
    ready: false,
    last_error: null,
    checks: null,
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
  custom_models: [], // [{ custom_id, name, api_base, api_key? }]
};

export const LOCAL_CONFIG_KEY = 'pokemon_chat_config_v1';

