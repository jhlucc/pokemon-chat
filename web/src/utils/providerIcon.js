// Centralized provider icon resolver.
// Keep filenames in `src/assets/providers/*.png` and map provider keys -> icon filenames here.

// Vite glob: use absolute `/src/...` (alias `@` may not work in glob patterns).
/** @type {Record<string, string>} */
const providerIcons = import.meta.glob('/src/assets/providers/*.png', {
  eager: true,
  import: 'default'
})

// Provider key -> icon filename (without extension)
const PROVIDER_ICON_ALIAS = {
  zhipu: 'zhipuai',
  togetherai: 'together.ai',
  anthropic: 'claude',
  gemini: 'google',
  '01ai': 'lingyiwanwu',
  yi: 'lingyiwanwu',
  bytedance: 'doubao'
}

// Provider key -> friendly display name
const PROVIDER_NAME_MAP = {
  siliconflow: '硅基流动',
  openai: 'OpenAI',
  deepseek: 'DeepSeek',
  zhipuai: '智谱 AI',
  zhipu: '智谱 AI',
  dashscope: '通义千问',
  qwen: '通义千问',
  ollama: 'Ollama',
  custom: '自定义',
  openrouter: 'OpenRouter',
  togetherai: 'Together AI',
  lingyiwanwu: '零一万物',
  '01ai': '零一万物',
  yi: '零一万物',
  ark: '火山方舟',
  qianfan: '百度千帆',
  anthropic: 'Anthropic',
  claude: 'Claude',
  google: 'Google',
  gemini: 'Gemini',
  mistral: 'Mistral',
  baichuan: '百川智能',
  doubao: '豆包',
  bytedance: '字节跳动',
  kimi: 'Kimi',
  moonshot: 'Moonshot',
  minimax: 'MiniMax',
  stepfun: '阶跃星辰',
  sensetime: '商汤',
  xunfei: '讯飞星火',
  ernie: '文心一言',
  wenxin: '文心一言'
}

// Model name -> friendly display name (common models)
// Updated: 2026-01
const MODEL_NAME_MAP = {
  // OpenAI GPT 系列 (最新: GPT-5.2)
  'gpt-5.2': 'GPT-5.2',
  'gpt-5.1': 'GPT-5.1',
  'gpt-5.1-codex-max': 'GPT-5.1 Codex Max',
  'gpt-5': 'GPT-5',
  'gpt-4o': 'GPT-4o',
  'gpt-4o-mini': 'GPT-4o Mini',
  'gpt-4-turbo': 'GPT-4 Turbo',
  'o4-mini': 'o4 Mini',
  'o3': 'o3',
  'o3-mini': 'o3 Mini',
  'o1': 'o1',
  'o1-mini': 'o1 Mini',

  // Claude 系列 (最新: Claude 4.5)
  'claude-opus-4.5': 'Claude Opus 4.5',
  'claude-sonnet-4.5': 'Claude Sonnet 4.5',
  'claude-4.5-opus': 'Claude 4.5 Opus',
  'claude-4.5-sonnet': 'Claude 4.5 Sonnet',
  'claude-4-opus': 'Claude 4 Opus',
  'claude-4-sonnet': 'Claude 4 Sonnet',
  'claude-3.5-sonnet': 'Claude 3.5 Sonnet',
  'claude-3.5-haiku': 'Claude 3.5 Haiku',
  'claude-3-opus': 'Claude 3 Opus',

  // Google Gemini 系列 (最新: Gemini 3)
  'gemini-3-pro': 'Gemini 3 Pro',
  'gemini-3-pro-preview': 'Gemini 3 Pro Preview',
  'gemini-3-flash': 'Gemini 3 Flash',
  'gemini-2.5-pro': 'Gemini 2.5 Pro',
  'gemini-2.5-flash': 'Gemini 2.5 Flash',
  'gemini-2.0-pro': 'Gemini 2.0 Pro',
  'gemini-2.0-flash': 'Gemini 2.0 Flash',
  'gemini-1.5-pro': 'Gemini 1.5 Pro',

  // 智谱 GLM 系列 (最新: GLM-4.7)
  'glm-4.7': 'GLM-4.7',
  'glm-4.6': 'GLM-4.6',
  'glm-4-plus': 'GLM-4 Plus',
  'glm-4-flash': 'GLM-4 Flash',
  'glm-4-air': 'GLM-4 Air',
  'glm-4-airx': 'GLM-4 AirX',
  'glm-4-long': 'GLM-4 Long',
  'glm-4v-plus': 'GLM-4V Plus',
  'glm-4': 'GLM-4',

  // DeepSeek 系列 (最新: V3.2)
  'deepseek-v3.2': 'DeepSeek V3.2',
  'deepseek-v3': 'DeepSeek V3',
  'DeepSeek-V3.2': 'DeepSeek V3.2',
  'DeepSeek-V3': 'DeepSeek V3',
  'DeepSeek-R1': 'DeepSeek R1',
  'deepseek-ai/DeepSeek-V3': 'DeepSeek V3',
  'deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
  'deepseek-chat': 'DeepSeek Chat',
  'deepseek-coder': 'DeepSeek Coder',
  'deepseek-reasoner': 'DeepSeek R1',

  // Qwen 系列 (最新: Qwen 3)
  'qwen3-max': 'Qwen3 Max',
  'qwen3-plus': 'Qwen3 Plus',
  'qwen3-turbo': 'Qwen3 Turbo',
  'Qwen/Qwen3-235B': 'Qwen3 (235B)',
  'Qwen/Qwen2.5-72B-Instruct': 'Qwen 2.5 (72B)',
  'Qwen/Qwen2.5-32B-Instruct': 'Qwen 2.5 (32B)',
  'Qwen/Qwen2.5-14B-Instruct': 'Qwen 2.5 (14B)',
  'Qwen/Qwen2.5-7B-Instruct': 'Qwen 2.5 (7B)',
  'Qwen/QwQ-32B': 'QwQ (32B)',
  'qwen-max': '通义千问 Max',
  'qwen-plus': '通义千问 Plus',
  'qwen-turbo': '通义千问 Turbo',
  'qwen-long': '通义千问 Long',

  // Kimi 系列 (最新: K2)
  'kimi-k2': 'Kimi K2',
  'kimi-k2-thinking': 'Kimi K2 Thinking',
  'moonshot-v1-128k': 'Moonshot V1 128K',
  'moonshot-v1-32k': 'Moonshot V1 32K',

  // Mistral 系列
  'mistral-large-latest': 'Mistral Large',
  'mistral-medium-latest': 'Mistral Medium',
  'mistral-small-latest': 'Mistral Small',
  'codestral-latest': 'Codestral',

  // 豆包/字节
  'doubao-pro-256k': '豆包 Pro 256K',
  'doubao-pro-32k': '豆包 Pro 32K',
  'doubao-lite-32k': '豆包 Lite 32K',

  // 百川
  'Baichuan4': '百川4',
  'Baichuan3-Turbo': '百川3 Turbo',

  // 零一万物
  'yi-lightning': 'Yi Lightning',
  'yi-large': 'Yi Large',
  'yi-medium': 'Yi Medium',

  // 文心 ERNIE
  'ernie-5.0': 'ERNIE 5.0',
  'ernie-x1': 'ERNIE X1',
  'ernie-4.5': 'ERNIE 4.5',
  'ernie-4.0-turbo': 'ERNIE 4.0 Turbo'
}

/**
 * Resolve provider key to icon path.
 * @param {string | undefined | null} provider
 * @returns {string}
 */
export function getProviderIcon(provider) {
  const p = PROVIDER_ICON_ALIAS[provider] || provider
  return (
    providerIcons[`/src/assets/providers/${p}.png`] || providerIcons['/src/assets/providers/default.png']
  )
}

/**
 * Get friendly provider display name
 * @param {string | undefined | null} provider
 * @returns {string}
 */
export function getProviderDisplayName(provider) {
  if (!provider) return '-'
  return PROVIDER_NAME_MAP[provider] || provider
}

/**
 * Get friendly model display name
 * @param {string | undefined | null} modelName
 * @returns {string}
 */
export function getModelDisplayName(modelName) {
  if (!modelName) return '-'
  return MODEL_NAME_MAP[modelName] || modelName
}

/**
 * Get combined friendly display for provider/model
 * @param {string | undefined | null} provider
 * @param {string | undefined | null} modelName
 * @returns {string}
 */
export function getModelLabel(provider, modelName) {
  const providerName = getProviderDisplayName(provider)
  const modelLabel = getModelDisplayName(modelName)
  return `${providerName} / ${modelLabel}`
}

