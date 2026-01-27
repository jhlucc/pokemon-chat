// Centralized provider icon resolver.
// Keep filenames in `src/assets/providers/*.png` and map provider keys -> icon filenames here.

// Vite glob: use absolute `/src/...` (alias `@` may not work in glob patterns).
const providerIcons = import.meta.glob('/src/assets/providers/*.png', {
  eager: true,
  import: 'default'
})

// Provider key -> icon filename (without extension)
const PROVIDER_ICON_ALIAS = {
  zhipu: 'zhipuai',
  togetherai: 'together.ai'
}

// Provider key -> friendly display name
const PROVIDER_NAME_MAP = {
  siliconflow: '硅基流动',
  openai: 'OpenAI',
  deepseek: 'DeepSeek',
  zhipuai: '智谱 AI',
  zhipu: '智谱 AI',
  dashscope: '通义千问',
  ollama: 'Ollama',
  custom: '自定义',
  openrouter: 'OpenRouter',
  togetherai: 'Together AI',
  lingyiwanwu: '零一万物',
  ark: '火山方舟',
  qianfan: '百度千帆'
}

// Model name -> friendly display name (common models)
const MODEL_NAME_MAP = {
  'Qwen/Qwen2.5-7B-Instruct': 'Qwen 2.5 (7B)',
  'Qwen/Qwen2.5-14B-Instruct': 'Qwen 2.5 (14B)',
  'Qwen/Qwen2.5-32B-Instruct': 'Qwen 2.5 (32B)',
  'Qwen/Qwen2.5-72B-Instruct': 'Qwen 2.5 (72B)',
  'Qwen/Qwen2-7B-Instruct': 'Qwen 2 (7B)',
  'Qwen/Qwen2-72B-Instruct': 'Qwen 2 (72B)',
  'deepseek-chat': 'DeepSeek Chat',
  'deepseek-coder': 'DeepSeek Coder',
  'gpt-4o': 'GPT-4o',
  'gpt-4o-mini': 'GPT-4o Mini',
  'gpt-4-turbo': 'GPT-4 Turbo',
  'gpt-3.5-turbo': 'GPT-3.5 Turbo',
  'glm-4': 'GLM-4',
  'glm-4-flash': 'GLM-4 Flash',
  'qwen-turbo': '通义千问 Turbo',
  'qwen-plus': '通义千问 Plus',
  'qwen-max': '通义千问 Max'
}

export function getProviderIcon(provider) {
  const p = PROVIDER_ICON_ALIAS[provider] || provider
  return (
    providerIcons[`/src/assets/providers/${p}.png`] || providerIcons['/src/assets/providers/default.png']
  )
}

/**
 * Get friendly provider display name
 */
export function getProviderDisplayName(provider) {
  if (!provider) return '-'
  return PROVIDER_NAME_MAP[provider] || provider
}

/**
 * Get friendly model display name
 */
export function getModelDisplayName(modelName) {
  if (!modelName) return '-'
  return MODEL_NAME_MAP[modelName] || modelName
}

/**
 * Get combined friendly display for provider/model
 */
export function getModelLabel(provider, modelName) {
  const providerName = getProviderDisplayName(provider)
  const modelLabel = getModelDisplayName(modelName)
  return `${providerName} / ${modelLabel}`
}

