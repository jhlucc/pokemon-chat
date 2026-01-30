// Static model catalog for the frontend (works without backend).
// Keep this file lightweight: it is only used for UI selections.
// Synced with backend src/static/models.yaml

export const MODEL_CATALOG = {
  openai: {
    name: 'OpenAI',
    url: 'https://platform.openai.com/docs/models',
    default: 'gpt-3.5-turbo',
    base_url: 'https://api.openai.com/v1',
    models: ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
  },
  deepseek: {
    name: 'DeepSeek',
    url: 'https://platform.deepseek.com/api-docs/zh-cn/pricing',
    default: 'deepseek-chat',
    base_url: 'https://api.deepseek.com/v1',
    models: ['deepseek-chat', 'deepseek-reasoner']
  },
  zhipu: {
    name: '智谱AI',
    url: 'https://open.bigmodel.cn/dev/api',
    default: 'glm-4-flash',
    base_url: 'https://open.bigmodel.cn/api/paas/v4/',
    models: ['glm-4', 'glm-4-plus', 'glm-4-air', 'glm-4-flash', 'glm-z1-air']
  },
  siliconflow: {
    name: 'SiliconFlow',
    url: 'https://cloud.siliconflow.cn/models',
    default: 'Qwen/Qwen2.5-7B-Instruct',
    base_url: 'https://api.siliconflow.cn/v1',
    models: [
      'Qwen/Qwen3-235B-A22B',
      'THUDM/GLM-Z1-32B-0414',
      'THUDM/GLM-Z1-9B-0414',
      'deepseek-ai/DeepSeek-V3',
      'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
      'Qwen/Qwen2.5-72B-Instruct',
      'Qwen/Qwen2.5-7B-Instruct',
      'Qwen/QwQ-32B'
    ]
  },
  togetherai: {
    name: 'Together AI',
    url: 'https://api.together.ai/models',
    default: 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
    base_url: 'https://api.together.xyz/v1/',
    models: [
      'meta-llama/Llama-3.3-70B-Instruct-Turbo',
      'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
      'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'
    ]
  },
  Bailian: {
    name: '阿里百炼',
    url: 'https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3',
    default: 'qwen2.5-72b-instruct',
    base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    models: [
      'qwen-max-latest',
      'qwen3-235b-a22b',
      'qwen3-30b-a3b',
      'qwen2.5-72b-instruct',
      'qwen3-8b',
      'qwq-plus-latest',
      'qwen3-4b'
    ]
  },
  ark: {
    name: '豆包（Ark）',
    url: 'https://console.volcengine.com/ark/region:ark+cn-beijing/model',
    default: 'doubao-1-5-lite-32k-250115',
    base_url: 'https://ark.cn-beijing.volces.com/api/v3',
    models: [
      'doubao-1-5-pro-32k-250115',
      'doubao-1-5-lite-32k-250115',
      'deepseek-r1-250120'
    ]
  },
  lingyiwanwu: {
    name: '零一万物',
    url: 'https://platform.lingyiwanwu.com/docs#%E6%A8%A1%E5%9E%8B%E4%B8%8E%E8%AE%A1%E8%B4%B9',
    default: 'yi-lightning',
    base_url: 'https://api.lingyiwanwu.com/v1',
    models: ['yi-lightning']
  },
  openrouter: {
    name: 'OpenRouter',
    url: 'https://openrouter.ai/models',
    default: 'openai/gpt-4o',
    base_url: 'https://openrouter.ai/api/v1',
    models: [
      'openai/gpt-4o',
      'openai/gpt-4o-mini',
      'google/gemini-2.5-pro-exp-03-25:free',
      'x-ai/grok-3-beta',
      'meta-llama/llama-4-maverick',
      'meta-llama/llama-4-maverick:free',
      'anthropic/claude-3.7-sonnet',
      'anthropic/claude-3.7-sonnet:thinking'
    ]
  },
  qianfan: {
    name: '千帆 (Baidu)',
    url: 'https://qianfan.baidubce.com/',
    default: '',
    base_url: 'https://qianfan.baidubce.com/v2',
    models: []
  },
  // Keep room for user-defined OpenAI-compatible models.
  custom: {
    name: 'Custom (OpenAI-compatible)',
    url: '',
    default: '',
    base_url: '',
    models: []
  }
}
