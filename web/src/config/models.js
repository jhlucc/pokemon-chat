// Static model catalog for the frontend (works without backend).
// Keep this file lightweight: it is only used for UI selections.

export const MODEL_CATALOG = {
  siliconflow: {
    name: 'SiliconFlow',
    url: 'https://cloud.siliconflow.cn/models',
    default: 'Qwen/Qwen2.5-7B-Instruct',
    base_url: 'https://api.siliconflow.cn/v1',
    models: [
      'Qwen/Qwen2.5-7B-Instruct',
      'Qwen/Qwen2.5-72B-Instruct',
      'Qwen/QwQ-32B',
      'deepseek-ai/DeepSeek-V3',
      'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    ]
  },
  dashscope: {
    name: 'DashScope (百炼)',
    url: 'https://dashscope.aliyun.com/',
    default: 'qwen-turbo-latest',
    base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    models: ['qwen-turbo-latest', 'qwen-plus', 'qwen-max']
  },
  openai: {
    name: 'OpenAI',
    url: 'https://platform.openai.com/docs/models',
    default: 'gpt-4o-mini',
    base_url: 'https://api.openai.com/v1',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
  },
  deepseek: {
    name: 'DeepSeek',
    url: 'https://platform.deepseek.com/api-docs',
    default: 'deepseek-chat',
    base_url: 'https://api.deepseek.com/v1',
    models: ['deepseek-chat', 'deepseek-reasoner']
  },
  openrouterai: {
    name: 'OpenRouter',
    url: 'https://openrouter.ai/docs',
    default: 'openai/gpt-4o-mini',
    base_url: 'https://openrouter.ai/api/v1',
    models: ['openai/gpt-4o-mini', 'anthropic/claude-3.5-sonnet', 'google/gemini-1.5-pro']
  },
  zhipu: {
    name: '智谱AI',
    url: 'https://open.bigmodel.cn/dev/api',
    default: 'glm-4-flash',
    base_url: 'https://open.bigmodel.cn/api/paas/v4',
    models: ['glm-4', 'glm-4-plus', 'glm-4-air', 'glm-4-flash']
  },
  togetherai: {
    name: 'Together AI',
    url: 'https://docs.together.ai/docs/quickstart',
    default: 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    base_url: 'https://api.together.xyz/v1',
    models: ['meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo']
  },
  ark: {
    name: 'ARK (Volcengine)',
    url: '',
    default: '',
    // OpenAI-compatible endpoint (override in Settings if your region differs)
    base_url: 'https://ark.cn-beijing.volces.com/api/v3',
    models: []
  },
  qianfan: {
    name: '千帆 (Baidu)',
    url: '',
    default: '',
    // OpenAI-compatible endpoint
    base_url: 'https://qianfan.baidubce.com/v2',
    models: []
  },
  lingyiwanwu: {
    name: '零一万物',
    url: '',
    default: '',
    // OpenAI-compatible endpoint
    base_url: 'https://api.01.ai/v1',
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
