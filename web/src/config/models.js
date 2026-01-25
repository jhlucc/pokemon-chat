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
  zhipu: {
    name: '智谱AI',
    url: 'https://open.bigmodel.cn/dev/api',
    default: 'glm-4-flash',
    base_url: 'https://open.bigmodel.cn/api/paas/v4/',
    models: ['glm-4', 'glm-4-plus', 'glm-4-air', 'glm-4-flash']
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
