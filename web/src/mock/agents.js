import { MOCK_TOOLS } from './tools';

export const MOCK_AGENTS = [
  {
    name: 'demo',
    description: '离线演示智能体：用于展示对话/工具调用 UI。',
    requirements: [],
    all_tools: MOCK_TOOLS.map((t) => t.name),
    config_schema: {
      system_prompt: '你是一个离线演示智能体。请明确告知用户当前为 Mock 模式。',
      model: 'mock/offline',
      tools: MOCK_TOOLS.map((t) => t.name),
      configurable_items: {
        temperature: { description: '生成温度（演示项）', default: 0.7 },
        use_web: { description: '是否启用联网搜索（演示项）', default: true },
        style: {
          description: '回答风格（演示项）',
          default: 'balanced',
          options: ['balanced', 'concise', 'detailed'],
        },
      },
    },
  },
  {
    name: 'pokemon',
    description: '宝可梦百科智能体（离线演示）：可以做问答与基础科普。',
    requirements: [],
    all_tools: MOCK_TOOLS.map((t) => t.name),
    config_schema: {
      system_prompt: '你是宝可梦百科助手（离线演示）。无法联网与调用后端时请给出提示。',
      model: 'mock/offline',
      tools: MOCK_TOOLS.map((t) => t.name),
      configurable_items: {
        safe_mode: { description: '安全模式（演示项）', default: true },
      },
    },
  },
];

export function mockAgentListResponse() {
  return {
    agents: MOCK_AGENTS.map((a) => ({ ...a })),
  };
}

