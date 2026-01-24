import { randomId } from '@/utils/id';
import { readJson, writeJson } from '@/utils/storage';

export const MOCK_STATE_STORAGE_KEY = 'pokemon_chat_mock_state_v1';

function nowSeconds() {
  return Math.floor(Date.now() / 1000);
}

function makeDefaultState() {
  const demoDbId = 'demo';
  const fileId1 = 'pokemon_intro.md';
  const fileId2 = 'pikachu.txt';

  return {
    databases: [
      {
        db_id: demoDbId,
        name: 'Demo 知识库',
        description: '离线演示：用于展示知识库列表/文件/片段查看等完整 UI。',
        embed_model: 'BAAI/bge-m3',
        dimension: 1024,
        created_at: nowSeconds(),
        files: {
          [fileId1]: {
            file_id: fileId1,
            filename: fileId1,
            type: 'md',
            status: 'done',
            created_at: nowSeconds() - 3600,
            // For mock /data/document
            lines: [
              { id: randomId(8), text: '宝可梦（Pokemon）是一系列由任天堂、GAME FREAK、Creatures 联合推出的作品。' },
              { id: randomId(8), text: '离线模式下，这些内容仅用于演示 UI，不代表真实检索结果。' },
            ],
          },
          [fileId2]: {
            file_id: fileId2,
            filename: fileId2,
            type: 'txt',
            status: 'done',
            created_at: nowSeconds() - 7200,
            lines: [
              { id: randomId(8), text: '皮卡丘是电属性宝可梦，最早出现在《宝可梦 红/绿》。' },
              { id: randomId(8), text: '如果你启用后端并导入真实文档，这里会显示真实分块。' },
            ],
          },
        },
      },
    ],
    tokensByAgent: {
      demo: [
        {
          id: randomId(10),
          name: 'Demo Token',
          token: `demo_${randomId(20)}`,
          created_at: new Date().toISOString(),
        },
      ],
    },
  };
}

export function getMockState() {
  const state = readJson(MOCK_STATE_STORAGE_KEY, null);
  if (state && typeof state === 'object') return state;
  const init = makeDefaultState();
  writeJson(MOCK_STATE_STORAGE_KEY, init);
  return init;
}

export function setMockState(next) {
  writeJson(MOCK_STATE_STORAGE_KEY, next);
  return next;
}

export function updateMockState(updater) {
  const cur = getMockState();
  const next = typeof updater === 'function' ? updater(cur) : cur;
  return setMockState(next);
}

