<template>
  <GlassPanel title="检索测试" class="search-test-panel">
    <template #header-actions>
      <a-tooltip title="测试知识库的检索效果">
        <QuestionCircleOutlined />
      </a-tooltip>
    </template>

    <div class="search-input-area">
      <a-textarea
        v-model:value="query"
        placeholder="输入查询语句测试检索效果..."
        :auto-size="{ minRows: 2, maxRows: 4 }"
        @keydown.ctrl.enter="runSearch"
      />
      <a-button
        type="primary"
        :loading="loading"
        :disabled="!query.trim() || !dbId"
        @click="runSearch"
      >
        <SearchOutlined /> 检索
      </a-button>
    </div>

    <a-collapse v-model:activeKey="paramsOpen" class="params-collapse">
      <a-collapse-panel key="params" header="检索参数">
        <div class="params-grid">
          <div class="param-item">
            <label>Top K</label>
            <a-input-number v-model:value="params.topK" :min="1" :max="20" />
          </div>
          <div class="param-item">
            <label>距离阈值</label>
            <a-slider v-model:value="params.distanceThreshold" :min="0" :max="1" :step="0.01" />
          </div>
          <div class="param-item">
            <label>启用重排序</label>
            <a-switch v-model:checked="params.rerank" />
          </div>
        </div>
      </a-collapse-panel>
    </a-collapse>

    <div v-if="results" class="results-section">
      <div class="results-header">
        <span>找到 {{ results.results?.length || 0 }} 条结果</span>
        <span class="results-time" v-if="searchTime">耗时 {{ searchTime }}ms</span>
      </div>

      <div class="results-list">
        <div v-for="(item, idx) in results.results" :key="idx" class="result-card">
          <div class="result-meta">
            <span class="result-rank">#{{ idx + 1 }}</span>
            <span class="result-file">{{ item.file?.filename || 'Unknown' }}</span>
            <span class="result-score">
              相似度: {{ (1 - item.distance).toFixed(3) }}
              <template v-if="item.rerank_score"> · 重排序: {{ item.rerank_score.toFixed(3) }}</template>
            </span>
          </div>
          <p class="result-text">{{ item.entity?.text }}</p>
        </div>
      </div>

      <a-empty v-if="results.results?.length === 0" description="未找到相关结果" />
    </div>
  </GlassPanel>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { message } from 'ant-design-vue'
import { SearchOutlined, QuestionCircleOutlined } from '@ant-design/icons-vue'
import GlassPanel from '@/components/common/GlassPanel.vue'
import { apiFetch } from '@/api/http'

const props = defineProps({
  dbId: { type: String, required: true }
})

const query = ref('')
const loading = ref(false)
const results = ref(null)
const searchTime = ref(null)
const paramsOpen = ref([])

const params = reactive({
  topK: 5,
  distanceThreshold: 0.5,
  rerank: true
})

const runSearch = async () => {
  if (!query.value.trim() || !props.dbId) return

  loading.value = true
  const startTime = Date.now()

  try {
    const data = await apiFetch('/data/query-test', {
      method: 'POST',
      body: {
        query: query.value,
        meta: {
          db_id: props.dbId,
          topK: params.topK,
          distanceThreshold: params.distanceThreshold,
          rerank: params.rerank
        }
      },
      timeoutMs: 30000
    })
    results.value = data
    searchTime.value = Date.now() - startTime
  } catch (e) {
    message.error(e?.message || '检索失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped lang="less">
.search-test-panel {
  margin-top: 16px;
}

.search-input-area {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;

  :deep(.ant-input) {
    flex: 1;
  }
}

.params-collapse {
  margin-bottom: 16px;
  background: transparent;

  :deep(.ant-collapse-item) {
    border: 1px solid var(--border-color);
    border-radius: 12px;
  }
}

.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.param-item {
  display: flex;
  flex-direction: column;
  gap: 8px;

  label {
    font-size: 13px;
    color: var(--gray-600);
    font-weight: 500;
  }

  /* 优化滑动条样式 */
  :deep(.ant-slider) {
    margin: 4px 0;

    .ant-slider-rail {
      height: 4px;
      background-color: var(--gray-200);
      border-radius: 2px;
    }

    .ant-slider-track {
      height: 4px;
      background-color: var(--primary-color);
      border-radius: 2px;
    }

    .ant-slider-handle {
      width: 16px;
      height: 16px;
      margin-top: -6px;
      border: 2px solid var(--primary-color);
      background-color: #fff;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
      transition: all 0.2s ease;

      &:hover, &:focus {
        border-color: var(--primary-color);
        box-shadow: 0 2px 8px rgba(255, 125, 0, 0.3);
      }
    }
  }
}

.results-section {
  margin-top: 16px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  font-size: 14px;
  color: var(--gray-600);
}

.results-time {
  color: var(--gray-400);
  font-size: 12px;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-card {
  padding: 16px;
  background: color-mix(in srgb, var(--primary-color) 4%, transparent);
  border: 1px solid color-mix(in srgb, var(--primary-color) 10%, transparent);
  border-radius: 12px;
  transition: all 0.2s ease;

  &:hover {
    background: color-mix(in srgb, var(--primary-color) 8%, transparent);
    transform: translateY(-1px);
  }
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
}

.result-rank {
  font-weight: 600;
  color: var(--primary-color);
}

.result-file {
  color: var(--gray-600);
}

.result-score {
  margin-left: auto;
  color: var(--gray-500);
}

.result-text {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-color);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

:root[data-theme='dark'] {
  .result-card {
    background: color-mix(in srgb, var(--primary-color) 8%, transparent);
    border-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
  }
}
</style>
