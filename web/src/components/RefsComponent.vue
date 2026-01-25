<template>
  <div class="refs">
    <div class="tags">
      <span v-if="msg.meta?.server_model_name" class="item">
        <BulbOutlined /> {{ msg.meta.server_model_name }}
      </span>

      <span v-if="showKey('copy')" class="item btn" @click="copyText(msg.content)" title="复制">
        <CopyOutlined />
      </span>

      <span
        v-if="showKey('regenerate')"
        class="item btn"
        @click="regenerateMessage()"
        title="重新生成"
      >
        <ReloadOutlined />
      </span>

      <span
        v-if="showKey('subGraph') && hasSubGraphData(msg)"
        class="item btn"
        @click="openSubGraph(msg)"
      >
        <DeploymentUnitOutlined /> 关系图
      </span>

      <span
        v-if="showKey('webSearch') && msg.refs?.web_search?.results?.length > 0"
        class="item btn"
        @click="showWebResult(msg)"
      >
        <GlobalOutlined /> 网页搜索 {{ msg.refs.web_search.results.length }}
      </span>

      <span
        class="filetag item btn"
        v-for="(results, filename) in kbGroupedResults"
        :key="filename"
        @click="toggleDrawer(filename)"
      >
        <FileTextOutlined /> {{ filename }}
        <a-drawer
          v-model:open="openDetail[filename]"
          :title="filename"
          width="700"
          :contentWrapperStyle="{ maxWidth: '100%' }"
          placement="right"
          class="retrieval-detail"
          rootClassName="root"
        >
          <div class="fileinfo">
            <p><FileOutlined /> {{ results[0].file.type }}</p>
            <p><ClockCircleOutlined /> {{ formatDate(results[0].file.created_at) }}</p>
          </div>
          <div class="results-list">
            <div v-for="res in results" :key="res.id" class="result-item">
              <div class="result-meta">
                <div class="score-info">
                  <span>
                    <strong>相似度：</strong>
                    <a-progress :percent="getPercent(res.distance)" />
                  </span>
                  <span v-if="res.rerank_score">
                    <strong>重排序：</strong>
                    <a-progress :percent="getPercent(res.rerank_score)" />
                  </span>
                </div>
                <div class="result-id">ID: #{{ res.id }}</div>
              </div>
              <div class="result-text">{{ res.entity.text }}</div>
            </div>
          </div>
        </a-drawer>
      </span>
    </div>

    <a-modal v-model:open="subGraphVisible" title="相关实体与关系" :width="800" :footer="null">
      <GraphContainer v-if="subGraphVisible && subGraphData" :graphData="subGraphData" />
    </a-modal>

    <a-drawer
      v-model:open="webResultVisible"
      title="网页搜索结果"
      width="700"
      :contentWrapperStyle="{ maxWidth: '100%' }"
      placement="right"
      class="web-result-detail"
      rootClassName="root"
    >
      <div class="results-list">
        <div v-for="result in webResults" :key="result.url" class="result-item">
          <div class="result-meta">
            <div class="score-info">
              <span>
                <strong>相关度：</strong>
                <a-progress :percent="getPercent(result.score)" />
              </span>
            </div>
            <div class="result-url">
              <a :href="result.url" target="_blank" rel="noopener noreferrer">{{ result.url }}</a>
            </div>
          </div>
          <div class="result-content">
            <h3 class="result-title">{{ result.title }}</h3>
            <div class="result-text">{{ result.content }}</div>
          </div>
        </div>
      </div>
    </a-drawer>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, defineAsyncComponent } from 'vue'
import { useClipboard } from '@vueuse/core'
import { message as antdMessage } from 'ant-design-vue'
import {
  GlobalOutlined,
  FileTextOutlined,
  CopyOutlined,
  DeploymentUnitOutlined,
  BulbOutlined,
  FileOutlined,
  ClockCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons-vue'
const GraphContainer = defineAsyncComponent(() => import('./GraphContainer.vue'))

const emit = defineEmits(['retry'])

const props = defineProps({
  message: Object,
  showRefs: {
    type: [Array, Boolean],
    default: () => false
  }
})

const msg = computed(() => props.message || {})
const displayKeys = computed(() => props.showRefs)

const kbGroupedResults = computed(() => {
  const direct = msg.value?.groupedResults
  if (direct && typeof direct === 'object' && !Array.isArray(direct)) {
    return direct
  }
  const results = msg.value?.refs?.knowledge_base?.results
  if (!Array.isArray(results) || results.length === 0) return {}
  return results
    .filter((r) => r?.file?.filename)
    .reduce((acc, r) => {
      const filename = r.file.filename
      if (!acc[filename]) acc[filename] = []
      acc[filename].push(r)
      return acc
    }, {})
})

const showKey = (key) => {
  if (displayKeys.value === true) return true
  if (Array.isArray(displayKeys.value)) return displayKeys.value.includes(key)
  return false
}

const { copy, isSupported } = useClipboard()
const copyText = async (text) => {
  if (!isSupported.value) {
    antdMessage.error('当前浏览器不支持复制')
    return
  }
  try {
    await copy(text)
    antdMessage.success('已复制到剪贴板')
  } catch {
    antdMessage.error('复制失败')
  }
}

const regenerateMessage = () => emit('retry')

const openDetail = reactive({})
const ensureOpenDetailKeys = () => {
  const grouped =
    kbGroupedResults.value && typeof kbGroupedResults.value === 'object'
      ? kbGroupedResults.value
      : {}
  Object.keys(grouped).forEach((filename) => {
    if (openDetail[filename] === undefined) openDetail[filename] = false
  })
}
watch(
  () => kbGroupedResults.value,
  () => ensureOpenDetailKeys(),
  { immediate: true, deep: true }
)
const toggleDrawer = (filename) => {
  openDetail[filename] = !openDetail[filename]
}

const hasSubGraphData = (msg) =>
  msg.refs && msg.refs.graph_base && msg.refs.graph_base.results?.nodes?.length > 0

const subGraphVisible = ref(false)
const subGraphData = ref(null)
const openSubGraph = (msg) => {
  if (hasSubGraphData(msg)) {
    subGraphData.value = msg.refs.graph_base.results
    subGraphVisible.value = true
  }
}

const webResultVisible = ref(false)
const webResults = ref(null)
const showWebResult = (msg) => {
  webResults.value = msg.refs.web_search.results
  webResultVisible.value = true
}

const formatDate = (timestamp) => new Date(timestamp * 1000).toLocaleString()

const getPercent = (value) => parseFloat((value * 100).toFixed(2))
</script>

<style lang="less" scoped>
.refs {
  display: flex;
  margin-bottom: 20px;
  //color: var(--gray-500);
  background: transparent;
  font-size: 13px;
  gap: 10px;

  .item {
    color: var(--gray-700);
    background: transparent; // ✅ 不要小气泡背景
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 13px;

    user-select: none;

    &.btn {
      cursor: pointer;
      &:hover {
        background: var(--hover-bg);
        color: var(--text-color);
      }
      &:active {
        background: var(--gray-200);
      }
    }
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;

    .filetag {
      display: flex;
      align-items: center;
      gap: 5px;
    }
  }
}

.retrieval-detail {
  .fileinfo {
    display: flex;
    justify-content: space-between;
    padding: 12px 16px;
    background-color: var(--surface-color-2);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    margin-bottom: 16px;

    p {
      margin: 0;
      color: var(--gray-700);
    }
  }

  .score-info {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    margin-bottom: 8px;

    span {
      display: flex;
      align-items: center;

      strong {
        margin-right: 8px;
        white-space: nowrap;
        color: var(--gray-700);
      }

      .ant-progress {
        width: 170px;
        margin-bottom: 0;
        margin-inline: 10px;

        .ant-progress-bg {
          background-color: var(--main-500);
        }
      }
    }
  }

  .result-id {
    font-size: 12px;
    color: var(--gray-600);
    margin-bottom: 8px;
  }

  .result-text {
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    background-color: var(--surface-color-2);
    padding: 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
  }
}

.results-list {
  .result-item {
    border-bottom: 1px solid var(--border-color);
    padding: 16px 0;

    &:last-child {
      border-bottom: none;
    }
  }

  .result-meta {
    margin-bottom: 12px;
  }
}

.web-result-detail {
  .results-list {
    .result-item {
      border-bottom: 1px solid var(--border-color);
      padding: 16px 0;

      &:last-child {
        border-bottom: none;
      }
    }

    .result-meta {
      margin-bottom: 12px;

      .score-info {
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
        margin-bottom: 8px;

        span {
          display: flex;
          align-items: center;

          strong {
            margin-right: 8px;
            white-space: nowrap;
            color: var(--gray-700);
          }

          .ant-progress {
            width: 170px;
            margin-bottom: 0;
            margin-inline: 10px;

            .ant-progress-bg {
              background-color: var(--main-500);
            }
          }
        }
      }

      .result-url {
        font-size: 12px;
        color: var(--primary-color);
        margin-bottom: 8px;
        word-break: break-all;
      }
    }

    .result-content {
      .result-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 8px;
        color: var(--text-color);
      }

      .result-text {
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
        background-color: var(--surface-color-2);
        padding: 12px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-color);
      }
    }
  }
}
</style>
