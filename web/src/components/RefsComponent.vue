<template>
  <div class="refs">
    <div class="tags">
      <span v-if="msg.meta?.server_model_name" class="item">
        <BulbOutlined /> {{ msg.meta.server_model_name }}
      </span>

      <span v-if="showKey('copy')" class="item btn" @click="copyText(msg.content)" title="复制">
        <CopyOutlined />
      </span>

      <span v-if="showKey('regenerate')" class="item btn" @click="regenerateMessage()" title="重新生成">
        <ReloadOutlined />
      </span>

      <span v-if="showKey('subGraph') && hasSubGraphData(msg)" class="item btn" @click="openSubGraph(msg)">
        <DeploymentUnitOutlined /> 关系图
      </span>

      <span
        v-if="showKey('webSearch') && msg.refs?.web_search?.results?.length > 0"
        class="item btn"
        @click="showWebResult(msg)">
        <GlobalOutlined /> 网页搜索 {{ msg.refs.web_search.results.length }}
      </span>

      <span class="filetag item btn"
        v-for="(results, filename) in msg.groupedResults"
        :key="filename"
        @click="toggleDrawer(filename)">
        <FileTextOutlined /> {{ filename }}
        <a-drawer
          v-model:open="openDetail[filename]"
          :title="filename"
          width="700"
          :contentWrapperStyle="{ maxWidth: '100%'}"
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
      <GraphContainer :graphData="subGraphData" />
    </a-modal>

    <a-drawer
      v-model:open="webResultVisible"
      title="网页搜索结果"
      width="700"
      :contentWrapperStyle="{ maxWidth: '100%'}"
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
              <a :href="result.url" target="_blank">{{ result.url }}</a>
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
import { ref, reactive } from 'vue'
import { useClipboard } from '@vueuse/core'
import { message } from 'ant-design-vue'
import {
  GlobalOutlined,
  FileTextOutlined,
  CopyOutlined,
  DeploymentUnitOutlined,
  BulbOutlined,
  FileOutlined,
  ClockCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons-vue'
import GraphContainer from './GraphContainer.vue'

const emit = defineEmits(['retry'])

const props = defineProps({
  message: Object,
  showRefs: {
    type: [Array, Boolean],
    default: () => false
  }
})

const msg = ref(props.message)
const displayKeys = ref(props.showRefs)

const showKey = (key) => {
  if (displayKeys.value === true) return true
  if (Array.isArray(displayKeys.value)) return displayKeys.value.includes(key)
  return false
}

const { copy, isSupported } = useClipboard()
const copyText = async (text) => {
  try {
    await copy(text)
    message.success('已复制到剪贴板')
  } catch (e) {
    message.error('复制失败')
  }
}

const regenerateMessage = () => emit('retry')

const openDetail = reactive({})
for (const filename in msg.value.groupedResults) {
  openDetail[filename] = false
}
const toggleDrawer = (filename) => {
  openDetail[filename] = !openDetail[filename]
}

const hasSubGraphData = (msg) =>
  msg.refs &&
  msg.refs.graph_base &&
  msg.refs.graph_base.results?.nodes?.length > 0

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

const formatDate = (timestamp) =>
  new Date(timestamp * 1000).toLocaleString()

const getPercent = (value) =>
  parseFloat((value * 100).toFixed(2))
</script>


<style lang="less" scoped>
.refs {
  display: flex;
  margin-bottom: 20px;
  background: transparent;
  font-size: 13px;
  gap: 10px;

  .item {
    color: var(--subtext-color);
    background: transparent;
    padding: 2px 8px;
    border-radius: var(--radius-md);
    font-size: 13px;

    user-select: none;

    &.btn {
      cursor: pointer;
      &:hover {
        background: var(--surface-secondary);
        color: var(--primary-color);
      }
      &:active {
        background: var(--gray-300); // Or a darker shade of surface-secondary
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
    background-color: var(--surface-secondary);
    border-radius: var(--radius-md);
    margin-bottom: 16px;

    p {
      margin: 0;
      color: var(--text-color);
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
        color: var(--subtext-color);
      }

      .ant-progress {
        width: 170px;
        margin-bottom: 0;
        margin-inline: 10px;

        :deep(.ant-progress-bg) {
          background-color: var(--primary-color) !important;
        }
      }
    }
  }

  .result-id {
    font-size: 12px;
    color: var(--subtext-color);
    margin-bottom: 8px;
  }

  .result-text {
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    background-color: var(--surface-secondary);
    padding: 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    color: var(--text-color);
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
            color: var(--subtext-color);
          }

          .ant-progress {
            width: 170px;
            margin-bottom: 0;
            margin-inline: 10px;

            :deep(.ant-progress-bg) {
              background-color: var(--primary-color) !important;
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
        background-color: var(--surface-secondary);
        padding: 12px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        color: var(--text-color);
      }
    }
  }
}
</style>