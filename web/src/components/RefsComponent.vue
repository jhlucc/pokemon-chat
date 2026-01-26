<template>
  <div class="refs-container">
    <!-- 引用来源标签 -->
    <div v-if="hasAnyRefs" class="refs-section">
      <div class="refs-header">
        <FolderOutlined class="refs-icon" />
        <span class="refs-title">参考来源</span>
        <span class="refs-count">{{ totalRefsCount }}</span>
      </div>

      <div class="refs-tags">
        <!-- 知识图谱 -->
        <button
          v-if="showKey('subGraph') && hasSubGraphData(msg)"
          class="ref-tag"
          @click="openSubGraph(msg)"
        >
          <ApartmentOutlined class="tag-icon" />
          <span class="tag-label">关系图</span>
        </button>

        <!-- 网页搜索 -->
        <button
          v-if="showKey('webSearch') && webSearchCount > 0"
          class="ref-tag"
          @click="showWebResult(msg)"
        >
          <GlobalOutlined class="tag-icon" />
          <span class="tag-label">网页搜索</span>
          <span class="tag-count">{{ webSearchCount }}</span>
        </button>

        <!-- 知识库文件 -->
        <button
          v-for="(results, filename) in kbGroupedResults"
          :key="filename"
          class="ref-tag"
          @click="toggleDrawer(filename)"
        >
          <FileTextOutlined class="tag-icon" />
          <span class="tag-label">{{ truncateFilename(filename) }}</span>
          <span class="tag-count">{{ results.length }}</span>
        </button>
      </div>
    </div>

    <!-- 知识图谱弹窗 -->
    <a-modal
      v-model:open="subGraphVisible"
      title="相关实体与关系"
      :width="800"
      :footer="null"
      class="graph-modal"
    >
      <GraphContainer v-if="subGraphVisible && subGraphData" :graphData="subGraphData" />
    </a-modal>

    <!-- 网页搜索结果抽屉 -->
    <a-drawer
      v-model:open="webResultVisible"
      title="网页搜索结果"
      width="600"
      :contentWrapperStyle="{ maxWidth: '100%' }"
      placement="right"
      class="refs-drawer"
    >
      <div class="results-list">
        <div v-for="result in webResults" :key="result.url" class="web-result-card">
          <div class="web-result-header">
            <div class="web-favicon">
              <GlobalOutlined />
            </div>
            <div class="web-meta">
              <a class="web-title" :href="result.url" target="_blank" rel="noopener noreferrer">
                {{ result.title }}
              </a>
              <span class="web-url">{{ formatUrl(result.url) }}</span>
            </div>
            <div class="web-score">
              <span class="score-value">{{ (result.score * 100).toFixed(0) }}%</span>
              <span class="score-label">相关度</span>
            </div>
          </div>
          <div class="web-content">{{ truncateContent(result.content, 200) }}</div>
        </div>
      </div>
    </a-drawer>

    <!-- 知识库文件详情抽屉 -->
    <a-drawer
      v-for="(results, filename) in kbGroupedResults"
      :key="`drawer-${filename}`"
      v-model:open="openDetail[filename]"
      :title="filename"
      width="600"
      :contentWrapperStyle="{ maxWidth: '100%' }"
      placement="right"
      class="refs-drawer"
    >
      <div class="file-info">
        <div class="file-info-item">
          <FileOutlined class="info-icon" />
          <span>{{ results[0]?.file?.type || '未知类型' }}</span>
        </div>
        <div class="file-info-item">
          <ClockCircleOutlined class="info-icon" />
          <span>{{ formatDate(results[0]?.file?.created_at) }}</span>
        </div>
      </div>

      <div class="results-list">
        <div v-for="(res, idx) in results" :key="res.id" class="kb-result-card">
          <div class="kb-result-header">
            <span class="result-index">#{{ idx + 1 }}</span>
            <div class="result-scores">
              <div class="score-item">
                <span class="score-label">相似度</span>
                <div class="score-bar">
                  <div class="score-fill" :style="{ width: `${res.distance * 100}%` }"></div>
                </div>
                <span class="score-value">{{ (res.distance * 100).toFixed(0) }}%</span>
              </div>
              <div v-if="res.rerank_score" class="score-item">
                <span class="score-label">重排序</span>
                <div class="score-bar">
                  <div class="score-fill rerank" :style="{ width: `${res.rerank_score * 100}%` }"></div>
                </div>
                <span class="score-value">{{ (res.rerank_score * 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
          <div class="kb-result-text">{{ res.entity?.text }}</div>
        </div>
      </div>
    </a-drawer>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, defineAsyncComponent } from 'vue'
import {
  GlobalOutlined,
  FileTextOutlined,
  ApartmentOutlined,
  FolderOutlined,
  FileOutlined,
  ClockCircleOutlined
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

const webSearchCount = computed(() => msg.value?.refs?.web_search?.results?.length || 0)

const totalRefsCount = computed(() => {
  let count = 0
  if (hasSubGraphData(msg.value)) count++
  count += webSearchCount.value
  count += Object.keys(kbGroupedResults.value).length
  return count
})

const hasAnyRefs = computed(() => {
  return (
    (showKey('subGraph') && hasSubGraphData(msg.value)) ||
    (showKey('webSearch') && webSearchCount.value > 0) ||
    Object.keys(kbGroupedResults.value).length > 0
  )
})

const showKey = (key) => {
  if (displayKeys.value === true) return true
  if (Array.isArray(displayKeys.value)) return displayKeys.value.includes(key)
  return false
}

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
  msg?.refs?.graph_base?.results?.nodes?.length > 0

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
  webResults.value = msg.refs?.web_search?.results || []
  webResultVisible.value = true
}

const formatDate = (timestamp) => {
  if (!timestamp) return '未知时间'
  return new Date(timestamp * 1000).toLocaleString()
}

const formatUrl = (url) => {
  try {
    const u = new URL(url)
    return u.hostname
  } catch {
    return url
  }
}

const truncateFilename = (filename, maxLen = 20) => {
  if (!filename) return ''
  if (filename.length <= maxLen) return filename
  const ext = filename.split('.').pop()
  const name = filename.slice(0, -(ext.length + 1))
  return `${name.slice(0, maxLen - ext.length - 4)}...${ext}`
}

const truncateContent = (content, maxLen = 200) => {
  if (!content) return ''
  if (content.length <= maxLen) return content
  return content.slice(0, maxLen) + '...'
}
</script>

<style lang="less" scoped>
.refs-container {
  margin-top: var(--space-2);
}

.refs-section {
  margin-bottom: var(--space-2);
}

.refs-header {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
  font-size: var(--font-size-sm);
  color: var(--gray-600);

  .refs-icon {
    font-size: 14px;
  }

  .refs-title {
    font-weight: 500;
  }

  .refs-count {
    padding: 2px 6px;
    background: var(--gray-100);
    border-radius: 10px;
    font-size: var(--font-size-xs);
    color: var(--gray-500);
  }
}

.refs-tags {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.ref-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--surface-color-2);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-xs);
  color: var(--gray-700);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--main-5);
    border-color: var(--primary-color);
    color: var(--primary-color);
  }

  .tag-icon {
    font-size: 14px;
    opacity: 0.8;
  }

  .tag-label {
    font-weight: 500;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tag-count {
    padding: 2px 6px;
    background: var(--gray-200);
    border-radius: 8px;
    font-size: 11px;
    color: var(--gray-600);
  }
}

/* 抽屉样式 */
.refs-drawer {
  .file-info {
    display: flex;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    background: var(--surface-color-2);
    border-radius: var(--radius-sm);
    margin-bottom: var(--space-4);
  }

  .file-info-item {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--font-size-sm);
    color: var(--gray-600);

    .info-icon {
      font-size: 14px;
      color: var(--gray-500);
    }
  }
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

/* 网页搜索结果卡片 */
.web-result-card {
  padding: var(--space-4);
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    border-color: var(--primary-color);
    box-shadow: var(--shadow-xs);
  }
}

.web-result-header {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.web-favicon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface-color-2);
  border-radius: var(--radius-sm);
  color: var(--gray-500);
  flex-shrink: 0;
}

.web-meta {
  flex: 1;
  min-width: 0;
}

.web-title {
  display: block;
  font-size: var(--font-size-base);
  font-weight: 600;
  color: var(--text-color);
  text-decoration: none;
  margin-bottom: 4px;

  &:hover {
    color: var(--primary-color);
  }
}

.web-url {
  font-size: var(--font-size-xs);
  color: var(--gray-500);
}

.web-score {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--space-2) var(--space-3);
  background: var(--main-5);
  border-radius: var(--radius-sm);

  .score-value {
    font-size: var(--font-size-lg);
    font-weight: 700;
    color: var(--primary-color);
  }

  .score-label {
    font-size: var(--font-size-xs);
    color: var(--gray-500);
  }
}

.web-content {
  font-size: var(--font-size-sm);
  color: var(--gray-600);
  line-height: var(--line-height-relaxed);
}

/* 知识库结果卡片 */
.kb-result-card {
  padding: var(--space-4);
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
}

.kb-result-header {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.result-index {
  font-size: var(--font-size-sm);
  font-weight: 600;
  color: var(--gray-500);
  font-family: var(--font-family-mono);
}

.result-scores {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.score-item {
  display: flex;
  align-items: center;
  gap: var(--space-2);

  .score-label {
    font-size: var(--font-size-xs);
    color: var(--gray-500);
    width: 48px;
    flex-shrink: 0;
  }

  .score-bar {
    flex: 1;
    height: 6px;
    background: var(--gray-100);
    border-radius: 3px;
    overflow: hidden;
  }

  .score-fill {
    height: 100%;
    background: var(--primary-color);
    border-radius: 3px;
    transition: width var(--duration-slow) var(--ease-out);

    &.rerank {
      background: var(--holo-accent);
    }
  }

  .score-value {
    font-size: var(--font-size-xs);
    font-weight: 600;
    color: var(--gray-600);
    width: 36px;
    text-align: right;
    font-family: var(--font-family-mono);
  }
}

.kb-result-text {
  font-size: var(--font-size-sm);
  color: var(--text-color);
  line-height: var(--line-height-relaxed);
  white-space: pre-wrap;
  word-break: break-word;
  padding: var(--space-3);
  background: var(--surface-color-2);
  border-radius: var(--radius-sm);
}
</style>
