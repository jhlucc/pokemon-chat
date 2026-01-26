<template>
  <div class="retrieval-status" :class="{ collapsed: isCollapsed }">
    <div class="retrieval-header" @click="isCollapsed = !isCollapsed">
      <div class="header-left">
        <div class="status-indicator" :class="overallStatus">
          <LoadingOutlined v-if="isActive" spin class="spin-icon" />
          <CheckCircleOutlined v-else-if="overallStatus === 'completed'" class="check-icon" />
          <SearchOutlined v-else class="search-icon" />
        </div>
        <span class="header-title">{{ headerTitle }}</span>
        <span v-if="completedCount > 0" class="completed-badge">
          {{ completedCount }}/{{ totalSources }}
        </span>
      </div>
      <CaretRightOutlined class="collapse-caret" :class="{ open: !isCollapsed }" />
    </div>

    <transition name="expand">
      <div v-if="!isCollapsed" class="retrieval-steps">
        <!-- Knowledge Base -->
        <div
          v-if="showKnowledgeBase"
          class="step-item"
          :class="getStepStatus('knowledge_base')"
        >
          <div class="step-icon">
            <BookOutlined />
          </div>
          <div class="step-content">
            <span class="step-label">知识库检索</span>
            <span v-if="getStepResults('knowledge_base')" class="step-result">
              {{ getStepResults('knowledge_base') }} 条结果
            </span>
          </div>
          <div class="step-status-indicator">
            <LoadingOutlined v-if="isStepActive('knowledge_base')" spin />
            <CheckOutlined v-else-if="isStepCompleted('knowledge_base')" />
            <span v-else class="dot"></span>
          </div>
        </div>

        <!-- Knowledge Graph -->
        <div
          v-if="showKnowledgeGraph"
          class="step-item"
          :class="getStepStatus('graph_base')"
        >
          <div class="step-icon">
            <ApartmentOutlined />
          </div>
          <div class="step-content">
            <span class="step-label">图谱查询</span>
            <span v-if="getStepResults('graph_base')" class="step-result">
              {{ getStepResults('graph_base') }} 个节点
            </span>
          </div>
          <div class="step-status-indicator">
            <LoadingOutlined v-if="isStepActive('graph_base')" spin />
            <CheckOutlined v-else-if="isStepCompleted('graph_base')" />
            <span v-else class="dot"></span>
          </div>
        </div>

        <!-- Web Search -->
        <div
          v-if="showWebSearch"
          class="step-item"
          :class="getStepStatus('web_search')"
        >
          <div class="step-icon">
            <GlobalOutlined />
          </div>
          <div class="step-content">
            <span class="step-label">联网搜索</span>
            <span v-if="getStepResults('web_search')" class="step-result">
              {{ getStepResults('web_search') }} 条结果
            </span>
          </div>
          <div class="step-status-indicator">
            <LoadingOutlined v-if="isStepActive('web_search')" spin />
            <CheckOutlined v-else-if="isStepCompleted('web_search')" />
            <span v-else class="dot"></span>
          </div>
        </div>

        <!-- MCP Tools -->
        <div
          v-if="showMcp"
          class="step-item"
          :class="getStepStatus('mcp')"
        >
          <div class="step-icon">
            <ApiOutlined />
          </div>
          <div class="step-content">
            <span class="step-label">MCP 工具</span>
            <span v-if="getStepResults('mcp')" class="step-result">
              {{ getStepResults('mcp') }} 次调用
            </span>
          </div>
          <div class="step-status-indicator">
            <LoadingOutlined v-if="isStepActive('mcp')" spin />
            <CheckOutlined v-else-if="isStepCompleted('mcp')" />
            <span v-else class="dot"></span>
          </div>
        </div>

        <!-- Progress Bar -->
        <div v-if="isActive" class="progress-section">
          <div class="progress-bar">
            <div
              class="progress-fill"
              :style="{ width: `${progressPercent}%` }"
            ></div>
          </div>
          <span class="progress-text">{{ progressText }}</span>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import {
  LoadingOutlined,
  CheckCircleOutlined,
  CheckOutlined,
  SearchOutlined,
  CaretRightOutlined,
  BookOutlined,
  ApartmentOutlined,
  GlobalOutlined,
  ApiOutlined
} from '@ant-design/icons-vue'

const props = defineProps({
  status: {
    type: String,
    default: 'init' // init, searching, reasoning, generating, finished, error
  },
  refs: {
    type: Object,
    default: () => ({})
  },
  activeStep: {
    type: String,
    default: null // knowledge_base, graph_base, web_search, mcp
  },
  showKnowledgeBase: {
    type: Boolean,
    default: true
  },
  showKnowledgeGraph: {
    type: Boolean,
    default: true
  },
  showWebSearch: {
    type: Boolean,
    default: true
  },
  showMcp: {
    type: Boolean,
    default: false
  }
})

const isCollapsed = ref(false)

const isActive = computed(() =>
  ['searching', 'reasoning'].includes(props.status)
)

const overallStatus = computed(() => {
  if (props.status === 'finished') return 'completed'
  if (props.status === 'error') return 'error'
  if (isActive.value) return 'active'
  return 'pending'
})

const headerTitle = computed(() => {
  if (props.status === 'searching') return '正在检索...'
  if (props.status === 'reasoning') return '正在推理...'
  if (props.status === 'generating') return '检索完成'
  if (props.status === 'finished') return '检索完成'
  if (props.status === 'error') return '检索出错'
  return '准备检索'
})

const totalSources = computed(() => {
  let count = 0
  if (props.showKnowledgeBase) count++
  if (props.showKnowledgeGraph) count++
  if (props.showWebSearch) count++
  if (props.showMcp) count++
  return count
})

const completedCount = computed(() => {
  let count = 0
  if (props.showKnowledgeBase && isStepCompleted('knowledge_base')) count++
  if (props.showKnowledgeGraph && isStepCompleted('graph_base')) count++
  if (props.showWebSearch && isStepCompleted('web_search')) count++
  if (props.showMcp && isStepCompleted('mcp')) count++
  return count
})

const progressPercent = computed(() => {
  if (totalSources.value === 0) return 0
  return Math.round((completedCount.value / totalSources.value) * 100)
})

const progressText = computed(() => {
  if (props.activeStep === 'knowledge_base') return '查询知识库...'
  if (props.activeStep === 'graph_base') return '遍历知识图谱...'
  if (props.activeStep === 'web_search') return '搜索网络...'
  if (props.activeStep === 'mcp') return '调用工具...'
  return '处理中...'
})

const getStepStatus = (step) => {
  if (isStepActive(step)) return 'active'
  if (isStepCompleted(step)) return 'completed'
  return 'pending'
}

const isStepActive = (step) => {
  return props.activeStep === step
}

const isStepCompleted = (step) => {
  if (!props.refs) return false

  switch (step) {
    case 'knowledge_base':
      return props.refs.knowledge_base?.results?.length > 0
    case 'graph_base':
      return props.refs.graph_base?.results?.nodes?.length > 0
    case 'web_search':
      return props.refs.web_search?.results?.length > 0
    case 'mcp':
      return props.refs.mcp?.calls?.length > 0
    default:
      return false
  }
}

const getStepResults = (step) => {
  if (!props.refs) return null

  switch (step) {
    case 'knowledge_base':
      return props.refs.knowledge_base?.results?.length || null
    case 'graph_base':
      return props.refs.graph_base?.results?.nodes?.length || null
    case 'web_search':
      return props.refs.web_search?.results?.length || null
    case 'mcp':
      return props.refs.mcp?.calls?.length || null
    default:
      return null
  }
}
</script>

<style scoped lang="less">
.retrieval-status {
  margin: var(--space-2) 0 var(--space-3);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  background: var(--surface-color-2);
  overflow: hidden;
  animation: fadeIn var(--duration-base) var(--ease-out);
}

.retrieval-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  cursor: pointer;
  transition: background var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--hover-bg);
  }
}

.header-left {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.status-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  font-size: 12px;

  &.pending {
    background: var(--gray-200);
    color: var(--gray-500);
  }

  &.active {
    background: var(--main-10);
    color: var(--primary-color);

    .spin-icon {
      animation: spin 1s linear infinite;
    }
  }

  &.completed {
    background: var(--success-color);
    color: white;

    .check-icon {
      font-size: 11px;
    }
  }

  &.error {
    background: var(--error-color);
    color: white;
  }
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.header-title {
  font-size: var(--font-size-sm);
  font-weight: 500;
  color: var(--gray-700);
}

.completed-badge {
  padding: 2px 8px;
  background: var(--gray-100);
  border-radius: 10px;
  font-size: var(--font-size-xs);
  color: var(--gray-500);
  font-family: var(--font-family-mono);
}

.collapse-caret {
  font-size: 12px;
  color: var(--gray-400);
  transition: transform var(--duration-fast) var(--ease-default);

  &.open {
    transform: rotate(90deg);
  }
}

.retrieval-steps {
  padding: 0 var(--space-3) var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.step-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  background: var(--surface-color);
  border: 1px solid transparent;
  transition: all var(--duration-fast) var(--ease-default);

  &.pending {
    opacity: 0.6;
  }

  &.active {
    border-color: var(--primary-color);
    background: var(--main-5);
    box-shadow: 0 0 0 2px var(--main-10);

    .step-icon {
      color: var(--primary-color);
    }

    .step-label {
      color: var(--primary-color);
      font-weight: 600;
    }
  }

  &.completed {
    .step-icon {
      color: var(--success-color);
    }

    .step-status-indicator {
      color: var(--success-color);
    }
  }
}

.step-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  background: var(--surface-color-2);
  color: var(--gray-500);
  font-size: 14px;
  flex-shrink: 0;
  transition: all var(--duration-fast) var(--ease-default);
}

.step-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.step-label {
  font-size: var(--font-size-sm);
  color: var(--gray-700);
  font-weight: 500;
}

.step-result {
  font-size: var(--font-size-xs);
  color: var(--gray-500);
}

.step-status-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  color: var(--gray-400);
  font-size: 12px;

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--gray-300);
  }
}

.progress-section {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px solid var(--border-color);
}

.progress-bar {
  flex: 1;
  height: 4px;
  background: var(--gray-200);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary-color), var(--main-400));
  border-radius: 2px;
  transition: width var(--duration-base) var(--ease-out);
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

.progress-text {
  font-size: var(--font-size-xs);
  color: var(--gray-500);
  white-space: nowrap;
}

/* Expand transition */
.expand-enter-active,
.expand-leave-active {
  transition: all var(--duration-base) var(--ease-default);
  max-height: 300px;
  overflow: hidden;
}

.expand-enter-from,
.expand-leave-to {
  max-height: 0;
  opacity: 0;
  padding-top: 0;
  padding-bottom: 0;
}

/* Responsive */
@media (max-width: 640px) {
  .step-item {
    padding: var(--space-2);
    gap: var(--space-2);
  }

  .step-icon {
    width: 24px;
    height: 24px;
    font-size: 12px;
  }

  .step-result {
    display: none;
  }

  .progress-text {
    display: none;
  }
}
</style>
