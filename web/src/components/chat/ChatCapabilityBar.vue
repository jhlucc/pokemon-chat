<template>
  <div class="capability-bar">
    <!-- Agent 模式主开关 -->
    <a-tooltip :title="agentTooltip">
      <div
        :class="{
          'capability-chip': true,
          'capability-chip--agent': true,
          active: useAgent,
          disabled: !canAgent
        }"
        role="button"
        :tabindex="canAgent ? 0 : -1"
        :aria-disabled="!canAgent"
        aria-label="切换 Agent 模式"
        @click="emit('toggle-agent')"
        @keydown.enter.prevent="emit('toggle-agent')"
        @keydown.space.prevent="emit('toggle-agent')"
      >
        <RobotOutlined class="chip-icon" />
        <span class="chip-label">智能模式</span>
        <span class="chip-status" :class="{ on: useAgent }">
          {{ useAgent ? 'ON' : 'OFF' }}
        </span>
      </div>
    </a-tooltip>

    <!-- 分隔线 (Agent 关闭时显示) -->
    <div v-if="!useAgent && hasAnyCapability" class="capability-divider"></div>

    <!-- 手动配置选项 (Agent 关闭时显示) -->
    <template v-if="!useAgent">
      <a-tooltip v-if="showWebSearch" :title="webTooltip">
        <div
          :class="{
            'capability-chip': true,
            active: useWeb,
            disabled: !canWebSearch
          }"
          role="button"
          :tabindex="canWebSearch ? 0 : -1"
          :aria-disabled="!canWebSearch"
          aria-label="切换联网搜索"
          @click="emit('toggle-web')"
          @keydown.enter.prevent="emit('toggle-web')"
          @keydown.space.prevent="emit('toggle-web')"
        >
          <GlobalOutlined class="chip-icon" />
          <span class="chip-label">联网</span>
        </div>
      </a-tooltip>

      <a-tooltip v-if="showKnowledgeGraph" :title="graphTooltip">
        <div
          :class="{
            'capability-chip': true,
            active: useGraph,
            disabled: !canGraph
          }"
          role="button"
          :tabindex="canGraph ? 0 : -1"
          :aria-disabled="!canGraph"
          aria-label="切换知识图谱"
          @click="emit('toggle-graph')"
          @keydown.enter.prevent="emit('toggle-graph')"
          @keydown.space.prevent="emit('toggle-graph')"
        >
          <ApartmentOutlined class="chip-icon" />
          <span class="chip-label">图谱</span>
        </div>
      </a-tooltip>

      <a-tooltip v-if="showMcp" :title="mcpTooltip">
        <div
          :class="{
            'capability-chip': true,
            active: useMcp,
            disabled: !canMcp
          }"
          role="button"
          :tabindex="canMcp ? 0 : -1"
          :aria-disabled="!canMcp"
          aria-label="切换 MCP"
          @click="emit('toggle-mcp')"
          @keydown.enter.prevent="emit('toggle-mcp')"
          @keydown.space.prevent="emit('toggle-mcp')"
        >
          <ApiOutlined class="chip-icon" />
          <span class="chip-label">MCP</span>
        </div>
      </a-tooltip>

      <a-tooltip v-if="showKnowledgeBase" :title="kbTooltip">
        <a-dropdown
          :disabled="!canKb || databases.length === 0"
          placement="topLeft"
        >
          <div
            :class="{
              'capability-chip': true,
              'capability-chip--dropdown': true,
              active: selectedKbIndex !== null,
              disabled: !canKb || databases.length === 0
            }"
            role="button"
            :tabindex="canKb ? 0 : -1"
            aria-label="选择知识库"
          >
            <BookOutlined class="chip-icon" />
            <span class="chip-label">
              {{ selectedKbIndex === null ? '知识库' : databases[selectedKbIndex]?.name }}
            </span>
            <DownOutlined class="chip-caret" />
          </div>
          <template #overlay>
            <a-menu class="kb-menu">
              <a-menu-item
                v-for="(db, index) in databases"
                :key="index"
                :class="{ 'is-selected': selectedKbIndex === index }"
                @click="emit('select-kb', index)"
              >
                <div class="kb-menu-item">
                  <BookOutlined />
                  <span>{{ db.name }}</span>
                  <CheckOutlined v-if="selectedKbIndex === index" class="check-icon" />
                </div>
              </a-menu-item>
              <a-menu-divider />
              <a-menu-item @click="emit('select-kb', null)">
                <div class="kb-menu-item">
                  <CloseCircleOutlined />
                  <span>不使用知识库</span>
                </div>
              </a-menu-item>
            </a-menu>
          </template>
        </a-dropdown>
      </a-tooltip>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  BookOutlined,
  GlobalOutlined,
  ApiOutlined,
  ApartmentOutlined,
  RobotOutlined,
  DownOutlined,
  CheckOutlined,
  CloseCircleOutlined
} from '@ant-design/icons-vue'

const props = defineProps<{
  backendOnline: boolean
  useAgent: boolean
  canAgent: boolean
  useWeb: boolean
  canWebSearch: boolean
  useGraph: boolean
  canGraph: boolean
  useMcp: boolean
  canMcp: boolean
  selectedKbIndex: number | null
  canKb: boolean
  databases: any[]
  showWebSearch?: boolean
  showKnowledgeGraph?: boolean
  showMcp?: boolean
  showKnowledgeBase?: boolean
}>()

const emit = defineEmits<{
  (e: 'toggle-agent'): void
  (e: 'toggle-web'): void
  (e: 'toggle-graph'): void
  (e: 'toggle-mcp'): void
  (e: 'select-kb', index: number | null): void
}>()

const showWebSearch = computed(() => props.showWebSearch !== false)
const showKnowledgeGraph = computed(() => props.showKnowledgeGraph !== false)
const showMcp = computed(() => props.showMcp !== false)
const showKnowledgeBase = computed(() => props.showKnowledgeBase !== false)

const hasAnyCapability = computed(
  () => showWebSearch.value || showKnowledgeGraph.value || showMcp.value || showKnowledgeBase.value
)

const agentTooltip = computed(() => {
  if (!props.canAgent) return '服务未连接'

  if (props.useAgent) {
    const features = []
    if (props.canWebSearch) features.push('联网')
    if (props.canGraph) features.push('图谱')
    if (props.canMcp) features.push('MCP')

    const featureText = features.length > 0
      ? `（已自动启用: ${features.join('、')}）`
      : ''

    return `智能模式：AI 自动选择最佳检索策略 ${featureText}。点击关闭可手动控制`
  }

  return '开启智能模式（推荐）：AI 将自动决策使用哪些工具'
})
const webTooltip = computed(() => {
  if (props.canWebSearch) return props.useWeb ? '已启用联网搜索' : '启用联网搜索'
  return props.backendOnline ? '后端未启用联网搜索' : '服务未连接'
})
const graphTooltip = computed(() => {
  if (props.canGraph) return props.useGraph ? '已启用知识图谱' : '启用知识图谱'
  return props.backendOnline ? '后端未启用知识图谱' : '服务未连接'
})
const mcpTooltip = computed(() => {
  if (props.canMcp) return props.useMcp ? '已启用 MCP' : '启用 MCP 服务'
  return props.backendOnline ? '后端未启用 MCP' : '服务未连接'
})
const kbTooltip = computed(() => {
  if (props.canKb) return '选择知识库'
  return props.backendOnline ? '后端未启用知识库' : '服务未连接'
})
</script>

<style scoped lang="less">
.capability-bar {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: nowrap;
  overflow-x: auto;

  /* 隐藏滚动条但保留滚动功能 */
  &::-webkit-scrollbar {
    height: 0;
    display: none;
  }
  scrollbar-width: none;
}

.capability-divider {
  width: 1px;
  height: 16px;
  background: var(--gray-200);
  margin: 0 var(--space-1);
  flex-shrink: 0;
}

.capability-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  border: none;
  background: var(--gray-100);
  font-size: var(--font-size-xs);
  color: var(--gray-600);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  user-select: none;
  white-space: nowrap;
  flex-shrink: 0;

  .chip-icon {
    font-size: 12px;
    opacity: 0.7;
  }

  .chip-label {
    font-weight: 500;
  }

  .chip-caret {
    font-size: 10px;
    opacity: 0.5;
    margin-left: 2px;
  }

  &:hover:not(.disabled) {
    color: var(--gray-700);
    background: var(--gray-200);
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
  }

  /* 激活状态：非常柔和的橙色背景 */
  &.active {
    color: var(--primary-color);
    background: rgba(255, 125, 0, 0.1);

    .chip-icon {
      opacity: 1;
    }
  }

  &.disabled {
    opacity: var(--opacity-disabled);
    cursor: not-allowed;
    pointer-events: none;
    background: transparent;
  }

  /* Agent 模式特殊样式 */
  &.capability-chip--agent {
    background: var(--gray-100);

    .chip-status {
      font-size: 10px;
      font-weight: 600;
      padding: 2px 5px;
      border-radius: 3px;
      background: var(--gray-300);
      color: var(--gray-600);
      margin-left: 4px;
      transition: all var(--duration-fast) var(--ease-default);

      &.on {
        background: #FFF7E6;
        color: #FA541C;
      }
    }

    &:hover:not(.disabled) {
      background: var(--gray-200);
    }

    &.active {
      background: #FFF7E6;
      color: #FA541C;

      .chip-icon {
        color: #FA541C;
      }
    }
  }
}

/* 知识库下拉菜单 */
.kb-menu {
  min-width: 180px;

  .kb-menu-item {
    display: flex;
    align-items: center;
    gap: var(--space-2);

    .check-icon {
      margin-left: auto;
      color: var(--primary-color);
    }
  }

  :deep(.ant-dropdown-menu-item) {
    &.is-selected {
      background: color-mix(in srgb, var(--primary-color) 5%, transparent);
      color: var(--primary-color);
    }
  }
}

/* 响应式 */
@media (max-width: 640px) {
  .capability-chip {
    padding: var(--space-1) var(--space-2);

    .chip-label {
      display: none;
    }

    &.capability-chip--agent .chip-label {
      display: inline;
    }

    &.capability-chip--dropdown .chip-label {
      display: inline;
      max-width: 60px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  }

  .capability-divider {
    display: none;
  }
}
</style>
