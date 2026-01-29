<template>
  <div class="capability-bar">
    <!-- 工具选择弹出框 -->
    <a-popover
      v-if="hasAnyCapability"
      v-model:open="popoverOpen"
      trigger="click"
      placement="topLeft"
      overlay-class-name="capability-popover"
    >
      <template #content>
        <div class="capability-grid">
          <!-- 联网搜索 -->
          <div
            v-if="showWebSearch"
            :class="{
              'grid-item': true,
              active: useWeb,
              disabled: !canWebSearch
            }"
            @click="handleToggleWeb"
          >
            <div class="grid-item__icon">
              <GlobalOutlined />
            </div>
            <div class="grid-item__info">
              <span class="grid-item__name">联网搜索</span>
              <span class="grid-item__desc">搜索实时网络信息</span>
            </div>
            <div class="grid-item__check" v-if="useWeb">
              <CheckOutlined />
            </div>
          </div>

          <!-- 知识图谱 -->
          <div
            v-if="showKnowledgeGraph"
            :class="{
              'grid-item': true,
              active: useGraph,
              disabled: !canGraph
            }"
            @click="handleToggleGraph"
          >
            <div class="grid-item__icon">
              <ApartmentOutlined />
            </div>
            <div class="grid-item__info">
              <span class="grid-item__name">知识图谱</span>
              <span class="grid-item__desc">关联知识推理</span>
            </div>
            <div class="grid-item__check" v-if="useGraph">
              <CheckOutlined />
            </div>
          </div>

          <!-- MCP -->
          <div
            v-if="showMcp"
            :class="{
              'grid-item': true,
              active: useMcp,
              disabled: !canMcp
            }"
            @click="handleToggleMcp"
          >
            <div class="grid-item__icon">
              <ApiOutlined />
            </div>
            <div class="grid-item__info">
              <span class="grid-item__name">MCP 服务</span>
              <span class="grid-item__desc">外部工具调用</span>
            </div>
            <div class="grid-item__check" v-if="useMcp">
              <CheckOutlined />
            </div>
          </div>

          <!-- 知识库选择 -->
          <div
            v-if="showKnowledgeBase"
            :class="{
              'grid-item': true,
              'grid-item--has-submenu': true,
              active: selectedKbIndex !== null,
              disabled: !canKb || databases.length === 0
            }"
            @click="toggleKbSubmenu"
          >
            <div class="grid-item__icon">
              <BookOutlined />
            </div>
            <div class="grid-item__info">
              <span class="grid-item__name">知识库</span>
              <span class="grid-item__desc">
                {{ selectedKbIndex !== null ? databases[selectedKbIndex]?.name : '选择知识库' }}
              </span>
            </div>
            <div class="grid-item__arrow">
              <RightOutlined />
            </div>
          </div>

          <!-- 知识库子菜单 -->
          <div v-if="showKbSubmenu && showKnowledgeBase" class="kb-submenu">
            <div
              v-for="(db, index) in databases"
              :key="index"
              :class="{ 'kb-option': true, active: selectedKbIndex === index }"
              @click.stop="selectKb(index)"
            >
              <BookOutlined />
              <span>{{ db.name }}</span>
              <CheckOutlined v-if="selectedKbIndex === index" class="check-icon" />
            </div>
            <div class="kb-option kb-option--clear" @click.stop="selectKb(null)">
              <CloseCircleOutlined />
              <span>不使用知识库</span>
            </div>
          </div>
        </div>
      </template>

      <!-- 触发按钮: + 号 -->
      <a-tooltip title="选择工具">
        <div
          :class="{
            'capability-trigger': true,
            'has-active': hasActiveTools
          }"
          role="button"
          tabindex="0"
          aria-label="选择工具"
        >
          <PlusOutlined class="trigger-icon" />
          <span v-if="hasActiveTools" class="active-count">{{ activeToolCount }}</span>
        </div>
      </a-tooltip>
    </a-popover>

    <!-- 智能模式主开关 — 始终可见 -->
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
        aria-label="切换智能模式"
        @click="handleToggleAgent"
        @keydown.enter.prevent="handleToggleAgent"
        @keydown.space.prevent="handleToggleAgent"
      >
        <RobotOutlined class="chip-icon" />
        <span class="chip-label">智能模式</span>
        <span class="chip-status" :class="{ on: useAgent }">
          {{ useAgent ? 'ON' : 'OFF' }}
        </span>
      </div>
    </a-tooltip>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  BookOutlined,
  GlobalOutlined,
  ApiOutlined,
  ApartmentOutlined,
  RobotOutlined,
  CheckOutlined,
  CloseCircleOutlined,
  PlusOutlined,
  RightOutlined
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

// Popover state
const popoverOpen = ref(false)
const showKbSubmenu = ref(false)

const showWebSearch = computed(() => props.showWebSearch !== false)
const showKnowledgeGraph = computed(() => props.showKnowledgeGraph !== false)
const showMcp = computed(() => props.showMcp !== false)
const showKnowledgeBase = computed(() => props.showKnowledgeBase !== false)

const hasAnyCapability = computed(
  () => showWebSearch.value || showKnowledgeGraph.value || showMcp.value || showKnowledgeBase.value
)

// Count active tools for badge (only counts tools in the popover, not Agent chip)
const activeToolCount = computed(() => {
  let count = 0
  if (props.useWeb) count++
  if (props.useGraph) count++
  if (props.useMcp) count++
  if (props.selectedKbIndex !== null) count++
  return count
})

const hasActiveTools = computed(() => activeToolCount.value > 0)

// Toggle handlers
const handleToggleWeb = () => {
  if (props.canWebSearch) {
    emit('toggle-web')
  }
}

const handleToggleGraph = () => {
  if (props.canGraph) {
    emit('toggle-graph')
  }
}

const handleToggleMcp = () => {
  if (props.canMcp) {
    emit('toggle-mcp')
  }
}

const handleToggleAgent = () => {
  if (props.canAgent) {
    emit('toggle-agent')
  }
}

const toggleKbSubmenu = () => {
  if (props.canKb && props.databases.length > 0) {
    showKbSubmenu.value = !showKbSubmenu.value
  }
}

const selectKb = (index: number | null) => {
  emit('select-kb', index)
  showKbSubmenu.value = false
}

const agentTooltip = computed(() => {
  if (!props.canAgent) return '服务未连接'

  if (props.useAgent) {
    const features = []
    if (props.useWeb) features.push('联网')
    if (props.useGraph) features.push('图谱')
    if (props.useMcp) features.push('MCP')

    const featureText = features.length > 0 ? `（允许使用: ${features.join('、')}）` : '（未允许外部工具）'
    return `智能模式：AI 自动决策使用哪些工具 ${featureText}。点击关闭可手动控制`
  }

  return '开启智能模式（推荐）：AI 将自动决策使用哪些工具'
})
</script>

<style scoped lang="less">
.capability-bar {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: nowrap;
}

/* 触发按钮: + 号 */
.capability-trigger {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  border: 1px solid var(--gray-200);
  background: var(--gray-50);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  flex-shrink: 0;

  .trigger-icon {
    font-size: 16px;
    color: var(--gray-600);
    transition: all var(--duration-fast) var(--ease-default);
  }

  .active-count {
    position: absolute;
    top: -4px;
    right: -4px;
    min-width: 16px;
    height: 16px;
    padding: 0 4px;
    border-radius: 8px;
    background: var(--primary-color);
    color: white;
    font-size: 10px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  &:hover {
    background: color-mix(in srgb, var(--primary-color) 10%, var(--surface-color));
    border-color: var(--primary-color);

    .trigger-icon {
      color: var(--primary-color);
    }
  }

  &.has-active {
    border-color: var(--primary-color);
    background: color-mix(in srgb, var(--primary-color) 8%, var(--surface-color));

    .trigger-icon {
      color: var(--primary-color);
    }
  }
}

/* 智能模式芯片 */
.capability-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--gray-200);
  background: var(--gray-50);
  font-size: var(--font-size-xs);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  user-select: none;
  white-space: nowrap;
  flex-shrink: 0;

  .chip-icon {
    font-size: 15px;
    color: var(--gray-700);
  }

  .chip-label {
    font-weight: 600;
    color: var(--text-color);
  }

  &:hover:not(.disabled) {
    background: color-mix(in srgb, var(--primary-color) 10%, var(--surface-color));
    border-color: var(--primary-color);

    .chip-icon,
    .chip-label {
      color: var(--primary-color);
    }
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
  }

  &.active {
    background: color-mix(in srgb, var(--primary-color) 12%, var(--surface-color));
    border-color: var(--primary-color);

    .chip-icon,
    .chip-label {
      color: var(--primary-color);
    }
  }

  &.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    pointer-events: none;
  }

  /* Agent 模式特殊样式 */
  &.capability-chip--agent {
    .chip-status {
      font-size: 10px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: var(--gray-100);
      color: var(--gray-600);
      margin-left: 4px;
      transition: all var(--duration-fast) var(--ease-default);

      &.on {
        background: color-mix(in srgb, var(--primary-color) 18%, var(--surface-color));
        color: var(--primary-color);
      }
    }

    &.active {
      background: color-mix(in srgb, var(--primary-color) 10%, var(--surface-color));
      border-color: var(--primary-color);

      .chip-icon,
      .chip-label {
        color: var(--primary-color);
      }
    }
  }
}

/* Popover 内的网格布局 */
.capability-grid {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  min-width: 220px;
  max-width: 280px;
}

.grid-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);
  background: transparent;

  &:hover:not(.disabled) {
    background: var(--hover-bg);
  }

  &.active {
    background: color-mix(in srgb, var(--primary-color) 10%, transparent);

    .grid-item__icon {
      color: var(--primary-color);
      background: color-mix(in srgb, var(--primary-color) 15%, transparent);
    }

    .grid-item__name {
      color: var(--primary-color);
    }
  }

  &.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .grid-item__icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    background: var(--gray-100);
    color: var(--gray-600);
    font-size: 18px;
    flex-shrink: 0;
    transition: all var(--duration-fast) var(--ease-default);
  }

  .grid-item__info {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 0;
  }

  .grid-item__name {
    font-size: var(--font-size-sm);
    font-weight: 600;
    color: var(--text-color);
  }

  .grid-item__desc {
    font-size: var(--font-size-xs);
    color: var(--gray-500);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .grid-item__check {
    color: var(--primary-color);
    font-size: 14px;
    flex-shrink: 0;
  }

  .grid-item__arrow {
    color: var(--gray-400);
    font-size: 12px;
    flex-shrink: 0;
  }
}

/* 知识库子菜单 */
.kb-submenu {
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.kb-option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: var(--font-size-sm);
  color: var(--text-color);
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--hover-bg);
  }

  &.active {
    background: color-mix(in srgb, var(--primary-color) 10%, transparent);
    color: var(--primary-color);
  }

  .check-icon {
    margin-left: auto;
    color: var(--primary-color);
  }

  &.kb-option--clear {
    color: var(--gray-500);
    border-top: 1px solid var(--border-color);
    margin-top: var(--space-1);
    padding-top: var(--space-2);
  }
}

/* 响应式 */
@media (max-width: 640px) {
  .capability-chip {
    padding: var(--space-1) var(--space-2);

    &.capability-chip--agent .chip-label {
      display: none;
    }
  }
}
</style>

<!-- Popover 全局样式 -->
<style lang="less">
.capability-popover {
  .ant-popover-inner {
    padding: var(--space-3);
    border-radius: var(--radius-lg);
    background: color-mix(in srgb, var(--surface-color) 98%, transparent);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    box-shadow:
      0 4px 20px rgba(0, 0, 0, 0.08),
      0 8px 40px rgba(0, 0, 0, 0.04);
  }

  .ant-popover-arrow {
    display: none;
  }
}
</style>
