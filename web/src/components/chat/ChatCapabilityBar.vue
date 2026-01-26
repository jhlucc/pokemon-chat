<template>
  <div class="capability-bar">
    <a-tooltip :title="agentTooltip">
      <div
        :class="{
          'opt-item': true,
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
        <RobotOutlined style="margin-right: 3px" />
        Agent 模式
      </div>
    </a-tooltip>

    <template v-if="!useAgent">
      <a-tooltip v-if="showWebSearch" :title="webTooltip">
        <div
          :class="{
            'opt-item': true,
            active: useWeb,
            disabled: !canWebSearch
          }"
          role="button"
          :tabindex="canWebSearch ? 0 : -1"
          :aria-disabled="!canWebSearch"
          aria-label="切换 联网搜索"
          @click="emit('toggle-web')"
          @keydown.enter.prevent="emit('toggle-web')"
          @keydown.space.prevent="emit('toggle-web')"
        >
          <CompassOutlined style="margin-right: 3px" />
          联网搜索
        </div>
      </a-tooltip>

      <a-tooltip v-if="showKnowledgeGraph" :title="graphTooltip">
        <div
          :class="{
            'opt-item': true,
            active: useGraph,
            disabled: !canGraph
          }"
          role="button"
          :tabindex="canGraph ? 0 : -1"
          :aria-disabled="!canGraph"
          aria-label="切换 知识图谱"
          @click="emit('toggle-graph')"
          @keydown.enter.prevent="emit('toggle-graph')"
          @keydown.space.prevent="emit('toggle-graph')"
        >
          <DeploymentUnitOutlined style="margin-right: 3px" />
          知识图谱
        </div>
      </a-tooltip>

      <a-tooltip v-if="showMcp" :title="mcpTooltip">
        <div
          :class="{
            'opt-item': true,
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
          <DatabaseOutlined style="margin-right: 3px" />
          MCP
        </div>
      </a-tooltip>

      <a-tooltip v-if="showKnowledgeBase" :title="kbTooltip">
        <a-dropdown
          :disabled="!canKb || databases.length === 0"
          :class="{
            'opt-item': true,
            active: selectedKbIndex !== null,
            disabled: !canKb || databases.length === 0
          }"
        >
          <a class="ant-dropdown-link" @click.prevent aria-label="选择知识库">
            <BookOutlined style="margin-right: 3px" />
            <span class="text">
              {{
                selectedKbIndex === null
                  ? databases.length > 0
                    ? '不使用知识库'
                    : '暂无知识库'
                  : databases[selectedKbIndex]?.name
              }}
            </span>
          </a>
          <template #overlay>
            <a-menu>
              <a-menu-item v-for="(db, index) in databases" :key="index" :disabled="!canKb" @click="emit('select-kb', index)">
                <a href="javascript:;">{{ db.name }}</a>
              </a-menu-item>
              <a-menu-item :disabled="!canKb" @click="emit('select-kb', null)">
                <a href="javascript:;">不使用</a>
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
  CompassOutlined,
  DatabaseOutlined,
  DeploymentUnitOutlined,
  RobotOutlined
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

const agentTooltip = computed(() =>
  !props.canAgent
    ? '服务未连接'
    : props.useAgent
      ? 'Agent 模式已开启：由 supervisor_agent 自动路由到子 worker（无需手动选择）'
      : '开启总 Agent（supervisor_agent）统一编排'
)
const webTooltip = computed(() => {
  if (props.canWebSearch) return ''
  return props.backendOnline ? '后端未启用联网搜索' : '服务未连接'
})
const graphTooltip = computed(() => {
  if (props.canGraph) return ''
  return props.backendOnline ? '后端未启用知识图谱' : '服务未连接'
})
const mcpTooltip = computed(() => {
  if (props.canMcp) return ''
  return props.backendOnline ? '后端未启用 MCP' : '服务未连接'
})
const kbTooltip = computed(() => {
  if (props.canKb) return ''
  return props.backendOnline ? '后端未启用知识库' : '服务未连接'
})
</script>

<style scoped lang="less">
.capability-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.ant-dropdown-link {
  color: inherit;
}
</style>
