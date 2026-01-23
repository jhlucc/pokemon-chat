<template>
  <div class="database-empty" v-if="!state.showPage">
    <a-empty>
      <template #description>
        <span>
          前往 <router-link to="/setting" style="color: var(--main-color); font-weight: bold;">设置</router-link> 页面启用知识图谱。
        </span>
      </template>
    </a-empty>
  </div>
  <div class="graph-container layout-container" v-else>
    <HeaderComponent
      title="图数据库"
      :description="graphDescription"
    >
      <template #actions>
        <div class="status-wrapper">
          <div class="status-indicator" :class="graphStatusClass"></div>
        </div>
      </template>
    </HeaderComponent>

    <div class="actions">
      <div class="actions-left">
        <input
          v-model="state.searchInput"
          placeholder="输入要查询的实体"
          style="width: 200px"
          @keydown.enter="onSearch"
        />
        <a-button
          type="primary"
          :loading="state.searchLoading"
          :disabled="state.searchLoading"
          @click="onSearch"
        >
          检索实体关系
        </a-button>
      </div>
      <div class="actions-right">
        <input v-model="sampleNodeCount">
        <a-button @click="loadSampleNodes" :loading="state.fetching">获取节点</a-button>
      </div>
    </div>
    <div class="main" id="container" ref="container" v-show="graphData.nodes.length > 0"></div>
    <a-empty v-show="graphData.nodes.length === 0" style="padding: 4rem 0;"/>
  </div>
</template>

<script setup>
import { Graph } from "@antv/g6";
import { computed, onMounted, reactive, ref } from 'vue';
import { message } from 'ant-design-vue';
import { useConfigStore } from '@/stores/config';
import HeaderComponent from '@/components/HeaderComponent.vue';

const configStore = useConfigStore()
const container = ref(null);
const sampleNodeCount = ref(100);
const graphData = reactive({ nodes: [], edges: [] });
let graphInstance

const state = reactive({
  fetching: false,
  loadingGraphInfo: false,
  searchInput: '',
  searchLoading: false,
  showPage: computed(() => configStore.config.enable_knowledge_base && configStore.config.enable_knowledge_graph),
})

const graphInfo = ref({})

const loadGraphInfo = () => {
  state.loadingGraphInfo = true
  fetch('/api/data/graph')
    .then(response => response.json())
    .then(data => {
      graphInfo.value = data
    })
    .catch(error => {
      message.error(error.message)
    })
    .finally(() => state.loadingGraphInfo = false)
}

const loadSampleNodes = () => {
  state.fetching = true
  fetch(`/api/data/graph/nodes?kgdb_name=neo4j&num=${sampleNodeCount.value}`)
    .then(res => res.json())
    .then(data => {
      graphData.nodes = data.result.nodes
      graphData.edges = data.result.edges
      setTimeout(() => renderGraph(), 500)
    })
    .catch(err => message.error(err.message))
    .finally(() => state.fetching = false)
}

const onSearch = () => {
  if (!state.searchInput) return message.error('请输入要查询的实体')
  state.searchLoading = true
  fetch(`/api/data/graph/node?entity_name=${state.searchInput}`)
    .then(res => res.json())
    .then(data => {
      graphData.nodes = data.result.nodes
      graphData.edges = data.result.edges
      if (graphData.nodes.length === 0) message.info('未找到相关实体')
      renderGraph()
    })
    .catch(err => message.error(err.message))
    .finally(() => state.searchLoading = false)
}

const getGraphData = () => {
  const nodeDegrees = {};
  graphData.nodes.forEach(n => nodeDegrees[n.id] = 0);
  graphData.edges.forEach(e => {
    nodeDegrees[e.source_id]++;
    nodeDegrees[e.target_id]++;
  });
  return {
    nodes: graphData.nodes.map(n => ({
      id: n.id,
      data: { label: n.name, degree: nodeDegrees[n.id] }
    })),
    edges: graphData.edges.map(e => ({
      source: e.source_id,
      target: e.target_id,
      data: { label: e.type }
    }))
  }
}

const renderGraph = () => {
  if (graphInstance) graphInstance.destroy();
  graphInstance = new Graph({
    container: container.value,
    width: container.value.offsetWidth,
    height: container.value.offsetHeight,
    autoFit: true,
    layout: {
      type: 'd3-force',
      preventOverlap: true,
      collide: { radius: 40, strength: 0.5 },
    },
    node: {
      type: 'circle',
      style: {
        labelText: d => d.data.label,
        size: d => Math.min(15 + d.data.degree * 5, 50),
      },
    },
    edge: {
      type: 'line',
      style: {
        labelText: d => d.data.label,
        labelBackground: '#fff',
        endArrow: true,
      },
    },
    behaviors: ['drag-element', 'zoom-canvas', 'drag-canvas'],
  });
  graphInstance.setData(getGraphData());
  graphInstance.render();
}

const graphStatusClass = computed(() => {
  if (state.loadingGraphInfo) return 'loading';
  return graphInfo.value?.status === 'open' ? 'open' : 'closed';
});

const graphDescription = computed(() => {
  const { graph_name, entity_count, relationship_count } = graphInfo.value || {}
  return `${graph_name || ''} - 共 ${entity_count || 0} 实体，${relationship_count || 0} 个关系。`
});

onMounted(() => {
  loadGraphInfo()
  loadSampleNodes()
})
</script>

<style lang="less" scoped>
.graph-container { padding: 0; }
.status-wrapper {
  display: flex;
  align-items: center;
  margin-right: 16px;
  font-size: 14px;
  color: rgba(0, 0, 0, 0.65);
}
.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  &.loading { background: #faad14; animation: pulse 1.5s infinite ease-in-out; }
  &.open { background: #52c41a; }
  &.closed { background: #f5222d; }
}
@keyframes pulse {
  0% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(0.8); opacity: 0.5; }
}
.actions {
  display: flex;
  justify-content: space-between;
  margin: 20px 0;
  padding: 0 24px;
  .actions-left, .actions-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  input {
    width: 100px;
    border-radius: 8px;
    padding: 4px 12px;
    border: 2px solid var(--main-300);
    height: 42px;
    &:focus { border-color: var(--main-color); }
  }
  button {
    border-width: 2px;
    height: 40px;
    box-shadow: none;
  }
}
#container {
  background: #F7F7F7;
  margin: 20px 24px;
  border-radius: 16px;
  width: calc(100% - 48px);
  height: calc(100vh - 200px);
  overflow: hidden;
}
.database-empty {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  flex-direction: column;
  color: var(--gray-900);
}
</style>
