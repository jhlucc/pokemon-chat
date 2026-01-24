<template>
  <div class="graph-page layout-container">
    <HeaderComponent title="知识图谱" :description="graphDescription">
      <template #actions>
        <div class="status-line">
          <a-tag :color="backendStatus.color">
            {{ backendStatus.label }}
          </a-tag>
          <a-tag :color="kgStatus.color">
            {{ kgStatus.label }}
          </a-tag>
        </div>
      </template>
    </HeaderComponent>

    <a-alert
      v-if="backendMock"
      type="info"
      show-icon
      message="当前为 Mock 演示图谱：数据来自本地 Demo / Mock API。"
      style="margin: 0 24px 16px;"
    />
    <a-alert
      v-else-if="!canUseGraph"
      type="warning"
      show-icon
      :message="backendOnline ? '后端未启用知识图谱（enable_knowledge_graph=false）' : '后端未启动/不可用：已切换为离线 Demo 图谱'"
      style="margin: 0 24px 16px;"
    />

    <div class="toolbar">
      <div class="left">
        <a-input
          v-model:value="state.searchInput"
          placeholder="输入要查询的实体（如：皮卡丘）"
          style="width: 260px"
          @pressEnter="onSearch"
        />
        <a-button
          type="primary"
          :loading="state.searchLoading"
          :disabled="!canUseGraph || state.searchLoading"
          @click="onSearch"
        >
          检索关系
        </a-button>
        <a-button @click="loadDemoGraph">加载 Demo</a-button>
      </div>
      <div class="right">
        <a-input-number v-model:value="sampleNodeCount" :min="20" :max="500" />
        <a-button :loading="state.fetching" :disabled="!canUseGraph || state.fetching" @click="loadSampleNodes">
          采样节点
        </a-button>
        <a-select v-model:value="state.layout" style="width: 140px" @change="renderGraph">
          <a-select-option value="force">力导向</a-select-option>
          <a-select-option value="radial">径向</a-select-option>
        </a-select>
      </div>
    </div>

    <div class="canvas-wrap">
      <div class="main" ref="container" v-show="graphData.nodes.length > 0"></div>
      <a-empty v-show="graphData.nodes.length === 0" style="padding: 4rem 0;" />
    </div>

    <a-drawer v-model:open="state.detailOpen" placement="right" width="380" title="节点详情">
      <div v-if="selectedNode">
        <div class="kv"><span class="k">名称</span><span class="v">{{ selectedNode.name }}</span></div>
        <div class="kv"><span class="k">ID</span><span class="v">{{ selectedNode.id }}</span></div>
        <div class="kv"><span class="k">度</span><span class="v">{{ selectedNodeDegree }}</span></div>
        <a-divider />
        <div class="section-title">邻居</div>
        <div v-if="neighborNodes.length === 0" class="muted">暂无邻居</div>
        <a-list v-else :data-source="neighborNodes" size="small">
          <template #renderItem="{ item }">
            <a-list-item class="neighbor-item" @click="focusNode(item.id)">
              <span class="name">{{ item.name }}</span>
              <span class="muted">({{ item.id }})</span>
            </a-list-item>
          </template>
        </a-list>
      </div>
      <div v-else class="muted">点击画布中的节点查看详情</div>
    </a-drawer>
  </div>
</template>

<script setup>
 import { Graph } from '@antv/g6';
 import { computed, reactive, ref, onMounted, onUnmounted, watch } from 'vue';
 import { message } from 'ant-design-vue';
 import { useConfigStore } from '@/stores/config';
 import HeaderComponent from '@/components/HeaderComponent.vue';
 import { apiFetch } from '@/api/http';
 import demoGraph from '@/assets/mock/graph.sample.json';

const configStore = useConfigStore();

 const container = ref(null);
const sampleNodeCount = ref(100);
const graphData = reactive({ nodes: [], edges: [] });

 let graphInstance;
 let layoutKey = null;
 let resizeObserver = null;

const state = reactive({
  fetching: false,
  loadingGraphInfo: false,
  searchInput: '',
  searchLoading: false,
  detailOpen: false,
  selectedNodeId: null,
  layout: 'force',
});

const cssVar = (name, fallback) => {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
};

const backendOnline = computed(() => Boolean(configStore.config.backend?.online));
const backendMock = computed(() => Boolean(configStore.config.backend?.mock));
const canUseGraph = computed(() => (backendOnline.value && Boolean(configStore.config.enable_knowledge_graph)) || backendMock.value);

const backendStatus = computed(() => {
  if (backendMock.value) return { color: 'blue', label: 'Mock' };
  if (backendOnline.value) return { color: 'green', label: 'Backend Online' };
  return { color: 'red', label: 'Offline' };
});

const kgStatus = computed(() => {
  if (backendMock.value) return { color: 'blue', label: 'Demo' };
  if (backendOnline.value && Boolean(configStore.config.enable_knowledge_graph)) return { color: 'green', label: 'KG Enabled' };
  if (backendOnline.value) return { color: 'orange', label: 'KG Disabled' };
  return { color: 'orange', label: 'KG Offline' };
});

const graphInfo = ref({});

const selectedNode = computed(() => {
  if (!state.selectedNodeId) return null;
  return graphData.nodes.find((n) => n.id === state.selectedNodeId) || null;
});

const selectedNodeDegree = computed(() => {
  if (!state.selectedNodeId) return 0;
  let d = 0;
  graphData.edges.forEach((e) => {
    if (e.source_id === state.selectedNodeId || e.target_id === state.selectedNodeId) d += 1;
  });
  return d;
});

const neighborNodes = computed(() => {
  if (!state.selectedNodeId) return [];
  const neighborIds = new Set();
  graphData.edges.forEach((e) => {
    if (e.source_id === state.selectedNodeId) neighborIds.add(e.target_id);
    if (e.target_id === state.selectedNodeId) neighborIds.add(e.source_id);
  });
  return graphData.nodes.filter((n) => neighborIds.has(n.id));
});

const loadGraphInfo = async () => {
  state.loadingGraphInfo = true;
  try {
    if (!canUseGraph.value) {
      graphInfo.value = {
        status: backendOnline.value ? 'closed' : 'offline',
        graph_name: backendOnline.value ? 'neo4j' : 'demo',
        entity_count: graphData.nodes.length,
        relationship_count: graphData.edges.length,
      };
      return;
    }

    graphInfo.value = await apiFetch('/data/graph', { method: 'GET' });
  } catch {
    graphInfo.value = { status: 'closed' };
  } finally {
    state.loadingGraphInfo = false;
  }
};

 const loadDemoGraph = () => {
    graphData.nodes = demoGraph.nodes;
    graphData.edges = demoGraph.edges;
  state.selectedNodeId = null;
  state.detailOpen = false;
    loadGraphInfo();
    setTimeout(() => renderGraph(), 0);
  };

const loadSampleNodes = async () => {
  if (!canUseGraph.value) return loadDemoGraph();
  state.fetching = true;
  try {
    const data = await apiFetch('/data/graph/nodes', {
      method: 'GET',
      query: { kgdb_name: 'neo4j', num: sampleNodeCount.value },
    });
    graphData.nodes = data?.result?.nodes || [];
    graphData.edges = data?.result?.edges || [];
    state.selectedNodeId = null;
    state.detailOpen = false;
    await loadGraphInfo();
    setTimeout(() => renderGraph(), 0);
  } catch (e) {
    message.error(e?.message || '获取节点失败');
  } finally {
    state.fetching = false;
  }
};

const onSearch = async () => {
  if (!state.searchInput) return message.error('请输入要查询的实体');
  if (!canUseGraph.value) return loadDemoGraph();

  state.searchLoading = true;
  try {
    const data = await apiFetch('/data/graph/node', {
      method: 'GET',
      query: { entity_name: state.searchInput },
    });
    graphData.nodes = data?.result?.nodes || [];
    graphData.edges = data?.result?.edges || [];
    if (graphData.nodes.length === 0) message.info('未找到相关实体');
    state.selectedNodeId = null;
    state.detailOpen = false;
    await loadGraphInfo();
    setTimeout(() => renderGraph(), 0);
  } catch (e) {
    message.error(e?.message || '检索失败');
  } finally {
    state.searchLoading = false;
  }
};

const getG6Data = () => {
  const degree = {};
  graphData.nodes.forEach((n) => (degree[n.id] = 0));
  graphData.edges.forEach((e) => {
    degree[e.source_id] = (degree[e.source_id] || 0) + 1;
    degree[e.target_id] = (degree[e.target_id] || 0) + 1;
  });

  return {
    nodes: graphData.nodes.map((n) => ({
      id: n.id,
      data: { label: n.name, degree: degree[n.id] || 0 },
    })),
    edges: graphData.edges.map((e) => ({
      id: `${e.source_id}-${e.type}-${e.target_id}`,
      source: e.source_id,
      target: e.target_id,
      data: { label: e.type },
    })),
  };
};

 const ensureGraph = () => {
   if (!container.value) return;

   const key = state.layout;
   const primary = cssVar('--primary-color', '#1677ff');
   const surface = cssVar('--surface-color', '#ffffff');
   const text = cssVar('--text-color', '#000000');
   const edgeStroke = cssVar('--gray-500', '#999');

   const layout =
     key === 'radial'
       ? { type: 'radial', unitRadius: 120, preventOverlap: true }
       : { type: 'd3-force', preventOverlap: true, collide: { radius: 40, strength: 0.6 } };

   // Only recreate when switching layout types (layout isn't always hot-swappable).
   if (!graphInstance || layoutKey !== key) {
     if (graphInstance) {
       try {
         graphInstance.destroy();
       } catch {
         // ignore
       }
       graphInstance = null;
     }

     graphInstance = new Graph({
       container: container.value,
       width: container.value.offsetWidth,
       height: container.value.offsetHeight,
       autoFit: true,
       layout,
       node: {
         type: 'circle',
         style: {
           labelText: (d) => d.data.label,
           size: (d) => Math.min(18 + (d.data.degree || 0) * 5, 52),
           labelFill: text,
           fill: (d) => (d.id === state.selectedNodeId ? primary : surface),
           stroke: primary,
           lineWidth: 2,
         },
       },
       edge: {
         type: 'line',
         style: {
           labelText: (d) => d.data.label,
           labelFill: text,
           labelBackground: surface,
           endArrow: true,
           stroke: edgeStroke,
         },
       },
       behaviors: ['drag-element', 'zoom-canvas', 'drag-canvas'],
     });

     graphInstance.on('node:click', (evt) => {
       const id =
         evt?.item?.getID?.() ||
         evt?.item?.id ||
         evt?.data?.id ||
         evt?.target?.id ||
         null;
       if (!id) return;
       state.selectedNodeId = id;
       state.detailOpen = true;
       // Re-render to update selected styling.
       setTimeout(() => renderGraph(), 0);
     });

     layoutKey = key;
   } else {
     // Best-effort resize when the container size changes.
     try {
       graphInstance.resize?.(container.value.offsetWidth, container.value.offsetHeight);
     } catch {
       // ignore
     }
   }
 };

 const renderGraph = () => {
   if (!container.value || graphData.nodes.length === 0) return;
   ensureGraph();
   graphInstance.setData(getG6Data());
   graphInstance.render();
 };

 const focusNode = (id) => {
   state.selectedNodeId = id;
   state.detailOpen = true;
   setTimeout(() => renderGraph(), 0);
 };

const graphDescription = computed(() => {
  const { graph_name, entity_count, relationship_count } = graphInfo.value || {};
  return `${graph_name || ''} - 共 ${entity_count || graphData.nodes.length || 0} 实体，${relationship_count || graphData.edges.length || 0} 个关系。`;
});

onMounted(() => {
  // In offline/mock mode, show a demo graph by default so the page is not empty.
  if (configStore.config?.backend?.mock || !backendOnline.value) {
    loadDemoGraph();
  }

  // Resize graph when the container resizes (e.g. drawer open/close, window resize).
  watch(
    () => container.value,
    (el) => {
      try {
        resizeObserver?.disconnect?.();
      } catch {
        // ignore
      }
      resizeObserver = null;
      if (!el || typeof ResizeObserver === 'undefined') return;

      resizeObserver = new ResizeObserver(() => {
        if (!graphInstance || !container.value) return;
        try {
          graphInstance.resize?.(container.value.offsetWidth, container.value.offsetHeight);
          graphInstance.render?.();
        } catch {
          // ignore
        }
      });
      resizeObserver.observe(el);
    },
    { immediate: true }
  );
});

onUnmounted(() => {
  try {
    resizeObserver?.disconnect?.();
  } catch {
    // ignore
  }
  resizeObserver = null;

  if (graphInstance) {
    try {
      graphInstance.destroy();
    } catch {
      // ignore
    }
    graphInstance = null;
  }
  layoutKey = null;
});
</script>

<style lang="less" scoped>
.graph-page {
  padding: 0;
}

.status-line {
  display: flex;
  gap: 8px;
  align-items: center;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  margin: 16px 0;
  padding: 0 24px;
  gap: 12px;

  .left,
  .right {
    display: flex;
    gap: 10px;
    align-items: center;
  }
}

.canvas-wrap {
  padding: 0 24px 24px;
}

.main {
  background: var(--surface-color);
  border-radius: 16px;
  width: 100%;
  height: calc(100vh - 240px);
  overflow: hidden;
}

.kv {
  display: flex;
  justify-content: space-between;
  margin: 6px 0;

  .k {
    color: var(--gray-700);
  }
  .v {
    font-weight: 600;
  }
}

.section-title {
  font-weight: 600;
  margin-bottom: 8px;
}

.neighbor-item {
  cursor: pointer;
}

.muted {
  color: var(--gray-600);
}
</style>
