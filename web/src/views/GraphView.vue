<template>
  <div class="database-empty" v-if="!state.showPage">
    <a-empty>
      <template #description>
        <span>
          前往 <router-link to="/setting" style="color: var(--primary-color); font-weight: bold;">设置</router-link> 页面启用知识图谱。
        </span>
      </template>
    </a-empty>
  </div>
  <div class="graph-container" v-else>
    <HeaderComponent
      title="图数据库"
      :description="graphDescription"
    >
      <template #actions>
        <div class="header-actions">
            <div class="status-wrapper">
              <div class="status-indicator" :class="graphStatusClass"></div>
              <span class="status-text">{{ graphStatusText }}</span>
            </div>
            
             <a-button class="icon-btn" @click="loadSampleNodes" :loading="state.fetching" title="刷新样本">
                <template #icon><ReloadOutlined /></template>
            </a-button>
        </div>
      </template>
    </HeaderComponent>

    <div class="main-content">
        <!-- Floating Toolbar -->
        <div class="toolbar window-card glass">
            <div class="search-box">
                 <a-input-search
                  v-model:value="state.searchInput"
                  placeholder="搜索实体..."
                  class="custom-search"
                  @search="onSearch"
                  :loading="state.searchLoading"
                  allowClear
                />
            </div>
            <div class="divider"></div>
            <div class="tool-item">
                <span class="label">节点数:</span>
                <a-input-number v-model:value="sampleNodeCount" :min="10" :max="500" size="small" :bordered="false" class="count-input"/>
            </div>
            <div class="divider"></div>
            <a-tooltip title="导出当前视图数据">
                 <a-button type="text" size="small" @click="exportData" class="tool-btn">
                    <ExportOutlined />
                </a-button>
            </a-tooltip>
        </div>

        <div class="canvas-wrapper" ref="wrapper">
             <div id="container" ref="container"></div>
             <!-- Empty State Overlay -->
             <div v-if="graphData.nodes.length === 0 && !state.fetching" class="empty-overlay">
                <a-empty description="暂无数据" />
                <p class="sub-text">请尝试调整搜索条件或刷新图谱</p>
             </div>
             
             <!-- Loading Overlay -->
             <div v-if="state.fetching || state.searchLoading" class="loading-overlay">
                 <div class="loading-content">
                     <a-spin size="large" />
                     <span class="loading-text">正在分析图谱关联...</span>
                 </div>
             </div>
        </div>
        
        <!-- Detail Panel -->
        <GraphDetailPanel 
            :visible="detailState.visible" 
            :item="detailState.item" 
            :type="detailState.type"
            @close="closeDetail"
        />
    </div>
  </div>
</template>

<script setup>
import { Graph, register, ExtensionCategory, Rect } from "@antv/g6";
import { computed, onMounted, reactive, ref, onBeforeUnmount } from 'vue';
import { message } from 'ant-design-vue';
import { ReloadOutlined, ExportOutlined } from '@ant-design/icons-vue';
import { useConfigStore } from '@/stores/config';
import HeaderComponent from '@/components/HeaderComponent.vue';
import GraphDetailPanel from '@/components/GraphDetailPanel.vue';
import axios from 'axios';

const configStore = useConfigStore()
const container = ref(null);
const wrapper = ref(null);
const sampleNodeCount = ref(80);
const graphData = reactive({ nodes: [], edges: [] });
let graphInstance = null

const state = reactive({
  fetching: false,
  loadingGraphInfo: false,
  searchInput: '',
  searchLoading: false,
  showPage: computed(() => configStore.config.enable_knowledge_base && configStore.config.enable_knowledge_graph),
})

const detailState = reactive({
    visible: false,
    item: null,
    type: 'node'
})

const graphInfo = ref({})

const loadGraphInfo = async () => {
  state.loadingGraphInfo = true
  try {
     const response = await axios.get('/api/data/graph');
     graphInfo.value = response.data;
  } catch (error) {
      console.warn("Failed to load graph info", error);
  } finally {
      state.loadingGraphInfo = false;
  }
}

const loadSampleNodes = async () => {
  state.fetching = true
  closeDetail();
  
  try {
    const response = await axios.get('/api/data/graph/nodes', {
        params: { kgdb_name: 'neo4j', num: sampleNodeCount.value }
    });
    
    const data = response.data;
    if(data.result) {
        graphData.nodes = data.result.nodes || []
        graphData.edges = data.result.edges || []
    } else {
        graphData.nodes = []
        graphData.edges = []
    }
    renderGraph();
  } catch (err) {
      message.error(err.response?.data?.message || err.message);
  } finally {
      state.fetching = false;
  }
}

const onSearch = async () => {
  if (!state.searchInput) return loadSampleNodes();
  state.searchLoading = true
  closeDetail();
  
  try {
      const response = await axios.get('/api/data/graph/node', {
          params: { entity_name: state.searchInput }
      });
      const data = response.data;
      if (data.result) {
          graphData.nodes = data.result.nodes || []
          graphData.edges = data.result.edges || []
          if (graphData.nodes.length === 0) message.info('未找到相关实体')
          renderGraph()
      }
  } catch (err) {
       message.error(err.response?.data?.message || err.message);
  } finally {
      state.searchLoading = false
  }
}

const getGraphData = () => {
  const nodeDegrees = {};
  graphData.nodes.forEach(n => nodeDegrees[n.id] = 0);
  graphData.edges.forEach(e => {
    nodeDegrees[e.source_id] = (nodeDegrees[e.source_id] || 0) + 1;
    nodeDegrees[e.target_id] = (nodeDegrees[e.target_id] || 0) + 1;
  });
  
  return {
    nodes: graphData.nodes.map(n => ({
      id: n.id,
      data: { 
          label: n.name, 
          degree: nodeDegrees[n.id],
          original: n 
      }
    })),
    edges: graphData.edges.map(e => ({
      source: e.source_id,
      target: e.target_id,
      data: { 
          label: e.type,
          original: e 
      }
    }))
  }
}

// --- G6 v5 Rendering ---

const renderGraph = () => {
    if(!container.value || !wrapper.value) return;
    
    // Clean up
    if (graphInstance) {
        graphInstance.destroy();
        graphInstance = null;
    }
    
    if(graphData.nodes.length === 0) return;

    // Use current dimensions
    const width = wrapper.value.offsetWidth;
    const height = wrapper.value.offsetHeight;

    graphInstance = new Graph({
        container: container.value,
        width,
        height,
        autoFit: 'view',
        data: getGraphData(),
        layout: {
            type: 'd3-force',
            preventOverlap: true,
            nodeSize: [160, 60],
            linkDistance: 150,
            collide: { strength: 0.8 },
            alphaDecay: 0.03
        },
        node: {
            type: 'rect',
            style: {
                size: [160, 60],
                radius: 12,
                fill: '#FFFFFF',
                stroke: '#E2E8F0',
                lineWidth: 1,
                shadowColor: 'rgba(0, 0, 0, 0.05)',
                shadowBlur: 10,
                cursor: 'pointer',
                labelText: (d) => {
                    const label = d.data?.label || '';
                    return label.length > 18 ? label.substring(0, 16) + '...' : label;
                },
                labelFill: '#1E293B',
                labelFontSize: 13,
                labelFontWeight: 600,
                labelFontFamily: 'Inter, sans-serif',
            },
            state: {
                selected: {
                    stroke: '#F97316',
                    lineWidth: 2,
                    shadowColor: 'rgba(249, 115, 22, 0.2)',
                    shadowBlur: 20,
                },
                active: {
                    stroke: '#F97316',
                    lineWidth: 2,
                },
            },
        },
        edge: {
            type: 'cubic-horizontal',
            style: {
                stroke: '#CBD5E1',
                lineWidth: 1.5,
                endArrow: true,
                endArrowType: 'vee',
                endArrowFill: '#CBD5E1',
                labelText: (d) => d.data?.label || '',
                labelFill: '#64748B',
                labelFontSize: 11,
                labelBackground: true,
                labelBackgroundFill: '#F1F5F9',
                labelBackgroundRadius: 4,
            },
            state: {
                active: {
                    stroke: '#F97316',
                    lineWidth: 2,
                    endArrowFill: '#F97316',
                },
            },
        },
        behaviors: ['drag-canvas', 'zoom-canvas', 'drag-element', 'hover-activate'],
    });

    graphInstance.render();
    
    // Interactions - G6 v5 event API
    graphInstance.on('node:click', (e) => {
        const nodeId = e.target.id;
        const nodeData = graphInstance.getNodeData(nodeId);
        
        if(nodeData) {
            detailState.item = nodeData;
            detailState.type = 'node';
            detailState.visible = true;
        }
    });

    graphInstance.on('edge:click', (e) => {
        const edgeId = e.target.id;
        const edgeData = graphInstance.getEdgeData(edgeId);
        
        if(edgeData) {
            detailState.item = edgeData;
            detailState.type = 'edge';
            detailState.visible = true;
        }
    });
    
    graphInstance.on('canvas:click', () => {
        closeDetail();
    });
}


const closeDetail = () => {
    detailState.visible = false;
    detailState.item = null;
}

const exportData = () => {
    const dataStr = JSON.stringify(graphData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `graph_export_${new Date().toISOString()}.json`;
    document.body.click(); 
    link.click();
    URL.revokeObjectURL(url);
    message.success("导出成功");
}

const graphStatusClass = computed(() => {
  if (state.loadingGraphInfo) return 'loading';
  return graphInfo.value?.status === 'open' ? 'open' : 'closed';
});

const graphStatusText = computed(() => {
    if (state.loadingGraphInfo) return '连接中...';
    return graphInfo.value?.status === 'open' ? '已连接' : '未连接';
});

const graphDescription = computed(() => {
  const { graph_name, entity_count, relationship_count } = graphInfo.value || {}
  if(!graph_name && !entity_count) return "探索知识图谱中的实体关系 network";
  return `${graph_name || 'KB'} • ${entity_count || 0} 实体 • ${relationship_count || 0} 关系`
});

// Resize handler
const handleResize = () => {
   if(graphInstance && wrapper.value) {
      graphInstance.changeSize(wrapper.value.offsetWidth, wrapper.value.offsetHeight);
   }
};

onMounted(() => {
  loadGraphInfo();
  window.addEventListener('resize', handleResize);
  // Initial load
  setTimeout(() => loadSampleNodes(), 100);
});

onBeforeUnmount(() => {
    window.removeEventListener('resize', handleResize);
    if(graphInstance) graphInstance.destroy();
});
</script>

<style lang="less" scoped>
.graph-container { 
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    background-color: var(--background-color);
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
    
    .status-wrapper {
      display: flex;
      align-items: center;
      padding: 4px 12px;
      background: var(--surface-secondary);
      border-radius: 99px;
      font-size: 12px;
      border: 1px solid var(--border-color);
      
      .status-text {
          margin-left: 8px;
          color: var(--subtext-color);
          font-weight: 500;
      }
    }
    
    .icon-btn {
        border-radius: var(--radius-md);
        color: var(--subtext-color);
        background: transparent;
        border: 1px solid transparent;
        &:hover {
         color: var(--primary-color);
         background: var(--surface-secondary);
        }
    }
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  &.loading { background: #faad14; animation: pulse 1.5s infinite ease-in-out; }
  &.open { background: #10B981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
  &.closed { background: #EF4444; }
}

@keyframes pulse {
  0% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(0.8); opacity: 0.5; }
}

.main-content {
    position: relative;
    flex: 1;
    display: flex; /* Ensure canvas wrapper fills space */
    width: 100%;
    overflow: hidden;
}

.canvas-wrapper {
    flex: 1;
    position: relative;
    width: 100%;
    height: 100%;
    
    #container {
        width: 100%;
        height: 100%;
        /* Grid pattern background */
        background-color: var(--surface-ground);
        background-image: radial-gradient(var(--slate-300) 1px, transparent 1px);
        background-size: 24px 24px;
    }
}

[data-theme='dark'] .canvas-wrapper #container {
     background-image: radial-gradient(var(--slate-700) 1px, transparent 1px);
}


/* Toolbar Styles */
.toolbar {
    position: absolute;
    top: 24px;
    left: 24px;
    z-index: 10;
    display: flex;
    align-items: center;
    padding: 8px 12px;
    gap: 12px;
    /* Glass effect inherited from global .glass + .window-card */
    background: var(--surface-overlay);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    
    .divider {
        width: 1px;
        height: 20px;
        background: var(--border-color);
    }
    
    .tool-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--subtext-color);
        
        .label {
            font-weight: 500;
        }
    }
    
    .tool-btn {
        color: var(--subtext-color);
        &:hover {
            color: var(--primary-color);
        }
    }
    
    .custom-search {
        width: 220px;
        :deep(.ant-input) {
            background: transparent !important;
            color: var(--text-color);
        }
    }
}

/* Overlays */
.empty-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    pointer-events: none;
    
    .sub-text {
        margin-top: 8px;
        color: var(--subtext-color);
        font-size: 13px;
    }
}

.loading-overlay {
    position: absolute;
    inset: 0;
    background: var(--surface-overlay);
    backdrop-filter: blur(4px);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 20;
    
    .loading-content {
        background: var(--surface-card);
        padding: 24px;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 16px;
        border: 1px solid var(--border-color);
        
        .loading-text {
            font-weight: 500;
            color: var(--primary-color);
            font-size: 14px;
        }
    }
}

.database-empty {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  flex-direction: column;
  background-color: var(--background-color);
}
</style>
