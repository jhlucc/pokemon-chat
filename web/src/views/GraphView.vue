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
  <div class="graph-container layout-container" v-else>
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
            
             <a-button type="default" class="icon-btn" @click="loadSampleNodes" :loading="state.fetching" title="刷新样本">
                <template #icon><ReloadOutlined /></template>
            </a-button>
        </div>
      </template>
    </HeaderComponent>

    <div class="main-content">
        <!-- Floating Toolbar -->
        <div class="toolbar glass">
            <div class="search-box">
                 <a-input-search
                  v-model:value="state.searchInput"
                  placeholder="搜索实体..."
                  style="width: 240px"
                  @search="onSearch"
                  :loading="state.searchLoading"
                  allowClear
                />
            </div>
            <div class="divider"></div>
            <div class="tool-item">
                <span class="label">数量:</span>
                <a-input-number v-model:value="sampleNodeCount" :min="10" :max="500" style="width: 70px" size="small" :bordered="false"/>
            </div>
            <div class="divider"></div>
            <a-tooltip title="导出当前视图数据">
                 <a-button type="text" size="small" @click="exportData">
                    <ExportOutlined />
                </a-button>
            </a-tooltip>
        </div>

        <div class="canvas-wrapper" ref="wrapper">
             <div id="container" ref="container"></div>
             <a-empty v-if="graphData.nodes.length === 0 && !state.fetching" description="暂无数据，请尝试刷新或搜索" class="empty-state"/>
             <div v-if="state.fetching || state.searchLoading" class="loading-overlay">
                 <a-spin tip="加载图谱数据..." />
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
import { Graph } from "@antv/g6";
import { computed, onMounted, reactive, ref, watch } from 'vue';
import { message } from 'ant-design-vue';
import { ReloadOutlined, ExportOutlined } from '@ant-design/icons-vue';
import { useConfigStore } from '@/stores/config';
import HeaderComponent from '@/components/HeaderComponent.vue';
import GraphDetailPanel from '@/components/GraphDetailPanel.vue';

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

const loadGraphInfo = () => {
  state.loadingGraphInfo = true
  fetch('/api/data/graph')
    .then(response => response.json())
    .then(data => {
      graphInfo.value = data
    })
    .catch(error => {
      // Quiet fail for info
      console.warn("Failed to load graph info", error)
    })
    .finally(() => state.loadingGraphInfo = false)
}

const loadSampleNodes = () => {
  state.fetching = true
  // Close detail when reloading
  closeDetail();
  
  fetch(`/api/data/graph/nodes?kgdb_name=neo4j&num=${sampleNodeCount.value}`)
    .then(res => res.json())
    .then(data => {
      if(data.result) {
          graphData.nodes = data.result.nodes || []
          graphData.edges = data.result.edges || []
          renderGraph()
      } else {
          graphData.nodes = []
          graphData.edges = []
          renderGraph() // Clear
      }
    })
    .catch(err => message.error(err.message))
    .finally(() => state.fetching = false)
}

const onSearch = () => {
  if (!state.searchInput) return loadSampleNodes();
  state.searchLoading = true
  closeDetail();
  
  fetch(`/api/data/graph/node?entity_name=${state.searchInput}`)
    .then(res => res.json())
    .then(data => {
      if (data.result) {
          graphData.nodes = data.result.nodes || []
          graphData.edges = data.result.edges || []
          if (graphData.nodes.length === 0) message.info('未找到相关实体')
          renderGraph()
      }
    })
    .catch(err => message.error(err.message))
    .finally(() => state.searchLoading = false)
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
          original: n // Pass original data for detail panel
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

const renderGraph = () => {
    if(!container.value) return;
    
    // Clean up
    if (graphInstance) {
        graphInstance.destroy();
        graphInstance = null;
    }
    
    if(graphData.nodes.length === 0) return;

    const width = wrapper.value.offsetWidth;
    const height = wrapper.value.offsetHeight;

    graphInstance = new Graph({
        container: container.value,
        width,
        height,
        autoFit: 'view',
        layout: {
            type: 'd3-force',
            preventOverlap: true,
            nodeSize: 30,
            linkDistance: 100,
            collide: { strength: 0.8 },
        },
        node: {
            type: 'circle',
            style: {
                labelText: d => d.data.label.length > 5 ? d.data.label.substring(0,5)+'...' : d.data.label,
                labelFill: '#1e293b', // var(--text-color) approximation
                labelFontSize: 12,
                labelPlacement: 'bottom',
                size: d => Math.max(20, Math.min(15 + (d.data.degree || 0) * 3, 60)),
                fill: '#6366f1', // Indigo-500
                stroke: '#ffffff',
                lineWidth: 2,
                cursor: 'pointer',
            },
            state: {
                active: {
                    fill: '#4338ca', // Indigo-700
                    shadowColor: 'rgba(99, 102, 241, 0.4)',
                    shadowBlur: 10
                },
                selected: {
                    fill: '#e11d48', // Rose-600
                    stroke: '#000',
                    lineWidth: 3
                }
            }
        },
        edge: {
            type: 'line',
            style: {
                labelText: d => d.data.label,
                labelBackground: true,
                labelBackgroundFill: '#f8fafc',
                labelBackgroundRadius: 4,
                labelFontSize: 10,
                labelFill: '#64748b',
                stroke: '#cbd5e1', // Slate-300
                endArrow: true,
                cursor: 'pointer'
            },
            state: {
                active: {
                    stroke: '#6366f1',
                    lineWidth: 2
                },
                selected: {
                    stroke: '#e11d48',
                    lineWidth: 3
                }
            }
        },
        behaviors: [
            'drag-element', 
            'zoom-canvas', 
            'drag-canvas',
            {
                type: 'click-select',
                multiple: false,
                trigger: 'click', // Required to trigger selection
                onClick: (e) => {
                   // Handled in event listener below for more control if needed
                }
            },
            {
                type: 'hover-activate',
                degree: 1 // Highlight neighbors
            }
        ],
    });

    graphInstance.setData(getGraphData());
    graphInstance.render();
    
    // Event Listeners
    graphInstance.on('node:click', (e) => {
        const model = e.target.id ? graphInstance.getNodeData(e.target.id) : null;
        if(model) {
            detailState.item = model;
            detailState.type = 'node';
            detailState.visible = true;
        }
    });

    graphInstance.on('edge:click', (e) => {
         const model = e.target.id ? graphInstance.getEdgeData(e.target.id) : null;
         if(model) {
            detailState.item = model;
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
    // Optional: Clear selection in graph if needed
    // if(graphInstance) graphInstance.setItemState(item, 'selected', false);
}

const exportData = () => {
    const dataStr = JSON.stringify(graphData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `graph_export_${new Date().toISOString()}.json`;
    document.body.click(); // Fix for firefox?
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
  if(!graph_name && !entity_count) return "探索知识图谱中的实体关系";
  return `${graph_name || 'KB'} • ${entity_count || 0} 实体 • ${relationship_count || 0} 关系`
});

onMounted(() => {
  loadGraphInfo()
  // Slight delay to ensure container dimension
  setTimeout(() => loadSampleNodes(), 100);
  
  window.addEventListener('resize', () => {
      if(graphInstance && wrapper.value) {
          graphInstance.setSize(wrapper.value.offsetWidth, wrapper.value.offsetHeight);
      }
  })
})
</script>

<style lang="less" scoped>
.graph-container { 
    padding: 0; 
    display: flex;
    flex-direction: column;
    height: 100%;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
    
    .status-wrapper {
      display: flex;
      align-items: center;
      padding: 4px 12px;
      background: var(--gray-100);
      border-radius: 20px;
      font-size: 12px;
      
      .status-text {
          margin-left: 6px;
          color: var(--subtext-color);
          font-weight: 500;
      }
    }
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  &.loading { background: #faad14; animation: pulse 1.5s infinite ease-in-out; }
  &.open { background: #52c41a; box-shadow: 0 0 8px rgba(82, 196, 26, 0.4); }
  &.closed { background: #f5222d; }
}

@keyframes pulse {
  0% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(0.8); opacity: 0.5; }
}

.main-content {
    position: relative;
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--surface-ground); // Checkered or simple bg
    overflow: hidden;
}

.canvas-wrapper {
    flex: 1;
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    
    #container {
        width: 100%;
        height: 100%;
        /* Grid pattern background */
        background-color: var(--surface-card);
        background-image: radial-gradient(var(--gray-200) 1px, transparent 1px);
        background-size: 20px 20px;
    }
}

/* Floating Toolbar */
.toolbar {
    position: absolute;
    top: 16px;
    left: 24px;
    z-index: 10;
    display: flex;
    align-items: center;
    padding: 8px 12px;
    gap: 12px;
    border-radius: 12px;
    background: var(--surface-overlay);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-md);
    
    .divider {
        width: 1px;
        height: 16px;
        background: var(--border-color);
    }
    
    .tool-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--subtext-color);
    }
}

.empty-state {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
}

.loading-overlay {
    position: absolute;
    inset: 0;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(2px);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 5;
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
