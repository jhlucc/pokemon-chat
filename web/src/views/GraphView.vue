<template>
  <div class="graph-universe" :class="{ 'dark-mode': isDarkMode }">
    <!-- 全屏画布 -->
    <div class="graph-canvas" ref="container">
      <!-- 画布背景网格 -->
      <div class="canvas-grid"></div>
    </div>

    <!-- 空状态 (悬浮在画布上) -->
    <Transition name="fade">
      <div v-if="graphData.nodes.length === 0 && !state.fetching" class="empty-overlay">
        <div class="empty-content">
          <div class="empty-icon">🌌</div>
          <h2 class="empty-title">探索知识宇宙</h2>
          <p class="empty-desc">搜索实体开始探索，或采样随机节点</p>
          <a-space :size="12">
            <a-button
              type="primary"
              size="large"
              :loading="state.fetching"
              :disabled="!canUseGraph"
              @click="loadSampleNodes"
            >
              <template #icon><ThunderboltOutlined /></template>
              随机探索
            </a-button>
          </a-space>
        </div>
      </div>
    </Transition>

    <!-- 顶部悬浮搜索栏 (Spotlight Style) -->
    <div class="floating-search">
      <div class="search-bar">
        <SearchOutlined class="search-icon" />
        <input
          v-model="state.searchInput"
          type="text"
          placeholder="搜索实体（如：皮卡丘、小智、关东地区...）"
          @keydown.enter="onSearch"
          :disabled="!canUseGraph"
        />
        <a-button
          type="text"
          class="search-btn"
          :loading="state.searchLoading"
          :disabled="!canUseGraph || !state.searchInput"
          @click="onSearch"
        >
          <SendOutlined v-if="!state.searchLoading" />
        </a-button>
      </div>

      <!-- 状态指示器 -->
      <div class="status-indicator">
        <span class="status-dot" :class="kgStatus.status"></span>
        <span class="status-text">{{ kgStatus.label }}</span>
        <span v-if="graphData.nodes.length > 0" class="node-count">
          {{ graphData.nodes.length }} 节点 · {{ graphData.edges.length }} 关系
        </span>
      </div>
    </div>

    <!-- 底部悬浮工具栏 (Dock Style) -->
    <div class="floating-dock">
      <div class="dock-group">
        <a-tooltip title="力导向布局">
          <button
            class="dock-btn"
            :class="{ active: state.layout === 'force' }"
            @click="setLayout('force')"
          >
            <ApartmentOutlined />
          </button>
        </a-tooltip>
        <a-tooltip title="径向布局">
          <button
            class="dock-btn"
            :class="{ active: state.layout === 'radial' }"
            @click="setLayout('radial')"
          >
            <RadarChartOutlined />
          </button>
        </a-tooltip>
      </div>

      <div class="dock-divider"></div>

      <div class="dock-group">
        <a-tooltip title="放大">
          <button class="dock-btn" @click="zoomIn">
            <ZoomInOutlined />
          </button>
        </a-tooltip>
        <a-tooltip title="缩小">
          <button class="dock-btn" @click="zoomOut">
            <ZoomOutOutlined />
          </button>
        </a-tooltip>
        <a-tooltip title="适应画布">
          <button class="dock-btn" @click="fitView">
            <ExpandOutlined />
          </button>
        </a-tooltip>
      </div>

      <div class="dock-divider"></div>

      <div class="dock-group">
        <a-popover placement="top" trigger="click">
          <template #content>
            <div class="sample-popover">
              <div class="sample-label">采样数量</div>
              <a-slider v-model:value="sampleNodeCount" :min="20" :max="500" :step="10" />
              <a-button
                type="primary"
                block
                :loading="state.fetching"
                :disabled="!canUseGraph"
                @click="loadSampleNodes"
              >
                采样 {{ sampleNodeCount }} 节点
              </a-button>
            </div>
          </template>
          <a-tooltip title="随机采样">
            <button class="dock-btn">
              <ExperimentOutlined />
            </button>
          </a-tooltip>
        </a-popover>

        <a-tooltip :title="isDarkMode ? '浅色模式' : '深色模式'">
          <button class="dock-btn" @click="toggleDarkMode">
            <BulbOutlined v-if="isDarkMode" />
            <BulbFilled v-else />
          </button>
        </a-tooltip>
      </div>

      <div class="dock-divider"></div>

      <div class="dock-group">
        <a-tooltip title="返回首页">
          <router-link to="/" class="dock-btn">
            <HomeOutlined />
          </router-link>
        </a-tooltip>
      </div>
    </div>

    <!-- 右侧详情抽屉 -->
    <Transition name="slide-right">
      <div v-if="state.detailOpen && selectedNode" class="detail-panel">
        <div class="detail-header">
          <div class="detail-avatar">
            <span class="avatar-text">{{ selectedNode.name?.charAt(0) || '?' }}</span>
          </div>
          <div class="detail-title">
            <h3>{{ selectedNode.name }}</h3>
            <span class="detail-id">ID: {{ selectedNode.id }}</span>
          </div>
          <button class="close-btn" @click="closeDetail">
            <CloseOutlined />
          </button>
        </div>

        <div class="detail-stats">
          <div class="stat-item">
            <span class="stat-value">{{ selectedNodeDegree }}</span>
            <span class="stat-label">连接数</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ neighborNodes.length }}</span>
            <span class="stat-label">邻居</span>
          </div>
        </div>

        <div class="detail-section">
          <div class="section-header">
            <NodeIndexOutlined />
            <span>相邻节点</span>
          </div>
          <div v-if="neighborNodes.length === 0" class="empty-neighbors">
            暂无相邻节点
          </div>
          <div v-else class="neighbor-list">
            <button
              v-for="node in neighborNodes"
              :key="node.id"
              class="neighbor-item"
              @click="focusNode(node.id)"
            >
              <span class="neighbor-avatar">{{ node.name?.charAt(0) || '?' }}</span>
              <span class="neighbor-name">{{ node.name }}</span>
              <RightOutlined class="neighbor-arrow" />
            </button>
          </div>
        </div>
      </div>
    </Transition>

    <!-- 加载遮罩 -->
    <Transition name="fade">
      <div v-if="state.vizLoading || state.fetching" class="loading-overlay">
        <div class="loading-content">
          <LoadingOutlined class="loading-icon" spin />
          <span>{{ state.vizLoading ? '加载渲染器...' : '获取数据...' }}</span>
        </div>
      </div>
    </Transition>

    <!-- 警告提示 (服务未连接时) -->
    <Transition name="slide-down">
      <div v-if="!canUseGraph" class="warning-banner">
        <ExclamationCircleOutlined />
        <span>{{ backendOnline ? '后端未启用知识图谱' : '服务已断开' }}</span>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { computed, reactive, ref, onMounted, onUnmounted, watch } from 'vue'
import { message } from 'ant-design-vue'
import { useConfigStore } from '@/stores/config'
import { apiFetch } from '@/api/http'
import { notifyApiError } from '@/utils/notify'
import {
  SearchOutlined,
  SendOutlined,
  ApartmentOutlined,
  RadarChartOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  ExpandOutlined,
  ExperimentOutlined,
  BulbOutlined,
  BulbFilled,
  HomeOutlined,
  CloseOutlined,
  NodeIndexOutlined,
  RightOutlined,
  LoadingOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons-vue'

const configStore = useConfigStore()

const container = ref(null)
const sampleNodeCount = ref(100)
const graphData = reactive({ nodes: [], edges: [] })
const isDarkMode = ref(true)

let graphInstance
let GraphCtor = null
let layoutKey = null
let resizeObserver = null

const state = reactive({
  fetching: false,
  vizLoading: false,
  searchInput: '',
  searchLoading: false,
  detailOpen: false,
  selectedNodeId: null,
  layout: 'force'
})

const ensureG6 = async () => {
  if (GraphCtor) return GraphCtor
  state.vizLoading = true
  try {
    const mod = await import('@antv/g6')
    GraphCtor = mod.Graph
    return GraphCtor
  } catch (e) {
    notifyApiError(e, { context: '加载图谱渲染器', fallback: '加载失败' })
    return null
  } finally {
    state.vizLoading = false
  }
}

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const canUseGraph = computed(() => backendOnline.value && Boolean(configStore.config.enable_knowledge_graph))

const kgStatus = computed(() => {
  if (backendOnline.value && Boolean(configStore.config.enable_knowledge_graph))
    return { status: 'online', label: '图谱已连接' }
  if (backendOnline.value) return { status: 'warning', label: '图谱未启用' }
  return { status: 'offline', label: '已断开' }
})

const selectedNode = computed(() => {
  if (!state.selectedNodeId) return null
  return graphData.nodes.find((n) => n.id === state.selectedNodeId) || null
})

const selectedNodeDegree = computed(() => {
  if (!state.selectedNodeId) return 0
  let d = 0
  graphData.edges.forEach((e) => {
    if (e.source_id === state.selectedNodeId || e.target_id === state.selectedNodeId) d += 1
  })
  return d
})

const neighborNodes = computed(() => {
  if (!state.selectedNodeId) return []
  const neighborIds = new Set()
  graphData.edges.forEach((e) => {
    if (e.source_id === state.selectedNodeId) neighborIds.add(e.target_id)
    if (e.target_id === state.selectedNodeId) neighborIds.add(e.source_id)
  })
  return graphData.nodes.filter((n) => neighborIds.has(n.id))
})

const loadSampleNodes = async () => {
  if (!canUseGraph.value) return
  state.fetching = true
  try {
    const data = await apiFetch('/data/graph/nodes', {
      method: 'GET',
      query: { kgdb_name: 'neo4j', num: sampleNodeCount.value }
    })
    graphData.nodes = data?.result?.nodes || []
    graphData.edges = data?.result?.edges || []
    state.selectedNodeId = null
    state.detailOpen = false
    setTimeout(() => void renderGraph(), 0)
  } catch (e) {
    notifyApiError(e, { context: '获取节点', fallback: '获取节点失败' })
  } finally {
    state.fetching = false
  }
}

const onSearch = async () => {
  if (!state.searchInput) return message.error('请输入要查询的实体')
  if (!canUseGraph.value) return

  state.searchLoading = true
  try {
    const data = await apiFetch('/data/graph/node', {
      method: 'GET',
      query: { entity_name: state.searchInput }
    })
    graphData.nodes = data?.result?.nodes || []
    graphData.edges = data?.result?.edges || []
    if (graphData.nodes.length === 0) message.info('未找到相关实体')
    state.selectedNodeId = null
    state.detailOpen = false
    setTimeout(() => void renderGraph(), 0)
  } catch (e) {
    notifyApiError(e, { context: '检索关系', fallback: '检索失败' })
  } finally {
    state.searchLoading = false
  }
}

const getG6Data = () => {
  const degree = {}
  graphData.nodes.forEach((n) => (degree[n.id] = 0))
  graphData.edges.forEach((e) => {
    degree[e.source_id] = (degree[e.source_id] || 0) + 1
    degree[e.target_id] = (degree[e.target_id] || 0) + 1
  })

  return {
    nodes: graphData.nodes.map((n) => ({
      id: n.id,
      data: { label: n.name, degree: degree[n.id] || 0 }
    })),
    edges: graphData.edges.map((e) => ({
      id: `${e.source_id}-${e.type}-${e.target_id}`,
      source: e.source_id,
      target: e.target_id,
      data: { label: e.type }
    }))
  }
}

const getThemeColors = () => {
  if (isDarkMode.value) {
    return {
      primary: '#FFA940',
      background: 'transparent',
      surface: 'rgba(255, 255, 255, 0.9)',
      text: 'rgba(255, 255, 255, 0.85)',
      textSecondary: 'rgba(255, 255, 255, 0.45)',
      edgeStroke: 'rgba(255, 255, 255, 0.2)',
      nodeFill: 'rgba(30, 30, 30, 0.8)',
      nodeStroke: '#FFA940',
      selectedFill: '#FFA940'
    }
  }
  return {
    primary: '#FF7D00',
    background: 'transparent',
    surface: 'rgba(255, 255, 255, 0.9)',
    text: '#333',
    textSecondary: '#666',
    edgeStroke: 'rgba(0, 0, 0, 0.15)',
    nodeFill: '#fff',
    nodeStroke: '#FF7D00',
    selectedFill: '#FF7D00'
  }
}

const ensureGraph = async () => {
  if (!container.value) return
  const Graph = await ensureG6()
  if (!Graph) return

  const key = `${state.layout}-${isDarkMode.value}`
  const colors = getThemeColors()

  const layout =
    state.layout === 'radial'
      ? { type: 'radial', unitRadius: 120, preventOverlap: true }
      : { type: 'd3-force', preventOverlap: true, collide: { radius: 40, strength: 0.6 } }

  if (!graphInstance || layoutKey !== key) {
    if (graphInstance) {
      try {
        graphInstance.destroy()
      } catch {}
      graphInstance = null
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
          size: (d) => Math.min(20 + (d.data.degree || 0) * 4, 60),
          labelFill: colors.text,
          labelFontSize: 11,
          labelFontWeight: 500,
          fill: (d) => (d.id === state.selectedNodeId ? colors.selectedFill : colors.nodeFill),
          stroke: colors.nodeStroke,
          lineWidth: 2,
          shadowColor: 'rgba(255, 125, 0, 0.3)',
          shadowBlur: (d) => (d.id === state.selectedNodeId ? 20 : 0)
        }
      },
      edge: {
        type: 'line',
        style: {
          labelText: (d) => d.data.label,
          labelFill: colors.textSecondary,
          labelFontSize: 10,
          labelBackground: true,
          labelBackgroundFill: isDarkMode.value ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.8)',
          labelPadding: [2, 4],
          endArrow: true,
          stroke: colors.edgeStroke,
          lineWidth: 1.5
        }
      },
      behaviors: ['drag-element', 'zoom-canvas', 'drag-canvas']
    })

    graphInstance.on('node:click', (evt) => {
      const id = evt?.item?.getID?.() || evt?.item?.id || evt?.data?.id || evt?.target?.id || null
      if (!id) return
      state.selectedNodeId = id
      state.detailOpen = true
      setTimeout(() => void renderGraph(), 0)
    })

    graphInstance.on('canvas:click', () => {
      state.selectedNodeId = null
      state.detailOpen = false
      setTimeout(() => void renderGraph(), 0)
    })

    layoutKey = key
  } else {
    try {
      graphInstance.resize?.(container.value.offsetWidth, container.value.offsetHeight)
    } catch {}
  }
}

const renderGraph = async () => {
  if (!container.value || graphData.nodes.length === 0) return
  await ensureGraph()
  if (!graphInstance) return
  graphInstance.setData(getG6Data())
  graphInstance.render()
}

const setLayout = (layout) => {
  state.layout = layout
  renderGraph()
}

const zoomIn = () => {
  graphInstance?.zoom?.(1.2)
}

const zoomOut = () => {
  graphInstance?.zoom?.(0.8)
}

const fitView = () => {
  graphInstance?.fitView?.()
}

const toggleDarkMode = () => {
  isDarkMode.value = !isDarkMode.value
  // Force re-render with new theme
  layoutKey = null
  renderGraph()
}

const focusNode = (id) => {
  state.selectedNodeId = id
  state.detailOpen = true
  setTimeout(() => void renderGraph(), 0)
}

const closeDetail = () => {
  state.selectedNodeId = null
  state.detailOpen = false
  setTimeout(() => void renderGraph(), 0)
}

onMounted(() => {
  watch(
    () => container.value,
    (el) => {
      try {
        resizeObserver?.disconnect?.()
      } catch {}
      resizeObserver = null
      if (!el || typeof ResizeObserver === 'undefined') return

      resizeObserver = new ResizeObserver(() => {
        if (!graphInstance || !container.value) return
        try {
          graphInstance.resize?.(container.value.offsetWidth, container.value.offsetHeight)
          graphInstance.render?.()
        } catch {}
      })
      resizeObserver.observe(el)
    },
    { immediate: true }
  )
})

onUnmounted(() => {
  try {
    resizeObserver?.disconnect?.()
  } catch {}
  resizeObserver = null

  if (graphInstance) {
    try {
      graphInstance.destroy()
    } catch {}
    graphInstance = null
  }
  layoutKey = null
})
</script>

<style lang="less" scoped>
.graph-universe {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  overflow: hidden;
  transition: background 0.3s ease;

  &:not(.dark-mode) {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
  }
}

/* 画布背景网格 */
.canvas-grid {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.1) 1px, transparent 1px);
  background-size: 30px 30px;
  pointer-events: none;
  opacity: 0.5;

  .graph-universe:not(.dark-mode) & {
    background-image:
      radial-gradient(circle, rgba(0, 0, 0, 0.08) 1px, transparent 1px);
  }
}

/* 画布 */
.graph-canvas {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 1;
}

/* 空状态覆盖层 */
.empty-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 5;
  pointer-events: none;

  .empty-content {
    text-align: center;
    pointer-events: auto;

    .empty-icon {
      font-size: 64px;
      margin-bottom: 16px;
      animation: float 3s ease-in-out infinite;
    }

    .empty-title {
      font-size: 28px;
      font-weight: 600;
      color: rgba(255, 255, 255, 0.9);
      margin: 0 0 8px;

      .graph-universe:not(.dark-mode) & {
        color: #333;
      }
    }

    .empty-desc {
      font-size: 14px;
      color: rgba(255, 255, 255, 0.6);
      margin: 0 0 24px;

      .graph-universe:not(.dark-mode) & {
        color: #666;
      }
    }
  }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

/* 悬浮搜索栏 */
.floating-search {
  position: absolute;
  top: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;

  .search-bar {
    display: flex;
    align-items: center;
    width: 480px;
    max-width: calc(100vw - 48px);
    padding: 8px 16px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;

    .graph-universe:not(.dark-mode) & {
      background: rgba(255, 255, 255, 0.85);
      border-color: rgba(0, 0, 0, 0.1);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    &:focus-within {
      border-color: rgba(255, 125, 0, 0.5);
      box-shadow: 0 8px 32px rgba(255, 125, 0, 0.2);
    }

    .search-icon {
      font-size: 18px;
      color: rgba(255, 255, 255, 0.5);
      margin-right: 12px;

      .graph-universe:not(.dark-mode) & {
        color: #999;
      }
    }

    input {
      flex: 1;
      border: none;
      background: transparent;
      font-size: 15px;
      color: rgba(255, 255, 255, 0.9);
      outline: none;

      .graph-universe:not(.dark-mode) & {
        color: #333;
      }

      &::placeholder {
        color: rgba(255, 255, 255, 0.4);

        .graph-universe:not(.dark-mode) & {
          color: #999;
        }
      }
    }

    .search-btn {
      width: 36px;
      height: 36px;
      padding: 0;
      border-radius: 50%;
      color: #FFA940;

      &:hover:not(:disabled) {
        background: rgba(255, 125, 0, 0.2);
      }
    }
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);

    .graph-universe:not(.dark-mode) & {
      color: #666;
    }

    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #52c41a;

      &.warning { background: #faad14; }
      &.offline { background: #ff4d4f; }
    }

    .node-count {
      padding-left: 8px;
      border-left: 1px solid rgba(255, 255, 255, 0.2);

      .graph-universe:not(.dark-mode) & {
        border-color: rgba(0, 0, 0, 0.1);
      }
    }
  }
}

/* 底部悬浮工具栏 */
.floating-dock {
  position: absolute;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);

  .graph-universe:not(.dark-mode) & {
    background: rgba(255, 255, 255, 0.85);
    border-color: rgba(0, 0, 0, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  }

  .dock-group {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .dock-divider {
    width: 1px;
    height: 24px;
    background: rgba(255, 255, 255, 0.2);
    margin: 0 8px;

    .graph-universe:not(.dark-mode) & {
      background: rgba(0, 0, 0, 0.1);
    }
  }

  .dock-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 10px;
    background: transparent;
    color: rgba(255, 255, 255, 0.7);
    font-size: 16px;
    cursor: pointer;
    transition: all 0.15s ease;
    text-decoration: none;

    .graph-universe:not(.dark-mode) & {
      color: #666;
    }

    &:hover {
      background: rgba(255, 255, 255, 0.15);
      color: #fff;

      .graph-universe:not(.dark-mode) & {
        background: rgba(0, 0, 0, 0.08);
        color: #333;
      }
    }

    &.active {
      background: rgba(255, 125, 0, 0.2);
      color: #FFA940;
    }
  }
}

/* 采样弹出框 */
.sample-popover {
  width: 200px;

  .sample-label {
    font-size: 12px;
    color: #666;
    margin-bottom: 8px;
  }

  :deep(.ant-slider) {
    margin: 8px 0 16px;
  }
}

/* 右侧详情面板 */
.detail-panel {
  position: absolute;
  top: 0;
  right: 0;
  width: 320px;
  height: 100%;
  background: rgba(20, 20, 30, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  z-index: 200;
  overflow-y: auto;
  padding: 24px;

  .graph-universe:not(.dark-mode) & {
    background: rgba(255, 255, 255, 0.95);
    border-color: rgba(0, 0, 0, 0.1);
  }

  .detail-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 24px;

    .detail-avatar {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      background: linear-gradient(135deg, #FF7D00, #FFA940);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;

      .avatar-text {
        font-size: 20px;
        font-weight: 600;
        color: white;
      }
    }

    .detail-title {
      flex: 1;
      min-width: 0;

      h3 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;

        .graph-universe:not(.dark-mode) & {
          color: #333;
        }
      }

      .detail-id {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.5);

        .graph-universe:not(.dark-mode) & {
          color: #999;
        }
      }
    }

    .close-btn {
      width: 28px;
      height: 28px;
      border: none;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.1);
      color: rgba(255, 255, 255, 0.6);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;

      .graph-universe:not(.dark-mode) & {
        background: rgba(0, 0, 0, 0.05);
        color: #999;
      }

      &:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #fff;

        .graph-universe:not(.dark-mode) & {
          background: rgba(0, 0, 0, 0.1);
          color: #333;
        }
      }
    }
  }

  .detail-stats {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;

    .stat-item {
      flex: 1;
      padding: 16px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 12px;
      text-align: center;

      .graph-universe:not(.dark-mode) & {
        background: rgba(0, 0, 0, 0.03);
      }

      .stat-value {
        display: block;
        font-size: 24px;
        font-weight: 600;
        color: #FFA940;
      }

      .stat-label {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.5);

        .graph-universe:not(.dark-mode) & {
          color: #999;
        }
      }
    }
  }

  .detail-section {
    .section-header {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
      color: rgba(255, 255, 255, 0.9);
      margin-bottom: 12px;

      .graph-universe:not(.dark-mode) & {
        color: #333;
      }
    }

    .empty-neighbors {
      font-size: 13px;
      color: rgba(255, 255, 255, 0.4);
      text-align: center;
      padding: 24px 0;

      .graph-universe:not(.dark-mode) & {
        color: #999;
      }
    }

    .neighbor-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .neighbor-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.15s ease;
      width: 100%;
      text-align: left;

      .graph-universe:not(.dark-mode) & {
        background: rgba(0, 0, 0, 0.03);
      }

      &:hover {
        background: rgba(255, 125, 0, 0.15);

        .neighbor-arrow {
          opacity: 1;
          transform: translateX(0);
        }
      }

      .neighbor-avatar {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: rgba(255, 125, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
        color: #FFA940;
        flex-shrink: 0;
      }

      .neighbor-name {
        flex: 1;
        font-size: 13px;
        color: rgba(255, 255, 255, 0.8);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;

        .graph-universe:not(.dark-mode) & {
          color: #333;
        }
      }

      .neighbor-arrow {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.4);
        opacity: 0;
        transform: translateX(-4px);
        transition: all 0.15s ease;

        .graph-universe:not(.dark-mode) & {
          color: #999;
        }
      }
    }
  }
}

/* 加载遮罩 */
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
  z-index: 300;

  .loading-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: rgba(255, 255, 255, 0.9);

    .loading-icon {
      font-size: 32px;
      color: #FFA940;
    }
  }
}

/* 警告横幅 */
.warning-banner {
  position: absolute;
  top: 90px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: rgba(250, 173, 20, 0.15);
  border: 1px solid rgba(250, 173, 20, 0.3);
  border-radius: 8px;
  font-size: 13px;
  color: #faad14;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.slide-right-enter-active,
.slide-right-leave-active {
  transition: transform 0.3s ease;
}
.slide-right-enter-from,
.slide-right-leave-to {
  transform: translateX(100%);
}

.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease;
}
.slide-down-enter-from,
.slide-down-leave-to {
  opacity: 0;
  transform: translate(-50%, -20px);
}

/* 响应式 */
@media (max-width: 768px) {
  .floating-search .search-bar {
    width: calc(100vw - 32px);
  }

  .detail-panel {
    width: 100%;
    border-left: none;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    height: 60%;
    top: auto;
    bottom: 0;
    border-radius: 20px 20px 0 0;
  }

  .floating-dock {
    bottom: 16px;
    padding: 6px 10px;

    .dock-btn {
      width: 32px;
      height: 32px;
    }
  }
}
</style>
