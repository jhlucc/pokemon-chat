<template>
  <div class="graph-explorer">
    <!-- 全屏画布 -->
    <div class="graph-canvas" ref="container">
      <!-- 画布背景网格 - 图纸/地图风格 -->
      <div class="canvas-grid"></div>
    </div>

    <!-- 空状态 (悬浮在画布上) -->
    <Transition name="fade">
      <div v-if="graphData.nodes.length === 0 && !state.fetching" class="empty-overlay">
        <div class="empty-content">
          <div class="empty-illustration">
            <div class="empty-icon-group">
              <ApartmentOutlined class="icon-main" />
              <SearchOutlined class="icon-accent" />
            </div>
          </div>
          <h2 class="empty-title">探索知识图谱</h2>
          <p class="empty-desc">搜索实体开始探索，发现有趣的关系网络</p>
          <a-button
            type="primary"
            size="large"
            :loading="state.fetching"
            :disabled="!canUseGraph"
            @click="loadSampleNodes"
          >
            <template #icon><CompassOutlined /></template>
            随机探索
          </a-button>
        </div>
      </div>
    </Transition>

    <!-- 顶部悬浮搜索栏 -->
    <div class="floating-search">
      <div class="search-bar">
        <!-- 状态呼吸灯 (内嵌) -->
        <a-tooltip :title="kgStatus.label">
          <span class="status-breath" :class="kgStatus.status"></span>
        </a-tooltip>
        <input
          v-model="state.searchInput"
          type="text"
          placeholder="搜索宝可梦、技能、地区..."
          @keydown.enter="onSearch"
          :disabled="!canUseGraph"
        />
        <a-button
          type="primary"
          class="search-btn"
          :loading="state.searchLoading"
          :disabled="!canUseGraph || !state.searchInput"
          @click="onSearch"
        >
          <SearchOutlined v-if="!state.searchLoading" />
        </a-button>
      </div>

      <!-- 节点统计 (有数据时显示) -->
      <Transition name="fade">
        <div v-if="graphData.nodes.length > 0" class="node-stats">
          {{ graphData.nodes.length }} 节点 · {{ graphData.edges.length }} 关系
        </div>
      </Transition>
    </div>

    <!-- 底部悬浮工具栏 - 轻量化胶囊条 -->
    <div class="floating-toolbar">
      <div class="toolbar-group">
        <a-tooltip title="力导向布局">
          <button
            class="toolbar-btn"
            :class="{ active: state.layout === 'force' }"
            @click="setLayout('force')"
          >
            <ApartmentOutlined />
          </button>
        </a-tooltip>
        <a-tooltip title="径向布局">
          <button
            class="toolbar-btn"
            :class="{ active: state.layout === 'radial' }"
            @click="setLayout('radial')"
          >
            <RadarChartOutlined />
          </button>
        </a-tooltip>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
        <a-tooltip title="放大">
          <button class="toolbar-btn" @click="zoomIn">
            <ZoomInOutlined />
          </button>
        </a-tooltip>
        <a-tooltip title="缩小">
          <button class="toolbar-btn" @click="zoomOut">
            <ZoomOutOutlined />
          </button>
        </a-tooltip>
        <a-tooltip title="适应画布">
          <button class="toolbar-btn" @click="fitView">
            <ExpandOutlined />
          </button>
        </a-tooltip>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
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
                探索 {{ sampleNodeCount }} 个节点
              </a-button>
            </div>
          </template>
          <a-tooltip title="随机采样">
            <button class="toolbar-btn">
              <ExperimentOutlined />
            </button>
          </a-tooltip>
        </a-popover>
      </div>

      <div class="toolbar-divider"></div>

      <div class="toolbar-group">
        <a-tooltip title="返回对话">
          <router-link to="/" class="toolbar-btn">
            <HomeOutlined />
          </router-link>
        </a-tooltip>
      </div>
    </div>

    <!-- 右侧详情面板 -->
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
          <span>{{ state.vizLoading ? '加载渲染器...' : '探索中...' }}</span>
        </div>
      </div>
    </Transition>

    <!-- 警告提示 -->
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
  ApartmentOutlined,
  RadarChartOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  ExpandOutlined,
  ExperimentOutlined,
  HomeOutlined,
  CloseOutlined,
  NodeIndexOutlined,
  RightOutlined,
  LoadingOutlined,
  ExclamationCircleOutlined,
  CompassOutlined
} from '@ant-design/icons-vue'

const configStore = useConfigStore()

const container = ref(null)
const sampleNodeCount = ref(100)
const graphData = reactive({ nodes: [], edges: [] })

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

  // 计算最大度数用于归一化
  const maxDegree = Math.max(...Object.values(degree), 1)

  return {
    nodes: graphData.nodes.map((n, index) => ({
      id: n.id,
      data: {
        label: n.name,
        degree: degree[n.id] || 0,
        maxDegree,
        colorIndex: index % candyPalette.length,
        original: n
      }
    })),
    edges: graphData.edges.map((e) => ({
      id: `${e.source_id}-${e.type}-${e.target_id}`,
      source: e.source_id,
      target: e.target_id,
      data: { label: e.type }
    }))
  }
}

// 🍬 糖果色暖色调色板 - 可萌风格
const candyPalette = [
  '#FF7D00',  // 品牌橙 - 核心宝可梦
  '#FFA940',  // 金橙 - 重要实体
  '#FFD666',  // 蜜黄 - 技能类
  '#FF9A9E',  // 珊瑚粉 - 地点类
  '#FECACA',  // 浅粉 - 道具类
  '#FED7AA',  // 杏色 - 属性类
  '#FDE68A',  // 浅金 - 事件类
  '#E8D5B7',  // 暖米 - 通用
]

// 根据度数获取节点颜色 (度数越高颜色越深)
const getNodeColor = (d) => {
  if (d.id === state.selectedNodeId) return '#FF5722'
  const deg = d.data?.degree || 0
  const maxDeg = d.data?.maxDegree || 1
  const ratio = deg / maxDeg

  // 度数越高，颜色越偏向品牌橙
  if (ratio > 0.7) return candyPalette[0] // 核心节点
  if (ratio > 0.4) return candyPalette[1] // 重要节点
  if (ratio > 0.2) return candyPalette[2] // 普通节点

  // 其他节点根据索引分配颜色
  return candyPalette[d.data?.colorIndex || 0]
}

// 根据度数计算节点大小 (更明显的差异)
const getNodeSize = (d) => {
  const deg = d.data?.degree || 0
  // 基础大小 28px，每个连接 +5px，最大 70px
  return Math.min(28 + deg * 5, 70)
}

// 判断节点是否与选中节点相关
const isRelatedNode = (nodeId) => {
  if (!state.selectedNodeId) return true
  if (nodeId === state.selectedNodeId) return true

  // 检查是否是邻居
  return graphData.edges.some(
    (e) =>
      (e.source_id === state.selectedNodeId && e.target_id === nodeId) ||
      (e.target_id === state.selectedNodeId && e.source_id === nodeId)
  )
}

// 判断边是否与选中节点相关
const isRelatedEdge = (edge) => {
  if (!state.selectedNodeId) return true
  return edge.source === state.selectedNodeId || edge.target === state.selectedNodeId
}

const ensureGraph = async () => {
  if (!container.value) return
  const Graph = await ensureG6()
  if (!Graph) return

  const key = state.layout

  // 专业的力导向布局配置
  const layout =
    state.layout === 'radial'
      ? {
          type: 'radial',
          unitRadius: 100,
          preventOverlap: true,
          nodeSpacing: 30
        }
      : {
          type: 'd3-force',
          preventOverlap: true,
          alphaDecay: 0.08,
          alphaMin: 0.01,
          velocityDecay: 0.55,
          forceSimulation: {
            velocityDecay: 0.6
          },
          collide: {
            radius: (d) => getNodeSize(d) / 2 + 10,
            strength: 0.8,
            iterations: 3
          },
          manyBody: {
            strength: -350,
            distanceMax: 500
          },
          link: {
            distance: 120,
            strength: 0.7
          }
        }

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
          // 🍬 糖果风节点样式
          size: getNodeSize,
          fill: getNodeColor,
          stroke: (d) => {
            if (d.id === state.selectedNodeId) return '#E64A19'
            return 'rgba(255, 255, 255, 0.8)'
          },
          lineWidth: (d) => (d.id === state.selectedNodeId ? 3 : 2),

          // 内发光效果 - 糖果质感
          shadowColor: (d) => {
            if (d.id === state.selectedNodeId) return 'rgba(255, 87, 34, 0.5)'
            return 'rgba(255, 125, 0, 0.25)'
          },
          shadowBlur: (d) => (d.id === state.selectedNodeId ? 20 : 8),
          shadowOffsetY: 2,

          // 聚光灯效果 - 未选中的不相关节点变暗
          opacity: (d) => (isRelatedNode(d.id) ? 1 : 0.25),

          // 标签样式
          labelText: (d) => d.data.label,
          labelFill: '#333',
          labelFontSize: (d) => {
            const size = getNodeSize(d)
            return size > 50 ? 12 : size > 35 ? 11 : 10
          },
          labelFontWeight: 500,
          labelOffsetY: (d) => getNodeSize(d) / 2 + 12,
          labelBackground: true,
          labelBackgroundFill: 'rgba(255, 255, 255, 0.85)',
          labelBackgroundRadius: 4,
          labelPadding: [2, 6]
        }
      },
      edge: {
        type: 'quadratic', // 曲线更优雅
        style: {
          stroke: (d) => (isRelatedEdge(d) ? 'rgba(255, 125, 0, 0.4)' : 'rgba(200, 200, 200, 0.15)'),
          lineWidth: (d) => (isRelatedEdge(d) ? 1.5 : 1),
          opacity: (d) => (isRelatedEdge(d) ? 0.8 : 0.2),
          endArrow: {
            path: 'M 0,0 L 8,4 L 8,-4 Z',
            fill: 'rgba(255, 125, 0, 0.5)'
          },
          // 边标签
          labelText: (d) => d.data.label,
          labelFill: '#888',
          labelFontSize: 9,
          labelBackground: true,
          labelBackgroundFill: 'rgba(255, 255, 255, 0.9)',
          labelBackgroundStroke: 'rgba(255, 125, 0, 0.1)',
          labelBackgroundRadius: 3,
          labelPadding: [1, 4]
        }
      },
      behaviors: [
        'drag-element',
        'zoom-canvas',
        'drag-canvas',
        {
          type: 'hover-activate',
          degree: 1,
          state: 'active'
        }
      ]
    })

    graphInstance.on('node:click', (evt) => {
      const id = evt?.target?.id || evt?.data?.id || null
      if (!id) return
      state.selectedNodeId = id
      state.detailOpen = true
      // 重新渲染以应用聚光灯效果
      graphInstance.setData(getG6Data())
      graphInstance.render()
    })

    graphInstance.on('canvas:click', () => {
      state.selectedNodeId = null
      state.detailOpen = false
      // 清除聚光灯效果
      graphInstance.setData(getG6Data())
      graphInstance.render()
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
/* ==================== 探险地图风格 ==================== */
.graph-explorer {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--layout-bg-color, #F7F8FA);
  overflow: hidden;
}

/* 画布背景 - 图纸/地图网格 */
.canvas-grid {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.08) 1px, transparent 1px),
    radial-gradient(circle, rgba(255, 125, 0, 0.04) 1px, transparent 1px);
  background-size: 20px 20px, 100px 100px;
  background-position: 0 0, 10px 10px;
  pointer-events: none;
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

/* ==================== 空状态 ==================== */
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
}

.empty-content {
  text-align: center;
  pointer-events: auto;
  padding: 40px;
  /* 去卡片化 - 直接浮在网格背景上 */

  .empty-illustration {
    margin-bottom: 24px;

    .empty-icon-group {
      position: relative;
      width: 120px;
      height: 120px;
      margin: 0 auto;
      animation: float 3s ease-in-out infinite;
      /* 白色光晕 - 洗掉背景点阵，让图标"着陆" */
      background: radial-gradient(circle, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0.4) 40%, transparent 70%);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;

      .icon-main {
        font-size: 56px;
        color: var(--primary-color, #FF7D00);
        filter: drop-shadow(0 2px 8px rgba(255, 125, 0, 0.3));
      }

      .icon-accent {
        position: absolute;
        bottom: 12px;
        right: 12px;
        font-size: 24px;
        color: var(--primary-color, #FF7D00);
        background: #fff;
        border-radius: 50%;
        padding: 6px;
        box-shadow: 0 2px 8px rgba(255, 125, 0, 0.2);
      }
    }
  }

  .empty-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-color, #333);
    margin: 0 0 8px;
  }

  .empty-desc {
    font-size: 14px;
    color: var(--gray-500, #888);
    margin: 0 0 28px;
  }

  /* 按钮呼吸感优化 */
  :deep(.ant-btn-primary) {
    padding: 0 24px;
    height: 44px;
    font-size: 15px;
    border-radius: 22px;

    .anticon {
      margin-right: 8px;
    }
  }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

/* ==================== 悬浮搜索栏 ==================== */
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
}

.search-bar {
  display: flex;
  align-items: center;
  width: 420px;
  max-width: calc(100vw - 48px);
  padding: 6px 6px 6px 14px;
  background: #fff;
  border: 1px solid var(--border-color, #e5e5e5);
  border-radius: 24px;
  box-shadow: 0 4px 20px rgba(255, 125, 0, 0.08);
  transition: all 0.2s ease;

  &:focus-within {
    border-color: var(--primary-color, #FF7D00);
    box-shadow: 0 4px 20px rgba(255, 125, 0, 0.15);
  }

  /* 状态呼吸灯 */
  .status-breath {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #52c41a;
    margin-right: 10px;
    flex-shrink: 0;
    animation: breath 2s ease-in-out infinite;

    &.warning {
      background: #faad14;
    }

    &.offline {
      background: #ff4d4f;
      animation: none;
    }
  }

  input {
    flex: 1;
    border: none;
    background: transparent;
    font-size: 14px;
    color: var(--text-color, #333);
    outline: none;

    &::placeholder {
      color: var(--gray-400, #999);
    }
  }

  .search-btn {
    width: 36px;
    height: 36px;
    padding: 0;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
}

@keyframes breath {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(0.9); }
}

/* 节点统计 */
.node-stats {
  font-size: 12px;
  color: var(--gray-500, #888);
  padding: 4px 12px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

/* ==================== 底部工具栏 - 轻量化胶囊条 ==================== */
.floating-toolbar {
  position: absolute;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  background: #fff;
  border: 1px solid var(--border-color, #e5e5e5);
  border-radius: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 2px;
}

.toolbar-divider {
  width: 1px;
  height: 20px;
  background: var(--border-color, #e5e5e5);
  margin: 0 6px;
}

.toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: var(--gray-600, #666);
  font-size: 15px;
  cursor: pointer;
  transition: all 0.15s ease;
  text-decoration: none;

  &:hover {
    background: var(--gray-100, #f5f5f5);
    color: var(--text-color, #333);
  }

  &.active {
    background: rgba(255, 125, 0, 0.1);
    color: var(--primary-color, #FF7D00);
  }
}

/* 采样弹出框 */
.sample-popover {
  width: 200px;

  .sample-label {
    font-size: 12px;
    color: var(--gray-600, #666);
    margin-bottom: 8px;
  }

  :deep(.ant-slider) {
    margin: 8px 0 16px;
  }
}

/* ==================== 右侧详情面板 ==================== */
.detail-panel {
  position: absolute;
  top: 0;
  right: 0;
  width: 300px;
  height: 100%;
  background: #fff;
  border-left: 1px solid var(--border-color, #e5e5e5);
  box-shadow: -4px 0 20px rgba(0, 0, 0, 0.05);
  z-index: 200;
  overflow-y: auto;
  padding: 20px;
}

.detail-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 20px;

  .detail-avatar {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #FF7D00, #FFA940);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(255, 125, 0, 0.25);

    .avatar-text {
      font-size: 18px;
      font-weight: 600;
      color: white;
    }
  }

  .detail-title {
    flex: 1;
    min-width: 0;

    h3 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      color: var(--text-color, #333);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .detail-id {
      font-size: 11px;
      color: var(--gray-500, #999);
    }
  }

  .close-btn {
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 8px;
    background: var(--gray-100, #f5f5f5);
    color: var(--gray-500, #999);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;

    &:hover {
      background: var(--gray-200, #e5e5e5);
      color: var(--text-color, #333);
    }
  }
}

.detail-stats {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;

  .stat-item {
    flex: 1;
    padding: 14px;
    background: var(--gray-50, #fafafa);
    border-radius: 12px;
    text-align: center;

    .stat-value {
      display: block;
      font-size: 22px;
      font-weight: 600;
      color: var(--primary-color, #FF7D00);
    }

    .stat-label {
      font-size: 11px;
      color: var(--gray-500, #999);
    }
  }
}

.detail-section {
  .section-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-color, #333);
    margin-bottom: 10px;
  }

  .empty-neighbors {
    font-size: 13px;
    color: var(--gray-500, #999);
    text-align: center;
    padding: 20px 0;
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
    background: var(--gray-50, #fafafa);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    text-align: left;

    &:hover {
      background: rgba(255, 125, 0, 0.08);

      .neighbor-arrow {
        opacity: 1;
        transform: translateX(0);
      }
    }

    .neighbor-avatar {
      width: 28px;
      height: 28px;
      border-radius: 8px;
      background: rgba(255, 125, 0, 0.15);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: 600;
      color: var(--primary-color, #FF7D00);
      flex-shrink: 0;
    }

    .neighbor-name {
      flex: 1;
      font-size: 13px;
      color: var(--text-color, #333);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .neighbor-arrow {
      font-size: 11px;
      color: var(--gray-400, #bbb);
      opacity: 0;
      transform: translateX(-4px);
      transition: all 0.15s ease;
    }
  }
}

/* ==================== 加载遮罩 ==================== */
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.8);
  z-index: 300;

  .loading-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: var(--text-color, #333);

    .loading-icon {
      font-size: 28px;
      color: var(--primary-color, #FF7D00);
    }
  }
}

/* ==================== 警告横幅 ==================== */
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
  background: #fffbe6;
  border: 1px solid #ffe58f;
  border-radius: 8px;
  font-size: 13px;
  color: #ad6800;
}

/* ==================== 过渡动画 ==================== */
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

/* ==================== 响应式 ==================== */
@media (max-width: 768px) {
  .floating-search .search-bar {
    width: calc(100vw - 32px);
  }

  .detail-panel {
    width: 100%;
    border-left: none;
    border-top: 1px solid var(--border-color, #e5e5e5);
    height: 55%;
    top: auto;
    bottom: 0;
    border-radius: 16px 16px 0 0;
  }

  .floating-toolbar {
    bottom: 16px;
    padding: 5px 8px;

    .toolbar-btn {
      width: 28px;
      height: 28px;
    }
  }
}
</style>
