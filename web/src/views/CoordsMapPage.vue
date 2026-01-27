<template>
  <div class="coords-page">
    <!-- 背景层：暖光氛围 -->
    <div class="ambient-glow glow--orange"></div>
    <div class="ambient-glow glow--purple"></div>
    <div class="dot-grid"></div>

    <!-- 悬浮搜索栏 - 毛玻璃胶囊 -->
    <div class="search-float">
      <div class="search-capsule">
        <input
          v-model="place"
          type="text"
          class="search-input"
          placeholder="输入地点 / 宝可梦名，例如：皮卡丘"
          @keydown.enter="handleSearch"
        />
        <button
          class="search-btn"
          :class="{ loading }"
          :disabled="loading"
          @click="handleSearch"
        >
          <span v-if="!loading">搜索</span>
          <span v-else class="loading-spinner"></span>
        </button>
      </div>
    </div>

    <!-- 返回首页按钮 -->
    <router-link to="/" class="back-btn">
      <LeftOutlined />
    </router-link>

    <!-- 地图容器 - 全屏沉浸式 -->
    <div class="map-wrapper">
      <a-spin :spinning="leafletLoading" tip="加载地图组件...">
        <div id="map" class="map-container"></div>
      </a-spin>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import { LeftOutlined } from '@ant-design/icons-vue'
import { apiFetch } from '@/api/http'
import { notifyApiError } from '@/utils/notify'

const place = ref('')
const loading = ref(false)
const leafletLoading = ref(false)

let Leaflet = null
let map, markersLayer, tileLayer
let warnedTileError = false

const ensureLeaflet = async () => {
  if (Leaflet) return Leaflet
  leafletLoading.value = true
  try {
    const [mod] = await Promise.all([import('leaflet'), import('leaflet/dist/leaflet.css')])
    Leaflet = mod.default
    return Leaflet
  } catch {
    message.error('地图组件加载失败（Leaflet）')
    return null
  } finally {
    leafletLoading.value = false
  }
}

onMounted(async () => {
  const L = await ensureLeaflet()
  if (!L) return

  // 初始化地图
  map = L.map('map', {
    zoomControl: false // 禁用默认缩放控件，后面自定义
  }).setView([35, 105], 4) // 默认显示中国区域

  // 使用 CartoDB Voyager 暖色系地图瓦片
  tileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
  })
    .on('tileerror', () => {
      if (warnedTileError) return
      warnedTileError = true
      message.warning('地图瓦片加载失败（可能离线或网络受限），但坐标查询仍可用。')
    })
    .addTo(map)

  // 添加自定义缩放控件
  L.control.zoom({
    position: 'bottomright'
  }).addTo(map)

  markersLayer = L.layerGroup().addTo(map)
})

onUnmounted(() => {
  try {
    tileLayer?.off?.()
  } catch {
    // ignore
  }
  tileLayer = null

  try {
    markersLayer?.clearLayers?.()
  } catch {
    // ignore
  }
  markersLayer = null

  try {
    map?.remove?.()
  } catch {
    // ignore
  }
  map = null
})

const handleSearch = async () => {
  if (!place.value.trim()) return
  if (!map || !markersLayer) return
  loading.value = true
  try {
    const data = await apiFetch('/mcp/coords', {
      method: 'GET',
      query: { place: place.value },
      timeoutMs: 15000
    })
    renderCoords(data.coords)
  } catch (e) {
    notifyApiError(e, { context: '坐标查询', fallback: '查询失败' })
    markersLayer.clearLayers()
  } finally {
    loading.value = false
  }
}

function renderCoords(coords) {
  if (!Leaflet) return
  markersLayer.clearLayers()
  if (!coords.length) {
    message.warning('未查询到坐标')
    return
  }
  const bounds = []
  coords.forEach(({ lat, lng, location }) => {
    const marker = Leaflet.marker([lat, lng]).addTo(markersLayer)
    marker.bindPopup(`<b>${location}</b><br/>${lat.toFixed(4)}, ${lng.toFixed(4)}`)
    bounds.push([lat, lng])
  })
  if (bounds.length === 1) {
    map.setView(bounds[0], 8)
  } else {
    map.fitBounds(bounds, { padding: [40, 40] })
  }
}
</script>

<style scoped lang="less">
.coords-page {
  position: relative;
  width: 100%;
  height: 100vh;
  overflow: hidden;
  background: transparent;
}

/* 背景层：暖光氛围 */
.ambient-glow {
  position: fixed;
  border-radius: 50%;
  filter: blur(100px);
  pointer-events: none;
  z-index: 0;
  mix-blend-mode: normal;
  animation: glow-drift 20s ease-in-out infinite;
}

.glow--orange {
  width: 600px;
  height: 600px;
  top: -20%;
  left: -15%;
  background: radial-gradient(circle, rgba(255, 125, 0, 0.3) 0%, rgba(255, 180, 100, 0.12) 40%, transparent 70%);
}

.glow--purple {
  width: 500px;
  height: 500px;
  bottom: -10%;
  right: -10%;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(180, 150, 255, 0.08) 40%, transparent 70%);
  animation-delay: -7s;
}

@keyframes glow-drift {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(20px, -15px) scale(1.03); }
  66% { transform: translate(-15px, 15px) scale(0.97); }
}

/* 点阵背景 */
.dot-grid {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.04) 1px, transparent 1px),
    radial-gradient(circle, rgba(255, 125, 0, 0.02) 1px, transparent 1px);
  background-size: 24px 24px, 96px 96px;
  pointer-events: none;
  z-index: 0;
}

/* 返回按钮 */
.back-btn {
  position: fixed;
  top: 24px;
  left: 24px;
  z-index: 1001;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  border-radius: 50%;
  color: var(--gray-600);
  font-size: 18px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transition: all 0.2s ease;
  text-decoration: none;

  &:hover {
    background: rgba(255, 255, 255, 0.95);
    color: var(--primary-color);
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
  }
}

/* 悬浮搜索栏 */
.search-float {
  position: fixed;
  top: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1001;
  width: 90%;
  max-width: 480px;
}

.search-capsule {
  display: flex;
  align-items: center;
  padding: 6px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  border-radius: 100px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 4px 16px rgba(255, 125, 0, 0.06);
  transition: all 0.25s ease;

  &:focus-within {
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12), 0 0 0 3px rgba(255, 125, 0, 0.15);
  }
}

.search-input {
  flex: 1;
  padding: 10px 20px;
  border: none;
  background: transparent;
  font-size: 15px;
  color: var(--text-color);
  outline: none;

  &::placeholder {
    color: var(--gray-400);
  }
}

.search-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 80px;
  padding: 10px 24px;
  border: none;
  border-radius: 100px;
  background: linear-gradient(180deg, #FFA940 0%, #FF7D00 100%);
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(255, 125, 0, 0.35);
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    background: linear-gradient(180deg, #FFB347 0%, #FF8C1A 100%);
    box-shadow: 0 6px 18px rgba(255, 125, 0, 0.45);
    transform: translateY(-1px);
  }

  &:active:not(:disabled) {
    transform: translateY(0) scale(0.98);
  }

  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
}

.loading-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 地图容器 */
.map-wrapper {
  position: relative;
  z-index: 1;
  width: 100%;
  height: 100%;
  padding: 80px 24px 24px;
}

.map-container {
  width: 100%;
  height: 100%;
  border-radius: 24px;
  overflow: hidden;
  /* 内发光边框 - 全息投影感 */
  border: 3px solid rgba(255, 255, 255, 0.5);
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.08),
    inset 0 0 0 1px rgba(255, 255, 255, 0.3),
    0 4px 16px rgba(255, 125, 0, 0.05);
  background: var(--surface-color);
}

/* Leaflet 自定义缩放控件样式 */
:deep(.leaflet-control-zoom) {
  border: none !important;
  box-shadow: none !important;
  margin: 16px !important;
}

:deep(.leaflet-control-zoom a) {
  width: 40px !important;
  height: 40px !important;
  line-height: 40px !important;
  background: rgba(255, 255, 255, 0.9) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.6) !important;
  color: var(--gray-600) !important;
  font-size: 18px !important;
  font-weight: 500 !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
  transition: all 0.2s ease !important;
}

:deep(.leaflet-control-zoom a:first-child) {
  border-radius: 12px 12px 0 0 !important;
}

:deep(.leaflet-control-zoom a:last-child) {
  border-radius: 0 0 12px 12px !important;
}

:deep(.leaflet-control-zoom a:hover) {
  background: rgba(255, 255, 255, 1) !important;
  color: var(--primary-color) !important;
}

/* Leaflet Popup 样式 */
:deep(.leaflet-popup-content-wrapper) {
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  padding: 4px;
}

:deep(.leaflet-popup-content) {
  margin: 12px 16px;
  font-size: 14px;
  line-height: 1.5;
}

:deep(.leaflet-popup-tip) {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Leaflet Attribution 样式 */
:deep(.leaflet-control-attribution) {
  background: rgba(255, 255, 255, 0.7) !important;
  backdrop-filter: blur(8px);
  border-radius: 8px 0 0 0;
  padding: 4px 8px !important;
  font-size: 11px;
}

/* 响应式 */
@media (max-width: 640px) {
  .search-float {
    top: 16px;
    width: calc(100% - 80px);
    left: auto;
    right: 16px;
    transform: none;
  }

  .back-btn {
    top: 16px;
    left: 16px;
    width: 40px;
    height: 40px;
  }

  .map-wrapper {
    padding: 72px 12px 12px;
  }

  .map-container {
    border-radius: 20px;
  }
}

/* 暗色模式 */
:root[data-theme='dark'] {
  .ambient-glow {
    mix-blend-mode: screen;
    opacity: 0.7;
  }

  .glow--orange {
    background: radial-gradient(circle, rgba(255, 125, 0, 0.25) 0%, rgba(255, 180, 100, 0.1) 40%, transparent 70%);
  }

  .glow--purple {
    background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(180, 150, 255, 0.08) 40%, transparent 70%);
  }

  .dot-grid {
    background-image:
      radial-gradient(circle, rgba(255, 125, 0, 0.06) 1px, transparent 1px),
      radial-gradient(circle, rgba(255, 125, 0, 0.03) 1px, transparent 1px);
  }

  .back-btn {
    background: rgba(40, 40, 40, 0.85);
    border-color: rgba(255, 255, 255, 0.1);
    color: var(--gray-300);

    &:hover {
      background: rgba(50, 50, 50, 0.95);
      color: var(--primary-color);
    }
  }

  .search-capsule {
    background: rgba(40, 40, 40, 0.85);
    border-color: rgba(255, 255, 255, 0.1);
  }

  .search-input {
    color: var(--gray-200);

    &::placeholder {
      color: var(--gray-500);
    }
  }

  .map-container {
    border-color: rgba(255, 255, 255, 0.15);
    background: var(--surface-color);
  }

  :deep(.leaflet-control-zoom a) {
    background: rgba(40, 40, 40, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: var(--gray-300) !important;
  }

  :deep(.leaflet-control-zoom a:hover) {
    background: rgba(50, 50, 50, 1) !important;
    color: var(--primary-color) !important;
  }

  :deep(.leaflet-popup-content-wrapper) {
    background: var(--surface-color);
    color: var(--text-color);
  }

  :deep(.leaflet-popup-tip) {
    background: var(--surface-color);
  }

  :deep(.leaflet-control-attribution) {
    background: rgba(30, 30, 30, 0.7) !important;
    color: var(--gray-500);
  }
}
</style>
