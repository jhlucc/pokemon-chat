<template>
  <div class="coords-page layout-container">
    <HeaderComponent
      title="地图"
      description="坐标查询与可视化"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '地图' }]"
    />

    <div class="ui-page">
      <div class="ui-container">
        <div class="search-bar">
          <a-input-search
            v-model:value="place"
            placeholder="输入地点 / 宝可梦名，例如：皮卡丘"
            enter-button="搜索"
            @search="handleSearch"
            :loading="loading"
            style="max-width: 500px"
          />
        </div>

        <a-spin :spinning="leafletLoading" tip="加载地图组件...">
          <div id="map" class="map-container ui-card map-card"></div>
        </a-spin>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
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
  map = L.map('map').setView([20, 0], 2)
  tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  })
    .on('tileerror', () => {
      if (warnedTileError) return
      warnedTileError = true
      message.warning('地图瓦片加载失败（可能离线或网络受限），但坐标查询仍可用。')
    })
    .addTo(map)
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

<style scoped>
.coords-page {
  padding: 0;
}

.search-bar {
  display: flex;
  justify-content: center;
  margin-bottom: 12px;
}

.search-bar > .ant-input-search {
  box-shadow: var(--shadow-sm);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: box-shadow 0.3s ease;
}

.search-bar > .ant-input-search:hover {
  box-shadow: var(--shadow-md);
}

.map-container {
  width: 100%;
  height: min(72vh, 720px);
  min-height: 420px;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: var(--surface-color);
}

.map-card:hover {
  transform: none;
  background: var(--surface-color);
  box-shadow: var(--shadow-xs);
}

.leaflet-popup-content-wrapper {
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
</style>
