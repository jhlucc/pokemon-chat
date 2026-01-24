<template>
  <div class="coords-page">
    <HeaderComponent title="📍 PokéMap" />

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

    <div id="map" class="map-container"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import { apiFetch } from '@/api/http'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

const place = ref('')
const loading = ref(false)
let map, markersLayer, tileLayer
let warnedTileError = false

onMounted(() => {
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
    const data = await apiFetch('/mcp/coords', { method: 'GET', query: { place: place.value }, timeoutMs: 15000 })
    renderCoords(data.coords)
  } catch (e) {
    console.error(e)
    message.error('查询失败: ' + (e.message || e))
    markersLayer.clearLayers()
  } finally {
    loading.value = false
  }
}

function renderCoords(coords) {
  markersLayer.clearLayers()
  if (!coords.length) {
    message.warning('未查询到坐标')
    return
  }
  const bounds = []
  coords.forEach(({ lat, lng, location }) => {
    const marker = L.marker([lat, lng]).addTo(markersLayer)
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
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: var(--layout-bg-color);
}

.search-bar {
  padding: 24px;
  display: flex;
  justify-content: center;
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
  flex: 1;
  border-top: 1px solid var(--border-color);
  min-height: 500px;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: var(--surface-color);
  box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.05);
}

.leaflet-popup-content-wrapper {
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
</style>
