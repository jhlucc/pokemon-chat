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
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

const place = ref('')
const loading = ref(false)
let map, markersLayer

onMounted(() => {
  map = L.map('map').setView([20, 0], 2)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map)
  markersLayer = L.layerGroup().addTo(map)
})

const handleSearch = async () => {
  if (!place.value.trim()) return
  loading.value = true
  try {
    const resp = await fetch(`/api/mcp/coords?place=${encodeURIComponent(place.value)}`)
    if (!resp.ok) throw new Error(await resp.text())
    const data = await resp.json()
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
  background-color: #f5f7fa;
}

.search-bar {
  padding: 24px;
  display: flex;
  justify-content: center;
}

.search-bar > .ant-input-search {
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  overflow: hidden;
  transition: box-shadow 0.3s ease;
}

.search-bar > .ant-input-search:hover {
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

.map-container {
  flex: 1;
  border-top: 1px solid #dcdfe6;
  min-height: 500px;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.05);
}

.leaflet-popup-content-wrapper {
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
</style>
