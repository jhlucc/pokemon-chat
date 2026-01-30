<template>
  <div class="supported-formats">
    <div class="formats-header">
      <span class="formats-title">支持的文件格式</span>
      <a-tooltip v-if="ocrAvailable" title="OCR 服务可用">
        <a-tag color="green">OCR ✓</a-tag>
      </a-tooltip>
    </div>
    <div class="formats-grid">
      <div v-for="fmt in formats" :key="fmt.ext" class="format-item">
        <span class="format-ext">{{ fmt.ext }}</span>
        <span class="format-parser">{{ fmt.parser }}</span>
        <a-tag v-if="fmt.ocr_support" size="small" color="blue">OCR</a-tag>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { apiFetch } from '@/api/http'

const formats = ref([
  { ext: '.pdf', parser: 'PDF Parser', ocr_support: true },
  { ext: '.docx', parser: 'MarkItDown', ocr_support: false },
  { ext: '.txt', parser: 'Direct', ocr_support: false },
  { ext: '.md', parser: 'MarkItDown', ocr_support: false },
])
const ocrAvailable = ref(false)

onMounted(async () => {
  try {
    const data = await apiFetch('/data/parsers', { method: 'GET', timeoutMs: 5000 })
    if (data?.supported_formats) {
      formats.value = data.supported_formats
    }
    ocrAvailable.value = data?.ocr_available || false
  } catch {
    // Use defaults
  }
})
</script>

<style scoped lang="less">
.supported-formats {
  padding: 12px 16px;
  background: color-mix(in srgb, var(--primary-color) 4%, transparent);
  border-radius: 12px;
  border: 1px solid color-mix(in srgb, var(--primary-color) 10%, transparent);
}

.formats-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.formats-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--gray-600);
}

.formats-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.format-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  background: rgba(255, 255, 255, 0.6);
  border-radius: 6px;
  font-size: 12px;
}

.format-ext {
  font-weight: 600;
  color: var(--primary-color);
}

.format-parser {
  color: var(--gray-500);
  font-size: 11px;
}

:root[data-theme='dark'] {
  .format-item {
    background: rgba(255, 255, 255, 0.08);
  }
}
</style>
