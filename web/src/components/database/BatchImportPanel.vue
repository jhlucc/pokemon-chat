<template>
  <GlassPanel title="批量导入" class="batch-import">
    <template #header-actions>
      <a-tooltip title="从服务器目录批量导入文件">
        <QuestionCircleOutlined />
      </a-tooltip>
    </template>

    <a-form layout="vertical">
      <a-form-item label="服务器目录路径">
        <a-input
          v-model:value="folderPath"
          placeholder="/path/to/documents"
        />
        <p class="field-hint">输入服务器上的绝对路径，适用于大批量文档处理</p>
      </a-form-item>

      <a-form-item label="文件后缀过滤（可选）">
        <a-select
          v-model:value="suffixes"
          mode="tags"
          placeholder="留空则导入所有支持的格式"
          :options="suffixOptions"
        />
      </a-form-item>

      <a-form-item label="切块参数">
        <div class="params-row">
          <div class="param">
            <span>Chunk Size</span>
            <a-input-number v-model:value="params.chunkSize" :min="100" :max="10000" />
          </div>
          <div class="param">
            <span>Overlap</span>
            <a-input-number v-model:value="params.chunkOverlap" :min="0" :max="2000" />
          </div>
          <div class="param">
            <span>启用 OCR</span>
            <a-switch v-model:checked="params.doOcr" />
          </div>
        </div>
      </a-form-item>

      <a-button
        type="primary"
        :loading="loading"
        :disabled="!folderPath || !dbId"
        @click="startImport"
        block
      >
        <ImportOutlined /> 开始导入
      </a-button>
    </a-form>

    <div v-if="result" class="import-result">
      <a-result
        :status="result.status === 'success' ? 'success' : 'error'"
        :title="result.status === 'success' ? '导入完成' : '导入失败'"
        :sub-title="`共导入 ${result.file_ids?.length || 0} 个文件`"
      />
    </div>
  </GlassPanel>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { message } from 'ant-design-vue'
import { ImportOutlined, QuestionCircleOutlined } from '@ant-design/icons-vue'
import GlassPanel from '@/components/common/GlassPanel.vue'
import { apiFetch } from '@/api/http'

const props = defineProps({
  dbId: { type: String, required: true }
})

const folderPath = ref('')
const suffixes = ref([])
const loading = ref(false)
const result = ref(null)

const params = reactive({
  chunkSize: 1000,
  chunkOverlap: 200,
  doOcr: false
})

const suffixOptions = [
  { value: '.pdf', label: 'PDF' },
  { value: '.docx', label: 'Word' },
  { value: '.txt', label: 'Text' },
  { value: '.md', label: 'Markdown' },
  { value: '.xlsx', label: 'Excel' },
]

const startImport = async () => {
  if (!folderPath.value || !props.dbId) return

  loading.value = true
  result.value = null

  try {
    const data = await apiFetch('/data/ingest/dir', {
      method: 'POST',
      body: {
        db_id: props.dbId,
        folder: folderPath.value,
        suffixes: suffixes.value.length > 0 ? suffixes.value : null
      },
      timeoutMs: 600000
    })
    result.value = data
    message.success(`成功导入 ${data.file_ids?.length || 0} 个文件`)
  } catch (e) {
    result.value = { status: 'error', message: e?.message }
    message.error(e?.message || '导入失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped lang="less">
.batch-import {
  margin-top: 16px;
}

.params-row {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}

.param {
  display: flex;
  flex-direction: column;
  gap: 6px;

  span {
    font-size: 12px;
    color: var(--gray-500);
  }
}

.import-result {
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid var(--border-color);
}

.field-hint {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--gray-400);
  line-height: 1.4;
}
</style>
