<template>
  <div class="workbench">
    <a-alert
      v-if="!backendOnline"
      type="warning"
      show-icon
      message="后端未启动/不可用（离线模式）"
      style="margin-bottom: 12px"
    />
    <a-alert
      v-else-if="!canUseKb"
      type="warning"
      show-icon
      message="后端未启用知识库功能（enable_knowledge_base=false），无法写入索引"
      style="margin-bottom: 12px"
    />

    <div class="topbar ui-card">
      <div class="topbar-left">
        <div class="label">目标知识库</div>
        <a-select
          v-model:value="selectedDbId"
          style="min-width: 260px"
          :disabled="!backendOnline"
          placeholder="选择一个知识库"
          @change="onDatabaseChange"
        >
          <a-select-option v-for="db in databases" :key="db.db_id" :value="db.db_id">
            {{ db.name }} ({{ db.db_id }})
          </a-select-option>
        </a-select>
        <a-button size="small" @click="loadDatabases" :loading="state.loadingDatabases">刷新</a-button>
      </div>
      <div class="topbar-right ui-muted">
        <span v-if="selectedDb">向量模型：{{ selectedDb.embed_model || '-' }} · 维度：{{ selectedDb.dimension || '-' }}</span>
      </div>
    </div>

    <a-tabs v-model:activeKey="state.activeTab" type="card" class="workbench-tabs">
      <a-tab-pane key="ingest">
        <template #tab><span><CloudUploadOutlined /> 切块与索引</span></template>

        <div class="ingest-grid">
          <div class="panel ui-card">
            <div class="panel-title">切块参数</div>
            <a-form layout="vertical">
              <a-form-item label="Chunk Size">
                <a-input-number v-model:value="ingestParams.chunkSize" :min="50" :max="10000" style="width: 100%" />
                <div class="hint ui-muted">每个分块的最大字符数</div>
              </a-form-item>
              <a-form-item label="Chunk Overlap">
                <a-input-number v-model:value="ingestParams.chunkOverlap" :min="0" :max="2000" style="width: 100%" />
                <div class="hint ui-muted">相邻分块的重叠字符数</div>
              </a-form-item>
              <a-form-item>
                <a-space>
                  <a-switch v-model:checked="ingestParams.doOcr" :disabled="!backendOnline" />
                  <span>启用 OCR（PDF/图片扫描件）</span>
                </a-space>
              </a-form-item>

              <a-divider style="margin: 12px 0" />

              <a-space wrap>
                <a-button
                  type="primary"
                  @click="chunkFiles"
                  :loading="state.chunking"
                  :disabled="uploadedFiles.length === 0"
                >
                  生成分块
                </a-button>
                <a-button @click="resetIngest" :disabled="uploadedFiles.length === 0 && chunkResults.length === 0">
                  清空
                </a-button>
              </a-space>
            </a-form>
          </div>

          <div class="panel ui-card">
            <div class="panel-title">上传文件</div>
            <a-upload-dragger
              v-model:fileList="fileList"
              name="file"
              :multiple="true"
              :disabled="state.uploading"
              :customRequest="customUpload"
            >
              <p class="ant-upload-text">点击或拖拽文件到这里上传</p>
              <p class="ant-upload-hint">支持 .pdf/.txt/.md 等。上传完成后点击“生成分块”。</p>
            </a-upload-dragger>

            <div class="ingest-actions">
              <a-space wrap>
                <a-button
                  type="primary"
                  @click="indexToDatabase"
                  :loading="state.indexing"
                  :disabled="!canIndex"
                >
                  写入索引
                </a-button>
                <span class="ui-muted" v-if="chunkResults.length > 0">
                  {{ chunkResults.length }} 个文件 · {{ totalChunks }} 个分块
                </span>
              </a-space>
            </div>
          </div>
        </div>

        <div v-if="chunkResults.length > 0" class="preview ui-card">
          <div class="preview-title">
            分块预览
            <span class="ui-muted">（点击文件展开查看）</span>
          </div>
          <a-collapse v-model:activeKey="activeFileKeys">
            <a-collapse-panel
              v-for="file in chunkResults"
              :key="file.file_id"
              :header="`${file.filename}（${file.nodes.length}）`"
            >
              <div class="chunk-grid">
                <div v-for="(node, idx) in file.nodes" :key="idx" class="chunk-card">
                  <div class="chunk-meta">#{{ idx + 1 }}</div>
                  <div class="chunk-text">{{ node.text }}</div>
                </div>
              </div>
            </a-collapse-panel>
          </a-collapse>
        </div>
      </a-tab-pane>

      <a-tab-pane key="pdf">
        <template #tab><span><FilePdfOutlined /> PDF 转文本</span></template>

        <div class="pdf-grid">
          <div class="panel ui-card">
            <div class="panel-title">上传 PDF</div>
            <a-upload-dragger
              v-model:fileList="pdfFileList"
              name="file"
              :max-count="1"
              :disabled="state.pdfUploading"
              :customRequest="customPdfUpload"
            >
              <p class="ant-upload-text">点击或拖拽 PDF 文件到这里上传</p>
              <p class="ant-upload-hint">用于检查解析质量（OCR/抽取）。</p>
            </a-upload-dragger>

            <div class="pdf-actions">
              <a-space wrap>
                <a-button type="primary" @click="convertPdf" :loading="state.converting" :disabled="pdfFileList.length === 0">
                  开始转换
                </a-button>
                <a-button @click="resetPdf" :disabled="pdfFileList.length === 0 && !pdfText">清空</a-button>
              </a-space>
            </div>
          </div>

          <div class="panel ui-card">
            <div class="panel-title">输出文本</div>
            <a-textarea :value="pdfText" :auto-size="{ minRows: 10, maxRows: 22 }" readonly />
            <div class="hint ui-muted" style="margin-top: 8px">
              字符数：{{ pdfText.length }} · Token 估算：{{ estimateTokens(pdfText) }}
            </div>
          </div>
        </div>
      </a-tab-pane>

      <a-tab-pane key="text">
        <template #tab><span><ScissorOutlined /> 文本试切块</span></template>

        <div class="text-grid">
          <div class="panel ui-card">
            <div class="panel-title">输入文本</div>
            <a-textarea v-model:value="plainText" :auto-size="{ minRows: 8, maxRows: 16 }" placeholder="粘贴一段文本用于试切块（离线可用）" />
            <div class="hint ui-muted" style="margin-top: 8px">
              字符数：{{ plainText.length }} · Token 估算：{{ estimateTokens(plainText) }}
            </div>
          </div>
          <div class="panel ui-card">
            <div class="panel-title">预览</div>
            <a-space wrap style="margin-bottom: 10px">
              <a-button type="primary" @click="chunkPlainTextNow" :disabled="!plainText">生成分块</a-button>
              <a-button @click="resetPlainText" :disabled="!plainText && plainChunks.length === 0">清空</a-button>
              <span class="ui-muted" v-if="plainChunks.length > 0">{{ plainChunks.length }} 个分块</span>
            </a-space>
            <div class="chunk-grid" v-if="plainChunks.length > 0">
              <div v-for="(c, idx) in plainChunks" :key="idx" class="chunk-card">
                <div class="chunk-meta">#{{ idx + 1 }}</div>
                <div class="chunk-text">{{ c.text }}</div>
              </div>
            </div>
            <a-empty v-else description="暂无分块" />
          </div>
        </div>
      </a-tab-pane>
    </a-tabs>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { message } from 'ant-design-vue'
import { CloudUploadOutlined, FilePdfOutlined, ScissorOutlined } from '@ant-design/icons-vue'
import { apiFetch } from '@/api/http'
import { useConfigStore } from '@/stores/config'
import { chunkPlainText } from '@/utils/chunking'

const configStore = useConfigStore()

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const canUseKb = computed(
  () => Boolean(configStore.config.backend?.online) && Boolean(configStore.config.enable_knowledge_base)
)

const state = reactive({
  loadingDatabases: false,
  uploading: false,
  chunking: false,
  indexing: false,
  activeTab: 'ingest',

  pdfUploading: false,
  converting: false
})

const databases = ref([])
const selectedDbId = ref(null)

const selectedDb = computed(() => (databases.value || []).find((d) => d.db_id === selectedDbId.value) || null)

const loadDatabases = async () => {
  state.loadingDatabases = true
  try {
    const data = await apiFetch('/data/', { method: 'GET', timeoutMs: 8000 })
    databases.value = data?.databases || []
    if (!selectedDbId.value && databases.value.length > 0) {
      selectedDbId.value = databases.value[0].db_id
    }
  } catch (e) {
    databases.value = []
    if (backendOnline.value) message.error(e?.message || '获取知识库列表失败')
  } finally {
    state.loadingDatabases = false
  }
}

const onDatabaseChange = () => {
  // When switching database, keep workbench state but uploads may be stored under the previous db.
  // That is OK because indexing is controlled by selectedDbId at write time.
}

// ---------------------------------------------------------------------------
// Ingest: upload -> chunk -> index
// ---------------------------------------------------------------------------

const ingestParams = reactive({
  chunkSize: 1000,
  chunkOverlap: 200,
  doOcr: false
})

const fileList = ref([])
const chunkResults = ref([])
const activeFileKeys = ref([])

const uploadedFiles = computed(() =>
  (fileList.value || []).filter((f) => f && (f.status === 'done' || f.response))
)

const totalChunks = computed(() =>
  (chunkResults.value || []).reduce((sum, f) => sum + (Array.isArray(f.nodes) ? f.nodes.length : 0), 0)
)

const canIndex = computed(() => {
  if (!canUseKb.value) return false
  if (!selectedDbId.value) return false
  return (chunkResults.value || []).length > 0
})

const customUpload = async ({ file, onSuccess, onError }) => {
  state.uploading = true
  try {
    // Prefer backend upload (so /data/file-to-chunk and /tools/pdf2txt can access the server path).
    if (!backendOnline.value) throw new Error('offline')
    const formData = new FormData()
    formData.append('file', file)
    if (selectedDbId.value) formData.append('db_id', String(selectedDbId.value))
    const data = await apiFetch('/data/upload', { method: 'POST', body: formData, timeoutMs: 60000 })
    onSuccess?.(data, file)
    return
  } catch (e) {
    // Local fallback (text files only).
    try {
      const content = await file.text()
      onSuccess?.({ file_path: file.name, file_content: content }, file)
      return
    } catch (e2) {
      onError?.(e2)
    }
  } finally {
    state.uploading = false
  }
}

const chunkFiles = async () => {
  const files = uploadedFiles.value
  if (files.length === 0) {
    message.error('请先上传文件')
    return
  }

  state.chunking = true
  chunkResults.value = []
  activeFileKeys.value = []

  try {
    const results = await Promise.all(
      files.map(async (f) => {
        const resp = f?.response || {}
        const filePath = resp.file_path || f.name
        const fileContent = resp.file_content

        let nodes = []
        // If we already have text content (offline/local), chunk in-browser.
        if (typeof fileContent === 'string' && fileContent.trim()) {
          nodes = chunkPlainText(fileContent, {
            chunkSize: ingestParams.chunkSize,
            chunkOverlap: ingestParams.chunkOverlap
          }).map((c) => ({ text: c.text, meta: c.meta }))
        } else {
          const data = await apiFetch('/data/file-to-chunk', {
            method: 'POST',
            body: {
              file: String(filePath),
              chunk_size: ingestParams.chunkSize,
              chunk_overlap: ingestParams.chunkOverlap,
              do_ocr: Boolean(ingestParams.doOcr)
            },
            timeoutMs: 120000
          })
          nodes = data?.chunks || []
        }

        const fileId = String(f.uid || filePath || f.name)
        return { filename: f.name, file_id: fileId, file_path: String(filePath), nodes }
      })
    )

    chunkResults.value = results
    activeFileKeys.value = results.slice(0, 1).map((r) => r.file_id)
  } catch (e) {
    console.error(e)
    message.error(e?.message || '分块失败')
  } finally {
    state.chunking = false
  }
}

const indexToDatabase = async () => {
  if (!selectedDbId.value) {
    message.error('请选择目标知识库')
    return
  }
  if (chunkResults.value.length === 0) {
    message.error('请先生成分块')
    return
  }

  state.indexing = true
  try {
    const file_chunks = {}
    for (const f of chunkResults.value) {
      file_chunks[f.file_id] = { filename: f.filename, file_id: f.file_id, nodes: f.nodes }
    }

    const data = await apiFetch('/data/add-by-chunks', {
      method: 'POST',
      body: { db_id: selectedDbId.value, file_chunks },
      timeoutMs: 300000
    })

    if (data?.status === 'failed') {
      message.error(data?.message || '写入失败')
      return
    }
    message.success(data?.message || '已写入索引')
    // Keep uploaded files so users can re-run with different params if needed.
  } catch (e) {
    console.error(e)
    message.error(e?.message || '写入失败')
  } finally {
    state.indexing = false
  }
}

const resetIngest = () => {
  fileList.value = []
  chunkResults.value = []
  activeFileKeys.value = []
}

// ---------------------------------------------------------------------------
// PDF -> text
// ---------------------------------------------------------------------------

const pdfFileList = ref([])
const pdfText = ref('')

const customPdfUpload = async ({ file, onSuccess, onError }) => {
  state.pdfUploading = true
  try {
    if (!backendOnline.value) throw new Error('offline')
    const formData = new FormData()
    formData.append('file', file)
    const data = await apiFetch('/data/upload', { method: 'POST', body: formData, timeoutMs: 60000 })
    onSuccess?.(data, file)
    return
  } catch (e) {
    // Local fallback: only for non-pdf text preview
    try {
      const isPdf = String(file.name || '').toLowerCase().endsWith('.pdf')
      const content = isPdf ? '' : await file.text()
      onSuccess?.({ file_path: file.name, file_content: content }, file)
    } catch (e2) {
      onError?.(e2)
    }
  } finally {
    state.pdfUploading = false
  }
}

const convertPdf = async () => {
  if (pdfFileList.value.length === 0) {
    message.error('请先上传文件')
    return
  }
  const f = pdfFileList.value.find((x) => x.status === 'done') || pdfFileList.value[0]
  const resp = f?.response || {}
  const filePath = resp.file_path || f.name
  const fileContent = resp.file_content

  state.converting = true
  try {
    if (fileContent && typeof fileContent === 'string' && fileContent.trim()) {
      pdfText.value = String(fileContent)
      return
    }

    const data = await apiFetch('/tools/pdf2txt', {
      method: 'POST',
      body: { file: String(filePath) },
      timeoutMs: 120000
    })
    pdfText.value = String(data?.text || '')
  } catch (e) {
    console.error(e)
    message.error(e?.message || '转换失败')
  } finally {
    state.converting = false
  }
}

const resetPdf = () => {
  pdfFileList.value = []
  pdfText.value = ''
}

// ---------------------------------------------------------------------------
// Plain text chunking (offline-friendly)
// ---------------------------------------------------------------------------

const plainText = ref('')
const plainChunks = ref([])

const chunkPlainTextNow = () => {
  plainChunks.value = chunkPlainText(plainText.value, {
    chunkSize: ingestParams.chunkSize,
    chunkOverlap: ingestParams.chunkOverlap
  })
}

const resetPlainText = () => {
  plainText.value = ''
  plainChunks.value = []
}

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

const estimateTokens = (text) => {
  const s = String(text || '')
  let tokenCount = 0
  for (const ch of s) {
    if (/[\u4e00-\u9fff]/.test(ch)) tokenCount += 1
    else if (/[a-zA-Z]/.test(ch)) tokenCount += 0.25
    else tokenCount += 0.5
  }
  return Math.ceil(tokenCount)
}

onMounted(() => {
  loadDatabases()
})
</script>

<style scoped lang="less">
.workbench {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  gap: 12px;
}

.topbar-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;

  .label {
    font-weight: 600;
  }
}

.workbench-tabs :deep(.ant-tabs-content) {
  padding-top: 8px;
}

.ingest-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 12px;
}

.pdf-grid,
.text-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.panel {
  padding: 12px;
}

.panel-title {
  font-weight: 650;
  margin-bottom: 10px;
}

.hint {
  font-size: 12px;
  margin-top: 6px;
}

.ingest-actions,
.pdf-actions {
  margin-top: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.preview {
  padding: 12px;
}

.preview-title {
  font-weight: 650;
  margin-bottom: 10px;
}

.chunk-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 10px;
}

.chunk-card {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  background: color-mix(in srgb, var(--main-500) 4%, var(--surface-color));
  box-shadow: var(--shadow-xs);
}

.chunk-meta {
  font-size: 12px;
  color: var(--subtext-color);
  margin-bottom: 6px;
}

.chunk-text {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
}

@media (max-width: 980px) {
  .ingest-grid {
    grid-template-columns: 1fr;
  }
  .pdf-grid,
  .text-grid {
    grid-template-columns: 1fr;
  }
}
</style>

