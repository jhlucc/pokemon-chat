<template>
  <div class="workbench">
    <!-- 警告提示 -->
    <a-alert
      v-if="!backendOnline"
      type="warning"
      show-icon
      message="后端未启动/不可用（离线模式）"
      class="workbench-alert"
    />
    <a-alert
      v-else-if="!canUseKb"
      type="warning"
      show-icon
      message="后端未启用知识库功能（enable_knowledge_base=false），无法写入索引"
      class="workbench-alert"
    />

    <!-- 顶部步骤条 - 醒目 -->
    <div class="stepper-section">
      <a-steps :current="currentStep" class="workbench-stepper">
        <a-step title="目标设定" description="选择知识库" />
        <a-step title="数据投喂" description="上传文档" />
        <a-step title="智能切分" description="生成分块" />
        <a-step title="索引构建" description="写入向量" />
      </a-steps>
    </div>

    <!-- 主布局：左配置 + 右操作 -->
    <div class="main-grid">
      <!-- 左侧：配置区 -->
      <div class="config-panel panel">
        <div class="panel-header">
          <span class="panel-title">配置区</span>
        </div>

        <!-- 目标知识库 -->
        <div class="config-section">
          <label class="config-label">目标知识库</label>
          <div class="config-row">
            <a-select
              v-model:value="selectedDbId"
              style="flex: 1"
              :disabled="!backendOnline"
              placeholder="选择一个知识库"
              @change="onDatabaseChange"
            >
              <a-select-option v-for="db in databases" :key="db.db_id" :value="db.db_id">
                {{ db.name }}
              </a-select-option>
            </a-select>
            <a-button @click="loadDatabases" :loading="state.loadingDatabases">
              <ReloadOutlined />
            </a-button>
          </div>
          <div v-if="selectedDb" class="config-hint">
            向量模型：{{ selectedDb.embed_model || '-' }} · 维度：{{ selectedDb.dimension || '-' }}
          </div>
        </div>

        <a-divider style="margin: 16px 0" />

        <!-- 切块参数 -->
        <div class="config-section">
          <label class="config-label">切块参数</label>

          <div class="param-item">
            <span class="param-name">Chunk Size</span>
            <a-input-number v-model:value="ingestParams.chunkSize" :min="50" :max="10000" style="width: 100%" />
            <span class="param-hint">每个分块的最大字符数</span>
          </div>

          <div class="param-item">
            <span class="param-name">Chunk Overlap</span>
            <a-input-number v-model:value="ingestParams.chunkOverlap" :min="0" :max="2000" style="width: 100%" />
            <span class="param-hint">相邻分块的重叠字符数</span>
          </div>

          <div class="param-item param-item--switch">
            <a-switch v-model:checked="ingestParams.doOcr" :disabled="!backendOnline" />
            <span class="param-name">启用 OCR</span>
            <span class="param-hint">PDF/图片扫描件</span>
          </div>
        </div>
      </div>

      <!-- 右侧：操作区 -->
      <div class="upload-panel panel">
        <div class="panel-header">
          <span class="panel-title">数据投喂</span>
          <span v-if="uploadedFiles.length > 0" class="panel-badge">{{ uploadedFiles.length }} 文件</span>
        </div>

        <!-- 上传区域 -->
        <a-upload-dragger
          v-model:fileList="fileList"
          name="file"
          :multiple="true"
          :disabled="state.uploading"
          :customRequest="customUpload"
          class="upload-dragger"
        >
          <div class="upload-content">
            <div class="upload-icon">
              <InboxOutlined />
            </div>
            <p class="upload-title">拖入文档，喂给大脑</p>
            <p class="upload-hint">支持 .pdf / .txt / .md 等格式，上传后点击下方"生成分块"</p>
          </div>
        </a-upload-dragger>

        <!-- 分块预览 -->
        <div v-if="chunkResults.length > 0" class="preview-section">
          <div class="preview-header">
            <span class="preview-title">分块预览</span>
            <span class="preview-stats">{{ chunkResults.length }} 文件 · {{ totalChunks }} 个分块</span>
          </div>
          <a-collapse v-model:activeKey="activeFileKeys" class="preview-collapse">
            <a-collapse-panel
              v-for="file in chunkResults"
              :key="file.file_id"
              :header="`${file.filename}（${file.nodes.length} 块）`"
            >
              <div class="chunk-grid">
                <div v-for="(node, idx) in file.nodes.slice(0, 6)" :key="idx" class="chunk-card">
                  <div class="chunk-meta">#{{ idx + 1 }}</div>
                  <div class="chunk-text">{{ node.text }}</div>
                </div>
                <div v-if="file.nodes.length > 6" class="chunk-more">
                  还有 {{ file.nodes.length - 6 }} 个分块...
                </div>
              </div>
            </a-collapse-panel>
          </a-collapse>
        </div>
      </div>
    </div>

    <!-- 高级工具 -->
    <div class="advanced-section panel">
      <a-collapse v-model:activeKey="advancedKeys" class="advanced-collapse">
        <a-collapse-panel key="pdf" header="高级：PDF 转文本（检查解析质量）">
          <div class="advanced-grid">
            <div class="advanced-item">
              <label class="config-label">上传 PDF</label>
              <a-upload-dragger
                v-model:fileList="pdfFileList"
                name="file"
                :max-count="1"
                :disabled="state.pdfUploading"
                :customRequest="customPdfUpload"
                class="upload-dragger upload-dragger--mini"
              >
                <p class="ant-upload-text">点击或拖拽 PDF</p>
              </a-upload-dragger>
              <a-space style="margin-top: 12px">
                <a-button type="primary" @click="convertPdf" :loading="state.converting" :disabled="pdfFileList.length === 0">
                  开始转换
                </a-button>
                <a-button @click="resetPdf" :disabled="pdfFileList.length === 0 && !pdfText">清空</a-button>
              </a-space>
            </div>
            <div class="advanced-item">
              <label class="config-label">输出文本</label>
              <a-textarea :value="pdfText" :auto-size="{ minRows: 6, maxRows: 12 }" readonly placeholder="转换结果将显示在这里" />
              <div class="param-hint" style="margin-top: 8px">
                字符数：{{ pdfText.length }} · Token 估算：{{ estimateTokens(pdfText) }}
              </div>
            </div>
          </div>
        </a-collapse-panel>

        <a-collapse-panel key="text" header="高级：文本试切块（离线可用）">
          <div class="advanced-grid">
            <div class="advanced-item">
              <label class="config-label">输入文本</label>
              <a-textarea v-model:value="plainText" :auto-size="{ minRows: 6, maxRows: 12 }" placeholder="粘贴一段文本用于试切块" />
              <div class="param-hint" style="margin-top: 8px">
                字符数：{{ plainText.length }} · Token 估算：{{ estimateTokens(plainText) }}
              </div>
            </div>
            <div class="advanced-item">
              <label class="config-label">切块预览</label>
              <a-space style="margin-bottom: 12px">
                <a-button type="primary" @click="chunkPlainTextNow" :disabled="!plainText">生成分块</a-button>
                <a-button @click="resetPlainText" :disabled="!plainText && plainChunks.length === 0">清空</a-button>
                <span class="param-hint" v-if="plainChunks.length > 0">{{ plainChunks.length }} 个分块</span>
              </a-space>
              <div class="chunk-grid" v-if="plainChunks.length > 0">
                <div v-for="(c, idx) in plainChunks.slice(0, 4)" :key="idx" class="chunk-card">
                  <div class="chunk-meta">#{{ idx + 1 }}</div>
                  <div class="chunk-text">{{ c.text }}</div>
                </div>
              </div>
              <a-empty v-else description="暂无分块" :image="null" />
            </div>
          </div>
        </a-collapse-panel>
      </a-collapse>
    </div>

    <!-- 底部固定操作栏 -->
    <div class="action-bar">
      <div class="action-bar-inner">
        <div class="action-left">
          <span v-if="chunkResults.length > 0" class="action-status">
            <CheckCircleOutlined class="status-icon status-icon--success" />
            {{ totalChunks }} 个分块已就绪
          </span>
          <span v-else-if="uploadedFiles.length > 0" class="action-status">
            <FileTextOutlined class="status-icon" />
            {{ uploadedFiles.length }} 个文件待处理
          </span>
          <span v-else class="action-status action-status--muted">
            请上传文件开始
          </span>
        </div>
        <div class="action-right">
          <a-button @click="resetIngest" :disabled="uploadedFiles.length === 0 && chunkResults.length === 0">
            清空
          </a-button>
          <a-button
            @click="chunkFiles"
            :loading="state.chunking"
            :disabled="uploadedFiles.length === 0"
          >
            生成分块
          </a-button>
          <a-button
            type="primary"
            @click="indexToDatabase"
            :loading="state.indexing"
            :disabled="!canIndex"
            class="action-primary"
          >
            <ThunderboltOutlined />
            写入索引
          </a-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { message } from 'ant-design-vue'
import {
  InboxOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  FileTextOutlined
} from '@ant-design/icons-vue'
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

const currentStep = computed(() => {
  if (!selectedDbId.value) return 0
  if (uploadedFiles.value.length === 0) return 1
  if (chunkResults.value.length === 0) return 2
  return 3
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
  } catch {
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
  } catch {
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

const advancedKeys = ref([])
</script>

<style scoped lang="less">
.workbench {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-bottom: 100px; /* 为底部操作栏留空间 */
}

.workbench-alert {
  margin-bottom: 0;
}

/* 步骤条区域 */
.stepper-section {
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  border-radius: 20px;
  padding: 24px 32px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.workbench-stepper {
  :deep(.ant-steps-item-title) {
    font-weight: 600;
    font-size: 15px;
  }

  :deep(.ant-steps-item-description) {
    font-size: 12px;
    color: var(--gray-500);
  }

  :deep(.ant-steps-item-process .ant-steps-item-icon) {
    background: var(--primary-color);
    border-color: var(--primary-color);
  }

  :deep(.ant-steps-item-finish .ant-steps-item-icon) {
    border-color: var(--primary-color);

    .ant-steps-icon {
      color: var(--primary-color);
    }
  }

  :deep(.ant-steps-item-finish .ant-steps-item-tail::after) {
    background: var(--primary-color);
  }
}

/* 面板通用样式 */
.panel {
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color);
}

.panel-title {
  font-size: 16px;
  font-weight: 650;
  color: var(--text-color);
}

.panel-badge {
  font-size: 12px;
  padding: 2px 10px;
  background: rgba(255, 125, 0, 0.1);
  color: var(--primary-color);
  border-radius: 100px;
  font-weight: 500;
}

/* 主布局 */
.main-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 16px;
}

/* 配置区 */
.config-section {
  margin-bottom: 16px;

  &:last-child {
    margin-bottom: 0;
  }
}

.config-label {
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: var(--gray-600);
  margin-bottom: 8px;
}

.config-row {
  display: flex;
  gap: 8px;
}

.config-hint {
  margin-top: 8px;
  font-size: 12px;
  color: var(--gray-500);
}

.param-item {
  margin-bottom: 14px;

  &:last-child {
    margin-bottom: 0;
  }
}

.param-item--switch {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;

  .param-name {
    margin-bottom: 0;
  }
}

.param-name {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-color);
  margin-bottom: 6px;
}

.param-hint {
  font-size: 12px;
  color: var(--gray-500);
  margin-top: 4px;
}

/* 上传区域 */
.upload-dragger {
  :deep(.ant-upload-drag) {
    background: rgba(255, 125, 0, 0.02);
    border: 2px dashed rgba(255, 125, 0, 0.2);
    border-radius: 16px;
    transition: all 0.3s ease;

    &:hover {
      border-color: var(--primary-color);
      background: rgba(255, 125, 0, 0.04);
    }
  }

  :deep(.ant-upload-btn) {
    padding: 32px 20px !important;
  }
}

.upload-dragger--mini {
  :deep(.ant-upload-btn) {
    padding: 16px !important;
  }
}

.upload-content {
  text-align: center;
}

.upload-icon {
  font-size: 48px;
  color: var(--primary-color);
  margin-bottom: 12px;
  opacity: 0.8;
}

.upload-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-color);
  margin: 0 0 8px;
}

.upload-hint {
  font-size: 13px;
  color: var(--gray-500);
  margin: 0;
}

/* 分块预览 */
.preview-section {
  margin-top: 20px;
}

.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.preview-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-color);
}

.preview-stats {
  font-size: 12px;
  color: var(--gray-500);
}

.preview-collapse {
  :deep(.ant-collapse-header) {
    font-weight: 500;
  }
}

.chunk-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}

.chunk-card {
  padding: 12px;
  border-radius: 12px;
  background: rgba(255, 125, 0, 0.04);
  border: 1px solid rgba(255, 125, 0, 0.1);
}

.chunk-meta {
  font-size: 11px;
  font-weight: 600;
  color: var(--primary-color);
  margin-bottom: 6px;
}

.chunk-text {
  font-size: 12px;
  line-height: 1.5;
  color: var(--gray-600);
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.chunk-more {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px;
  font-size: 12px;
  color: var(--gray-500);
  background: var(--gray-100);
  border-radius: 12px;
}

/* 高级工具 */
.advanced-section {
  padding: 0;

  :deep(.ant-collapse) {
    background: transparent;
    border: none;
  }

  :deep(.ant-collapse-item) {
    border: none;
  }

  :deep(.ant-collapse-header) {
    padding: 16px 20px !important;
    font-weight: 600;
    color: var(--gray-600);
  }

  :deep(.ant-collapse-content-box) {
    padding: 0 20px 20px;
  }
}

.advanced-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.advanced-item {
  display: flex;
  flex-direction: column;
}

/* 底部操作栏 */
.action-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 100;
  padding: 0 24px 24px;
  pointer-events: none;
}

.action-bar-inner {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.5) inset;
  pointer-events: auto;
}

.action-left {
  display: flex;
  align-items: center;
}

.action-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-color);
}

.action-status--muted {
  color: var(--gray-500);
}

.status-icon {
  font-size: 16px;
  color: var(--gray-500);
}

.status-icon--success {
  color: #22C55E;
}

.action-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.action-primary {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 20px;
  height: 40px;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(255, 125, 0, 0.25);

  &:hover:not(:disabled) {
    box-shadow: 0 4px 16px rgba(255, 125, 0, 0.35);
  }
}

/* 响应式 */
@media (max-width: 900px) {
  .main-grid {
    grid-template-columns: 1fr;
  }

  .advanced-grid {
    grid-template-columns: 1fr;
  }

  .stepper-section {
    padding: 16px;
  }

  .workbench-stepper {
    :deep(.ant-steps-item-description) {
      display: none;
    }
  }
}

@media (max-width: 640px) {
  .action-bar {
    padding: 0 12px 12px;
  }

  .action-bar-inner {
    flex-direction: column;
    gap: 12px;
    padding: 16px;
  }

  .action-left {
    width: 100%;
    justify-content: center;
  }

  .action-right {
    width: 100%;
    justify-content: center;
    flex-wrap: wrap;
  }
}

/* 暗色模式 */
:root[data-theme='dark'] {
  .stepper-section,
  .panel {
    background: rgba(40, 40, 40, 0.85);
    border-color: rgba(255, 255, 255, 0.08);
  }

  .upload-dragger {
    :deep(.ant-upload-drag) {
      background: rgba(255, 125, 0, 0.05);
      border-color: rgba(255, 125, 0, 0.2);

      &:hover {
        background: rgba(255, 125, 0, 0.08);
      }
    }
  }

  .chunk-card {
    background: rgba(255, 125, 0, 0.08);
    border-color: rgba(255, 125, 0, 0.15);
  }

  .chunk-more {
    background: var(--gray-800);
  }

  .action-bar-inner {
    background: rgba(30, 30, 30, 0.92);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
  }
}
</style>
