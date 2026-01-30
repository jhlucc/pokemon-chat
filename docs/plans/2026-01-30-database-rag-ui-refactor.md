# Database & RAG Knowledge Base UI Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the database/knowledge base management UI to align with modern RAG platforms (Dify, RAGFlow), unify styling with the existing design system, integrate all backend APIs, and enhance document parsing capabilities.

**Architecture:** Three-tier refactoring approach:
1. UI/UX Layer - Unified glassmorphism design, improved layout with sidebar navigation
2. Feature Layer - Complete API integration, search testing, document management
3. Parser Layer - Extended format support via MarkItDown, better parsing feedback

**Tech Stack:** Vue 3, Ant Design Vue, TypeScript, FastAPI, MarkItDown, LangChain

---

## Current State Analysis

### Existing Backend APIs (data_router.py)
| Endpoint | Method | Status | Frontend Integration |
|----------|--------|--------|---------------------|
| `/data/` | GET | ✅ | ✅ List databases |
| `/data/` | POST | ✅ | ✅ Create database |
| `/data/` | DELETE | ✅ | ✅ Delete database |
| `/data/info` | GET | ✅ | ✅ Database details |
| `/data/file-to-chunk` | POST | ✅ | ✅ Parse & chunk |
| `/data/add-by-chunks` | POST | ✅ | ✅ Index chunks |
| `/data/upload` | POST | ✅ | ✅ File upload |
| `/data/document` | GET | ✅ | ✅ Get document |
| `/data/document` | DELETE | ✅ | ✅ Delete document |
| `/data/search` | GET | ✅ | ❌ **Not integrated** |
| `/data/query-test` | POST | ✅ | ❌ **Commented out in UI** |
| `/data/ingest/file` | POST | ✅ | ❌ **Not integrated** |
| `/data/ingest/dir` | POST | ✅ | ❌ **Not integrated** |
| `/data/graph/*` | GET | ✅ | ❌ **Separate page** |

### Current Pages
- `/database` - DatabaseHubView.vue (list + workbench tabs)
- `/database/workbench` - DatabaseRagWorkbench.vue (embedded)
- `/database/:id` - DataBaseInfoView.vue (file management)

### Design System Variables (from existing components)
```css
--primary-color: #ff7d00 (orange)
--primary-light-color: lighter orange
--gray-100 to --gray-900
--border-color
--surface-color
--radius-sm/md/lg
backdrop-filter: blur(16px)
```

---

## Phase 1: UI/UX Unification & Layout Refactor

### Task 1: Create Unified Database Layout Component

**Files:**
- Create: `web/src/components/database/DatabaseLayout.vue`
- Modify: `web/src/views/DatabaseHubView.vue`

**Step 1: Create DatabaseLayout component skeleton**

```vue
<!-- web/src/components/database/DatabaseLayout.vue -->
<template>
  <div class="db-layout">
    <!-- Background effects -->
    <div class="ambient-glow glow--orange"></div>
    <div class="ambient-glow glow--purple"></div>
    <div class="dot-grid"></div>

    <!-- Main content -->
    <div class="db-layout__content">
      <slot />
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped lang="less">
.db-layout {
  position: relative;
  min-height: 100vh;
  background: transparent;
  overflow-x: hidden;
}

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
  width: 700px;
  height: 700px;
  top: -25%;
  left: -20%;
  background: radial-gradient(circle, rgba(255, 125, 0, 0.35) 0%, rgba(255, 180, 100, 0.15) 40%, transparent 70%);
}

.glow--purple {
  width: 550px;
  height: 550px;
  bottom: -15%;
  right: -15%;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, rgba(180, 150, 255, 0.1) 40%, transparent 70%);
  animation-delay: -7s;
}

@keyframes glow-drift {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(20px, -15px) scale(1.03); }
  66% { transform: translate(-15px, 15px) scale(0.97); }
}

.dot-grid {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.05) 1px, transparent 1px),
    radial-gradient(circle, rgba(255, 125, 0, 0.025) 1px, transparent 1px);
  background-size: 24px 24px, 96px 96px;
  pointer-events: none;
  z-index: 0;
}

.db-layout__content {
  position: relative;
  z-index: 1;
}

:root[data-theme='dark'] {
  .ambient-glow { mix-blend-mode: screen; opacity: 0.7; }
  .glow--orange { background: radial-gradient(circle, rgba(255, 125, 0, 0.25) 0%, rgba(255, 180, 100, 0.1) 40%, transparent 70%); }
  .glow--purple { background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(180, 150, 255, 0.08) 40%, transparent 70%); }
  .dot-grid { background-image: radial-gradient(circle, rgba(255, 125, 0, 0.06) 1px, transparent 1px), radial-gradient(circle, rgba(255, 125, 0, 0.03) 1px, transparent 1px); }
}
</style>
```

**Step 2: Verify component renders**

Run: `cd /data/temp34/pokemon-chat/web && npm run dev`
Expected: Dev server starts without errors

**Step 3: Commit**

```bash
git add web/src/components/database/DatabaseLayout.vue
git commit -m "feat(database): add unified DatabaseLayout component with glassmorphism background"
```

---

### Task 2: Create GlassPanel Reusable Component

**Files:**
- Create: `web/src/components/common/GlassPanel.vue`

**Step 1: Create GlassPanel component**

```vue
<!-- web/src/components/common/GlassPanel.vue -->
<template>
  <div class="glass-panel" :class="{ 'glass-panel--compact': compact }">
    <div v-if="title || $slots.header" class="glass-panel__header">
      <span v-if="title" class="glass-panel__title">{{ title }}</span>
      <slot name="header" />
      <span v-if="badge" class="glass-panel__badge">{{ badge }}</span>
      <slot name="header-actions" />
    </div>
    <div class="glass-panel__body">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, default: '' },
  badge: { type: [String, Number], default: '' },
  compact: { type: Boolean, default: false }
})
</script>

<style scoped lang="less">
.glass-panel {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.5);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03), 0 4px 12px color-mix(in srgb, var(--primary-color) 3%, transparent);

  &--compact {
    padding: 16px;
    border-radius: 16px;
  }
}

.glass-panel__header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color);
}

.glass-panel__title {
  font-size: 16px;
  font-weight: 650;
  color: var(--text-color);
}

.glass-panel__badge {
  font-size: 12px;
  padding: 2px 10px;
  background: color-mix(in srgb, var(--primary-color) 10%, transparent);
  color: var(--primary-color);
  border-radius: 100px;
  font-weight: 500;
  margin-left: auto;
}

.glass-panel__body {
  /* Content container */
}

:root[data-theme='dark'] {
  .glass-panel {
    background: rgba(40, 40, 40, 0.85);
    border-color: rgba(255, 255, 255, 0.08);
  }
}
</style>
```

**Step 2: Commit**

```bash
git add web/src/components/common/GlassPanel.vue
git commit -m "feat(ui): add reusable GlassPanel component"
```

---

### Task 3: Refactor DatabaseHubView with New Layout

**Files:**
- Modify: `web/src/views/DatabaseHubView.vue:1-130`

**Step 1: Update template to use DatabaseLayout**

Replace the existing background elements with the new layout component:

```vue
<template>
  <DatabaseLayout>
    <HeaderComponent
      :title="headerTitle"
      :description="headerDescription"
      :breadcrumbs="breadcrumbs"
    >
      <!-- Keep existing header actions -->
    </HeaderComponent>

    <div class="ui-page">
      <div class="ui-container">
        <!-- Keep existing tab content -->
      </div>
    </div>
  </DatabaseLayout>
</template>
```

**Step 2: Import DatabaseLayout**

```javascript
import DatabaseLayout from '@/components/database/DatabaseLayout.vue'
```

**Step 3: Remove duplicate background styles**

Remove `.ambient-glow`, `.dot-grid` styles from DatabaseHubView since they're now in DatabaseLayout.

**Step 4: Verify the page renders correctly**

Run: `npm run dev` and navigate to `/database`
Expected: Same visual appearance, background from layout component

**Step 5: Commit**

```bash
git add web/src/views/DatabaseHubView.vue
git commit -m "refactor(database): use DatabaseLayout component for background"
```

---

## Phase 2: Search & Query Testing Feature (Re-enable)

### Task 4: Create SearchTestPanel Component

**Files:**
- Create: `web/src/components/database/SearchTestPanel.vue`

**Step 1: Create the search testing component**

```vue
<!-- web/src/components/database/SearchTestPanel.vue -->
<template>
  <GlassPanel title="检索测试" class="search-test-panel">
    <template #header-actions>
      <a-tooltip title="测试知识库的检索效果">
        <QuestionCircleOutlined />
      </a-tooltip>
    </template>

    <!-- Query Input -->
    <div class="search-input-area">
      <a-textarea
        v-model:value="query"
        placeholder="输入查询语句测试检索效果..."
        :auto-size="{ minRows: 2, maxRows: 4 }"
        @keydown.ctrl.enter="runSearch"
      />
      <a-button
        type="primary"
        :loading="loading"
        :disabled="!query.trim() || !dbId"
        @click="runSearch"
      >
        <SearchOutlined /> 检索
      </a-button>
    </div>

    <!-- Search Parameters -->
    <a-collapse v-model:activeKey="paramsOpen" class="params-collapse">
      <a-collapse-panel key="params" header="检索参数">
        <div class="params-grid">
          <div class="param-item">
            <label>Top K</label>
            <a-input-number v-model:value="params.topK" :min="1" :max="20" />
          </div>
          <div class="param-item">
            <label>距离阈值</label>
            <a-slider v-model:value="params.distanceThreshold" :min="0" :max="1" :step="0.01" />
          </div>
          <div class="param-item">
            <label>启用重排序</label>
            <a-switch v-model:checked="params.rerank" />
          </div>
        </div>
      </a-collapse-panel>
    </a-collapse>

    <!-- Results -->
    <div v-if="results" class="results-section">
      <div class="results-header">
        <span>找到 {{ results.results?.length || 0 }} 条结果</span>
        <span class="results-time" v-if="searchTime">耗时 {{ searchTime }}ms</span>
      </div>

      <div class="results-list">
        <div v-for="(item, idx) in results.results" :key="idx" class="result-card">
          <div class="result-meta">
            <span class="result-rank">#{{ idx + 1 }}</span>
            <span class="result-file">{{ item.file?.filename || 'Unknown' }}</span>
            <span class="result-score">
              相似度: {{ (1 - item.distance).toFixed(3) }}
              <template v-if="item.rerank_score"> · 重排序: {{ item.rerank_score.toFixed(3) }}</template>
            </span>
          </div>
          <p class="result-text">{{ item.entity?.text }}</p>
        </div>
      </div>

      <a-empty v-if="results.results?.length === 0" description="未找到相关结果" />
    </div>
  </GlassPanel>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { message } from 'ant-design-vue'
import { SearchOutlined, QuestionCircleOutlined } from '@ant-design/icons-vue'
import GlassPanel from '@/components/common/GlassPanel.vue'
import { apiFetch } from '@/api/http'

const props = defineProps({
  dbId: { type: String, required: true }
})

const query = ref('')
const loading = ref(false)
const results = ref(null)
const searchTime = ref(null)
const paramsOpen = ref([])

const params = reactive({
  topK: 5,
  distanceThreshold: 0.5,
  rerank: true
})

const runSearch = async () => {
  if (!query.value.trim() || !props.dbId) return

  loading.value = true
  const startTime = Date.now()

  try {
    const data = await apiFetch('/data/query-test', {
      method: 'POST',
      body: {
        query: query.value,
        meta: {
          db_id: props.dbId,
          topK: params.topK,
          distanceThreshold: params.distanceThreshold,
          rerank: params.rerank
        }
      },
      timeoutMs: 30000
    })
    results.value = data
    searchTime.value = Date.now() - startTime
  } catch (e) {
    message.error(e?.message || '检索失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped lang="less">
.search-test-panel {
  margin-top: 16px;
}

.search-input-area {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;

  :deep(.ant-input) {
    flex: 1;
  }
}

.params-collapse {
  margin-bottom: 16px;
  background: transparent;

  :deep(.ant-collapse-item) {
    border: 1px solid var(--border-color);
    border-radius: 12px;
  }
}

.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.param-item {
  display: flex;
  flex-direction: column;
  gap: 8px;

  label {
    font-size: 13px;
    color: var(--gray-600);
    font-weight: 500;
  }
}

.results-section {
  margin-top: 16px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  font-size: 14px;
  color: var(--gray-600);
}

.results-time {
  color: var(--gray-400);
  font-size: 12px;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-card {
  padding: 16px;
  background: color-mix(in srgb, var(--primary-color) 4%, transparent);
  border: 1px solid color-mix(in srgb, var(--primary-color) 10%, transparent);
  border-radius: 12px;
  transition: all 0.2s ease;

  &:hover {
    background: color-mix(in srgb, var(--primary-color) 8%, transparent);
    transform: translateY(-1px);
  }
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
}

.result-rank {
  font-weight: 600;
  color: var(--primary-color);
}

.result-file {
  color: var(--gray-600);
}

.result-score {
  margin-left: auto;
  color: var(--gray-500);
}

.result-text {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-color);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

:root[data-theme='dark'] {
  .result-card {
    background: color-mix(in srgb, var(--primary-color) 8%, transparent);
    border-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
  }
}
</style>
```

**Step 2: Commit**

```bash
git add web/src/components/database/SearchTestPanel.vue
git commit -m "feat(database): add SearchTestPanel component for knowledge base testing"
```

---

### Task 5: Integrate SearchTestPanel into DataBaseInfoView

**Files:**
- Modify: `web/src/views/DataBaseInfoView.vue`

**Step 1: Import SearchTestPanel**

Add to imports:
```javascript
import SearchTestPanel from '@/components/database/SearchTestPanel.vue'
```

**Step 2: Add new tab for search testing**

Find the `<a-tabs>` section and add a new tab pane:

```vue
<a-tab-pane key="search">
  <template #tab><span><SearchOutlined />检索测试</span></template>
  <div class="db-tab-container">
    <SearchTestPanel :db-id="databaseId" />
  </div>
</a-tab-pane>
```

**Step 3: Import SearchOutlined icon**

Add to icon imports:
```javascript
import { SearchOutlined } from '@ant-design/icons-vue'
```

**Step 4: Verify search tab works**

Run: `npm run dev`, navigate to `/database/{id}`, click "检索测试" tab
Expected: Search panel renders, can input query and get results

**Step 5: Commit**

```bash
git add web/src/views/DataBaseInfoView.vue
git commit -m "feat(database): integrate SearchTestPanel into database detail view"
```

---

## Phase 3: Enhanced Document Parsing

### Task 6: Add Parser Status API Endpoint

**Files:**
- Modify: `server/routers/data_router.py`

**Step 1: Add parser info endpoint**

Add after line 359:

```python
@data.get("/parsers")
async def get_parser_info():
    """返回支持的文件格式和解析器信息"""
    parsers = {
        "supported_formats": [
            {"ext": ".pdf", "parser": "PyPDFLoader + OCR fallback", "ocr_support": True},
            {"ext": ".docx", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".doc", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".pptx", "parser": "DeepDoc", "ocr_support": False},
            {"ext": ".ppt", "parser": "DeepDoc", "ocr_support": False},
            {"ext": ".xlsx", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".xls", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".csv", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".txt", "parser": "Direct read", "ocr_support": False},
            {"ext": ".md", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".html", "parser": "MarkItDown", "ocr_support": False},
            {"ext": ".json", "parser": "MarkItDown", "ocr_support": False},
        ],
        "ocr_available": False,
        "markitdown_version": None
    }

    # Check OCR availability
    try:
        from src.plugins.vision._ocr import OCRHandler2
        parsers["ocr_available"] = True
    except ImportError:
        pass

    # Check MarkItDown version
    try:
        import markitdown
        parsers["markitdown_version"] = getattr(markitdown, "__version__", "unknown")
    except ImportError:
        pass

    return parsers
```

**Step 2: Verify endpoint works**

Run: `curl http://localhost:5050/data/parsers`
Expected: JSON with supported formats and parser info

**Step 3: Commit**

```bash
git add server/routers/data_router.py
git commit -m "feat(api): add /data/parsers endpoint for parser capabilities"
```

---

### Task 7: Create SupportedFormats Component

**Files:**
- Create: `web/src/components/database/SupportedFormats.vue`

**Step 1: Create the component**

```vue
<!-- web/src/components/database/SupportedFormats.vue -->
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
```

**Step 2: Commit**

```bash
git add web/src/components/database/SupportedFormats.vue
git commit -m "feat(database): add SupportedFormats component showing parser capabilities"
```

---

### Task 8: Integrate SupportedFormats into Workbench

**Files:**
- Modify: `web/src/components/database/DatabaseRagWorkbench.vue`

**Step 1: Import SupportedFormats**

Add to imports:
```javascript
import SupportedFormats from '@/components/database/SupportedFormats.vue'
```

**Step 2: Add to upload panel**

Find the upload section and add below the upload hint:

```vue
<SupportedFormats class="formats-info" />
```

**Step 3: Add spacing style**

```less
.formats-info {
  margin-top: 16px;
}
```

**Step 4: Verify it renders**

Run: `npm run dev`, navigate to `/database/workbench`
Expected: Supported formats panel visible below upload area

**Step 5: Commit**

```bash
git add web/src/components/database/DatabaseRagWorkbench.vue
git commit -m "feat(database): show supported file formats in workbench"
```

---

## Phase 4: Batch Import Feature

### Task 9: Create BatchImportPanel Component

**Files:**
- Create: `web/src/components/database/BatchImportPanel.vue`

**Step 1: Create batch import component**

```vue
<!-- web/src/components/database/BatchImportPanel.vue -->
<template>
  <GlassPanel title="批量导入" class="batch-import">
    <template #header-actions>
      <a-tooltip title="从服务器目录批量导入文件">
        <QuestionCircleOutlined />
      </a-tooltip>
    </template>

    <a-alert
      type="info"
      show-icon
      message="从服务器本地目录导入文件，适用于大批量文档处理"
      style="margin-bottom: 16px"
    />

    <a-form layout="vertical">
      <a-form-item label="服务器目录路径">
        <a-input
          v-model:value="folderPath"
          placeholder="/path/to/documents"
        />
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
      timeoutMs: 600000 // 10 minutes for large imports
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
</style>
```

**Step 2: Commit**

```bash
git add web/src/components/database/BatchImportPanel.vue
git commit -m "feat(database): add BatchImportPanel for server-side directory import"
```

---

### Task 10: Add Batch Import Tab to DataBaseInfoView

**Files:**
- Modify: `web/src/views/DataBaseInfoView.vue`

**Step 1: Import BatchImportPanel**

```javascript
import BatchImportPanel from '@/components/database/BatchImportPanel.vue'
import { FolderOpenOutlined } from '@ant-design/icons-vue'
```

**Step 2: Add tab pane**

After the "add" tab pane:

```vue
<a-tab-pane key="batch">
  <template #tab><span><FolderOpenOutlined />批量导入</span></template>
  <div class="db-tab-container">
    <BatchImportPanel :db-id="databaseId" />
  </div>
</a-tab-pane>
```

**Step 3: Verify**

Run: `npm run dev`, navigate to `/database/{id}`, click "批量导入" tab
Expected: Batch import panel renders with folder input

**Step 4: Commit**

```bash
git add web/src/views/DataBaseInfoView.vue
git commit -m "feat(database): add batch import tab for server directory ingestion"
```

---

## Phase 5: Styling Consistency & Polish

### Task 11: Unify Table Styling

**Files:**
- Modify: `web/src/views/DataBaseInfoView.vue:708-1020` (styles section)

**Step 1: Update table styles for consistency**

Find `.my-table` and update:

```less
.my-table {
  :deep(.ant-table) {
    background: transparent;
  }

  :deep(.ant-table-thead > tr > th) {
    background: color-mix(in srgb, var(--primary-color) 4%, transparent);
    font-weight: 600;
    color: var(--gray-700);
    border-bottom: 1px solid var(--border-color);
  }

  :deep(.ant-table-tbody > tr > td) {
    border-bottom: 1px solid var(--border-color);
  }

  :deep(.ant-table-tbody > tr:hover > td) {
    background: color-mix(in srgb, var(--primary-color) 6%, transparent);
  }

  button.ant-btn-link {
    padding: 0;
    font-weight: 500;
  }

  .span-type {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;

    &.pdf { background: #fee2e2; color: #dc2626; }
    &.txt { background: #e0f2fe; color: #0284c7; }
    &.docx, &.doc { background: #dbeafe; color: #2563eb; }
    &.md { background: #f3e8ff; color: #9333ea; }
    &.xlsx, &.xls { background: #dcfce7; color: #16a34a; }
  }

  button.main-btn {
    font-weight: 600;
    color: var(--text-color);

    &:hover {
      color: var(--primary-color);
    }
  }

  button.del-btn {
    color: var(--gray-400);

    &:hover:not(:disabled) {
      color: var(--error-color);
    }
  }
}
```

**Step 2: Commit**

```bash
git add web/src/views/DataBaseInfoView.vue
git commit -m "style(database): unify table styling with glassmorphism theme"
```

---

### Task 12: Add Loading States and Empty States

**Files:**
- Modify: `web/src/components/database/DatabaseRagWorkbench.vue`

**Step 1: Add skeleton loading for database list**

When `state.loadingDatabases` is true, show skeleton:

Find the database select and wrap with conditional:

```vue
<a-skeleton v-if="state.loadingDatabases" active :paragraph="{ rows: 1 }" />
<div v-else class="select-with-action">
  <!-- existing select -->
</div>
```

**Step 2: Improve empty state for no databases**

Add after the select:

```vue
<a-empty
  v-if="!state.loadingDatabases && databases.length === 0"
  description="暂无知识库"
  :image="Empty.PRESENTED_IMAGE_SIMPLE"
>
  <a-button type="primary" @click="$router.push('/database')">
    创建知识库
  </a-button>
</a-empty>
```

Import:
```javascript
import { Empty } from 'ant-design-vue'
```

**Step 3: Commit**

```bash
git add web/src/components/database/DatabaseRagWorkbench.vue
git commit -m "feat(database): add loading skeletons and empty states to workbench"
```

---

## Phase 6: Final Integration Testing

### Task 13: End-to-End Testing

**Files:**
- N/A (manual testing)

**Step 1: Start development server**

```bash
cd /data/temp34/pokemon-chat
# Start backend
python -m uvicorn server.main:app --reload --port 5050 &
# Start frontend
cd web && npm run dev
```

**Step 2: Test database list page**

Navigate to `http://localhost:3100/database`
- [ ] Background animations render
- [ ] Database cards load
- [ ] Create new database works
- [ ] Navigate to database detail works

**Step 3: Test workbench**

Navigate to `http://localhost:3100/database/workbench`
- [ ] Database selection works
- [ ] File upload works
- [ ] Chunk preview generates
- [ ] Index writing completes
- [ ] Supported formats panel shows

**Step 4: Test database detail**

Navigate to `http://localhost:3100/database/{id}`
- [ ] File list loads
- [ ] File detail drawer opens
- [ ] Search test tab works
- [ ] Batch import tab works
- [ ] Delete file works

**Step 5: Document any issues**

Create issues for any bugs found during testing.

**Step 6: Final commit**

```bash
git add -A
git commit -m "test: verify database UI refactor end-to-end functionality"
```

---

## Summary

### Files Created
- `web/src/components/database/DatabaseLayout.vue`
- `web/src/components/common/GlassPanel.vue`
- `web/src/components/database/SearchTestPanel.vue`
- `web/src/components/database/SupportedFormats.vue`
- `web/src/components/database/BatchImportPanel.vue`

### Files Modified
- `web/src/views/DatabaseHubView.vue`
- `web/src/views/DataBaseInfoView.vue`
- `web/src/components/database/DatabaseRagWorkbench.vue`
- `server/routers/data_router.py`

### Features Added
1. ✅ Unified glassmorphism layout component
2. ✅ Reusable GlassPanel component
3. ✅ Search/Query testing panel (re-enabled)
4. ✅ Parser capabilities API endpoint
5. ✅ Supported formats display component
6. ✅ Batch import from server directories
7. ✅ Unified table styling
8. ✅ Loading states and empty states

### API Integration Status
| Endpoint | Before | After |
|----------|--------|-------|
| `/data/search` | ❌ | ✅ Via query-test |
| `/data/query-test` | ❌ Commented | ✅ Integrated |
| `/data/ingest/dir` | ❌ | ✅ Batch import |
| `/data/parsers` | N/A | ✅ New endpoint |

### References
- [Dify Knowledge Base UI](https://docs.dify.ai/guides/knowledge-base) - Visual workflow editor inspiration
- [RAGFlow Features](https://ragflow.io/docs/dev/release_notes) - Cross-language search, enhanced image display
- [RAG Best Practices](https://www.kapa.ai/blog/rag-best-practices) - Knowledge base quality guidelines
