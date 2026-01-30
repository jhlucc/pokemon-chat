<template>
  <div>
    <HeaderComponent
      :title="database.name || '数据库信息'"
      :breadcrumbs="[
        { label: '首页', to: '/' },
        { label: '知识库', to: '/database' },
        { label: database.name || '详情' }
      ]"
      compact
    >
      <template #description>
        <div class="database-info">
          <a-tag color="blue" v-if="database.embed_model">{{ database.embed_model }}</a-tag>
          <a-tag color="green" v-if="database.dimension">{{ database.dimension }}</a-tag>
          <span class="row-count"
            >{{ database.files ? Object.keys(database.files).length : 0 }} 文件 ·
            {{ database.db_id }}</span
          >
        </div>
      </template>
      <template #actions>
        <a-button type="primary" @click="backToDatabase"> <LeftOutlined /> 返回 </a-button>
        <a-button type="primary" danger @click="deleteDatabse">
          <DeleteOutlined /> 删除数据库
        </a-button>
      </template>
    </HeaderComponent>

    <div class="ui-page">
      <div class="ui-container">
        <a-alert
          v-if="!canUseKb"
          type="warning"
          show-icon
          :message="
            configStore.config.backend?.online
              ? '后端未启用知识库功能（enable_knowledge_base=false）'
              : '后端未启动/不可用（离线模式）'
          "
          style="margin-bottom: 12px"
        />
        <!--  <a-alert v-if="configStore.config.embed_model &&database.embed_model != configStore.config.embed_model" message="向量模型不匹配，请重新选择" type="warning" style="margin: 10px 20px;" />-->
        <div class="db-main-container">
          <a-tabs v-model:activeKey="state.curPage" class="atab-container" type="card">
        <a-tab-pane key="files">
          <template #tab
            ><span><ReadOutlined />文件列表</span></template
          >
          <div class="db-tab-container">
            <div class="actions">
              <a-button @click="handleRefresh" :loading="state.refrashing">刷新</a-button>
            </div>

            <a-empty
              v-if="Object.values(database.files || {}).length === 0"
              style="padding: 24px 0"
              description="暂无文件"
            >
              <a-button type="primary" @click="state.curPage = 'add'">添加文件</a-button>
            </a-empty>

            <a-table
              v-else
              :columns="columns"
              :data-source="Object.values(database.files || {})"
              row-key="file_id"
              class="my-table"
              size="middle"
              :pagination="{ pageSize: 10, hideOnSinglePage: true }"
            >
              <template #bodyCell="{ column, text, record }">
                <template v-if="column.key === 'filename'">
                  <a-button class="main-btn" type="link" @click="openFileDetail(record)">{{
                    text
                  }}</a-button>
                </template>
                <template v-else-if="column.key === 'type'">
                  <span :class="['span-type', text]">{{ text?.toUpperCase() }}</span>
                </template>
                <template v-else-if="column.key === 'status' && text === 'done'">
                  <CheckCircleFilled style="color: var(--success-color)" />
                </template>
                <template v-else-if="column.key === 'status' && text === 'failed'">
                  <CloseCircleFilled style="color: var(--danger-500)" />
                </template>
                <template v-else-if="column.key === 'status' && text === 'processing'">
                  <HourglassFilled style="color: var(--primary-color)" />
                </template>
                <template v-else-if="column.key === 'status' && text === 'waiting'">
                  <ClockCircleFilled style="color: var(--warning-color)" />
                </template>
                <template v-else-if="column.key === 'action'">
                  <a-button
                    class="del-btn"
                    type="link"
                    @click="deleteFile(text)"
                    :disabled="
                      state.lock || record.status === 'processing' || record.status === 'waiting'
                    "
                    >删除
                  </a-button>
                </template>
                <span v-else-if="column.key === 'created_at'">{{
                  formatRelativeTime(Math.round(text * 1000))
                }}</span>
                <span v-else>{{ text }}</span>
              </template>
            </a-table>
            <a-drawer
              width="50%"
              v-model:open="state.drawer"
              class="custom-class"
              :title="selectedFile?.filename || '文件详情'"
              placement="right"
              @after-open-change="afterOpenChange"
            >
              <h2>共 {{ selectedFile?.lines?.length || 0 }} 个片段</h2>
              <p v-for="line in selectedFile?.lines || []" :key="line.id" class="line-text">
                {{ line.text }}
              </p>
            </a-drawer>
          </div>
        </a-tab-pane>

        <a-tab-pane key="add">
          <template #tab
            ><span><CloudUploadOutlined />添加文件</span></template
          >
          <div class="db-tab-container">
            <div class="upload-section">
              <div class="upload-sidebar">
                <div class="chunking-params">
                  <div class="params-info">
                    <p>调整分块参数可以控制文本的切分方式，影响检索质量和文档加载效率。</p>
                  </div>
                  <a-form :model="chunkParams" name="basic" autocomplete="off" layout="vertical">
                    <a-form-item label="Chunk Size" name="chunk_size">
                      <a-input-number
                        v-model:value="chunkParams.chunk_size"
                        :min="100"
                        :max="10000"
                      />
                      <p class="param-description">每个文本片段的最大字符数</p>
                    </a-form-item>
                    <a-form-item label="Chunk Overlap" name="chunk_overlap">
                      <a-input-number
                        v-model:value="chunkParams.chunk_overlap"
                        :min="0"
                        :max="1000"
                      />
                      <p class="param-description">相邻文本片段间的重叠字符数</p>
                    </a-form-item>
                    <a-form-item label="使用文件节点解析器" name="use_parser">
                      <a-switch v-model:checked="chunkParams.use_parser" />
                      <p class="param-description">启用特定文件格式的智能分析</p>
                    </a-form-item>
                  </a-form>
                </div>
              </div>
              <div class="upload-main">
                <div class="upload">
                  <a-upload-dragger
                    class="upload-dragger"
                    v-model:fileList="fileList"
                    name="file"
                    :multiple="true"
                    :disabled="state.loading"
                    :customRequest="customUpload"
                    @change="handleFileUpload"
                    @drop="handleDrop"
                  >
                    <p class="ant-upload-text">点击或者把文件拖拽到这里上传</p>
                    <p class="ant-upload-hint">
                      目前仅支持上传文本文件，如 .pdf, .txt, .md。且同名文件无法重复添加
                    </p>
                  </a-upload-dragger>
                </div>
                <div class="actions">
                  <a-button
                    type="primary"
                    @click="chunkFiles"
                    :loading="state.loading"
                    :disabled="fileList.length === 0"
                    style="margin: 0px 20px 20px 0"
                  >
                    生成分块
                  </a-button>
                </div>
              </div>
            </div>

            <!-- 分块结果预览区域 -->
            <div class="chunk-preview" v-if="chunkResults.length > 0">
              <div class="preview-header">
                <h3>
                  分块预览 (共 {{ chunkResults.length }} 个文件，{{ getTotalChunks() }} 个分块)
                </h3>
                <a-button type="primary" @click="addToDatabase" :loading="state.adding">
                  添加到数据库
                </a-button>
              </div>

              <a-collapse v-model:activeKey="activeFileKeys">
                <a-collapse-panel
                  v-for="(file, fileIdx) in chunkResults"
                  :key="fileIdx"
                  :header="file.filename + ' (' + file.nodes.length + ' 个分块)'"
                >
                  <div id="result-cards" class="result-cards">
                    <div v-for="(chunk, index) in file.nodes" :key="index" class="chunk">
                      <p>
                        <strong>#{{ index + 1 }}</strong> {{ chunk.text }}
                      </p>
                    </div>
                  </div>
                </a-collapse-panel>
              </a-collapse>
            </div>
          </div>
        </a-tab-pane>

        <a-tab-pane key="batch">
          <template #tab><span><FolderOpenOutlined />批量导入</span></template>
          <div class="db-tab-container">
            <BatchImportPanel :db-id="databaseId" />
          </div>
        </a-tab-pane>

        <a-tab-pane key="search">
          <template #tab><span><SearchOutlined />检索测试</span></template>
          <div class="db-tab-container">
            <SearchTestPanel :db-id="databaseId" />
          </div>
        </a-tab-pane>

        <!--      <a-tab-pane key="query-test" force-render>-->
        <!--        <template #tab><span><SearchOutlined />检索测试</span></template>-->
        <!--        <div class="query-test-container db-tab-container">-->
        <!--          <div class="sider">-->
        <!--            <div class="sider-top">-->
        <!--              <div class="query-params" v-if="state.curPage == 'query-test'">-->
        <!--                &lt;!&ndash; <h3 class="params-title">查询参数</h3> &ndash;&gt;-->
        <!--                <div class="params-group">-->
        <!--                  <div class="params-item">-->
        <!--                    <p>检索数量：</p>-->
        <!--                    <a-input-number size="small" v-model:value="meta.maxQueryCount" :min="1" :max="20" />-->
        <!--                  </div>-->
        <!--                  <div class="params-item">-->
        <!--                    <p>过滤低质量：</p>-->
        <!--                    <a-switch v-model:checked="meta.filter" />-->
        <!--                  </div>-->
        <!--                  <div class="params-item">-->
        <!--                    <p>筛选 TopK：</p>-->
        <!--                    <a-input-number size="small" v-model:value="meta.topK" :min="1" :max="meta.maxQueryCount" />-->
        <!--                  </div>-->
        <!--                  <div class="params-item" v-if="configStore.config.enable_reranker">-->
        <!--                    <p>排序方式：</p>-->
        <!--                    <a-radio-group v-model:value="meta.sortBy" button-style="solid" size="small">-->
        <!--                      <a-radio-button value="rerank_score">重排序分</a-radio-button>-->
        <!--                      <a-radio-button value="distance">相似度</a-radio-button>-->
        <!--                    </a-radio-group>-->
        <!--                  </div>-->
        <!--                </div>-->
        <!--                <div class="params-group">-->
        <!--                  <div class="params-item w100" v-if="configStore.config.enable_reranker">-->
        <!--                    <p>重排序阈值：</p>-->
        <!--                    <a-slider v-model:value="meta.rerankThreshold" :min="0" :max="1" :step="0.01" />-->
        <!--                  </div>-->
        <!--                  <div class="params-item w100">-->
        <!--                    <p>距离阈值：</p>-->
        <!--                    <a-slider v-model:value="meta.distanceThreshold" :min="0" :max="1" :step="0.01" />-->
        <!--                  </div>-->
        <!--                </div>-->
        <!--                <div class="params-group">-->
        <!--                  <div class="params-item col">-->
        <!--                    <p>重写查询<small>（修改后需重新检索）</small>：</p>-->
        <!--                    <a-segmented v-model:value="meta.use_rewrite_query" :options="use_rewrite_queryOptions">-->
        <!--                      <template #label="{ payload }">-->
        <!--                        <div>-->
        <!--                          <p style="margin: 4px 0">{{ payload.subTitle }}</p>-->
        <!--                        </div>-->
        <!--                      </template>-->
        <!--                    </a-segmented>-->
        <!--                  </div>-->
        <!--                </div>-->
        <!--              </div>-->
        <!--            </div>-->
        <!--            <div class="sider-bottom">-->
        <!--            </div>-->
        <!--          </div>-->
        <!--          <div class="query-result-container">-->
        <!--            <div class="query-action">-->
        <!--              <a-textarea-->
        <!--                v-model:value="queryText"-->
        <!--                placeholder="填写需要查询的句子"-->
        <!--                :auto-size="{ minRows: 2, maxRows: 10 }"-->
        <!--              />-->
        <!--              <a-button class="btn-query" @click="onQuery" :disabled="queryText.length == 0">-->
        <!--                <span v-if="!state.searchLoading"><SearchOutlined /> 检索</span>-->
        <!--                <span v-else><LoadingOutlined /></span>-->
        <!--              </a-button>-->
        <!--            </div>-->

        <!--            &lt;!&ndash; 新增示例按钮 &ndash;&gt;-->
        <!--            &lt;!&ndash; <div class="query-examples-container">-->
        <!--              <div class="examples-title">示例查询：</div>-->
        <!--              <div class="query-examples">-->
        <!--                <a-button v-for="example in queryExamples" :key="example" @click="useQueryExample(example)">-->
        <!--                  {{ example }}-->
        <!--                </a-button>-->
        <!--              </div>-->
        <!--            </div> &ndash;&gt;-->
        <!--            <div class="query-test" v-if="queryResult">-->
        <!--              <div class="results-overview">-->
        <!--                <div class="results-stats">-->
        <!--                  <span class="stat-item">-->
        <!--                    <strong>总数:</strong> {{ queryResult.all_results.length }}-->
        <!--                  </span>-->
        <!--                  <span class="stat-item">-->
        <!--                    <strong>过滤后:</strong> {{ filteredResults.length }}-->
        <!--                  </span>-->
        <!--                  <span class="stat-item">-->
        <!--                    <strong>TopK:</strong> {{ meta.topK }}-->
        <!--                  </span>-->
        <!--                  <span class="stat-item">-->
        <!--                    <strong>排序:</strong> {{ meta.sortBy === 'rerank_score' ? '重排序分' : '相似度' }}-->
        <!--                  </span>-->
        <!--                </div>-->
        <!--                <div class="rewritten-query" v-if="queryResult.rw_query">-->
        <!--                  <strong>重写后查询:</strong>-->
        <!--                  <span class="query-text">{{ queryResult.rw_query }}</span>-->
        <!--                </div>-->
        <!--              </div>-->
        <!--              <div class="query-result-card" v-for="(result, idx) in (filteredResults)" :key="idx">-->
        <!--                <p>-->
        <!--                  <strong>#{{ idx + 1 }}&nbsp;&nbsp;&nbsp;</strong>-->
        <!--                  <span>{{ result.file.filename }}&nbsp;&nbsp;&nbsp;</span>-->
        <!--                  <span><strong>相似度</strong>：{{ result.distance.toFixed(4) }}&nbsp;&nbsp;&nbsp;</span>-->
        <!--                  <span v-if="result.rerank_score"><strong>重排序分</strong>：{{ result.rerank_score.toFixed(4) }}</span>-->
        <!--                </p>-->
        <!--                <p class="query-text">{{ result.entity.text }}</p>-->
        <!--              </div>-->
        <!--            </div>-->
        <!--          </div>-->
        <!--        </div>-->
        <!--      </a-tab-pane>-->
        <!-- <a-tab-pane key="3" tab="Tab 3">Content of Tab Pane 3</a-tab-pane> -->
          </a-tabs>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref, watch, onUnmounted, computed } from 'vue'
import { message, Modal } from 'ant-design-vue'
import { useRoute, useRouter } from 'vue-router'
import { useConfigStore } from '@/stores/config'
import { apiFetch } from '@/api/http'
import HeaderComponent from '@/components/HeaderComponent.vue'
import SearchTestPanel from '@/components/database/SearchTestPanel.vue'
import {
  ReadOutlined,
  LeftOutlined,
  CheckCircleFilled,
  HourglassFilled,
  CloseCircleFilled,
  ClockCircleFilled,
  DeleteOutlined,
  CloudUploadOutlined,
  SearchOutlined,
  FolderOpenOutlined
} from '@ant-design/icons-vue'
import BatchImportPanel from '@/components/database/BatchImportPanel.vue'

const route = useRoute()
const router = useRouter()
const databaseId = ref(route.params.database_id)
const database = ref({})

const fileList = ref([])
const selectedFile = ref(null)

const configStore = useConfigStore()
const canUseKb = computed(
  () =>
    Boolean(configStore.config.backend?.online) && Boolean(configStore.config.enable_knowledge_base)
)

const state = reactive({
  loading: false,
  adding: false,
  refrashing: false,
  searchLoading: false,
  lock: false,
  drawer: false,
  refreshInterval: null,
  curPage: 'files'
})

const handleFileUpload = (event) => {
  void event
}

const handleDrop = (event) => {
  void event
}

const customUpload = async ({ file, onSuccess, onError }) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    // Backend expects db_id as form field.
    if (databaseId.value) formData.append('db_id', String(databaseId.value))

    const data = await apiFetch('/data/upload', {
      method: 'POST',
      body: formData,
      timeoutMs: 60000
    })
    onSuccess?.(data, file)
  } catch (e) {
    onError?.(e)
  }
}

const afterOpenChange = (visible) => {
  if (!visible) {
    selectedFile.value = null
  }
}

const backToDatabase = () => {
  router.push('/database')
}

const handleRefresh = () => {
  state.refrashing = true
  const p = getDatabaseInfo()
  if (p && typeof p.finally === 'function') {
    p.finally(() => {
      state.refrashing = false
    })
  } else {
    state.refrashing = false
  }
}

const deleteDatabse = () => {
  Modal.confirm({
    title: '删除数据库',
    content: '确定要删除该数据库吗？',
    okText: '确认',
    cancelText: '取消',
    onOk: () => {
      state.lock = true
      apiFetch('/data/', {
        method: 'DELETE',
        query: { db_id: databaseId.value },
        body: { db_id: databaseId.value },
        timeoutMs: 20000
      })
        .then((data) => {
          message.success(data?.message || '已删除')
          router.push('/database')
        })
        .catch((error) => {
          console.error(error)
          message.error(error?.message || '删除失败')
        })
        .finally(() => {
          state.lock = false
        })
    },
    onCancel: () => {}
  })
}

const openFileDetail = (record) => {
  state.lock = true
  apiFetch(`/data/document`, {
    method: 'GET',
    query: { db_id: databaseId.value, file_id: record.file_id },
    timeoutMs: 20000
  })
    .then((data) => {
      if (data?.status === 'failed') {
        message.error(data?.message || '获取失败')
        return
      }
      state.lock = false
      selectedFile.value = {
        ...record,
        lines: data.lines || []
      }
      state.drawer = true
    })
    .catch((error) => {
      console.error(error)
      message.error(error?.message || '获取失败')
    })
}

const formatRelativeTime = (timestamp) => {
  const now = Date.now()
  const secondsPast = (now - timestamp) / 1000
  if (secondsPast < 60) {
    return Math.round(secondsPast) + ' 秒前'
  } else if (secondsPast < 3600) {
    return Math.round(secondsPast / 60) + ' 分钟前'
  } else if (secondsPast < 86400) {
    return Math.round(secondsPast / 3600) + ' 小时前'
  } else {
    const date = new Date(timestamp)
    const year = date.getFullYear()
    const month = date.getMonth() + 1
    const day = date.getDate()
    return `${year} 年 ${month} 月 ${day} 日`
  }
}

const getDatabaseInfo = () => {
  const db_id = databaseId.value
  if (!db_id) {
    return
  }
  state.lock = true
  return new Promise((resolve, reject) => {
    apiFetch(`/data/info`, { method: 'GET', query: { db_id } })
      .then((data) => {
        database.value = data
        resolve(data)
      })
      .catch((error) => {
        console.error(error)
        message.error(error.message)
        reject(error)
      })
      .finally(() => {
        state.lock = false
      })
  })
}

const deleteFile = (fileId) => {
  //删除提示
  Modal.confirm({
    title: '删除文件',
    content: '确定要删除该文件吗？',
    okText: '确认',
    cancelText: '取消',
    onOk: () => {
      state.lock = true
      apiFetch('/data/document', {
        method: 'DELETE',
        body: {
          db_id: databaseId.value,
          file_id: fileId
        },
        timeoutMs: 20000
      })
        .then((data) => {
          message.success(data?.message || '已删除')
          getDatabaseInfo()
        })
        .catch((error) => {
          console.error(error)
          message.error(error?.message || '删除失败')
        })
        .finally(() => {
          state.lock = false
        })
    },
    onCancel: () => {}
  })
}

const chunkParams = ref({
  chunk_size: 1000,
  chunk_overlap: 200,
  use_parser: false
})

const chunkResults = ref([])
const activeFileKeys = ref([])

// 获取所有分块的总数
const getTotalChunks = () => {
  return chunkResults.value.reduce((total, file) => total + file.nodes.length, 0)
}

// 分块预览
const chunkFiles = () => {
  const files = fileList.value
    .filter((file) => file.status === 'done')
    .map((file) => ({
      name: file.name,
      file_path: file?.response?.file_path || file.name,
      file_content: file?.response?.file_content || ''
    }))

  if (files.length === 0) {
    message.error('请先上传文件')
    return
  }

  state.loading = true
  chunkResults.value = []

  Promise.all(
    files.map(async (file) => {
      let nodes = []
      const data = await apiFetch('/data/file-to-chunk', {
        method: 'POST',
        body: {
          file: file.file_path,
          chunk_size: chunkParams.value.chunk_size,
          chunk_overlap: chunkParams.value.chunk_overlap
        },
        timeoutMs: 60000
      })
      nodes = data?.chunks || []

      return {
        filename: file.name,
        file_id: file.name, // demo: use filename as id
        nodes
      }
    })
  )
    .then((results) => {
      chunkResults.value = results
      activeFileKeys.value = results.length > 0 ? [0] : []
    })
    .catch((error) => {
      console.error(error)
      message.error('分块失败：' + error.message)
    })
    .finally(() => {
      state.loading = false
    })
}

// 添加到数据库
const addToDatabase = () => {
  if (chunkResults.value.length === 0) {
    message.error('没有可添加的分块')
    return
  }

  state.adding = true
  state.lock = true

  // 转换为API需要的格式
  const fileChunks = {}
  chunkResults.value.forEach((file) => {
    fileChunks[file.file_id] = file
  })

  // 调用add-by-chunks接口将分块添加到数据库
  apiFetch('/data/add-by-chunks', {
    method: 'POST',
    body: {
      db_id: databaseId.value,
      file_chunks: fileChunks
    },
    timeoutMs: 300000
  })
    .then((data) => {
      if (data?.status === 'failed') {
        message.error(data?.message || '添加失败')
      } else {
        message.success(data?.message || '已添加')
        fileList.value = []
        chunkResults.value = []
        activeFileKeys.value = []
      }
    })
    .catch((error) => {
      console.error(error)
      message.error(error?.message || '添加失败')
    })
    .finally(() => {
      getDatabaseInfo()
      state.adding = false
      state.lock = false
    })
}

const columns = [
  // { title: '文件ID', dataIndex: 'file_id', key: 'file_id' },
  { title: '文件名', dataIndex: 'filename', key: 'filename' },
  { title: '上传时间', dataIndex: 'created_at', key: 'created_at' },
  { title: '状态', dataIndex: 'status', key: 'status' },
  { title: '类型', dataIndex: 'type', key: 'type' },
  { title: '操作', key: 'action', dataIndex: 'file_id' }
]

watch(
  () => route.params.database_id,
  (newId) => {
    databaseId.value = newId
    clearInterval(state.refreshInterval)
    getDatabaseInfo()
  }
)

onMounted(() => {
  getDatabaseInfo()
  state.refreshInterval = setInterval(() => {
    getDatabaseInfo()
  }, 10000)
})

// 添加 onUnmounted 钩子，在组件卸载时清除定时器
onUnmounted(() => {
  if (state.refreshInterval) {
    clearInterval(state.refreshInterval)
    state.refreshInterval = null
  }
})
</script>

<style lang="less" scoped>
.database-info {
  margin: 8px 0 0;
}

.db-main-container {
  display: flex;
  width: 100%;
}

.db-tab-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.query-test-container {
  display: flex;
  flex-direction: row;
  gap: 20px;

  .sider {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 325px;
    height: 100%;
    padding: 0;
    flex: 0 0 325px;

    .sider-top {
      .query-params {
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
        font-size: 15px;
        gap: 12px;
        padding-top: 12px;
        padding-right: 16px;
        border: 1px solid var(--main-light-3);
        background-color: var(--main-light-6);
        border-radius: 8px;
        padding: 16px;
        margin-right: 8px;

        .params-title {
          margin-top: 0;
          margin-bottom: 16px;
          color: var(--main-color);
          font-size: 18px;
          text-align: center;
          font-weight: bold;
        }

        .params-group {
          margin-bottom: 16px;
          padding-bottom: 16px;
          border-bottom: 1px solid var(--main-light-3);

          &:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
          }
        }

        .params-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;

          &:last-child {
            margin-bottom: 0;
          }

          p {
            margin: 0;
            color: var(--gray-900);
          }

          &.col {
            align-items: flex-start;
            flex-direction: column;
            width: 100%;
            height: auto;
          }

          &.w100,
          &.col {
            & > * {
              width: 100%;
            }
          }
        }

        .ant-slider {
          margin: 6px 0px;
        }
      }
    }
  }

  .query-result-container {
    flex: 1;
    padding-bottom: 20px;
  }

  .query-action {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;

    textarea {
      padding: 12px 16px;
      border: 1px solid var(--main-light-2);
    }

    button.btn-query {
      height: auto;
      width: 100px;
      box-shadow: none;
      border: none;
      font-weight: bold;
      background: var(--main-light-3);
      color: var(--main-color);

      &:disabled {
        cursor: not-allowed;
        background: var(--main-light-4);
        color: var(--gray-700);
      }
    }
  }

  .query-examples-container {
    margin-bottom: 20px;
    padding: 12px;
    background: var(--main-light-6);
    border-radius: 8px;
    border: 1px solid var(--main-light-3);

    .examples-title {
      font-weight: bold;
      margin-bottom: 10px;
      color: var(--main-color);
    }

    .query-examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 0;

      .ant-btn {
        font-size: 14px;
        padding: 4px 12px;
        height: auto;
        background-color: var(--gray-200);
        border: none;
        color: var(--gray-800);

        &:hover {
          color: var(--main-color);
        }
      }
    }
  }

  .query-test {
    display: flex;
    flex-direction: column;
    border-radius: 12px;
    gap: 20px;

    .results-overview {
      background-color: var(--surface-color);
      border-radius: 8px;
      padding: 16px;
      border: 1px solid var(--main-light-3);

      .results-stats {
        display: flex;
        justify-content: flex-start;

        .stat-item {
          border-radius: 4px;
          font-size: 14px;
          margin-right: 24px;
          padding: 4px 8px;
          strong {
            color: var(--main-color);
            margin-right: 4px;
          }
        }
      }

      .rewritten-query {
        border-radius: 4px;
        font-size: 14px;
        padding: 4px 8px;
        strong {
          color: var(--main-color);
          margin-right: 8px;
        }

        .query-text {
          font-style: italic;
          color: var(--gray-900);
        }
      }
    }

    .query-result-card {
      padding: 20px;
      border-radius: 8px;
      background: var(--surface-color);
      border: 1px solid var(--main-light-3);
      transition: box-shadow 0.3s ease;

      &:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      }

      p {
        margin-bottom: 8px;
        line-height: 1.6;
        color: var(--gray-900);

        &:last-child {
          margin-bottom: 0;
        }
      }

      strong {
        color: var(--main-color);
      }

      .query-text {
        font-size: 15px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid var(--main-light-3);
      }
    }
  }
}

.upload {
  margin-bottom: 20px;
  .upload-dragger {
    margin: 0px;
  }
}

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

.custom-class .line-text {
  padding: 10px;
  border-radius: 4px;

  &:hover {
    background-color: color-mix(in srgb, var(--primary-color) 8%, transparent);
  }
}

.upload-section {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 20px;

  .upload-sidebar {
    padding: 20px;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: var(--radius-lg);
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03), 0 4px 12px color-mix(in srgb, var(--primary-color) 3%, transparent);

    .chunking-params {
      h4 {
        margin-top: 0;
        margin-bottom: 16px;
        color: var(--primary-color);
        font-size: 16px;
        font-weight: 650;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
      }

      .params-info {
        background: color-mix(in srgb, var(--primary-color) 6%, var(--surface-color));
        border-radius: var(--radius-md);
        padding: 12px 14px;
        margin-bottom: 16px;
        border: 1px solid color-mix(in srgb, var(--primary-color) 10%, transparent);

        p {
          margin: 0;
          font-size: 13px;
          line-height: 1.5;
          color: var(--gray-600);
        }
      }

      .ant-form-item {
        margin-bottom: 16px;

        .ant-form-item-label {
          padding-bottom: 6px;

          label {
            color: var(--text-color);
            font-weight: 500;
            font-size: 14px;
          }
        }
      }

      .ant-input-number {
        width: 100%;
        border-radius: var(--radius-sm);

        &:hover,
        &:focus {
          border-color: var(--primary-color);
        }
      }

      .ant-switch {
        &.ant-switch-checked {
          background-color: var(--primary-color);
        }
      }

      .param-description {
        color: var(--gray-500);
        font-size: 12px;
        margin-top: 4px;
        margin-bottom: 0;
      }
    }
  }

  .upload-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 16px;

    .upload {
      :deep(.ant-upload-dragger) {
        background: rgba(255, 247, 237, 0.7);
        border: 2px dashed color-mix(in srgb, var(--primary-color) 50%, transparent);
        border-radius: var(--radius-lg);
        transition: all 0.25s ease;

        &:hover {
          border-color: var(--primary-color);
          background: rgba(255, 237, 213, 0.9);
          box-shadow: 0 0 0 4px color-mix(in srgb, var(--primary-color) 8%, transparent);
        }
      }

      :deep(.ant-upload-drag-hover) {
        border-style: solid !important;
        border-color: var(--primary-color) !important;
        background: rgba(255, 237, 213, 1) !important;
        box-shadow: 0 0 0 4px color-mix(in srgb, var(--primary-color) 15%, transparent) !important;
      }

      :deep(.ant-upload-text) {
        font-size: 15px;
        font-weight: 500;
        color: var(--text-color);
      }

      :deep(.ant-upload-hint) {
        color: var(--gray-500);
      }
    }
  }
}

.chunk-preview {
  margin-top: 24px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-radius: var(--radius-lg);
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);

  .preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);

    h3 {
      margin: 0;
      color: var(--text-color);
      font-size: 16px;
      font-weight: 650;
    }
  }

  .result-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 10px;
  }

  .chunk {
    background: color-mix(in srgb, var(--primary-color) 4%, var(--surface-color));
    border: 1px solid color-mix(in srgb, var(--primary-color) 10%, transparent);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    word-wrap: break-word;
    word-break: break-all;
    transition: all 0.2s ease;

    &:hover {
      background: color-mix(in srgb, var(--primary-color) 8%, var(--surface-color));
      border-color: color-mix(in srgb, var(--primary-color) 20%, transparent);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px color-mix(in srgb, var(--primary-color) 8%, transparent);
    }

    p {
      margin: 0;
      line-height: 1.6;
      font-size: 13px;
      color: var(--gray-700);
      display: -webkit-box;
      -webkit-line-clamp: 4;
      -webkit-box-orient: vertical;
      overflow: hidden;

      strong {
        color: var(--primary-color);
        margin-right: 6px;
        font-size: 12px;
      }
    }
  }

  :deep(.ant-collapse) {
    background: transparent;
    border: none;
  }

  :deep(.ant-collapse-item) {
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md) !important;
    margin-bottom: 8px;
    overflow: hidden;
  }

  :deep(.ant-collapse-header) {
    font-weight: 500;
    color: var(--text-color);
  }

  :deep(.ant-collapse-content) {
    border-top: 1px solid var(--border-color);
  }
}
</style>

<style lang="less">
.db-main-container {
  .atab-container {
    padding: 0;
    width: 100%;
    max-height: 100%;
    overflow: auto;

    div.ant-tabs-nav {
      background: var(--surface-color);
      padding: 8px 20px;
      padding-bottom: 0;
      border-bottom: 1px solid var(--border-color);
    }

    .ant-tabs-content-holder {
      padding: 20px;
    }
  }

  .params-item.col .ant-segmented {
    width: 100%;
    font-size: smaller;
    div.ant-segmented-group {
      display: flex;
      justify-content: space-around;
    }
    label.ant-segmented-item {
      flex: 1;
      text-align: center;
      div.ant-segmented-item-label > div > p {
        font-size: small;
      }
    }
  }
}

/* 暗色模式 */
:root[data-theme='dark'] {
  .upload-section {
    .upload-sidebar {
      background: rgba(40, 40, 40, 0.85);
      border-color: rgba(255, 255, 255, 0.08);
    }

    .upload-main .upload {
      :deep(.ant-upload-dragger) {
        background: color-mix(in srgb, var(--primary-color) 8%, transparent);
        border-color: color-mix(in srgb, var(--primary-color) 35%, transparent);

        &:hover {
          background: color-mix(in srgb, var(--primary-color) 12%, transparent);
        }
      }

      :deep(.ant-upload-drag-hover) {
        background: color-mix(in srgb, var(--primary-color) 15%, transparent) !important;
      }
    }
  }

  .chunk-preview {
    background: rgba(40, 40, 40, 0.85);
    border-color: rgba(255, 255, 255, 0.08);

    .chunk {
      background: color-mix(in srgb, var(--primary-color) 8%, transparent);
      border-color: color-mix(in srgb, var(--primary-color) 15%, transparent);

      &:hover {
        background: color-mix(in srgb, var(--primary-color) 12%, transparent);
      }

      p {
        color: var(--gray-300);
      }
    }
  }

  .custom-class .line-text {
    &:hover {
      background-color: var(--gray-800);
    }
  }
}
</style>
