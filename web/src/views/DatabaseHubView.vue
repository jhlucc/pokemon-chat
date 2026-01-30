<template>
  <DatabaseLayout>
    <div class="database-hub">
      <HeaderComponent
      :title="headerTitle"
      :description="headerDescription"
      :breadcrumbs="breadcrumbs"
    >
      <template #actions>
        <!-- 空状态时隐藏按钮，聚焦中间引导 -->
        <a-space wrap v-if="!isEmptyState">
          <a-button @click="refresh" :loading="state.loading" :disabled="!backendOnline">
            <ReloadOutlined /> 刷新
          </a-button>
          <a-button type="primary" @click="newDatabase.open = true">
            <PlusOutlined /> 新建知识库
          </a-button>
        </a-space>
      </template>
    </HeaderComponent>

    <a-modal
      :open="newDatabase.open"
      title="新建知识库"
      @ok="createDatabase"
      @cancel="newDatabase.open = false"
    >
      <h3 class="form-title">知识库名称 <span class="required">*</span></h3>
      <a-input v-model:value="newDatabase.name" placeholder="新建知识库名称" />
      <h3 class="form-title form-title--spaced">知识库描述</h3>
      <p class="form-hint">
        在智能体流程中，这里的描述会作为工具的描述。智能体会根据知识库的标题和描述来选择合适的工具。描述越清晰，效果越好。
      </p>
      <a-textarea
        v-model:value="newDatabase.description"
        placeholder="新建知识库描述"
        :auto-size="{ minRows: 5, maxRows: 10 }"
      />

      <template #footer>
        <a-button key="back" @click="newDatabase.open = false">取消</a-button>
        <a-button key="submit" type="primary" :loading="newDatabase.loading" @click="createDatabase"
          >创建</a-button
        >
      </template>
    </a-modal>

    <div class="ui-page">
      <div class="ui-container">
        <!-- 分段控制器 Tab -->
        <div class="segmented-tabs">
          <button
            class="seg-tab"
            :class="{ active: activeTab === 'list' }"
            @click="onTabChange('list')"
          >
            知识库列表
          </button>
          <button
            class="seg-tab"
            :class="{ active: activeTab === 'workbench' }"
            @click="onTabChange('workbench')"
          >
            RAG 工作台
          </button>
        </div>

        <!-- 知识库列表 -->
        <div v-if="activeTab === 'list'" class="tab-content">
          <!-- 加载骨架屏 -->
          <div v-if="state.loading" class="databases">
            <div v-for="n in 6" :key="n" class="dbcard dbcard--skeleton">
              <a-skeleton active :title="{ width: '60%' }" :paragraph="{ rows: 2 }" />
            </div>
          </div>

          <!-- 空状态：居中聚焦 -->
          <div v-else-if="databases.length === 0" class="empty-state">
            <div class="empty-illustration">
              <BookOutlined />
            </div>
            <h3 class="empty-title">开始构建你的知识粮仓</h3>
            <p class="empty-desc">导入文档数据，增强模型上下文。把文档变成 AI 的长期记忆。</p>
            <a-button type="primary" size="large" @click="newDatabase.open = true">
              <PlusOutlined /> 新建知识库
            </a-button>
          </div>

          <!-- 知识库列表 -->
          <div v-else class="databases">
            <div
              v-for="database in databases"
              :key="database.db_id"
              class="dbcard"
              @click="navigateToDatabase(database.db_id)"
              role="button"
              tabindex="0"
              @keydown.enter.prevent="navigateToDatabase(database.db_id)"
              @keydown.space.prevent="navigateToDatabase(database.db_id)"
            >
              <div class="top">
                <div class="icon"><ReadFilled /></div>
                <div class="info">
                  <h3>{{ database.name }}</h3>
                  <p>
                    <span>{{ database.files ? Object.keys(database.files).length : 0 }} 文件</span>
                    <span class="muted">· {{ database.db_id }}</span>
                  </p>
                </div>
              </div>
              <p class="description">{{ database.description || '暂无描述' }}</p>
              <div class="tags">
                <a-tag color="blue" v-if="database.embed_model">{{ database.embed_model }}</a-tag>
                <a-tag color="green" v-if="database.dimension">{{ database.dimension }}</a-tag>
              </div>
            </div>
          </div>
        </div>

        <!-- RAG 工作台 -->
        <div v-if="activeTab === 'workbench'" class="tab-content">
          <DatabaseRagWorkbench />
        </div>
      </div>
    </div>
  </div>
  </DatabaseLayout>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined, ReadFilled, BookOutlined, ReloadOutlined } from '@ant-design/icons-vue'

import HeaderComponent from '@/components/HeaderComponent.vue'
import DatabaseLayout from '@/components/database/DatabaseLayout.vue'
import DatabaseRagWorkbench from '@/components/database/DatabaseRagWorkbench.vue'
import { apiFetch } from '@/api/http'
import { useConfigStore } from '@/stores/config'

const route = useRoute()
const router = useRouter()
const configStore = useConfigStore()

const databases = ref([])
const state = reactive({ loading: false })

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const canUseKb = computed(
  () => Boolean(configStore.config.backend?.online) && Boolean(configStore.config.enable_knowledge_base)
)

// 空状态判断 - 用于弱化右上角按钮
const isEmptyState = computed(
  () => activeTab.value === 'list' && !state.loading && databases.value.length === 0
)

const inferTabFromRoute = () => {
  const metaKey = route.meta?.databaseTab
  if (metaKey === 'workbench') return 'workbench'
  if (metaKey === 'list') return 'list'
  return route.path.startsWith('/database/workbench') ? 'workbench' : 'list'
}

const activeTab = ref(inferTabFromRoute())

watch(
  () => route.fullPath,
  () => {
    activeTab.value = inferTabFromRoute()
  }
)

const onTabChange = (key) => {
  const nextKey = key === 'workbench' ? 'workbench' : 'list'
  const nextPath = nextKey === 'workbench' ? '/database/workbench' : '/database'
  if (route.path !== nextPath) router.push(nextPath)
}

const headerTitle = computed(() => (activeTab.value === 'workbench' ? '知识库工作台' : '文档知识库'))
const headerDescription = computed(() =>
  activeTab.value === 'workbench'
    ? '把文件解析/切块并写入知识库索引（原工具箱能力已合并到这里）'
    : '管理文档型知识库，并在工作台中完成解析、切块与索引写入。'
)
const breadcrumbs = computed(() => {
  if (activeTab.value === 'workbench') {
    return [{ label: '首页', to: '/' }, { label: '知识库', to: '/database' }, { label: '工作台' }]
  }
  return [{ label: '首页', to: '/' }, { label: '知识库' }]
})

const loadDatabases = async () => {
  state.loading = true
  try {
    const data = await apiFetch('/data/', { method: 'GET' })
    databases.value = data?.databases || []
  } catch (err) {
    databases.value = []
    // Avoid spamming errors in offline mode when users just browse UI.
    if (backendOnline.value) message.error(err?.message || '获取知识库列表失败')
  } finally {
    state.loading = false
  }
}

const refresh = async () => {
  await configStore.refreshConfig()
  await loadDatabases()
}

const newDatabase = reactive({
  open: false,
  name: '',
  description: '',
  loading: false
})

const createDatabase = async () => {
  if (!newDatabase.name) {
    message.error('知识库名称不能为空')
    return
  }
  newDatabase.loading = true
  try {
    await apiFetch('/data/', {
      method: 'POST',
      body: {
        database_name: newDatabase.name,
        description: newDatabase.description
      }
    })
    newDatabase.open = false
    newDatabase.name = ''
    newDatabase.description = ''
    await loadDatabases()
  } finally {
    newDatabase.loading = false
  }
}

const navigateToDatabase = (databaseId) => {
  router.push({ path: `/database/${databaseId}` })
}

onMounted(async () => {
  // Keep the page informative even if the user lands on the workbench tab first.
  await loadDatabases()
})
</script>

<style scoped lang="less">
.database-hub {
  position: relative;
  min-height: 100vh;
  /* 透明化：让光晕透出来 */
  background: transparent;
  overflow-x: hidden;
}

.ui-page {
  position: relative;
  z-index: 1;
}

/* 分段控制器 Tab - 去容器化 (De-containerization) */
.segmented-tabs {
  display: inline-flex;
  gap: 8px;
  margin-bottom: 24px;
  /* 容器完全透明，让暖色光晕直接透出 */
  background: transparent;
  border: none;
  padding: 0;
}

.seg-tab {
  padding: 10px 20px;
  border: none;
  /* 未选中态：完全透明，文字直接浮在背景上 */
  background: transparent;
  font-size: 14px;
  font-weight: 500;
  color: var(--gray-500);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover:not(.active) {
    color: var(--gray-700);
    /* hover时微微显现白底 */
    background: rgba(255, 255, 255, 0.4);
  }

  &.active {
    /* 选中态：实体悬浮按钮 - 唯一的"物理开关" */
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: var(--primary-color);
    font-weight: 600;
    /* 玻璃边缘高光 */
    border: 1px solid rgba(255, 255, 255, 0.6);
    /* 柔和阴影，悬浮感 */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06), 0 4px 12px rgba(255, 125, 0, 0.04);
  }
}

.tab-content {
  animation: fadeIn 0.2s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 空状态：居中聚焦 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 24px;
  text-align: center;
}

.empty-illustration {
  width: 120px;
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
  background: linear-gradient(135deg, rgba(255, 125, 0, 0.1) 0%, rgba(255, 125, 0, 0.03) 100%);
  border-radius: 50%;
  font-size: 48px;
  color: var(--primary-color);
}

.empty-title {
  margin: 0 0 8px;
  font-size: 20px;
  font-weight: 600;
  color: var(--text-color);
}

.empty-desc {
  margin: 0 0 24px;
  font-size: 14px;
  color: var(--gray-500);
  max-width: 320px;
  line-height: 1.6;
}

/* 知识库卡片网格 */
.databases {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

/* 知识库卡片 - 微磨砂质感 */
.dbcard {
  position: relative;
  width: 100%;
  padding: 24px;
  border-radius: 20px;
  min-height: 180px;
  cursor: pointer;
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.5);
  /* 有色阴影：微微带橙 */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03), 0 4px 12px rgba(255, 125, 0, 0.03);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    transform: translateY(-4px);
    /* hover时橙色阴影更明显 */
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06), 0 10px 30px rgba(255, 125, 0, 0.12);
    border-color: rgba(255, 125, 0, 0.2);
    background: rgba(255, 255, 255, 0.9);

    .icon {
      transform: scale(1.05);
      box-shadow: 0 4px 12px rgba(255, 125, 0, 0.2);
    }
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--primary-color);
  }

  .top {
    display: flex;
    align-items: center;
    height: 56px;
    margin-bottom: 12px;

    .icon {
      width: 56px;
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-right: 14px;
      border-radius: 16px;
      font-size: 26px;
      background: linear-gradient(135deg, rgba(255, 125, 0, 0.12) 0%, rgba(255, 125, 0, 0.06) 100%);
      color: var(--primary-color);
      transition: all 0.3s ease;
    }

    .info {
      flex: 1;
      min-width: 0;

      h3 {
        margin: 0;
        font-size: 17px;
        font-weight: 650;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: var(--text-color);
      }

      p {
        margin: 4px 0 0;
        font-size: 13px;
        color: var(--gray-500);
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
    }
  }

  .description {
    font-size: 14px;
    color: var(--gray-500);
    margin: 0 0 12px 0;
    line-height: 1.6;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
}

/* 骨架屏卡片 */
.dbcard--skeleton {
  cursor: default;

  &:hover {
    transform: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
  }
}

.muted {
  opacity: 0.7;
}

/* 表单样式 */
.form-title {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 650;
}

.form-title--spaced {
  margin-top: 14px;
}

.form-hint {
  margin: 0 0 10px;
  color: var(--gray-500);
  font-size: 12px;
  line-height: 1.55;
}

.required {
  color: var(--error-color);
}

/* 响应式 */
@media (max-width: 640px) {
  .databases {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .dbcard {
    padding: 20px;
    min-height: 160px;
  }

  .empty-state {
    padding: 60px 20px;
  }

  .empty-illustration {
    width: 100px;
    height: 100px;
    font-size: 40px;
  }

  .segmented-tabs {
    width: 100%;
    justify-content: center;
    gap: 6px;
  }

  .seg-tab {
    flex: 1;
    text-align: center;
    padding: 10px 12px;
    max-width: 160px;
  }
}

/* 暗色模式 */
:root[data-theme='dark'] {
  .database-hub {
    /* 暗色模式也透明，让光晕透出 */
    background: transparent;
  }

  .segmented-tabs {
    background: transparent;
    border: none;
  }

  .seg-tab {
    color: var(--gray-400);

    &:hover:not(.active) {
      color: var(--gray-300);
      background: rgba(255, 255, 255, 0.06);
    }

    &.active {
      background: rgba(255, 255, 255, 0.1);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      color: var(--primary-color);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
  }

  .dbcard {
    background: rgba(40, 40, 40, 0.75);
    border-color: rgba(255, 255, 255, 0.08);

    &:hover {
      background: rgba(50, 50, 50, 0.9);
      border-color: rgba(255, 125, 0, 0.3);
    }

    .top .icon {
      background: linear-gradient(135deg, rgba(255, 125, 0, 0.15) 0%, rgba(255, 125, 0, 0.08) 100%);
    }
  }

  .empty-illustration {
    background: linear-gradient(135deg, rgba(255, 125, 0, 0.15) 0%, rgba(255, 125, 0, 0.05) 100%);
  }
}
</style>
