<template>
  <div class="database-hub layout-container">
    <HeaderComponent
      :title="headerTitle"
      :description="headerDescription"
      :breadcrumbs="breadcrumbs"
    >
      <template #actions>
        <a-space wrap>
          <a-button @click="refresh" :loading="state.loading" :disabled="!backendOnline"
            >刷新</a-button
          >
          <a-button type="primary" @click="newDatabase.open = true">新建知识库</a-button>
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
        <a-alert
          v-if="activeTab === 'workbench' && !canUseKb"
          type="warning"
          show-icon
          :message="
            backendOnline
              ? '后端未启用知识库功能（enable_knowledge_base=false），无法写入索引'
              : '后端未启动/不可用（离线模式）'
          "
          style="margin-bottom: 12px"
        />

        <a-tabs v-model:activeKey="activeTab" type="card" class="hub-tabs" @change="onTabChange">
          <a-tab-pane key="list">
            <template #tab>知识库列表</template>

            <a-alert
              v-if="!canUseKb"
              type="warning"
              show-icon
              :message="
                backendOnline
                  ? '后端未启用知识库功能（enable_knowledge_base=false）'
                  : '后端未启动/不可用（离线模式）'
              "
              style="margin-bottom: 12px"
            />

            <div class="databases">
              <div class="new-database dbcard ui-card" @click="newDatabase.open = true">
                <div class="top">
                  <div class="icon"><PlusOutlined /></div>
                  <div class="info">
                    <h3>新建知识库</h3>
                  </div>
                </div>
                <p>导入文档数据，增强模型上下文。</p>
              </div>

              <template v-if="state.loading">
                <div v-for="n in 6" :key="n" class="dbcard ui-card dbcard--skeleton">
                  <a-skeleton active :title="{ width: '60%' }" :paragraph="{ rows: 2 }" />
                </div>
              </template>
              <template v-else>
                <a-empty v-if="databases.length === 0" class="db-empty" description="暂无知识库">
                  <a-button type="primary" @click="newDatabase.open = true">新建知识库</a-button>
                </a-empty>
                <div
                  v-for="database in databases"
                  :key="database.db_id"
                  class="database dbcard ui-card"
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
              </template>
            </div>
          </a-tab-pane>

          <a-tab-pane key="workbench">
            <template #tab>RAG 工作台</template>
            <DatabaseRagWorkbench />
          </a-tab-pane>
        </a-tabs>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined, ReadFilled } from '@ant-design/icons-vue'

import HeaderComponent from '@/components/HeaderComponent.vue'
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
  padding: 0;
}

.hub-tabs :deep(.ant-tabs-content) {
  padding-top: 8px;
}

.databases {
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;

  .new-database {
    border-style: dashed;
    background: color-mix(in srgb, var(--main-500) 6%, var(--surface-color));
  }
}

.dbcard,
.database {
  width: 100%;
  padding: 20px;
  border-radius: var(--radius-lg);
  min-height: 160px;
  cursor: pointer;

  .top {
    display: flex;
    align-items: center;
    height: 50px;
    margin-bottom: 10px;

    .icon {
      width: 50px;
      height: 50px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-right: 10px;
      border-radius: 12px;
      font-size: 22px;
      background: color-mix(in srgb, var(--main-500) 12%, var(--surface-color));
      color: var(--main-600);
      box-shadow: var(--shadow-xs);
    }

    .info {
      flex: 1;
      min-width: 0;

      h3 {
        margin: 0;
        font-size: 16px;
        font-weight: 650;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      p {
        margin: 0;
        font-size: 13px;
        color: var(--subtext-color);
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
    }
  }

  .description {
    font-size: 13px;
    color: var(--subtext-color);
    margin: 0 0 10px 0;
    line-height: 1.6;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
}

.muted {
  opacity: 0.8;
}

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
  color: var(--subtext-color);
  font-size: 12px;
  line-height: 1.55;
}

.required {
  color: var(--danger-500);
}
</style>
