<template>
  <div class="database-container layout-container">
    <HeaderComponent
      title="文档知识库"
      description="知识型数据库，主要是非结构化的文本组成，使用向量检索使用。如果出现问题，可以检查 saves/data/database.json 查看配置。"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '知识库' }]"
    >
      <template #actions>
        <a-button type="primary" @click="newDatabase.open = true">新建知识库</a-button>
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
        在智能体流程中，这里的描述会作为工具的描述。智能体会根据知识库的标题和描述来选择合适的工具。所以这里描述的越详细，智能体越容易选择到合适的工具。
      </p>
      <a-textarea
        v-model:value="newDatabase.description"
        placeholder="新建知识库描述"
        :auto-size="{ minRows: 5, maxRows: 10 }"
      />
      <!-- <h3 style="margin-top: 20px;">向量维度</h3>
      <p>必须与向量模型 {{ configStore.config.embed_model }} 一致</p>
      <a-input v-model:value="newDatabase.dimension" placeholder="向量维度 (e.g. 768, 1024)" /> -->
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

        <div class="databases">
          <div class="new-database dbcard ui-card" @click="newDatabase.open = true">
            <div class="top">
              <div class="icon"><PlusOutlined /></div>
              <div class="info">
                <h3>新建知识库</h3>
              </div>
            </div>
            <p>导入您自己的文本数据或通过 Webhook 实时写入数据以增强 LLM 的上下文。</p>
          </div>

          <div class="workbench dbcard ui-card" @click="navigateToWorkbench">
            <div class="top">
              <div class="icon"><ExperimentOutlined /></div>
              <div class="info">
                <h3>知识库工作台</h3>
              </div>
            </div>
            <p>切块、解析、索引写入等 RAG 工具集中在这里。</p>
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
            >
              <div class="top">
                <div class="icon"><ReadFilled /></div>
                <div class="info">
                  <h3>{{ database.name }}</h3>
                  <p>
                    <span>{{ database.files ? Object.keys(database.files).length : 0 }} 文件</span>
                  </p>
                </div>
              </div>
              <p class="description">{{ database.description || '暂无描述' }}</p>
              <div class="tags">
                <a-tag color="blue" v-if="database.embed_model">{{ database.embed_model }}</a-tag>
                <a-tag color="green" v-if="database.dimension">{{ database.dimension }}</a-tag>
              </div>
              <!-- <button @click="deleteDatabase(database.collection_name)">删除</button> -->
            </div>
          </template>
        </div>
      </div>
    </div>
    <!-- <h2>图数据库 &nbsp; <a-spin v-if="graphloading" :indicator="indicator" /></h2>
    <p>基于 neo4j 构建的图数据库。</p>
    <div :class="{'graphloading': graphloading, 'databases': true}" v-if="graph">
      <div class="dbcard graphbase" @click="navigateToGraph">
        <div class="top">
          <div class="icon"><AppstoreFilled /></div>
          <div class="info">
            <h3>{{ graph?.database_name }}</h3>
            <p>
              <span>{{ graph?.status }}</span> ·
              <span>{{ graph?.entity_count }}实体</span>
            </p>
          </div>
        </div>
        <p class="description">基于 neo4j 构建的图数据库。基于 neo4j 构建的图数据库。基于 neo4j 构建的图数据库。</p>
      </div>
    </div> -->
  </div>
</template>

<script setup>
import { ref, onMounted, reactive, watch, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import { ExperimentOutlined, ReadFilled, PlusOutlined } from '@ant-design/icons-vue'
import { useConfigStore } from '@/stores/config'
import HeaderComponent from '@/components/HeaderComponent.vue'
import { apiFetch } from '@/api/http'

const route = useRoute()
const router = useRouter()
const databases = ref([])

const configStore = useConfigStore()
const canUseKb = computed(
  () =>
    Boolean(configStore.config.backend?.online) && Boolean(configStore.config.enable_knowledge_base)
)

const state = reactive({
  loading: false
})

const newDatabase = reactive({
  name: '',
  description: '',
  dimension: '',
  loading: false
})

const loadDatabases = () => {
  state.loading = true
  apiFetch('/data/', { method: 'GET' })
    .then((data) => {
      databases.value = data?.databases || []
    })
    .catch((err) => {
      databases.value = []
      message.error(err?.message || '获取知识库列表失败')
    })
    .finally(() => {
      state.loading = false
    })
}

const createDatabase = () => {
  newDatabase.loading = true
  if (!newDatabase.name) {
    message.error('数据库名称不能为空')
    newDatabase.loading = false
    return
  }
  apiFetch('/data/', {
    method: 'POST',
    body: {
      database_name: newDatabase.name,
      description: newDatabase.description,
      dimension: newDatabase.dimension ? parseInt(newDatabase.dimension) : null
    }
  })
    .then(() => {
      loadDatabases()
      newDatabase.open = false
      newDatabase.name = ''
      ;(newDatabase.description = ''), (newDatabase.dimension = '')
    })
    .finally(() => {
      newDatabase.loading = false
    })
}

const navigateToDatabase = (databaseId) => {
  router.push({ path: `/database/${databaseId}` })
}

const navigateToWorkbench = () => {
  router.push({ path: `/database/workbench` })
}

watch(
  () => route.path,
  (newPath, _oldPath) => {
    if (newPath === '/database') {
      loadDatabases()
    }
  }
)

onMounted(() => {
  loadDatabases()
})
</script>

<style lang="less" scoped>
.database-container {
  padding: 0; // Let HeaderComponent span full width (consistent with Tools/Graph pages)
}

.database-actions,
.document-actions {
  margin-bottom: 20px;
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
      font-size: 28px;
      margin-right: 10px;
      display: flex;
      justify-content: center;
      align-items: center;
      background-color: var(--main-10);
      border-radius: 8px;
      border: 1px solid var(--main-10);
      color: var(--main-color);
    }

    .info {
      h3,
      p {
        margin: 0;
        color: var(--text-color);
      }

      h3 {
        font-size: 16px;
        font-weight: bold;
      }

      p {
        color: var(--gray-900);
        font-size: small;
      }
    }
  }

  .description {
    color: var(--gray-900);
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 1;
    -webkit-box-orient: vertical;
    text-overflow: ellipsis;
    margin-bottom: 10px;
  }
}

.dbcard--skeleton {
  cursor: default;
}

.db-empty {
  padding: 32px 0;

  :deep(.ant-empty-description) {
    color: var(--gray-600);
  }
}

.form-title {
  font-size: 14px;
  font-weight: 650;
  margin: 0 0 8px;
}

.form-title--spaced {
  margin-top: 20px;
}

.form-hint {
  color: var(--gray-600);
  font-size: 13px;
  margin: 8px 0 12px;
}

.required {
  color: var(--error-color);
}

// 整个卡片是模糊的
// .graphloading {
//   filter: blur(2px);
// }

.database-empty {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  flex-direction: column;
  color: var(--gray-900);
}

.database-container {
  padding: 0;
}
</style>
