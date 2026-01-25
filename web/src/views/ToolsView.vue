<template>
  <div class="tools-container layout-container">
    <HeaderComponent
      title="工具箱"
      description="这里展示了各种可用的工具"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '工具' }]"
    />
    <div class="ui-page">
      <div class="ui-container">
        <div class="tools-grid">
          <template v-if="state.loadingTools">
            <div v-for="n in 6" :key="n" class="tool-card ui-card tool-card--skeleton">
              <a-skeleton active :title="false" :paragraph="{ rows: 3 }" />
            </div>
          </template>
          <template v-else>
            <a-empty v-if="tools.length === 0" class="tools-empty" description="暂无工具">
              <a-space>
                <a-button type="primary" @click="getTools">刷新</a-button>
                <a-button @click="router.push('/setting')">去设置</a-button>
              </a-space>
            </a-empty>
            <div
              v-for="tool in tools"
              :key="tool.name"
              class="tool-card ui-card"
              @click="navigateToTool(tool.url)"
              role="button"
              tabindex="0"
              @keydown.enter.prevent="navigateToTool(tool.url)"
              @keydown.space.prevent="navigateToTool(tool.url)"
            >
              <div class="tool-top">
                <div class="tool-icon" aria-hidden="true">
                  <component :is="getToolIcon(tool.name)" />
                </div>
                <div class="tool-meta">
                  <div class="tool-title">{{ tool.title }}</div>
                  <div class="tool-desc ui-muted">{{ tool.description }}</div>
                </div>
              </div>
              <div class="tool-actions">
                <a-button size="small" type="primary" @click.stop="navigateToTool(tool.url)"
                  >打开</a-button
                >
              </div>
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { FilePdfOutlined, RobotOutlined, ScissorOutlined, ToolOutlined } from '@ant-design/icons-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import { apiFetch } from '@/api/http'

const router = useRouter()
const tools = ref([])

const state = reactive({
  loadingTools: true
})

const getToolIcon = (name) => {
  if (name === 'file-chunking') return ScissorOutlined
  if (name === 'pdf2txt') return FilePdfOutlined
  if (name === 'agent') return RobotOutlined
  return ToolOutlined
}

const getTools = () => {
  state.loadingTools = true
  apiFetch('/tools/', { method: 'GET', timeoutMs: 5000 })
    .then((data) => {
      tools.value = Array.isArray(data) ? data : data?.tools || []
    })
    .catch(() => {
      // Offline fallback so the page still renders.
      tools.value = [
        {
          name: 'file-chunking',
          title: '文件分块',
          description: '离线模式：仅展示',
          url: '/tools/file-chunking'
        },
        {
          name: 'pdf2txt',
          title: 'PDF 转文本',
          description: '离线模式：仅展示',
          url: '/tools/pdf2txt'
        },
        { name: 'agent', title: '智能体', description: '离线模式：仅展示', url: '/agent' }
      ]
    })
    .finally(() => {
      state.loadingTools = false
    })
}

const navigateToTool = (toolUrl) => {
  router.push(toolUrl)
}

onMounted(() => {
  getTools()
})
</script>

<style scoped lang="less">
.tools-container {
  padding: 0;
}

.tools-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;

  .tool-card {
    display: flex;
    flex-direction: column;
    padding: 16px;
    cursor: pointer;

    &:hover {
      // ui-card handles hover
    }

    .tool-top {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      min-width: 0;

      .tool-icon {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: color-mix(in srgb, var(--main-500) 12%, var(--surface-color));
        color: var(--main-600);
        flex: 0 0 auto;
        box-shadow: var(--shadow-xs);
      }

      .tool-meta {
        min-width: 0;
        flex: 1 1 auto;
      }

      .tool-title {
        font-size: 15px;
        font-weight: 650;
        line-height: 1.3;
      }

      .tool-desc {
        margin-top: 4px;
        font-size: 13px;
      }
    }

    .tool-actions {
      display: flex;
      justify-content: flex-end;
      margin-top: 14px;
    }
  }

  .tool-card--skeleton {
    cursor: default;
  }
}

.tools-empty {
  padding: 42px 0;
}
</style>
