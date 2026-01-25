<template>
  <div class="home-page">
    <header class="home-header">
      <div class="brand" role="banner">
        <img class="brand__logo" src="/logo.png" alt="Logo" loading="lazy" />
        <div class="brand__meta">
          <div class="brand__title">{{ APP_NAME }}</div>
          <div class="brand__sub">
            知识库 / 知识图谱增强的专域对话助手
            <span v-if="buildLabel" class="brand__build">· {{ buildLabel }}</span>
          </div>
        </div>
      </div>

      <a-space>
        <a-button :href="apiDocsUrl" target="_blank" rel="noopener noreferrer">
          <template #icon><ApiOutlined /></template>
          API 文档
        </a-button>
        <a-button @click="go('/setting')">
          <template #icon><SettingOutlined /></template>
          设置
        </a-button>
        <a-button type="primary" @click="go('/chat')">
          <template #icon><MessageOutlined /></template>
          进入对话
        </a-button>
      </a-space>
    </header>

    <main class="home-content">
      <a-row :gutter="[16, 16]">
        <a-col :xs="24" :md="12" :xl="8">
          <a-card class="entry-card" hoverable @click="go('/chat')">
            <template #title>
              <a-space><MessageOutlined /> 对话</a-space>
            </template>
            <div class="muted">多智能体编排 + 记忆 + RAG/图谱/工具协同</div>
          </a-card>
        </a-col>

        <a-col v-if="ui.show_knowledge_base" :xs="24" :md="12" :xl="8">
          <a-card class="entry-card" hoverable @click="go('/database')">
            <template #title>
              <a-space><BookOutlined /> 知识库</a-space>
            </template>
            <div class="muted">文档解析、切分、检索与可视化管理</div>
          </a-card>
        </a-col>

        <a-col v-if="ui.show_knowledge_graph" :xs="24" :md="12" :xl="8">
          <a-card class="entry-card" hoverable @click="go('/graph')">
            <template #title>
              <a-space><ProjectOutlined /> 知识图谱</a-space>
            </template>
            <div class="muted">实体关系探索 + GraphRAG</div>
          </a-card>
        </a-col>

        <a-col v-if="ui.show_agents" :xs="24" :md="12" :xl="8">
          <a-card class="entry-card" hoverable @click="go('/agent')">
            <template #title>
              <a-space><RobotOutlined /> 智能体</a-space>
            </template>
            <div class="muted">专业子代理：图鉴、数值、训练、深度研究</div>
          </a-card>
        </a-col>

        <a-col v-if="ui.show_tools" :xs="24" :md="12" :xl="8">
          <a-card class="entry-card" hoverable @click="go('/tools')">
            <template #title>
              <a-space><ToolOutlined /> 工具</a-space>
            </template>
            <div class="muted">文本切分、格式转换等辅助能力</div>
          </a-card>
        </a-col>

        <a-col v-if="ui.show_map" :xs="24" :md="12" :xl="8">
          <a-card class="entry-card" hoverable @click="go('/coords')">
            <template #title>
              <a-space><EnvironmentOutlined /> 地图</a-space>
            </template>
            <div class="muted">宝可梦地点与真实世界坐标映射</div>
          </a-card>
        </a-col>
      </a-row>

      <a-card class="status-card" title="系统状态" :bordered="false">
        <template #extra>
          <a-button size="small" @click="refreshStatus" :loading="refreshing">刷新</a-button>
        </template>

        <div class="status-grid">
          <div class="status-block">
            <div class="status-label">Backend</div>
            <a-space wrap>
              <StatusTag :status="backendOnline ? 'online' : 'offline'" />
              <StatusTag :status="backendReady ? 'ready' : 'not_ready'" />
            </a-space>
          </div>

          <div class="status-block">
            <div class="status-label">Capabilities</div>
            <a-space wrap>
              <a-tag :color="configStore.config.enable_knowledge_base ? 'green' : 'default'"
                >知识库</a-tag
              >
              <a-tag :color="configStore.config.enable_knowledge_graph ? 'green' : 'default'"
                >知识图谱</a-tag
              >
              <a-tag :color="configStore.config.enable_web_search ? 'green' : 'default'"
                >联网搜索</a-tag
              >
              <a-tag :color="configStore.config.enable_mcp ? 'green' : 'default'">MCP</a-tag>
              <a-tag :color="configStore.config.enable_reranker ? 'green' : 'default'"
                >Reranker</a-tag
              >
            </a-space>
          </div>
        </div>

        <a-alert
          v-if="configStore.config.backend?.last_error"
          style="margin-top: 12px"
          type="error"
          show-icon
          message="Last error"
          :description="configStore.config.backend?.last_error"
        />
      </a-card>
    </main>

    <footer class="home-footer">
      <span class="muted">© {{ year }} {{ APP_NAME }}. All rights reserved.</span>
      <span class="muted">{{ buildLabel }}</span>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  ApiOutlined,
  BookOutlined,
  EnvironmentOutlined,
  MessageOutlined,
  ProjectOutlined,
  RobotOutlined,
  SettingOutlined,
  ToolOutlined
} from '@ant-design/icons-vue'

import StatusTag from '@/components/StatusTag.vue'
import { useConfigStore } from '@/stores/config'
import { APP_NAME, getBuildLabel } from '@/config/appMeta'

const router = useRouter()
const configStore = useConfigStore()

const year = new Date().getFullYear()
const buildLabel = getBuildLabel()

const refreshing = ref(false)
const refreshStatus = async () => {
  refreshing.value = true
  try {
    await configStore.refreshConfig()
  } finally {
    refreshing.value = false
  }
}


const ui = computed(() => configStore.config?.ui || {})
const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const backendReady = computed(() => Boolean(configStore.config.backend?.ready))

const apiDocsUrl = computed(() => `${window.location.origin}/api/docs`)

const go = (path: string) => router.push(path)

onMounted(async () => {
  // Keep homepage informative even before entering AppLayout.
  await refreshStatus()
})

onUnmounted(() => {
})
</script>

<style scoped lang="less">
.home-page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--background-color);
  color: var(--text-color);
}

.home-header {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: color-mix(in srgb, var(--surface-color) 92%, transparent);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.brand__logo {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  box-shadow: var(--shadow-xs);
}

.brand__meta {
  min-width: 0;
}

.brand__title {
  font-size: 18px;
  font-weight: 650;
  line-height: 1.2;
}

.brand__sub {
  font-size: 12px;
  color: var(--subtext-color);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.brand__build {
  opacity: 0.9;
}

.home-content {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  flex: 1 1 auto;
}

.entry-card {
  height: 100%;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-xs);
}

.entry-card :deep(.ant-card-head-title) {
  font-weight: 600;
}

.status-card {
  margin-top: 16px;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-xs);
}

.status-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.status-block {
  flex: 1 1 360px;
  min-width: 260px;
}

.status-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--subtext-color);
  margin-bottom: 8px;
}

.muted {
  color: var(--subtext-color);
}

.home-footer {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 24px;
  border-top: 1px solid var(--border-color);
  background: var(--surface-color);
}
</style>
