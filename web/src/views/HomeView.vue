<template>
  <div class="home-universe">
    <!-- 点阵背景 -->
    <div class="dot-grid"></div>

    <!-- 极简 Header -->
    <header class="home-header">
      <div class="brand">
        <img class="brand-logo" src="/logo.png" alt="Logo" />
        <span class="brand-name">{{ APP_NAME }}</span>
      </div>
      <div class="header-actions">
        <ThemeToggle />
        <a class="header-link" :href="apiDocsUrl" target="_blank">API</a>
        <button class="header-link" @click="go('/setting')">
          <SettingOutlined />
        </button>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="home-main">
      <!-- Hero 区域：大标题 -->
      <section class="hero">
        <h1 class="hero-title">
          知识，<span class="hero-gradient">触手可及</span>
        </h1>
        <p class="hero-desc">
          基于知识图谱的下一代智能助手。多智能体编排，让数据不再是孤岛。
        </p>
      </section>

      <!-- Bento Grid -->
      <section class="bento-grid">
        <!-- 主卡片：对话 (2列) -->
        <div class="bento-card bento-card--hero" @click="go('/chat')">
          <div class="card-bg-icon">
            <MessageOutlined />
          </div>
          <div class="card-content">
            <div class="card-icon card-icon--orange">
              <MessageOutlined />
            </div>
            <h3 class="card-title">开始对话</h3>
            <p class="card-desc">多智能体编排 + 记忆 + RAG/图谱/工具协同。不仅是聊天，更是生产力工具。</p>
            <div class="card-cta">
              立即进入 <RightOutlined class="cta-arrow" />
            </div>
          </div>
        </div>

        <!-- 知识图谱 (1列) -->
        <div v-if="ui.show_knowledge_graph" class="bento-card" @click="go('/graph')">
          <div class="card-content">
            <div class="card-icon card-icon--blue">
              <ApartmentOutlined />
            </div>
            <h3 class="card-title">知识图谱</h3>
            <p class="card-desc">实体关系探索 + GraphRAG。看见数据之间的隐秘连接。</p>
          </div>
        </div>

        <!-- 知识库 (1列) -->
        <div v-if="ui.show_knowledge_base" class="bento-card" @click="go('/database')">
          <div class="card-content">
            <div class="card-icon card-icon--purple">
              <BookOutlined />
            </div>
            <h3 class="card-title">知识库</h3>
            <p class="card-desc">文档解析、切分与检索管理。</p>
          </div>
        </div>

        <!-- 地图 (2列) -->
        <div v-if="ui.show_map" class="bento-card bento-card--wide" @click="go('/coords')">
          <div class="card-bg-icon card-bg-icon--right">
            <EnvironmentOutlined />
          </div>
          <div class="card-content">
            <div class="card-icon card-icon--green">
              <EnvironmentOutlined />
            </div>
            <h3 class="card-title">地图探索</h3>
            <p class="card-desc">宝可梦地点与真实世界坐标映射。探索虚拟与现实的交汇。</p>
          </div>
        </div>
      </section>
    </main>

    <!-- 底部悬浮状态栏 -->
    <footer class="status-bar">
      <div class="status-pill">
        <div class="status-item">
          <span class="status-dot" :class="backendOnline ? 'online' : 'offline'"></span>
          <span>{{ backendOnline ? '系统正常' : '已断开' }}</span>
        </div>
        <div class="status-divider"></div>
        <div class="status-item">
          <ApiOutlined class="status-icon" />
          <span>v{{ configStore.config.backend?.version || '0.0.1' }}</span>
        </div>
        <template v-if="configStore.config.enable_reranker">
          <div class="status-divider"></div>
          <div class="status-item">
            <ThunderboltOutlined class="status-icon" />
            <span>Reranker</span>
          </div>
        </template>
        <template v-if="configStore.config.enable_knowledge_graph">
          <div class="status-divider"></div>
          <div class="status-item">
            <ApartmentOutlined class="status-icon" />
            <span>Graph</span>
          </div>
        </template>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  ApiOutlined,
  ApartmentOutlined,
  BookOutlined,
  EnvironmentOutlined,
  MessageOutlined,
  RightOutlined,
  SettingOutlined,
  ThunderboltOutlined
} from '@ant-design/icons-vue'

import ThemeToggle from '@/components/common/ThemeToggle.vue'
import { useConfigStore } from '@/stores/config'
import { APP_NAME } from '@/config/appMeta'

const router = useRouter()
const configStore = useConfigStore()

const ui = computed(() => configStore.config?.ui || {})
const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const apiDocsUrl = computed(() => `${window.location.origin}/api/docs`)

const go = (path: string) => router.push(path)

onMounted(async () => {
  await configStore.refreshConfig()
})
</script>

<style scoped lang="less">
/* ==================== 全局容器 ==================== */
.home-universe {
  min-height: 100vh;
  background: var(--layout-bg-color, #F7F8FA);
  position: relative;
  overflow-x: hidden;
}

/* 点阵背景 */
.dot-grid {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.06) 1px, transparent 1px),
    radial-gradient(circle, rgba(255, 125, 0, 0.03) 1px, transparent 1px);
  background-size: 24px 24px, 96px 96px;
  pointer-events: none;
  z-index: 0;
}

/* ==================== Header ==================== */
.home-header {
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 32px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-logo {
  width: 32px;
  height: 32px;
  border-radius: 8px;
}

.brand-name {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-color, #333);
  letter-spacing: -0.02em;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-link {
  font-size: 14px;
  font-weight: 500;
  color: var(--gray-600, #666);
  text-decoration: none;
  background: none;
  border: none;
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 6px;
  transition: all 0.15s ease;

  &:hover {
    color: var(--primary-color, #FF7D00);
    background: rgba(255, 125, 0, 0.08);
  }
}

/* ==================== Main ==================== */
.home-main {
  position: relative;
  z-index: 1;
  max-width: 1100px;
  margin: 0 auto;
  padding: 48px 24px 120px;
}

/* ==================== Hero ==================== */
.hero {
  text-align: center;
  margin-bottom: 56px;
}

.hero-title {
  font-size: clamp(36px, 6vw, 56px);
  font-weight: 800;
  color: var(--text-color, #1a1a1a);
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: 0 0 16px;
}

.hero-gradient {
  background: linear-gradient(135deg, #FF7D00 0%, #FF5722 50%, #E91E63 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-desc {
  font-size: 18px;
  color: var(--gray-500, #888);
  max-width: 500px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ==================== Bento Grid ==================== */
.bento-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

/* Bento 卡片基础样式 */
.bento-card {
  position: relative;
  background: #fff;
  border-radius: 24px;
  padding: 28px;
  cursor: pointer;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  border: 1px solid transparent;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(255, 125, 0, 0.12);
    border-color: rgba(255, 125, 0, 0.2);

    .card-title {
      color: var(--primary-color, #FF7D00);
    }

    .card-bg-icon {
      opacity: 0.12;
      transform: scale(1.05);
    }

    .cta-arrow {
      transform: translateX(4px);
    }
  }
}

/* 主卡片 (2列) */
.bento-card--hero {
  grid-column: span 2;
  min-height: 220px;
  background: linear-gradient(135deg, #FFF9F5 0%, #FFF 100%);
}

/* 宽卡片 (2列) */
.bento-card--wide {
  grid-column: span 2;
}

/* 背景装饰图标 */
.card-bg-icon {
  position: absolute;
  top: -20px;
  right: -20px;
  font-size: 140px;
  color: var(--primary-color, #FF7D00);
  opacity: 0.06;
  pointer-events: none;
  transition: all 0.3s ease;

  &.card-bg-icon--right {
    top: 50%;
    right: 24px;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.08;
  }
}

/* 卡片内容 */
.card-content {
  position: relative;
  z-index: 1;
}

/* 图标容器 */
.card-icon {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  margin-bottom: 16px;

  &.card-icon--orange {
    background: rgba(255, 125, 0, 0.1);
    color: #FF7D00;
  }

  &.card-icon--blue {
    background: rgba(59, 130, 246, 0.1);
    color: #3B82F6;
  }

  &.card-icon--purple {
    background: rgba(139, 92, 246, 0.1);
    color: #8B5CF6;
  }

  &.card-icon--green {
    background: rgba(34, 197, 94, 0.1);
    color: #22C55E;
  }
}

.card-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-color, #1a1a1a);
  margin: 0 0 8px;
  transition: color 0.2s ease;
}

.card-desc {
  font-size: 14px;
  color: var(--gray-500, #888);
  line-height: 1.5;
  margin: 0;
}

/* CTA 按钮 */
.card-cta {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 20px;
  font-size: 15px;
  font-weight: 600;
  color: var(--primary-color, #FF7D00);
}

.cta-arrow {
  font-size: 12px;
  transition: transform 0.2s ease;
}

/* ==================== 底部状态栏 ==================== */
.status-bar {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 20px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 100px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.05);
  font-size: 12px;
  font-weight: 500;
  color: var(--gray-600, #666);
}

.status-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #22C55E;

  &.online {
    animation: pulse 2s ease-in-out infinite;
  }

  &.offline {
    background: #EF4444;
    animation: none;
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-icon {
  font-size: 12px;
  opacity: 0.7;
}

.status-divider {
  width: 1px;
  height: 12px;
  background: var(--gray-200, #e5e5e5);
}

/* ==================== 响应式 ==================== */
@media (max-width: 900px) {
  .bento-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .bento-card--hero {
    grid-column: span 2;
  }

  .bento-card--wide {
    grid-column: span 2;
  }
}

@media (max-width: 640px) {
  .home-header {
    padding: 12px 16px;
  }

  .home-main {
    padding: 32px 16px 100px;
  }

  .hero {
    margin-bottom: 40px;
  }

  .hero-desc {
    font-size: 16px;
  }

  .bento-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .bento-card--hero,
  .bento-card--wide {
    grid-column: span 1;
  }

  .bento-card {
    padding: 24px;
  }

  .card-bg-icon {
    font-size: 100px;
  }

  .status-pill {
    padding: 6px 14px;
    gap: 8px;
    font-size: 11px;
  }
}

/* ==================== 暗色模式 ==================== */
:root[data-theme='dark'] {
  .home-universe {
    background: var(--background-color);
  }

  .dot-grid {
    background-image:
      radial-gradient(circle, rgba(255, 125, 0, 0.08) 1px, transparent 1px),
      radial-gradient(circle, rgba(255, 125, 0, 0.04) 1px, transparent 1px);
  }

  .home-header {
    background: rgba(30, 30, 30, 0.8);
    border-color: rgba(255, 255, 255, 0.05);
  }

  .bento-card {
    background: var(--surface-color);

    &:hover {
      border-color: rgba(255, 125, 0, 0.3);
    }
  }

  .bento-card--hero {
    background: linear-gradient(135deg, rgba(255, 125, 0, 0.05) 0%, var(--surface-color) 100%);
  }

  .status-pill {
    background: rgba(30, 30, 30, 0.9);
    border-color: rgba(255, 255, 255, 0.08);
  }
}
</style>
