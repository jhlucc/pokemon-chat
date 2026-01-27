<template>
  <div class="home-universe">
    <!-- 光感氛围层 Mesh Gradients -->
    <div class="ambient-glow glow--orange"></div>
    <div class="ambient-glow glow--purple"></div>
    <div class="ambient-glow glow--yellow"></div>
    <!-- 噪点纹理层 -->
    <div class="noise-overlay"></div>
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
        <!-- 主卡片：对话 (2列) - The Hero Card -->
        <div class="bento-card bento-card--hero" @click="go('/chat')">
          <!-- 有机形状装饰 -->
          <div class="organic-blob blob--1"></div>
          <div class="organic-blob blob--2"></div>
          <div class="card-content">
            <div class="card-icon card-icon--orange">
              <ThunderboltOutlined />
            </div>
            <h3 class="card-title">开始对话 <span class="title-spark">⚡</span></h3>
            <p class="card-desc">不只是聊天，是你的第二大脑。多智能体 + 长记忆，像训练宝可梦一样训练它。</p>
            <button class="card-btn">
              立即进入 <RightOutlined class="btn-arrow" />
            </button>
          </div>
        </div>

        <!-- 知识图谱 (1列) -->
        <div v-if="ui.show_knowledge_graph" class="bento-card bento-card--graph" @click="go('/graph')">
          <!-- 进化链装饰 -->
          <div class="deco-evolution"></div>
          <div class="card-content">
            <div class="card-icon card-icon--blue">
              <ApartmentOutlined />
            </div>
            <h3 class="card-title">知识图谱</h3>
            <p class="card-desc">数据的进化链。看见实体之间的隐秘连接。</p>
          </div>
          <div class="card-hover-arrow"><RightOutlined /></div>
        </div>

        <!-- 知识库 (1列) -->
        <div v-if="ui.show_knowledge_base" class="bento-card bento-card--kb" @click="go('/database')">
          <!-- 书页装饰 -->
          <div class="deco-pages"></div>
          <div class="card-content">
            <div class="card-icon card-icon--purple">
              <BookOutlined />
            </div>
            <h3 class="card-title">知识库</h3>
            <p class="card-desc">喂给 AI 的知识粮仓。把文档变成它的长期记忆。</p>
          </div>
          <div class="card-hover-arrow"><RightOutlined /></div>
        </div>

        <!-- 地图 (2列) -->
        <div v-if="ui.show_map" class="bento-card bento-card--wide bento-card--map" @click="go('/coords')">
          <!-- 等高线装饰 -->
          <div class="deco-contour"></div>
          <div class="card-content">
            <div class="card-icon card-icon--green">
              <EnvironmentOutlined />
            </div>
            <h3 class="card-title">地图探索</h3>
            <p class="card-desc">虚拟与现实的交汇点。在真实世界里寻找宝可梦的足迹。</p>
          </div>
          <div class="card-hover-arrow"><RightOutlined /></div>
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

/* 光感氛围层 - Mesh Gradients */
.ambient-glow {
  position: fixed;
  border-radius: 50%;
  filter: blur(120px);
  pointer-events: none;
  z-index: 0;
  mix-blend-mode: multiply;
  animation: glow-drift 20s ease-in-out infinite;
}

.glow--orange {
  width: 600px;
  height: 600px;
  top: -20%;
  left: -15%;
  background: radial-gradient(circle, rgba(255, 125, 0, 0.28) 0%, transparent 70%);
  animation-delay: 0s;
}

.glow--purple {
  width: 500px;
  height: 500px;
  bottom: -10%;
  right: -10%;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, transparent 70%);
  animation-delay: -7s;
}

.glow--yellow {
  width: 400px;
  height: 400px;
  top: 30%;
  left: 50%;
  transform: translateX(-50%);
  background: radial-gradient(circle, rgba(255, 214, 102, 0.2) 0%, transparent 70%);
  animation-delay: -14s;
}

@keyframes glow-drift {
  0%, 100% {
    transform: translate(0, 0) scale(1);
  }
  33% {
    transform: translate(30px, -20px) scale(1.05);
  }
  66% {
    transform: translate(-20px, 20px) scale(0.95);
  }
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

/* 噪点纹理层 - 印刷品质感 */
.noise-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 1000;
  opacity: 0.025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
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
  color: var(--gray-400, #9CA3AF);
  max-width: 520px;
  margin: 0 auto;
  line-height: 1.8;
  letter-spacing: 0.01em;
}

/* ==================== Bento Grid ==================== */
.bento-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

/* Bento 卡片基础样式 - 微磨砂质感 */
.bento-card {
  position: relative;
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 24px;
  padding: 28px;
  cursor: pointer;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.6);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
    border-color: rgba(255, 255, 255, 0.8);
    background: rgba(255, 255, 255, 0.95);

    .card-title {
      color: var(--primary-color, #FF7D00);
    }

    .card-hover-arrow {
      opacity: 1;
      transform: translateX(0);
    }

    .organic-blob {
      opacity: 0.18;
    }

    .blob--1 {
      transform: rotate(-8deg) scale(1.05);
    }

    .blob--2 {
      transform: rotate(20deg) scale(1.1);
      opacity: 0.12;
    }
  }
}

/* 主卡片 (2列) - The Hero Card */
.bento-card--hero {
  grid-column: span 2;
  min-height: 240px;
  background: linear-gradient(135deg, rgba(255, 247, 237, 0.92) 0%, rgba(255, 255, 255, 0.9) 60%);
  border: 1px solid rgba(255, 237, 213, 0.8);

  &:hover {
    box-shadow: 0 24px 60px rgba(255, 125, 0, 0.15);
    border-color: rgba(255, 200, 150, 0.6);
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.98) 0%, rgba(255, 255, 255, 0.98) 60%);

    .card-btn {
      background: #EA580C;
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(255, 125, 0, 0.4);
    }

    .btn-arrow {
      transform: translateX(3px);
    }

    .title-spark {
      animation: spark-bounce 0.6s ease;
    }
  }
}

/* 标题闪电动画 */
.title-spark {
  display: inline-block;
  margin-left: 4px;
  transition: transform 0.3s ease;
}

@keyframes spark-bounce {
  0%, 100% { transform: translateY(0) rotate(0); }
  25% { transform: translateY(-3px) rotate(-5deg); }
  50% { transform: translateY(0) rotate(5deg); }
  75% { transform: translateY(-2px) rotate(-3deg); }
}

/* 宽卡片 (2列) */
.bento-card--wide {
  grid-column: span 2;
}

/* ==================== 有机形状装饰 (Organic Blobs) ==================== */
.organic-blob {
  position: absolute;
  pointer-events: none;
  opacity: 0.12;
  transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.blob--1 {
  width: 280px;
  height: 280px;
  top: -60px;
  right: -40px;
  background: linear-gradient(135deg, #FF7D00 0%, #FFA940 100%);
  border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
  transform: rotate(-12deg);
}

.blob--2 {
  width: 120px;
  height: 120px;
  bottom: -20px;
  right: 80px;
  background: linear-gradient(135deg, #FFD666 0%, #FFA940 100%);
  border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
  transform: rotate(15deg);
  opacity: 0.08;
}

/* 进化链装饰 - 图谱卡片 */
.bento-card--graph {
  .deco-evolution {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 80px;
    height: 80px;
    opacity: 0.08;
    background:
      radial-gradient(circle at 20% 30%, #3B82F6 8px, transparent 8px),
      radial-gradient(circle at 50% 50%, #3B82F6 12px, transparent 12px),
      radial-gradient(circle at 80% 70%, #3B82F6 8px, transparent 8px);
    transform: rotate(8deg);
    transition: all 0.4s ease;

    &::before {
      content: '';
      position: absolute;
      top: 28%;
      left: 25%;
      width: 50%;
      height: 2px;
      background: linear-gradient(90deg, #3B82F6, transparent);
      transform: rotate(25deg);
    }

    &::after {
      content: '';
      position: absolute;
      top: 55%;
      left: 45%;
      width: 40%;
      height: 2px;
      background: linear-gradient(90deg, #3B82F6, transparent);
      transform: rotate(35deg);
    }
  }

  &:hover .deco-evolution {
    opacity: 0.15;
    transform: rotate(12deg) scale(1.1);
  }
}

/* 书页装饰 - 知识库卡片 */
.bento-card--kb {
  .deco-pages {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 60px;
    height: 70px;
    opacity: 0.06;
    transform: rotate(-8deg);
    transition: all 0.4s ease;

    &::before,
    &::after {
      content: '';
      position: absolute;
      border: 2px solid #8B5CF6;
      border-radius: 2px 6px 6px 2px;
    }

    &::before {
      width: 40px;
      height: 50px;
      top: 0;
      right: 0;
    }

    &::after {
      width: 40px;
      height: 50px;
      top: 8px;
      right: 8px;
      opacity: 0.6;
    }
  }

  &:hover .deco-pages {
    opacity: 0.12;
    transform: rotate(-4deg) scale(1.05);
  }
}

/* 等高线装饰 - 地图卡片 */
.bento-card--map {
  .deco-contour {
    position: absolute;
    top: 50%;
    right: 30px;
    width: 140px;
    height: 100px;
    transform: translateY(-50%) rotate(5deg);
    opacity: 0.06;
    transition: all 0.4s ease;

    &::before {
      content: '';
      position: absolute;
      inset: 0;
      border: 2px solid #22C55E;
      border-radius: 50% 30% 60% 40% / 40% 50% 30% 60%;
    }

    &::after {
      content: '';
      position: absolute;
      inset: 15px;
      border: 2px solid #22C55E;
      border-radius: 40% 60% 30% 70% / 60% 30% 70% 40%;
      opacity: 0.7;
    }
  }

  &:hover .deco-contour {
    opacity: 0.12;
    transform: translateY(-50%) rotate(8deg) scale(1.05);
  }
}

/* 卡片内容 */
.card-content {
  position: relative;
  z-index: 1;
}

/* 图标容器 - 放大 */
.card-icon {
  width: 56px;
  height: 56px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  margin-bottom: 20px;

  &.card-icon--orange {
    background: rgba(255, 125, 0, 0.12);
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

/* 实体按钮 - Hero Card CTA */
.card-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 24px;
  padding: 12px 24px;
  font-size: 15px;
  font-weight: 600;
  color: white;
  background: var(--primary-color, #FF7D00);
  border: none;
  border-radius: 100px;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(255, 125, 0, 0.3);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.btn-arrow {
  font-size: 12px;
  transition: transform 0.25s ease;
}

/* 悬停时出现的箭头 - Standard Cards */
.card-hover-arrow {
  position: absolute;
  bottom: 28px;
  right: 28px;
  font-size: 20px;
  color: var(--gray-300, #D1D5DB);
  opacity: 0;
  transform: translateX(8px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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
  padding: 10px 22px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-radius: 100px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.08),
    0 2px 8px rgba(0, 0, 0, 0.04),
    0 0 0 1px rgba(255, 255, 255, 0.5) inset;
  border: 1px solid rgba(255, 255, 255, 0.4);
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

  .blob--1 {
    width: 200px;
    height: 200px;
    right: -30px;
  }

  .blob--2 {
    width: 80px;
    height: 80px;
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

  .blob--1 {
    width: 150px;
    height: 150px;
    top: -40px;
    right: -20px;
  }

  .blob--2 {
    display: none;
  }

  .deco-evolution,
  .deco-pages,
  .deco-contour {
    display: none;
  }

  .card-icon {
    width: 48px;
    height: 48px;
    font-size: 24px;
  }

  .card-btn {
    padding: 10px 20px;
    font-size: 14px;
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

  .ambient-glow {
    mix-blend-mode: screen;
    opacity: 0.6;
  }

  .glow--orange {
    background: radial-gradient(circle, rgba(255, 125, 0, 0.2) 0%, transparent 70%);
  }

  .glow--purple {
    background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
  }

  .glow--yellow {
    background: radial-gradient(circle, rgba(255, 214, 102, 0.1) 0%, transparent 70%);
  }

  .dot-grid {
    background-image:
      radial-gradient(circle, rgba(255, 125, 0, 0.08) 1px, transparent 1px),
      radial-gradient(circle, rgba(255, 125, 0, 0.04) 1px, transparent 1px);
  }

  .noise-overlay {
    opacity: 0.03;
  }

  .home-header {
    background: rgba(30, 30, 30, 0.8);
    border-color: rgba(255, 255, 255, 0.05);
  }

  .bento-card {
    background: rgba(40, 40, 40, 0.85);
    backdrop-filter: blur(12px);
    border-color: rgba(255, 255, 255, 0.08);

    &:hover {
      border-color: rgba(255, 125, 0, 0.3);
      background: rgba(45, 45, 45, 0.95);
    }
  }

  .bento-card--hero {
    background: linear-gradient(135deg, rgba(255, 125, 0, 0.1) 0%, rgba(40, 40, 40, 0.85) 60%);
    border-color: rgba(255, 125, 0, 0.15);

    &:hover {
      background: linear-gradient(135deg, rgba(255, 125, 0, 0.15) 0%, rgba(45, 45, 45, 0.95) 60%);
    }
  }

  .organic-blob {
    opacity: 0.08;
  }

  .blob--2 {
    opacity: 0.05;
  }

  .deco-evolution,
  .deco-pages,
  .deco-contour {
    opacity: 0.05;
  }

  .card-hover-arrow {
    color: var(--gray-600, #6B7280);
  }

  .status-pill {
    background: rgba(30, 30, 30, 0.75);
    border-color: rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
  }
}
</style>
