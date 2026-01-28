<template>
  <div class="welcome-hero animate-fade-in">
    <!-- Hero Section -->
    <div class="hero-section">
      <div class="hero-avatar">
        <img :src="avatarSrc" alt="可萌" class="avatar-image animate-float" loading="lazy" />
      </div>

      <div class="hero-content">
        <h1 class="hero-title">
          <span class="wave">👋</span> 你好，我是<span class="brand-name">可萌</span>
        </h1>
        <p class="hero-subtitle">基于宝可梦知识图谱的智能助手</p>
      </div>
    </div>


    <!-- Quick Actions Grid -->
    <div class="quick-actions">
      <div class="actions-label">试试这些问题</div>
      <div class="actions-grid">
        <button
          v-for="action in quickActions"
          :key="action.id"
          type="button"
          class="action-card"
          @click="$emit('select', action.prompt)"
        >
          <span class="action-icon">{{ action.icon }}</span>
          <div class="action-content">
            <span class="action-label">{{ action.label }}</span>
            <span class="action-desc">{{ action.desc }}</span>
          </div>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  SearchOutlined,
  ThunderboltOutlined,
  ApartmentOutlined,
  MessageOutlined
} from '@ant-design/icons-vue'

const props = defineProps<{
  avatarSrc?: string
}>()

defineEmits<{
  (e: 'select', prompt: string): void
}>()

const avatarSrc = computed(() => props.avatarSrc || '/user.png')

const quickActions = [
  {
    id: 'query',
    icon: '🔍',
    label: '查询属性',
    desc: '皮卡丘的属性是什么？',
    prompt: '皮卡丘的属性是什么？'
  },
  {
    id: 'battle',
    icon: '⚔️',
    label: '对战分析',
    desc: '水系克制什么属性？',
    prompt: '水系克制什么属性？'
  },
  {
    id: 'evolution',
    icon: '🔄',
    label: '进化信息',
    desc: '小火龙的进化链',
    prompt: '小火龙的进化链是什么？'
  },
  {
    id: 'explore',
    icon: '🗺️',
    label: '地区探索',
    desc: '介绍一下关东地区',
    prompt: '介绍一下关东地区'
  },
  {
    id: 'character',
    icon: '👤',
    label: '角色故事',
    desc: '小智的冒险经历',
    prompt: '介绍一下小智'
  },
  {
    id: 'chat',
    icon: '💬',
    label: '随便聊聊',
    desc: '你喜欢哪只宝可梦？',
    prompt: '你喜欢哪只宝可梦？'
  }
]
</script>

<style lang="less" scoped>
.welcome-hero {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  padding: var(--space-6) var(--space-4);
  gap: var(--space-6);
  overflow: hidden;
}

/* Hero Section */
.hero-section {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: var(--space-4);
}

.hero-avatar {
  position: relative;
  width: 80px;
  height: 80px;

  .avatar-image {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid var(--surface-color);
    box-shadow: var(--shadow-md);
  }
}

.hero-content {
  .hero-title {
    font-size: clamp(1.75rem, 5vw, 2.5rem);
    font-weight: 800;
    color: var(--text-color);
    margin: 0 0 var(--space-2) 0;
    letter-spacing: -0.02em;

    .wave {
      display: inline-block;
      animation: wave 2.5s ease-in-out infinite;
      transform-origin: 70% 70%;
    }

    .brand-name {
      background: linear-gradient(135deg, #ff7d00 0%, #ff5350 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
  }

  .hero-subtitle {
    font-size: var(--font-size-base);
    color: var(--gray-500);
    margin: 0;
  }
}

@keyframes wave {
  0%, 100% { transform: rotate(0deg); }
  10%, 30% { transform: rotate(14deg); }
  20% { transform: rotate(-8deg); }
  40% { transform: rotate(-4deg); }
  50%, 100% { transform: rotate(0deg); }
}

/* Quick Actions */
.quick-actions {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 640px;
}

.actions-label {
  font-size: var(--font-size-sm);
  color: var(--gray-500);
  margin-bottom: var(--space-3);
  text-align: center;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-3);
}

.action-card {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  /* Glassmorphism Style */
  background: color-mix(in srgb, var(--surface-color) 70%, transparent);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid color-mix(in srgb, var(--border-color) 60%, transparent);
  border-radius: var(--radius-md);
  cursor: pointer;
  text-align: left;
  transition: all var(--duration-fast) var(--ease-default);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);

  &:hover {
    background: var(--surface-color);
    border-color: color-mix(in srgb, var(--primary-color) 40%, var(--border-color));
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(255, 125, 0, 0.12);

    .action-icon {
      transform: scale(1.15);
      background: color-mix(in srgb, var(--primary-color) 18%, transparent);
    }

    .action-label {
      color: var(--primary-color);
    }
  }

  &:active {
    transform: translateY(-1px);
  }

  .action-icon {
    font-size: 18px;
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: color-mix(in srgb, var(--primary-color) 10%, transparent);
    border-radius: 50%;
    transition: transform var(--duration-fast) var(--ease-default);
  }

  .action-content {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .action-label {
    font-size: var(--font-size-sm);
    font-weight: 600;
    color: var(--text-color);
  }

  .action-desc {
    font-size: var(--font-size-xs);
    color: var(--gray-500);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
}

/* Responsive */
@media (max-width: 640px) {
  .welcome-hero {
    min-height: 50vh;
    padding: var(--space-4);
    gap: var(--space-4);
  }

  .hero-avatar {
    width: 64px;
    height: 64px;
  }

  .hero-content .hero-title {
    font-size: var(--font-size-xl);
  }

  .capabilities-card {
    padding: var(--space-3);
  }

  .actions-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 400px) {
  .actions-grid {
    grid-template-columns: 1fr;
  }

  .action-card {
    flex-direction: row;
    align-items: center;
  }
}
</style>
