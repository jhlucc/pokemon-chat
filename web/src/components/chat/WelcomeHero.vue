<template>
  <div class="welcome-hero animate-fade-in">
    <!-- Hero Section -->
    <div class="hero-section">
      <div class="hero-avatar">
        <img :src="avatarSrc" alt="可萌" class="avatar-image animate-float" loading="lazy" />
      </div>

      <div class="hero-content">
        <h1 class="hero-title">
          <span class="wave">👋</span> 你好，我是可萌
        </h1>
        <p class="hero-subtitle">基于宝可梦知识图谱的智能助手</p>
      </div>
    </div>

    <!-- Capabilities Card -->
    <div class="capabilities-card">
      <div class="capabilities-header">
        <BulbOutlined class="capabilities-icon" />
        <span>我可以帮你</span>
      </div>
      <ul class="capabilities-list">
        <li v-for="(cap, index) in capabilities" :key="index">
          <component :is="cap.icon" class="cap-icon" />
          <span>{{ cap.text }}</span>
        </li>
      </ul>
      <div class="disclaimer">
        <InfoCircleOutlined />
        <span>回复基于知识库生成，可能存在不准确之处，请注意甄别</span>
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
  BulbOutlined,
  InfoCircleOutlined,
  SearchOutlined,
  ThunderboltOutlined,
  ApartmentOutlined,
  MessageOutlined,
  BookOutlined,
  CompassOutlined
} from '@ant-design/icons-vue'

const props = defineProps<{
  avatarSrc?: string
}>()

defineEmits<{
  (e: 'select', prompt: string): void
}>()

const avatarSrc = computed(() => props.avatarSrc || '/user.png')

const capabilities = [
  { icon: SearchOutlined, text: '查询宝可梦属性、技能、进化链' },
  { icon: ThunderboltOutlined, text: '分析对战数据和战力评估' },
  { icon: ApartmentOutlined, text: '探索角色关系和故事背景' },
  { icon: MessageOutlined, text: '回答宝可梦相关的各种问题' }
]

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
    font-size: var(--font-size-3xl);
    font-weight: 700;
    color: var(--text-color);
    margin: 0 0 var(--space-2) 0;

    .wave {
      display: inline-block;
      animation: wave 2.5s ease-in-out infinite;
      transform-origin: 70% 70%;
    }
  }

  .hero-subtitle {
    font-size: var(--font-size-md);
    color: var(--gray-600);
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

/* Capabilities Card */
.capabilities-card {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 480px;
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: var(--space-4);
  box-shadow: var(--shadow-xs);
}

.capabilities-header {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--font-size-base);
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: var(--space-3);

  .capabilities-icon {
    color: var(--primary-color);
  }
}

.capabilities-list {
  list-style: none;
  padding: 0;
  margin: 0 0 var(--space-3) 0;

  li {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) 0;
    font-size: var(--font-size-sm);
    color: var(--gray-700);

    .cap-icon {
      color: var(--gray-500);
      font-size: 14px;
    }
  }
}

.disclaimer {
  display: flex;
  align-items: flex-start;
  gap: var(--space-2);
  padding: var(--space-3);
  background: var(--gray-50);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-xs);
  color: var(--gray-600);
  line-height: var(--line-height-base);

  :deep(.anticon) {
    flex-shrink: 0;
    margin-top: 2px;
  }
}

:root[data-theme='dark'] .disclaimer {
  background: var(--gray-100);
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
  background: var(--surface-color);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  cursor: pointer;
  text-align: left;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--surface-color-2);
    border-color: color-mix(in srgb, var(--primary-color) 30%, var(--border-color));
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);

    .action-icon {
      transform: scale(1.1);
    }
  }

  &:active {
    transform: translateY(0);
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
