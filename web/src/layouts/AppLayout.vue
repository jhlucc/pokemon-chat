<script setup>
import { reactive, onMounted, computed, ref } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import {
  MessageOutlined,
  MessageFilled,
  SettingOutlined,
  SettingFilled,
  BookOutlined,
  BookFilled,
  ToolFilled,
  ToolOutlined,
  ProjectFilled,
  ProjectOutlined,
  RedoOutlined,
  BugOutlined,
} from '@ant-design/icons-vue'
import { themeConfig, setTheme } from '@/assets/theme'
import { useConfigStore } from '@/stores/config'
import { useDatabaseStore } from '@/stores/database'
import DebugComponent from '@/components/DebugComponent.vue'

const configStore = useConfigStore()
const databaseStore = useDatabaseStore()

const layoutSettings = reactive({
  showDebug: false,
  useTopBar: false,
})

const isDark = ref(false)

const toggleTheme = () => {
  isDark.value = !isDark.value
  setTheme(isDark.value ? 'dark' : 'light')
}

const getRemoteConfig = () => {
  configStore.refreshConfig()
}

const getRemoteDatabase = () => {
  if (!configStore.config.enable_knowledge_base) {
    return
  }
  databaseStore.refreshDatabase()
}

onMounted(() => {
  getRemoteConfig()
  getRemoteDatabase()
  const savedTheme = localStorage.getItem('theme') || 'light'
  isDark.value = savedTheme === 'dark'
  setTheme(savedTheme)
})

const route = useRoute()

// Navigation List
const mainList = [{
    name: '对话',
    path: '/chat',
    icon: MessageOutlined,
    activeIcon: MessageFilled,
  }, {
    name: '图谱',
    path: '/graph',
    icon: ProjectOutlined,
    activeIcon: ProjectFilled,
  }, {
    name: '知识库',
    path: '/database',
    icon: BookOutlined,
    activeIcon: BookFilled,
  }, {
    name: '工具',
    path: '/tools',
    icon: ToolOutlined,
    activeIcon: ToolFilled,
  },
   {
    name: '地图',
    path: '/coords',
    icon: RedoOutlined,
    activeIcon: RedoOutlined,
  }
]
</script>

<template>
  <div class="app-layout" :class="{ 'use-top-bar': layoutSettings.useTopBar }">
    <div class="debug-panel" >
      <a-float-button
        @click="layoutSettings.showDebug = !layoutSettings.showDebug"
        tooltip="调试面板"
        :style="{ right: '12px' }"
      >
        <template #icon>
          <BugOutlined />
        </template>
      </a-float-button>
      <a-drawer
        v-model:open="layoutSettings.showDebug"
        title="调试面板"
        width="800"
        :contentWrapperStyle="{ maxWidth: '100%'}"
        placement="right"
      >
        <DebugComponent />
      </a-drawer>
    </div>

    <!-- Sidebar -->
    <div class="sidebar" :class="{ 'top-bar': layoutSettings.useTopBar }">
      <!-- Top Action: New Chat -->
      <div class="sidebar-top">
         <RouterLink to="/chat" class="new-chat-btn" active-class="active-btn" title="New Chat" @click="$emit('newconv')">
             <MessageOutlined class="icon" />
         </RouterLink>
      </div>

      <!-- Main Navigation -->
      <div class="nav">
        <RouterLink
          v-for="(item, index) in mainList"
          :key="index"
          :to="item.path"
          v-show="!item.hidden"
          class="nav-item"
          active-class="active">
          <component class="icon" :is="route.path.startsWith(item.path) ? item.activeIcon : item.icon" />
          <div class="tooltip">{{ item.name }}</div>
        </RouterLink>
      </div>
      
      <div class="fill" style="flex-grow: 1;"></div>

      <!-- Bottom Actions -->
      <div class="sidebar-bottom">
        <!-- Theme Toggle -->
        <div class="nav-item" @click="toggleTheme" title="Toggle Theme">
             <div class="icon" style="font-size: 20px;">
                {{ isDark ? '🌙' : '☀️' }}
             </div>
             <div class="tooltip">{{ isDark ? 'Dark Mode' : 'Light Mode' }}</div>
        </div>

        <RouterLink class="nav-item round" to="/setting" active-class="active">
           <component class="icon" :is="route.path === '/setting' ? SettingFilled : SettingOutlined" />
        </RouterLink>
        <div class="user-avatar circle">
            <img src="/avatar.jpg" alt="User">
        </div>
      </div>
    </div>
    
    <div class="header-mobile">
      <RouterLink to="/chat" class="nav-item" active-class="active">对话</RouterLink>
      <RouterLink to="/database" class="nav-item" active-class="active">知识</RouterLink>
      <RouterLink to="/setting" class="nav-item" active-class="active">设置</RouterLink>
    </div>
    <a-config-provider :theme="themeConfig">
    <router-view v-slot="{ Component, route }" id="app-router-view">
      <keep-alive v-if="route.meta.keepAlive !== false">
        <component :is="Component" />
      </keep-alive>
      <component :is="Component" v-else />
    </router-view>
    </a-config-provider>
  </div>
</template>

<style lang="less" scoped>
@import '@/assets/main.css';

.app-layout {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: 100vh;
  min-width: var(--min-width);
  background-color: var(--background-color);

  .header-mobile {
    display: none;
  }

  .debug-panel {
    position: absolute;
    z-index: 100;
    right: 0;
    bottom: 50px;
    border-radius: 20px 0 0 20px;
    cursor: pointer;
  }
}

/* Sidebar Styling */
.sidebar {
  display: flex;
  flex-direction: column;
  flex: 0 0 72px;
  align-items: center;
  background-color: var(--surface-overlay); /* Variable for theme support */
  height: 100%;
  padding: 16px 0 24px 0;
  z-index: 10;
  border-right: 1px solid var(--border-color);

  .sidebar-top {
      margin-bottom: 24px;
      
      .new-chat-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 44px;
          height: 44px;
          border-radius: 12px;
          color: var(--text-color);
          background-color: var(--surface-card);
          box-shadow: var(--shadow-sm);
          transition: all 0.2s ease;
          border: 1px solid var(--border-color);
          
          &:hover {
              background-color: var(--primary-color);
              color: white;
              box-shadow: var(--shadow-md);
              transform: translateY(-1px);
          }
           &.active-btn {
              background-color: var(--gray-200);
          }
          
          .icon {
              font-size: 20px;
          }
      }
  }

  .nav {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    width: 100%;

    .nav-item {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      border-radius: 10px;
      color: var(--subtext-color);
      transition: all 0.2s ease;
      text-decoration: none;
      position: relative; 
      cursor: pointer;
      
      &:hover {
        background-color: var(--gray-100);
        color: var(--text-color);
        
        .tooltip {
            opacity: 1;
            transform: translateX(0);
        }
      }

      &.active {
        background-color: var(--gray-200);
        color: var(--text-color);
        
        .icon {
            transform: scale(1.05);
        }
      }

      .icon {
        font-size: 22px;
        transition: transform 0.2s ease;
      }
      
      /* Simple Tooltip on hover */
      .tooltip {
          position: absolute;
          left: 120%;
          background: rgba(0,0,0,0.8);
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          white-space: nowrap;
          pointer-events: none;
          opacity: 0;
          transform: translateX(-5px);
          transition: all 0.2s ease;
          z-index: 100;
      }
    }
  }
  
  .sidebar-bottom {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      
      .nav-item {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 40px;
          height: 40px;
          border-radius: 10px;
          color: var(--subtext-color);
          transition: all 0.2s ease;
          text-decoration: none;
          position: relative; 
          cursor: pointer;
          
          &:hover {
            background-color: var(--gray-100);
            color: var(--text-color);
            
            .tooltip {
                opacity: 1;
                transform: translateX(0);
            }
          }

          .icon {
            font-size: 22px;
            transition: transform 0.2s ease;
          }
          
          .tooltip {
              position: absolute;
              left: 120%;
              background: rgba(0,0,0,0.8);
              color: white;
              padding: 4px 8px;
              border-radius: 4px;
              font-size: 12px;
              white-space: nowrap;
              pointer-events: none;
              opacity: 0;
              transform: translateX(-5px);
              transition: all 0.2s ease;
              z-index: 100;
          }
      }
      
      .user-avatar {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          overflow: hidden;
          cursor: pointer;
          transition: transform 0.2s;
          border: 2px solid var(--border-color);
          
          &:hover {
              transform: scale(1.05);
          }
          
          img {
              width: 100%;
              height: 100%;
              object-fit: cover;
          }
      }
  }
}

#app-router-view {
  flex: 1 1 auto;
  height: 100%;
  max-width: 100%;
  overflow-y: auto;
  background-color: var(--background-color);
  scroll-behavior: smooth;
}

/* Mobile Responsiveness */
@media (max-width: 520px) {
  .app-layout {
    flex-direction: column-reverse;

    .sidebar {
      display: none;
    }

    .debug-panel {
      bottom: 10rem;
    }
  }

  .app-layout div.header-mobile {
    display: flex;
    flex-direction: row;
    width: 100%;
    padding: 0 20px;
    justify-content: space-around;
    align-items: center;
    flex: 0 0 72px; 
    border-top: 1px solid var(--border-color);
    z-index: 20;
    box-shadow: var(--shadow-sm);
    background-color: var(--surface-overlay);

    .nav-item {
      text-decoration: none;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 64px;
      height: 100%;
      color: var(--subtext-color);
      font-size: 12px;
      font-weight: 500;
      
      &.active {
        color: var(--primary-color);
        font-weight: 600;
      }
    }
  }
}

/* Top Bar Mode (Simplified) */
.app-layout.use-top-bar {
  flex-direction: column;
}

.sidebar.top-bar {
  flex-direction: row;
  flex: 0 0 64px;
  width: 100%;
  height: 64px;
  border-right: none;
  border-bottom: 1px solid var(--border-color);
  padding: 0 24px;
  background-color: var(--surface-overlay);

  .sidebar-top { margin: 0 16px 0 0; }
  .nav { flex-direction: row; gap: 8px; width: auto; }
}
</style>
