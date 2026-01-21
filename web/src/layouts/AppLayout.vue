<script setup>
import { reactive,onMounted, computed } from 'vue'
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
  BugOutlined,
  ProjectFilled,
  ProjectOutlined,
  RedoOutlined,
  ApiOutlined,
} from '@ant-design/icons-vue'
import { themeConfig } from '@/assets/theme'
import { useConfigStore } from '@/stores/config'
import { useDatabaseStore } from '@/stores/database'
import DebugComponent from '@/components/DebugComponent.vue'

const configStore = useConfigStore()
const databaseStore = useDatabaseStore()

const layoutSettings = reactive({
  showDebug: false,
  useTopBar: false, // 是否使用顶栏
})



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

})

// 打印当前页面的路由信息，使用vue3的setup composition API
const route = useRoute()
console.log(route)

const apiDocsUrl = computed(() => {
  // Use the same-origin reverse proxy in both dev (Vite proxy) and prod (Nginx).
  return `/api/docs`
})


// 下面是导航菜单部分，添加智能体项
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
    // hidden: !configStore.config.enable_knowledge_graph,
  }, {
    name: '知识库',
    path: '/database',
    icon: BookOutlined,
    activeIcon: BookFilled,
    // hidden: !configStore.config.enable_knowledge_base,
  }, {
    name: '工具',
    path: '/tools',
    icon: ToolOutlined,
    activeIcon: ToolFilled,
  },
   {
    name: '地图',
    path: '/coords',
    icon: RedoOutlined, // 你可以换成其他图标
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
        :style="{
          right: '12px',
        }"
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
    <div class="header" :class="{ 'top-bar': layoutSettings.useTopBar }">
      <div class="logo circle">
        <router-link to="/">
          <img src="/avatar.jpg">
          <span class="logo-text">可萌</span>
        </router-link>
      </div>
      <div class="nav">
        <!-- 使用mainList渲染导航项 -->
        <RouterLink
          v-for="(item, index) in mainList"
          :key="index"
          :to="item.path"
          v-show="!item.hidden"
          class="nav-item"
          active-class="active">
          <component class="icon" :is="route.path.startsWith(item.path) ? item.activeIcon : item.icon" />
          <span class="text">{{item.name}}</span>
        </RouterLink>
      </div>
      <div class="fill" style="flex-grow: 1;"></div>

      <div class="nav-item api-docs">
        <a-tooltip placement="right">
          <template #title>接口文档 {{ apiDocsUrl }}</template>
          <a :href="apiDocsUrl" target="_blank" class="github-link">
            <ApiOutlined class="icon" style="color: #222;"/>
          </a>
        </a-tooltip>
      </div>
      <RouterLink class="nav-item setting" to="/setting" active-class="active">
        <a-tooltip placement="right">
          <template #title>设置</template>
          <component class="icon" :is="route.path === '/setting' ? SettingFilled : SettingOutlined" />
        </a-tooltip>
      </RouterLink>
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

/* Glassy/Premium Sidebar */
.header {
  display: flex;
  flex-direction: column;
  flex: 0 0 80px; /* Slightly wider sidebar */
  justify-content: flex-start;
  align-items: center;
  background-color: var(--sidebar-background-color);
  height: 100%;
  border-right: 1px solid var(--border-color);
  z-index: 10;
  transition: all 0.3s ease;
  box-shadow: 1px 0 4px rgba(0, 0, 0, 0.02);

  .logo {
    width: 48px;
    height: 48px;
    margin: 24px 0;
    transition: transform 0.3s ease;

    &:hover {
      transform: scale(1.05);
    }

    img {
      width: 100%;
      height: 100%;
      border-radius: 12px; /* Soft rounding */
      box-shadow: var(--shadow-sm);
    }

    .logo-text {
      display: none;
    }

    & > a {
      text-decoration: none;
      font-size: 24px;
      font-weight: bold;
      color: var(--text-color);
    }
  }

  .nav {
    display: flex;
    flex-direction: column;
    justify-content: flex-start; /* Changed from space-between to avoid huge gaps */
    align-items: center;
    width: 100%;
    gap: 16px; /* consistent gap */

    /* Nav Item Styling */
    .nav-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 56px;
      height: 56px;
      border-radius: 12px;
      color: var(--subtext-color);
      background-color: transparent;
      transition: all 0.2s ease;
      text-decoration: none;
      cursor: pointer;
      position: relative;

      &.active {
        color: #FFFFFF;
        background-color: var(--primary-color);
        box-shadow: var(--shadow-md);
        
        .text {
            color: rgba(255,255,255,0.9);
        }
      }

      &:hover:not(.active) {
        background-color: var(--gray-100);
        color: var(--primary-color);
      }

      .icon {
        font-size: 22px;
      }

      .text {
        font-size: 11px;
        margin-top: 4px;
        font-weight: 500;
        transition: color 0.2s ease;
      }
    }
  }
  
  /* Bottom actions */
  .api-docs, .setting {
     margin-top: auto; /* Push to bottom */
     margin-bottom: 12px;
  }
}

#app-router-view {
  flex: 1 1 auto;
  height: 100%;
  max-width: 100%;
  overflow-y: auto;
  background-color: var(--background-color);
  /* Smooth scroll behavior */
  scroll-behavior: smooth;
}

/* Mobile Responsiveness */
@media (max-width: 520px) {
  .app-layout {
    flex-direction: column-reverse;

    div.header {
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
    flex: 0 0 64px;
    height: 64px;
    background-color: var(--sidebar-background-color);
    border-top: 1px solid var(--border-color);
    z-index: 20;

    .nav-item {
      text-decoration: none;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      width: 48px;
      color: var(--subtext-color);
      font-size: 12px;
      font-weight: 500;
      transition: color 0.2s ease;

      &.active {
        color: var(--primary-color);
        font-weight: 600;
      }
    }
  }
}

/* Top Bar Mode (Optional) */
.app-layout.use-top-bar {
  flex-direction: column;
}

.header.top-bar {
  flex-direction: row;
  flex: 0 0 64px;
  width: 100%;
  height: 64px;
  border-right: none;
  border-bottom: 1px solid var(--border-color);
  padding: 0 24px;
  background-color: var(--sidebar-background-color);

  .logo {
    margin: 0 24px 0 0;
    width: auto;
    
    img {
        width: 32px;
        height: 32px;
    }
    .logo-text {
        display: block;
        margin-left: 12px;
        font-size: 18px;
        color: var(--text-color);
    }
  }
  
  .nav {
      flex-direction: row;
      width: auto;
      gap: 8px;
      
      .nav-item {
          flex-direction: row;
          width: auto;
          height: 40px;
          padding: 0 16px;
          gap: 8px;
          
          .text {
              margin-top: 0;
              font-size: 14px;
          }
          
          .icon {
              font-size: 18px;
          }
      }
  }
  
  .fill {
      display: flex;
      flex-grow: 1;
  }
}
</style>
