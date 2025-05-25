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
  // return `${import.meta.env.VITE_API_URL || `http://${window.location.hostname}:${window.location.port}`}/docs`
  return `http://localhost:5050/docs`
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

div.header, #app-router-view {
  height: 100%;
  max-width: 100%;
  user-select: none;
}

#app-router-view {
  flex: 1 1 auto;
  overflow-y: auto;
}

.header {
  display: flex;
  flex-direction: column;
  flex: 0 0 70px;
  justify-content: flex-start;
  align-items: center;
  background-color: var(--gray-100);
  height: 100%;
  width: 74px;
  border-right: 1px solid var(--gray-300);

  .logo {
    width: 40px;
    height: 40px;
    margin: 18px 0 18px 0;

    img {
      width: 100%;
      height: 100%;
      border-radius: 4px;  // 50% for circle
    }

    .logo-text {
      display: none;
    }

    & > a {
      text-decoration: none;
      font-size: 24px;
      font-weight: bold;
      color: #333;
    }
  }

  .nav-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 52px;
    padding: 4px;
    padding-top: 10px;
    border: 1px solid transparent;
    border-radius: 8px;
    background-color: transparent;
    color: #222;
    font-size: 20px;
    transition: background-color 0.2s ease-in-out;
    margin: 0 10px;
    text-decoration: none;
    cursor: pointer;

    &.github {
      padding: 10px 12px;
      &:hover {
        background-color: transparent;
        border: 1px solid transparent;
      }

      .github-link {
        display: flex;
        flex-direction: column;
        align-items: center;
        color: inherit;
      }

      .github-stars {
        display: flex;
        align-items: center;
        font-size: 12px;
        margin-top: 4px;

        .star-icon {
          color: #f0a742;
          font-size: 12px;
          margin-right: 2px;
        }

        .star-count {
          font-weight: 600;
        }
      }
    }

    &.api-docs {
      padding: 10px 12px;
    }

    &.setting {
      padding: 16px 12px;
      width: 56px;
    }

    &.active {
      font-weight: bold;
      color: var(--main-600);
      background-color: white;
      border: 1px solid white;
    }

    &.warning {
      color: red;
    }

    &:hover {
      background-color: rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(10px);
    }

    .text {
      font-size: 12px;
      margin-top: 4px;
      text-align: center;
    }
  }

  .setting {
    width: auto;
    font-size: 20px;
    color: #333;
    margin-bottom: 20px;
    margin-top: 10px;

    &:hover {
      cursor: pointer;
    }
  }
}

.header .nav {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  position: relative;
  height: 45px;
  gap: 16px;
}

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
    flex: 0 0 60px;
    border-right: none;
    height: 40px;

    .nav-item {
      text-decoration: none;
      width: 40px;
      color: var(--gray-900);
      font-size: 1rem;
      font-weight: bold;
      transition: color 0.1s ease-in-out, font-size 0.1s ease-in-out;

      &.active {
        color: black;
        font-size: 1.1rem;
      }
    }
  }
  .app-layout .chat-box::webkit-scrollbar {
    width: 0;
  }
}

.app-layout.use-top-bar {
  flex-direction: column;
}

.header.top-bar {
  flex-direction: row;
  flex: 0 0 50px;
  width: 100%;
  height: 50px;
  border-right: none;
  border-bottom: 1px solid var(--main-light-2);
  background-color: var(--main-light-3);
  padding: 0 20px;
  gap: 24px;

  .logo {
    width: fit-content;
    height: 28px;
    margin-right: 16px;
    display: flex;
    align-items: center;

    a {
      display: flex;
      align-items: center;
      text-decoration: none;
      color: inherit;
    }

    img {
      width: 28px;
      height: 28px;
      margin-right: 8px;
    }

    .logo-text {
      display: block;
      font-size: 16px;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: var(--main-600);
      white-space: nowrap;
    }
  }

  .nav {
    flex-direction: row;
    height: auto;
    gap: 20px;
  }

  .nav-item {
    flex-direction: row;
    width: auto;
    padding: 4px 16px;
    margin: 0;

    .icon {
      margin-right: 8px;
      font-size: 15px; // 减小图标大小
    }

    .text {
      margin-top: 0;
      font-size: 15px;
    }

    &.github, &.setting {
      padding: 8px 12px;

      .icon {
        margin-right: 0;
        font-size: 18px;
      }

      &.active {
        color: var(--main-600);
      }
    }

    &.github {
      a {
        display: flex;
        align-items: center;
      }

      .github-stars {
        display: flex;
        align-items: center;
        margin-left: 6px;

        .star-icon {
          color: #f0a742;
          font-size: 14px;
          margin-right: 2px;
        }
      }
    }
  }
}
</style>