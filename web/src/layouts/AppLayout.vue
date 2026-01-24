<script setup>
import { reactive, onMounted, onUnmounted, computed, ref, watch } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { usePreferredDark } from '@vueuse/core'
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
  BulbFilled,
  DesktopOutlined,
  RobotOutlined,
  RobotFilled,
} from '@ant-design/icons-vue'
import { theme as antdTheme } from 'ant-design-vue'
import { applyTheme, getSavedThemeMode, setThemeMode, themeConfig } from '@/assets/theme'
import { useConfigStore } from '@/stores/config'
import { useDatabaseStore } from '@/stores/database'
import DebugComponent from '@/components/DebugComponent.vue'
import { getOfflineMode } from '@/utils/offlineMode'

const configStore = useConfigStore()
const databaseStore = useDatabaseStore()
const preferredDark = usePreferredDark()

const offlineMode = ref(getOfflineMode())
const onOfflineModeChanged = () => {
  offlineMode.value = getOfflineMode()
}

const layoutSettings = reactive({
  showDebug: false,
  useTopBar: false, // 是否使用顶栏
})

// Theme mode: only "system" | "dark" (stored in localStorage)
const themeMode = ref(getSavedThemeMode())
const appliedIsDark = computed(() => themeMode.value === 'dark' || (themeMode.value === 'system' && preferredDark.value))
const themeLabel = computed(() => (themeMode.value === 'dark' ? '暗色主题' : '跟随系统'))
const appliedLabel = computed(() => (appliedIsDark.value ? '暗色' : '亮色'))

watch(
  () => themeMode.value,
  (m) => {
    setThemeMode(m)
  },
  { immediate: true }
)
watch(
  () => preferredDark.value,
  () => {
    if (themeMode.value === 'system') {
      applyTheme(preferredDark.value ? 'dark' : 'light')
    }
  }
)

const toggleThemeMode = () => {
  themeMode.value = themeMode.value === 'dark' ? 'system' : 'dark'
}

const onThemeToggleKeydown = (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault()
    toggleThemeMode()
  }
}

const antdThemeConfig = computed(() => ({
  ...themeConfig,
  ...(appliedIsDark.value ? { algorithm: antdTheme.darkAlgorithm } : {}),
}))



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

  window.addEventListener('offline-mode-changed', onOfflineModeChanged)
})

onUnmounted(() => {
  window.removeEventListener('offline-mode-changed', onOfflineModeChanged)
})

// 打印当前页面的路由信息，使用vue3的setup composition API
const route = useRoute()

const apiDocsUrl = computed(() => {
  // Works both in dev (Vite proxy) and prod (Nginx /api reverse proxy).
  return `${window.location.origin}/api/docs`
})

const backendMode = computed(() => {
  const backend = configStore.config?.backend || {}
  const isMock = Boolean(backend.mock) || offlineMode.value === 'on'
  const isOnline = Boolean(backend.online)
  if (isMock) return { key: 'mock', short: 'MOCK', label: 'Mock（离线演示）' }
  if (isOnline) return { key: 'online', short: 'API', label: 'Backend Online' }
  return { key: 'offline', short: 'OFF', label: 'Backend Offline' }
})


// 下面是导航菜单部分，添加智能体项
const mainList = computed(() => {
  const ui = configStore.config?.ui || {}
  return [
    {
      name: '对话',
      path: '/chat',
      icon: MessageOutlined,
      activeIcon: MessageFilled,
    },
    {
      name: '图谱',
      path: '/graph',
      icon: ProjectOutlined,
      activeIcon: ProjectFilled,
      hidden: ui.show_knowledge_graph === false,
    },
    {
      name: '知识库',
      path: '/database',
      icon: BookOutlined,
      activeIcon: BookFilled,
      hidden: ui.show_knowledge_base === false,
    },
    {
      name: '工具',
      path: '/tools',
      icon: ToolOutlined,
      activeIcon: ToolFilled,
      hidden: ui.show_tools === false,
    },
    {
      name: '智能体',
      path: '/agent',
      icon: RobotOutlined,
      activeIcon: RobotFilled,
      hidden: ui.show_agents === false,
    },
    {
      name: '地图',
      path: '/coords',
      icon: RedoOutlined, // 你可以换成其他图标
      activeIcon: RedoOutlined,
      hidden: ui.show_map === false,
    },
  ]
})
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
	          <img src="/avatar.jpg" alt="Logo">
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

      <div class="nav-item mode-indicator" aria-label="运行模式">
        <a-tooltip placement="right">
          <template #title>
            <div>{{ backendMode.label }}</div>
            <div>离线模式：{{ offlineMode }}</div>
          </template>
          <span class="mode-pill" :class="backendMode.key">{{ backendMode.short }}</span>
        </a-tooltip>
      </div>

	      <div class="nav-item api-docs">
	        <a-tooltip placement="right">
	          <template #title>接口文档 {{ apiDocsUrl }}</template>
	          <a :href="apiDocsUrl" target="_blank" rel="noopener noreferrer" aria-label="接口文档" class="github-link">
	            <ApiOutlined class="icon" />
	          </a>
	        </a-tooltip>
	      </div>
      <div
        class="nav-item theme-toggle"
        @click="toggleThemeMode"
        @keydown="onThemeToggleKeydown"
        role="button"
        tabindex="0"
        aria-label="切换主题"
      >
        <a-tooltip placement="right">
          <template #title>主题：{{ themeLabel }}（当前：{{ appliedLabel }}）</template>
          <component
            class="icon"
            :is="themeMode === 'dark' ? BulbFilled : DesktopOutlined"
            aria-hidden="true"
          />
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
	      <RouterLink to="/chat" class="nav-item" active-class="active" aria-label="对话">
	        <MessageOutlined class="icon" aria-hidden="true" />
	        <span class="label">对话</span>
	      </RouterLink>
	      <RouterLink
	        v-if="configStore.config?.ui?.show_knowledge_base !== false"
	        to="/database"
	        class="nav-item"
	        active-class="active"
	        aria-label="知识库"
	      >
	        <BookOutlined class="icon" aria-hidden="true" />
	        <span class="label">知识</span>
	      </RouterLink>
	      <RouterLink to="/setting" class="nav-item" active-class="active" aria-label="设置">
	        <SettingOutlined class="icon" aria-hidden="true" />
	        <span class="label">设置</span>
	        <span class="mode-badge" :class="backendMode.key" aria-hidden="true">{{ backendMode.short }}</span>
	      </RouterLink>
	    </div>
    <a-config-provider :theme="antdThemeConfig">
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
  background-color: var(--bg-sider);
  height: 100%;
  width: 74px;
  border-right: 1px solid var(--border-color);

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
      color: var(--text-color);
    }
  }

  .nav-item {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 52px;
    padding: 4px;
    padding-top: 10px;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    background-color: transparent;
    color: var(--text-color);
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
          color: var(--warning-color);
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

	    &.mode-indicator {
	      padding: 10px 0;
	    }

	    &.setting {
	      padding: 16px 12px;
	      width: 56px;
	    }

    &.active {
      font-weight: bold;
      color: var(--main-600);
      background-color: var(--surface-color);
      border: 1px solid var(--border-color);

      &::before {
        content: '';
        position: absolute;
        left: -10px;
        top: 50%;
        width: 3px;
        height: 22px;
        border-radius: 3px;
        background: var(--main-500);
        transform: translateY(-50%);
      }
    }

    &.warning {
      color: red;
    }

    &:hover {
      background-color: var(--hover-bg);
    }

	    .text {
	      font-size: 12px;
	      margin-top: 4px;
	      text-align: center;
	    }

	    .mode-pill {
	      display: inline-flex;
	      align-items: center;
	      justify-content: center;
	      min-width: 44px;
	      padding: 4px 8px;
	      border-radius: 999px;
	      border: 1px solid var(--border-color);
	      background: var(--surface-color);
	      color: var(--gray-700);
	      font-size: 11px;
	      font-weight: 700;
	      letter-spacing: 0.4px;
	      user-select: none;
	    }

	    .mode-pill.mock {
	      color: var(--main-600);
	      border-color: color-mix(in srgb, var(--main-500) 35%, var(--border-color));
	      background: color-mix(in srgb, var(--main-500) 14%, var(--surface-color));
	    }

	    .mode-pill.online {
	      color: var(--success-color);
	      border-color: color-mix(in srgb, var(--success-color) 35%, var(--border-color));
	      background: color-mix(in srgb, var(--success-color) 12%, var(--surface-color));
	    }

	    .mode-pill.offline {
	      color: var(--danger-600);
	      border-color: color-mix(in srgb, var(--danger-600) 35%, var(--border-color));
	      background: color-mix(in srgb, var(--danger-600) 10%, var(--surface-color));
	    }
	  }

	  .setting {
	    width: auto;
    font-size: 20px;
    color: var(--text-color);
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
  justify-content: flex-start;
  align-items: center;
  position: relative;
  padding: 6px 0;
  gap: 12px;
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
    height: 60px;
    background-color: var(--bg-sider);
    border-top: 1px solid var(--border-color);

	    .nav-item {
	      position: relative;
	      text-decoration: none;
	      width: auto;
	      min-width: 52px;
	      color: var(--gray-700);
      font-size: 12px;
      font-weight: 600;
      transition: color 0.12s ease-in-out, transform 0.12s ease-in-out, background-color 0.12s ease-in-out;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 4px;
      padding: 6px 10px;
      border-radius: 12px;

      .icon {
        font-size: 18px;
      }

	      &.active {
	        color: var(--main-600);
	        background-color: var(--surface-color);
	      }

	      .mode-badge {
	        position: absolute;
	        top: 6px;
	        right: 6px;
	        display: inline-flex;
	        align-items: center;
	        justify-content: center;
	        min-width: 34px;
	        padding: 2px 6px;
	        border-radius: 999px;
	        border: 1px solid var(--border-color);
	        background: var(--surface-color);
	        font-size: 10px;
	        font-weight: 800;
	        letter-spacing: 0.3px;
	        color: var(--gray-700);
	        pointer-events: none;
	      }

	      .mode-badge.mock {
	        color: var(--main-600);
	        border-color: color-mix(in srgb, var(--main-500) 35%, var(--border-color));
	        background: color-mix(in srgb, var(--main-500) 14%, var(--surface-color));
	      }

	      .mode-badge.online {
	        color: var(--success-color);
	        border-color: color-mix(in srgb, var(--success-color) 35%, var(--border-color));
	        background: color-mix(in srgb, var(--success-color) 12%, var(--surface-color));
	      }

	      .mode-badge.offline {
	        color: var(--danger-600);
	        border-color: color-mix(in srgb, var(--danger-600) 35%, var(--border-color));
	        background: color-mix(in srgb, var(--danger-600) 10%, var(--surface-color));
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
  border-bottom: 1px solid var(--border-color);
  background-color: var(--bg-sider);
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
          color: var(--warning-color);
          font-size: 14px;
          margin-right: 2px;
        }
      }
    }
  }

  .nav-item.active::before {
    display: none;
  }

  .nav-item.active::after {
    content: '';
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: -6px;
    height: 2px;
    border-radius: 2px;
    background: var(--main-500);
  }
}
</style>
