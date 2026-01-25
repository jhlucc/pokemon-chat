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
  InfoCircleOutlined,
  BulbFilled,
  DesktopOutlined,
  RobotOutlined,
  RobotFilled,
  MenuFoldOutlined,
  MenuUnfoldOutlined
} from '@ant-design/icons-vue'
import { applyTheme, getSavedThemeMode, setThemeMode } from '@/assets/theme'
import { APP_NAME, BUILD_SHA, BUILD_TIME, getBuildLabel } from '@/config/appMeta'
import { useConfigStore } from '@/stores/config'
import { useDatabaseStore } from '@/stores/database'
import DebugComponent from '@/components/DebugComponent.vue'

const configStore = useConfigStore()
const databaseStore = useDatabaseStore()
const preferredDark = usePreferredDark()


const buildLabel = getBuildLabel()

const SIDER_COLLAPSED_STORAGE_KEY = 'pokemon_chat_sider_collapsed_v1'
const getSavedSiderCollapsed = () => {
  try {
    const raw = localStorage.getItem(SIDER_COLLAPSED_STORAGE_KEY)
    if (raw === '0') return false
    if (raw === '1') return true
  } catch {
    // ignore storage errors
  }

  try {
    // Default: expanded on desktop, collapsed on smaller screens.
    return window.matchMedia('(max-width: 960px)').matches
  } catch {
    return false
  }
}

const siderCollapsed = ref(getSavedSiderCollapsed())
watch(
  () => siderCollapsed.value,
  (v) => {
    try {
      localStorage.setItem(SIDER_COLLAPSED_STORAGE_KEY, v ? '1' : '0')
    } catch {
      // ignore quota / private mode errors
    }
  }
)

const toggleSider = () => {
  siderCollapsed.value = !siderCollapsed.value
}

const scrollContainer = ref<HTMLElement | null>(null)
const backTopTarget = () => scrollContainer.value || document.body


const layoutSettings = reactive({
  showDebug: false,
  useTopBar: false, // 是否使用顶栏
  showAbout: false
})

// Theme mode: only "system" | "dark" (stored in localStorage)
const themeMode = ref(getSavedThemeMode())
const appliedIsDark = computed(
  () => themeMode.value === 'dark' || (themeMode.value === 'system' && preferredDark.value)
)
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

onUnmounted(() => {
})

// 打印当前页面的路由信息，使用vue3的setup composition API
const route = useRoute()

const apiDocsUrl = computed(() => {
  // Works both in dev (Vite proxy) and prod (Nginx /api reverse proxy).
  return `${window.location.origin}/api/docs`
})

const backendMode = computed(() => {
  const backend = configStore.config?.backend || {}
  const isOnline = Boolean(backend.online)
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
      activeIcon: MessageFilled
    },
    {
      name: '图谱',
      path: '/graph',
      icon: ProjectOutlined,
      activeIcon: ProjectFilled,
      hidden: ui.show_knowledge_graph === false
    },
    {
      name: '知识库',
      path: '/database',
      icon: BookOutlined,
      activeIcon: BookFilled,
      hidden: ui.show_knowledge_base === false
    },
    {
      name: '工具',
      path: '/tools',
      icon: ToolOutlined,
      activeIcon: ToolFilled,
      hidden: ui.show_tools === false
    },
    {
      name: '智能体',
      path: '/agent',
      icon: RobotOutlined,
      activeIcon: RobotFilled,
      hidden: ui.show_agents === false
    },
    {
      name: '地图',
      path: '/coords',
      icon: RedoOutlined, // 你可以换成其他图标
      activeIcon: RedoOutlined,
      hidden: ui.show_map === false
    }
  ]
})
</script>

<template>
  <div class="app-layout" :class="{ 'use-top-bar': layoutSettings.useTopBar }">
    <a class="skip-link" href="#app-router-view">Skip to content</a>
    <div class="debug-panel">
      <a-float-button
        @click="layoutSettings.showDebug = !layoutSettings.showDebug"
        tooltip="调试面板"
        :style="{
          right: '12px'
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
        :contentWrapperStyle="{ maxWidth: '100%' }"
        placement="right"
      >
        <DebugComponent />
      </a-drawer>
    </div>
    <div
      class="header"
      :class="{
        'top-bar': layoutSettings.useTopBar,
        'is-collapsed': !layoutSettings.useTopBar && siderCollapsed,
        'is-expanded': !layoutSettings.useTopBar && !siderCollapsed
      }"
    >
      <div class="logo circle">
        <a-tooltip placement="right">
          <template #title>
            <div>{{ APP_NAME }}</div>
            <div v-if="buildLabel" style="opacity: 0.75; font-size: 12px">{{ buildLabel }}</div>
          </template>
          <router-link to="/" aria-label="返回首页">
            <img src="/avatar.jpg" alt="Logo" />
            <div class="logo-meta">
              <span class="logo-text">{{ APP_NAME }}</span>
              <span v-if="buildLabel" class="logo-sub">{{ buildLabel }}</span>
            </div>
          </router-link>
        </a-tooltip>
      </div>
      <div class="nav">
        <!-- 使用mainList渲染导航项 -->
        <div v-for="(item, index) in mainList" :key="index" v-show="!item.hidden" class="nav-item-wrap">
          <a-tooltip
            placement="right"
            :title="!layoutSettings.useTopBar && siderCollapsed ? item.name : null"
          >
            <RouterLink
              :to="item.path"
              class="nav-item"
              active-class="active"
              :aria-label="item.name"
            >
              <component
                class="icon"
                :is="route.path.startsWith(item.path) ? item.activeIcon : item.icon"
              />
              <span class="text">{{ item.name }}</span>
            </RouterLink>
          </a-tooltip>
        </div>
      </div>
      <div class="fill" style="flex-grow: 1"></div>

      <div
        v-if="!layoutSettings.useTopBar"
        class="nav-item collapse-toggle"
        @click="toggleSider"
        role="button"
        tabindex="0"
        aria-label="Toggle sidebar"
        @keydown.enter.prevent="toggleSider"
        @keydown.space.prevent="toggleSider"
      >
        <a-tooltip placement="right">
          <template #title>{{ siderCollapsed ? 'Expand sidebar' : 'Collapse sidebar' }}</template>
          <component class="icon" :is="siderCollapsed ? MenuUnfoldOutlined : MenuFoldOutlined" />
          <span class="text">{{ siderCollapsed ? 'Expand' : 'Collapse' }}</span>
        </a-tooltip>
      </div>

      <div
        class="nav-item mode-indicator"
        aria-label="????"
        role="status"
      >
        <a-tooltip placement="right">
          <template #title>
            <div>{{ backendMode.label }}</div>
          </template>
          <span class="mode-pill" :class="backendMode.key">{{ backendMode.short }}</span>
        </a-tooltip>
      </div>


      <div class="nav-item api-docs">
        <a-tooltip placement="right">
          <template #title>接口文档 {{ apiDocsUrl }}</template>
          <a
            :href="apiDocsUrl"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="接口文档"
            class="github-link"
          >
            <ApiOutlined class="icon" />
          </a>
        </a-tooltip>
      </div>
      <div
        class="nav-item about"
        @click="layoutSettings.showAbout = true"
        role="button"
        tabindex="0"
        aria-label="关于"
        @keydown.enter.prevent="layoutSettings.showAbout = true"
        @keydown.space.prevent="layoutSettings.showAbout = true"
      >
        <a-tooltip placement="right">
          <template #title>关于</template>
          <InfoCircleOutlined class="icon" aria-hidden="true" />
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
          <component
            class="icon"
            :is="route.path === '/setting' ? SettingFilled : SettingOutlined"
          />
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
        <span class="mode-badge" :class="backendMode.key" aria-hidden="true">{{
          backendMode.short
        }}</span>
      </RouterLink>
    </div>
    <div id="app-router-view" ref="scrollContainer" tabindex="-1" role="main">

      <router-view v-slot="{ Component, route }">
        <keep-alive v-if="route.meta.keepAlive !== false">
          <component :is="Component" />
        </keep-alive>
        <component :is="Component" v-else />
      </router-view>
    </div>

    <a-back-top :target="backTopTarget" :visibilityHeight="240" />

    <a-modal v-model:open="layoutSettings.showAbout" title="关于" :footer="null">
      <a-descriptions size="small" :column="1" bordered>
        <a-descriptions-item label="应用">{{ APP_NAME }}</a-descriptions-item>
        <a-descriptions-item label="前端">{{ buildLabel || 'dev' }}</a-descriptions-item>
        <a-descriptions-item v-if="BUILD_TIME" label="Build time">{{ BUILD_TIME }}</a-descriptions-item>
        <a-descriptions-item v-if="BUILD_SHA" label="Commit">{{ BUILD_SHA }}</a-descriptions-item>
        <a-descriptions-item label="API Docs">
          <a :href="apiDocsUrl" target="_blank" rel="noopener noreferrer">{{ apiDocsUrl }}</a>
        </a-descriptions-item>
      </a-descriptions>
      <a-alert
        style="margin-top: 12px"
        type="info"
        show-icon
        message="排查问题时请提供 Request ID（RID），可在错误提示或后端响应头 X-Request-ID 中找到。"
      />
    </a-modal>
  </div>
</template>

<style lang="less" scoped>
.app-layout {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: 100vh;
  min-width: var(--min-width);

  .skip-link {
    position: fixed;
    left: 12px;
    top: 12px;
    padding: 8px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    background: var(--surface-color);
    color: var(--text-color);
    box-shadow: var(--shadow-sm);
    text-decoration: none;
    z-index: 4000;
    transform: translateY(-200%);
    transition: transform 0.2s ease;
  }

  .skip-link:focus {
    transform: translateY(0);
  }

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

div.header,
#app-router-view {
  height: 100%;
  max-width: 100%;
  user-select: none;
}

#app-router-view {
  flex: 1 1 auto;
  overflow-y: auto;
}

.offline-banner {
  position: sticky;
  top: 0;
  z-index: 20;
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
  transition: width 0.18s ease-in-out;

  .logo {
    width: 40px;
    height: 40px;
    margin: 18px 0 18px 0;

    img {
      width: 100%;
      height: 100%;
      border-radius: 4px; // 50% for circle
    }

    .logo-meta {
      display: none;
      min-width: 0;
    }

    .logo-sub {
      display: block;
      font-size: 11px;
      color: var(--gray-600);
      opacity: 0.9;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    & > a {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
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
    min-height: 52px;
    padding: 4px;
    padding-top: 10px;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    background-color: transparent;
    color: var(--text-color);
    font-size: 20px;
    transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, color 0.2s ease-in-out,
      transform 0.12s ease-in-out;
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
      border-color: var(--border-color);
    }

    &:active {
      transform: translateY(1px);
    }

    &:focus-visible {
      border-color: color-mix(in srgb, var(--main-500) 32%, var(--border-color));
      background-color: var(--surface-color);
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

.header.is-expanded {
  width: 224px;
  align-items: stretch;
  padding: 0 8px;
}

.header.is-expanded .logo {
  width: auto;
  height: auto;
  margin: 14px 4px 12px;
}

.header.is-expanded .logo img {
  width: 32px;
  height: 32px;
}

.header.is-expanded .logo > a {
  justify-content: flex-start;
  padding: 8px 10px;
  border-radius: var(--radius-md);
}

.header.is-expanded .logo > a:hover {
  background: var(--hover-bg);
}

.header.is-expanded .logo-meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.header.is-expanded .logo-text {
  display: block;
  font-size: 14px;
  font-weight: 650;
  line-height: 1.2;
}

.header.is-expanded .nav {
  align-items: stretch;
}

.header.is-expanded .nav-item {
  flex-direction: row;
  justify-content: flex-start;
  width: 100%;
  padding: 10px 10px;
  margin: 0;
  gap: 10px;
}

.header.is-expanded .nav-item .text {
  margin-top: 0;
  font-size: 14px;
  text-align: left;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.header.is-expanded .nav-item.active::before {
  left: 0;
}

.header.is-expanded .nav-item.api-docs,
.header.is-expanded .nav-item.mode-indicator,
.header.is-expanded .nav-item.setting {
  padding: 10px 10px;
  width: 100%;
}

.header.is-collapsed .nav-item.collapse-toggle .text {
  display: none;
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

.header .nav .nav-item-wrap {
  width: 100%;
  display: flex;
  justify-content: center;
}

.header.is-expanded .nav .nav-item-wrap {
  justify-content: stretch;
}

.header.is-collapsed .nav .nav-item .text {
  display: none;
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
      transition: color 0.12s ease-in-out, transform 0.12s ease-in-out,
        background-color 0.12s ease-in-out;
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

    .logo-meta {
      display: flex;
      align-items: center;
    }

    .logo-text {
      display: block;
      font-size: 16px;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: var(--main-600);
      white-space: nowrap;
    }

    .logo-sub {
      display: none;
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

    &.github,
    &.setting {
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
