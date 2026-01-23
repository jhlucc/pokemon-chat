<template>
  <div class="app-layout">
    <!-- Side Navigation (Optional/Collapsible) -->
    <aside class="sidebar" :class="{ collapsed: collapsed }">
        <div class="logo-area">
            <div class="traffic-lights">
                <div class="traffic-light red"></div>
                <div class="traffic-light yellow"></div>
                <div class="traffic-light green"></div>
            </div>
            <span v-if="!collapsed" class="app-title">ThinkFlow</span>
        </div>

        <nav class="nav-menu">
            <router-link to="/chat" class="nav-item" active-class="active">
                <MessageOutlined />
                <span v-if="!collapsed">Chat</span>
            </router-link>
            <router-link to="/graph" class="nav-item" active-class="active">
                <ShareAltOutlined />
                <span v-if="!collapsed">Graph</span>
            </router-link>
            <router-link to="/database" class="nav-item" active-class="active">
                <DatabaseOutlined />
                <span v-if="!collapsed">Data</span>
            </router-link>
            <router-link to="/agent" class="nav-item" active-class="active">
                <RobotOutlined />
                <span v-if="!collapsed">Agents</span>
            </router-link>
             <router-link to="/setting" class="nav-item" active-class="active">
                <SettingOutlined />
                <span v-if="!collapsed">Settings</span>
            </router-link>
        </nav>

        <div class="sidebar-footer" @click="toggleCollapse">
            <MenuUnfoldOutlined v-if="collapsed" />
            <MenuFoldOutlined v-else />
        </div>
    </aside>

    <!-- Main Workspace -->
    <main class="main-content">
        <!-- Top Navigation / Status Bar -->
        <header class="top-nav glass">
             <div class="breadcrumbs mono-font">
                 <span class="path-segment">~/thinkflow</span>
                 <span class="path-segment">/workspace</span>
             </div>
             
             <div class="top-actions">
                 <a-button type="text" shape="circle">
                     <GithubOutlined />
                 </a-button>
                 <a-button type="text" shape="circle">
                     <BellOutlined />
                 </a-button>
                 <div class="user-avatar">
                     <UserOutlined />
                 </div>
             </div>
        </header>

        <div class="content-area">
            <div class="content-window">
                <router-view v-slot="{ Component }">
                  <keep-alive :include="['ChatView', 'GraphView']">
                    <component :is="Component" />
                  </keep-alive>
                </router-view>
            </div>
        </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { 
    MessageOutlined, 
    ShareAltOutlined, 
    DatabaseOutlined, 
    RobotOutlined,
    SettingOutlined,
    MenuFoldOutlined,
    MenuUnfoldOutlined,
    GithubOutlined,
    BellOutlined,
    UserOutlined
} from '@ant-design/icons-vue';

const collapsed = ref(false);

const toggleCollapse = () => {
    collapsed.value = !collapsed.value;
}
</script>

<style lang="less" scoped>
.app-layout {
    display: flex;
    height: 100vh;
    width: 100vw;
    background: transparent; /* Shows grid from body */
}

.sidebar {
    width: 240px;
    height: 100%;
    background: var(--surface-card);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    z-index: 50;
    
    &.collapsed {
        width: 64px;
        
        .logo-area { padding: 20px; justify-content: center; }
        .traffic-lights { display: none; } /* Hide dots on collapse or stack them vertical? Hide for now */
        .app-title { display: none; }
        .nav-item span { display: none; }
        .nav-item { justify-content: center; padding: 12px; }
    }
}

.logo-area {
    height: 60px;
    padding: 0 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid var(--border-color);
    
    .app-title {
        font-family: var(--font-sans);
        font-weight: 700;
        font-size: 16px;
        color: var(--text-color);
        letter-spacing: -0.02em;
    }
}

/* Traffic lights from base.css */
.traffic-lights {
    display: flex;
    gap: 6px;
}
.traffic-light {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    
    &.red { background-color: #FF5F56; }
    &.yellow { background-color: #FFBD2E; }
    &.green { background-color: #27C93F; }
}

.nav-menu {
    flex: 1;
    padding: 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    color: var(--subtext-color);
    text-decoration: none;
    border-radius: var(--radius-md);
    font-family: var(--font-mono); /* Developer feel */
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
    
    &:hover {
        background: var(--surface-secondary);
        color: var(--text-color);
    }
    
    &.active {
        background: var(--surface-secondary);
        color: var(--primary-color);
        border: 1px solid var(--border-color);
    }
    
    .anticon { font-size: 16px; }
}

.sidebar-footer {
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-top: 1px solid var(--border-color);
    cursor: pointer;
    color: var(--subtext-color);
    
    &:hover { color: var(--text-color); }
}

.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.top-nav {
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 40;
    background: transparent;
    
    .breadcrumbs {
        color: var(--subtext-color);
        font-size: 13px;
        
        .path-segment:after {
            content: '/';
            margin: 0 8px;
            color: var(--slate-300);
        }
        .path-segment:last-child:after { content: ''; }
        .path-segment:last-child { color: var(--text-color); font-weight: 600; }
    }
}

.top-actions {
    display: flex;
    gap: 12px;
    align-items: center;
    color: var(--text-color);
    
    .ant-btn {
        color: var(--subtext-color);
        &:hover { color: var(--primary-color); background: var(--surface-secondary); }
    }
}

.user-avatar {
    width: 32px;
    height: 32px;
    background: var(--surface-secondary);
    border: 1px solid var(--border-color);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--subtext-color);
    cursor: pointer;
    transition: all 0.2s;
    
    &:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
    }
}

.content-area {
    flex: 1;
    padding: 24px;
    overflow: hidden;
    display: flex;
}

.content-window {
    flex: 1;
    background: var(--surface-card); /* Pure white card on the grid */
    border: 1px solid var(--border-color); /* 1px border */
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg); /* Soft floating shadow */
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
    
    /* Ensure child components fill this window */
    & > * {
        height: 100%;
        width: 100%;
    }
}
</style>
