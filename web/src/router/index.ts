import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'
import BlankLayout from '@/layouts/BlankLayout.vue'
import { APP_NAME } from '@/config/appMeta'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: BlankLayout,
    children: [
      {
        path: '',
        name: 'Home',
        component: () => import('@/views/HomeView.vue'),
        meta: { title: '首页', keepAlive: true }
      }
    ]
  },
  {
    path: '/chat',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'Chat',
        component: () => import('@/views/ChatView.vue'),
        meta: { title: '对话', keepAlive: true }
      }
    ]
  },
  {
    path: '/agent',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'Agent',
        redirect: '/chat',
        meta: { title: '智能体', keepAlive: true }
      },
      {
        path: ':agent_id',
        name: 'AgentSinglePage',
        redirect: '/chat',
        meta: { title: '智能体详情', keepAlive: false }
      }
    ]
  },
  {
    path: '/graph',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'Graph',
        component: () => import('@/views/GraphView.vue'),
        meta: { title: '知识图谱', keepAlive: false }
      }
    ]
  },
  {
    path: '/database',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'Database',
        component: () => import('@/views/DataBaseView.vue'),
        meta: { title: '知识库', keepAlive: true }
      },
      {
        path: ':database_id',
        name: 'DatabaseInfo',
        component: () => import('@/views/DataBaseInfoView.vue'),
        meta: { title: '知识库详情', keepAlive: false }
      }
    ]
  },
  {
    path: '/coords',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'CoordsMap',
        component: () => import('@/views/CoordsMapPage.vue'),
        meta: { title: '地图', keepAlive: false }
      }
    ]
  },
  {
    path: '/tools',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'Tools',
        redirect: '/database',
        meta: { title: '工具', keepAlive: true }
      },
      {
        path: 'file-chunking',
        name: 'FileChunking',
        redirect: '/database',
        meta: { title: '文件切分', keepAlive: false }
      },
      {
        path: 'pdf2txt',
        name: 'PDF_to_TXT',
        redirect: '/database',
        meta: { title: 'PDF 转 TXT', keepAlive: false }
      },
      {
        path: 'agent',
        name: 'AgentRedirect',
        redirect: '/chat'
      },
      {
        path: ':pathMatch(.*)*',
        redirect: '/database'
      }
    ]
  },
  {
    path: '/setting',
    component: AppLayout,
    children: [
      {
        path: '',
        name: 'Setting',
        component: () => import('@/views/SettingView.vue'),
        meta: { title: '设置', keepAlive: true }
      }
    ]
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/EmptyView.vue'),
    meta: { title: '404', keepAlive: false }
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

router.afterEach((to) => {
  const pageTitle = to.meta?.title
  document.title = pageTitle ? `${pageTitle} - ${APP_NAME}` : APP_NAME
})

export default router
