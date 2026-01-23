import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import Antd from 'ant-design-vue';
import 'ant-design-vue/dist/reset.css';
import './assets/main.css'
import { initTheme } from './assets/theme'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(Antd)

// Apply theme ASAP to avoid flash (mode is persisted in localStorage).
initTheme()
app.mount('#app')
