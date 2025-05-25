import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import Antd from 'ant-design-vue';
import 'ant-design-vue/dist/reset.css';
import './assets/main.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(Antd)


const savedTheme = localStorage.getItem('theme') || 'system';
setTheme(savedTheme);

function setTheme(theme) {
  if (theme === 'system') {
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  } else {
    document.documentElement.setAttribute('data-theme', theme);
  }
}
app.mount('#app')
