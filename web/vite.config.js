import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    build: {
      rollupOptions: {
        output: {
          // Split large vendor bundles for better caching and faster initial load.
          manualChunks(id) {
            if (!id.includes('node_modules')) return

            // Vue core + router/pinia
            if (id.includes('/node_modules/vue/') || id.includes('/node_modules/@vue/') || id.includes('/node_modules/vue-router/') || id.includes('/node_modules/pinia/')) {
              return 'vue-vendor'
            }

            // Ant Design Vue + icons
            if (id.includes('/node_modules/ant-design-vue/') || id.includes('/node_modules/@ant-design/')) {
              return 'antd-vendor'
            }

            // Markdown preview/tooling
            if (id.includes('/node_modules/md-editor-v3/') || id.includes('/node_modules/marked') || id.includes('/node_modules/highlight.js/')) {
              return 'markdown-vendor'
            }

            // Charts / graph / maps
            // Keep antv + echarts families together to avoid circular chunk deps.
            if (
              id.includes('/node_modules/echarts') ||
              id.includes('/node_modules/zrender') ||
              id.includes('/node_modules/@antv/') ||
              id.includes('/node_modules/d3') ||
              id.includes('/node_modules/leaflet/')
            ) {
              return 'viz-vendor'
            }

            return 'vendor'
          },
        },
      },
    },
    server: {
      proxy: {
        '^/api': {
          target: env.VITE_API_URL || 'http://127.0.0.1:5050',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '')
        }
      },
      watch: {
        usePolling: true,
        ignored: ['**/node_modules/**', '**/dist/**'],
      },
      host: '0.0.0.0',
      port: 3100,
    }
  }
})
