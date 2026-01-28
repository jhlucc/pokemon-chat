import { fileURLToPath, URL } from 'node:url'
import { readFileSync } from 'node:fs'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  // Unify env: load from repo root (one level above /web).
  // Only load VITE_* keys to avoid pulling backend secrets into the build config.
  const repoRoot = fileURLToPath(new URL('..', import.meta.url))
  const env = loadEnv(mode, repoRoot, 'VITE_')
  const apiTarget = env.VITE_API_URL || 'http://127.0.0.1:5050'
  const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf-8'))
  const buildTime = env.VITE_BUILD_TIME || new Date().toISOString()
  const buildSha = env.VITE_BUILD_SHA || process.env.GITHUB_SHA || ''
  return {
    plugins: [vue()],
    envDir: repoRoot,
    define: {
      // Make app metadata available at runtime via import.meta.env.
      'import.meta.env.VITE_APP_VERSION': JSON.stringify(env.VITE_APP_VERSION || pkg.version),
      'import.meta.env.VITE_BUILD_TIME': JSON.stringify(buildTime),
      'import.meta.env.VITE_BUILD_SHA': JSON.stringify(buildSha),
    },
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    esbuild: mode === 'production' ? { drop: ['console', 'debugger'] } : undefined,
    build: {
      target: 'es2022',
      // Enable sourcemaps only when explicitly requested (useful for staging).
      sourcemap: env.VITE_SOURCEMAP === 'true',
      chunkSizeWarningLimit: 1500,
      // CSS 代码分割
      cssCodeSplit: true,
      // 小于 4KB 的资源内联为 base64
      assetsInlineLimit: 4096,
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
            // md-editor-v3 pulls in CodeMirror/Lezer packages; split them into their own chunk
            // so the main app stays small and the editor toolchain can be lazy-loaded.
            if (
              id.includes('/node_modules/@codemirror/lang-') ||
              id.includes('/node_modules/@codemirror/language-data/')
            ) {
              return 'codemirror-lang'
            }

            if (
              id.includes('/node_modules/@codemirror/') ||
              id.includes('/node_modules/@lezer/') ||
              id.includes('/node_modules/codemirror/') ||
              id.includes('/node_modules/style-mod/') ||
              id.includes('/node_modules/w3c-keyname/') ||
              id.includes('/node_modules/crelt/') ||
              id.includes('/node_modules/@uiw/')
            ) {
              return 'codemirror-core'
            }

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
        // Vite dev server proxy. Mirrors the docker/nginx reverse-proxy behavior:
        // - `/api/*` -> backend `/*`
        // - `/openapi.json` is required for Swagger UI (`/docs`) to render correctly.
        '^/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '')
        },
        '^/openapi.json': {
          target: apiTarget,
          changeOrigin: true
        },
        '^/docs': {
          target: apiTarget,
          changeOrigin: true
        },
        '^/redoc': {
          target: apiTarget,
          changeOrigin: true
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
