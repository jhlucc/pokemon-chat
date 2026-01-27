<template>
  <div class="setting-page">
    <!-- 背景层：暖光氛围 -->
    <div class="ambient-glow glow--orange"></div>
    <div class="ambient-glow glow--purple"></div>
    <div class="dot-grid"></div>

    <HeaderComponent
      title="控制中心"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '设置' }]"
      class="setting-header"
    >
      <template #actions>
        <a-button type="text" @click="refreshAll" :loading="state.refreshing" class="header-action">
          <SyncOutlined :spin="state.refreshing" /> 刷新系统
        </a-button>
      </template>
    </HeaderComponent>

    <div class="setting-container layout-container">
      <div class="setting-body ui-page">
        <div class="ui-container bento-wrapper">
          
          <!-- Row 1: Status & Actions -->
          <div class="bento-grid">
            
            <!-- Status Card (Span 8) - 白色毛玻璃风格 -->
            <div class="bento-card status-card" :class="{ online: backendOnline }">
              <div class="status-content">
                <div class="status-header">
                  <div class="status-indicator" :class="{ online: backendOnline }">
                    <span class="pulse-dot"></span>
                    <span class="status-text">{{ backendOnline ? '系统运行中' : '系统离线' }}</span>
                  </div>
                  <div class="version-badge">v{{ BUILD_SHA ? BUILD_SHA.slice(0, 7) : 'DEV' }}</div>
                </div>
                <div class="status-body">
                  <h2 class="hero-title">{{ backendOnline ? '洛托姆系统运转正常' : '与后端服务的连接已断开' }}</h2>
                  <p class="hero-sub">{{ backendOnline ? '所有服务连接就绪，随时准备响应任务指令。' : '信号丢失。请检查后端服务状态或尝试重新连接。' }}</p>
                </div>
                <div class="status-footer">
                   <div class="metric">
                      <span class="label">接口网关</span>
                      <span class="value" :class="{ active: backendReady }">{{ backendReady ? '就绪' : '等待中' }}</span>
                   </div>
                   <div class="metric">
                      <span class="label">响应延迟</span>
                      <span class="value active">24ms</span>
                   </div>
                </div>
              </div>
              <!-- 装饰：半透明洛托姆轮廓 -->
              <div class="status-visual">
                <div class="rotom-silhouette"></div>
              </div>
            </div>

            <!-- Quick Actions (Span 4) -->
            <div class="bento-card actions-card">
              <div class="card-label">系统操作</div>
              <div class="action-buttons">
                <button class="bento-btn primary" :disabled="!backendOnline" @click="restartBackend">
                  <div class="btn-content">
                    <RedoOutlined class="icon" />
                    <span class="btn-text">重启服务</span>
                  </div>
                  <RightOutlined class="arrow" />
                </button>
                <button class="bento-btn secondary" @click="refreshProviders" :disabled="providersState.loading">
                  <div class="btn-content">
                    <SyncOutlined class="icon" :spin="providersState.loading" />
                    <span class="btn-text">重新加载配置</span>
                  </div>
                </button>
              </div>
            </div>

            <!-- Row 2: Brain & Providers -->
            
            <!-- Brain / Model (Span 6) -->
            <div class="bento-card model-card">
              <div class="card-label">模型配置</div>
              <div class="model-selector-area">
                <div class="model-row">
                  <span class="label">供应商</span>
                  <a-select v-model:value="modelProvider" @change="onProviderChange" class="bento-select" :bordered="false" dropdownClassName="tech-dropdown">
                    <a-select-option v-for="p in providerKeys" :key="p" :value="p">
                      <div class="select-option">
                        <img :src="getProviderIcon(p)" class="opt-icon" />
                        {{ modelCatalog[p]?.name || p }}
                      </div>
                    </a-select-option>
                  </a-select>
                </div>
                <div class="divider"></div>
                <div class="model-row">
                  <span class="label">模型 ID</span>
                  <div class="model-input-wrap">
                    <a-select v-if="providerModels.length" v-model:value="modelName" @change="onModelChange" class="bento-select" :bordered="false" dropdownClassName="tech-dropdown">
                      <a-select-option v-for="m in providerModels" :key="m" :value="m">{{ m }}</a-select-option>
                    </a-select>
                    <a-input v-else v-model:value="modelName" placeholder="输入模型名" class="bento-input" :bordered="false" @blur="commitModelInput"/>
                  </div>
                </div>
              </div>
            </div>

            <!-- Providers (Span 6) -->
            <div class="bento-card providers-card">
              <div class="card-header">
                <div class="card-label">服务连接</div>
                <div class="header-tools">
                   <span class="count mono">{{ providerList.length }}</span>
                </div>
              </div>
              <div class="providers-scroll">
                <div v-for="p in providerList" :key="p" class="provider-row" @click="toggleProviderEdit(p)">
                  <div class="p-left">
                    <img :src="getProviderIcon(p)" class="p-icon" />
                    <div class="p-name">{{ modelCatalog[p]?.name || p }}</div>
                  </div>
                  <div class="p-right">
                    <div class="status-pill" :class="providerConfigured(p) ? 'active' : 'inactive'">
                       {{ providerConfigured(p) ? '已连接' : '未配置' }}
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- Inline Edit Modal -->
              <a-modal 
                v-model:open="state.showProviderModal" 
                :title="(modelCatalog[state.editingProvider]?.name || state.editingProvider) + ' 配置'" 
                :footer="null"
                width="400px"
                class="tech-modal"
              >
                <div v-if="state.editingProvider" class="provider-edit-form">
                   <a-form layout="vertical">
                      <a-form-item label="Endpoint URL">
                        <a-input v-model:value="providerForm[state.editingProvider].api_base" :placeholder="modelCatalog[state.editingProvider]?.base_url || 'Default'" class="tech-input"/>
                      </a-form-item>
                      <a-form-item label="Secret Key">
                        <a-input-password v-model:value="providerForm[state.editingProvider].api_key" placeholder="••••••••" class="tech-input"/>
                      </a-form-item>
                      <div class="form-actions">
                        <a-button @click="clearProvider(state.editingProvider)" danger ghost>清空</a-button>
                        <a-button
                          type="primary"
                          @click="saveProvider(state.editingProvider)"
                          :loading="Boolean(providersState?.saving?.[state.editingProvider])"
                        >
                          保存
                        </a-button>
                      </div>
                   </a-form>
                </div>
              </a-modal>
            </div>

            <!-- Row 3: Capabilities Grid (Span 12) -->
            <div class="bento-card capabilities-card">
              <div class="card-label">功能模块</div>
              <div class="modules-container">
                <div 
                  class="module-tile" 
                  v-for="mod in modulesList" 
                  :key="mod.key"
                  :class="{ active: configStore.config[mod.key] }"
                  @click="toggleFeature(mod.key)"
                >
                  <div class="tile-bg"></div>
                  <div class="tile-content">
                    <div class="tile-icon">
                      <component :is="mod.icon" />
                    </div>
                    <div class="tile-info">
                      <div class="tile-name">{{ mod.label }}</div>
                      <div class="tile-status">{{ configStore.config[mod.key] ? '已开启' : '已关闭' }}</div>
                    </div>
                  </div>
                  <div class="tile-bar"></div>
                  <div class="tile-spinner" v-if="Boolean(featureState?.saving?.[mod.key])">
                    <LoadingOutlined />
                  </div>
                </div>
              </div>
            </div>

            <!-- Row 4: Appearance (Span 12) -->
            <div class="bento-card ui-card">
              <div class="card-label">界面设置</div>
              <div class="ui-options">
                <div class="ui-group">
                  <span class="group-label">主题色</span>
                  <div class="theme-picker">
                    <div
                       v-for="(p, key) in THEME_PRESETS"
                       :key="key"
                       class="color-dot"
                       :style="{ background: p.primary }"
                       :class="{ selected: themePreset === key }"
                       @click="onThemePresetChange(key)"
                    ></div>
                  </div>
                </div>
                <div class="divider-v"></div>
                <div class="ui-group">
                  <span class="group-label">布局密度</span>
                  <a-segmented v-model:value="uiDensity" :options="densityOptions" @change="onUiDensityChange" size="small" class="tech-segment"/>
                </div>
                <div class="divider-v"></div>
                <div class="ui-group">
                  <span class="group-label">导航显示</span>
                  <div class="checkbox-group">
                    <a-checkbox :checked="uiVisibility.show_knowledge_base" @change="(e) => setUiVisibility('show_knowledge_base', e.target.checked)">知识库</a-checkbox>
                    <a-checkbox :checked="uiVisibility.show_knowledge_graph" @change="(e) => setUiVisibility('show_knowledge_graph', e.target.checked)">图谱</a-checkbox>
                    <a-checkbox :checked="uiVisibility.show_web_search" @change="(e) => setUiVisibility('show_web_search', e.target.checked)">联网搜索</a-checkbox>
                    <a-checkbox :checked="uiVisibility.show_mcp" @change="(e) => setUiVisibility('show_mcp', e.target.checked)">MCP</a-checkbox>
                    <a-checkbox :checked="uiVisibility.show_map" @change="(e) => setUiVisibility('show_map', e.target.checked)">地图</a-checkbox>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { message } from 'ant-design-vue'
import { 
  RedoOutlined, SyncOutlined, RightOutlined, LoadingOutlined,
  BookFilled, DeploymentUnitOutlined, CompassFilled, ApiFilled, 
  SortAscendingOutlined, AudioFilled, TagFilled 
} from '@ant-design/icons-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import { useConfigStore } from '@/stores/config'
import { DEFAULT_CONFIG } from '@/config/defaultConfig'
import { BUILD_SHA } from '@/config/appMeta'
import { apiFetch } from '@/api/http'
import { setUiDensity } from '@/utils/uiDensity'
import { THEME_PRESETS, getThemePreset, setThemePreset } from '@/utils/themePreset'
import { notifyApiError } from '@/utils/notify'
import { getProviderIcon } from '@/utils/providerIcon'

const configStore = useConfigStore()

const state = reactive({
  refreshing: false,
  showProviderModal: false,
  editingProvider: null
})

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const backendReady = computed(() => Boolean(configStore.config.backend?.ready))
const themePreset = ref(getThemePreset())
const uiDensity = ref('comfortable')

const modelCatalog = computed(() => configStore.config.model_names || {})
const providerKeys = computed(() => Object.keys(modelCatalog.value || {}).filter((k) => k !== 'custom'))
const modelProvider = ref(configStore.config.model_provider)
const modelName = ref(configStore.config.model_name)
const providerModels = computed(() => modelCatalog.value?.[modelProvider.value]?.models || [])

const uiVisibility = computed(() => ({ ...(DEFAULT_CONFIG.ui || {}), ...(configStore.config?.ui || {}) }))

const modulesList = [
  { key: 'enable_knowledge_base', label: '知识库', icon: BookFilled },
  { key: 'enable_knowledge_graph', label: '知识图谱', icon: DeploymentUnitOutlined },
  { key: 'enable_web_search', label: '联网搜索', icon: CompassFilled },
  { key: 'enable_mcp', label: 'MCP 工具', icon: ApiFilled },
  { key: 'enable_reranker', label: '重排序', icon: SortAscendingOutlined },
  { key: 'enable_asr', label: '语音识别', icon: AudioFilled },
  { key: 'enable_ner_bert', label: '实体识别', icon: TagFilled },
]

const densityOptions = [
  { label: '舒适', value: 'comfortable' },
  { label: '紧凑', value: 'compact' }
]

// Providers
const providersState = reactive({ loading: false, saving: {}, status: {} })
const providerForm = reactive({})
const providerList = computed(() => {
  const merged = new Set([...providerKeys.value, ...Object.keys(providersState.status || {})])
  return Array.from(merged).filter(k => k && k !== 'custom')
})

const ensureProviderForm = (p) => {
  if (!providerForm[p]) providerForm[p] = { api_base: '', api_key: '' }
  if (!providerForm[p].api_base) providerForm[p].api_base = providersState.status?.[p]?.api_base || modelCatalog.value?.[p]?.base_url || ''
}

const providerConfigured = (p) => {
  if (!backendOnline.value) return false
  return Boolean(providersState.status?.[p]?.configured)
}

const toggleProviderEdit = (p) => {
  ensureProviderForm(p)
  state.editingProvider = p
  state.showProviderModal = true
}

const refreshAll = async () => { state.refreshing = true; await configStore.refreshConfig(); await refreshProviders(); state.refreshing = false }
const restartBackend = async () => { if(!backendOnline.value) return; try { await apiFetch('/restart', {method:'POST'}); message.success('已触发重启'); await refreshAll(); } catch(e){ notifyApiError(e) } }

const featureState = reactive({ saving: {} })
const setBackendFeature = async (key, checked) => {
  if (!featureState.saving) featureState.saving = {}
  featureState.saving[key] = true
  try {
    const res = await apiFetch('/config', { method: 'PATCH', body: { [key]: checked } })
    configStore.patchLocal({ ...res, backend: { online: true, last_error: null, ...(res?.backend||{}) } })
  } catch(e) { notifyApiError(e) } finally { featureState.saving[key] = false }
}
const toggleFeature = (key) => setBackendFeature(key, !configStore.config[key])

const refreshProviders = async () => {
  providersState.loading = true
  try {
    const res = await apiFetch('/providers', {method:'GET', timeoutMs:5000})
    providersState.status = res?.providers || {}
    providerList.value.forEach(ensureProviderForm)
  } catch {} finally { providersState.loading = false }
}

const saveProvider = async (p) => {
  if (!providersState.saving) providersState.saving = {}
  providersState.saving[p] = true
  try {
    const body = { provider: p, ...(providerForm[p].api_base ? {api_base: providerForm[p].api_base} : {}), ...(providerForm[p].api_key ? {api_key: providerForm[p].api_key} : {}) }
    await apiFetch('/providers', {method:'PATCH', body})
    message.success('已保存')
    refreshProviders(); state.showProviderModal = false
  } catch(e) { notifyApiError(e) } finally { providersState.saving[p] = false }
}

const clearProvider = async (p) => {
  try {
    await apiFetch('/providers', {method:'PATCH', body:{provider:p, api_key:'', api_base:''}})
    message.success('已清空'); refreshProviders(); state.showProviderModal = false
  } catch(e) { notifyApiError(e) }
}

const onProviderChange = async (p) => {
  const def = modelCatalog.value?.[p]?.default || ''
  await configStore.setConfigValues({ model_provider: p, ...(def ? {model_name: def} : {}) })
}
const onModelChange = async (m) => await configStore.setConfigValue('model_name', m)
const commitModelInput = async () => { if(modelName.value) await configStore.setConfigValue('model_name', modelName.value) }
const onUiDensityChange = (v) => setUiDensity(v)
const onThemePresetChange = (v) => { themePreset.value = setThemePreset(v) }
const setUiVisibility = (k, v) => configStore.patchLocal({ ui: { ...uiVisibility.value, [k]: v } })

watch(() => configStore.config.model_provider, (v) => (modelProvider.value = v))
watch(() => configStore.config.model_name, (v) => (modelName.value = v))
onMounted(() => refreshProviders())
</script>

<style scoped lang="less">
.setting-page {
  position: relative;
  min-height: 100vh;
  background: transparent;
  overflow-x: hidden;
}

/* 背景层：暖光氛围 */
.ambient-glow {
  position: fixed;
  border-radius: 50%;
  filter: blur(100px);
  pointer-events: none;
  z-index: 0;
  mix-blend-mode: normal;
  animation: glow-drift 20s ease-in-out infinite;
}

.glow--orange {
  width: 700px;
  height: 700px;
  top: -25%;
  left: -20%;
  background: radial-gradient(circle, rgba(255, 125, 0, 0.35) 0%, rgba(255, 180, 100, 0.15) 40%, transparent 70%);
}

.glow--purple {
  width: 550px;
  height: 550px;
  bottom: -15%;
  right: -15%;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, rgba(180, 150, 255, 0.1) 40%, transparent 70%);
  animation-delay: -7s;
}

@keyframes glow-drift {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(20px, -15px) scale(1.03); }
  66% { transform: translate(-15px, 15px) scale(0.97); }
}

/* 点阵背景 */
.dot-grid {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.05) 1px, transparent 1px),
    radial-gradient(circle, rgba(255, 125, 0, 0.025) 1px, transparent 1px);
  background-size: 24px 24px, 96px 96px;
  pointer-events: none;
  z-index: 0;
}

.setting-container { padding: 0; }

.bento-wrapper {
  max-width: none;
  margin: 0;
  padding: 0 0 40px;
  position: relative;
  z-index: 1;
}

.mono { font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; }

.bento-grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 16px;
  grid-auto-rows: minmax(auto, auto);
}

.bento-card {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03), 0 4px 12px rgba(255, 125, 0, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.5);
  padding: 20px;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  overflow: hidden;
  position: relative;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06), 0 10px 30px rgba(255, 125, 0, 0.08);
    border-color: rgba(255, 125, 0, 0.15);
    background: rgba(255, 255, 255, 0.9);
  }
}

.card-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--gray-500);
  margin-bottom: 12px;
}

/* Status Card - 白色毛玻璃风格 */
.status-card {
  grid-column: span 12;
  @media (min-width: 768px) { grid-column: span 8; }
  display: flex;
  justify-content: space-between;
  min-height: 180px;
  position: relative;
  padding: 24px;

  &.online {
    .status-indicator.online {
      .pulse-dot {
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        animation: pulse 2s ease-in-out infinite;
      }
    }
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.1); }
}

.status-content { flex: 1; display: flex; flex-direction: column; z-index: 2; }

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.status-indicator {
  display: flex;
  align-items: center;
  background: rgba(255, 125, 0, 0.08);
  padding: 6px 12px;
  border-radius: 100px;
  border: 1px solid rgba(255, 125, 0, 0.15);

  &.online {
    background: rgba(34, 197, 94, 0.08);
    border-color: rgba(34, 197, 94, 0.2);
  }
}

.pulse-dot {
  width: 8px;
  height: 8px;
  background: var(--gray-400);
  border-radius: 50%;
  margin-right: 8px;
}

.status-text {
  font-size: 12px;
  font-weight: 600;
  color: var(--gray-600);
}

.version-badge {
  font-size: 12px;
  color: var(--gray-400);
  font-weight: 500;
}

.hero-title {
  font-size: 22px;
  margin: 0 0 8px;
  font-weight: 650;
  color: var(--text-color);
  letter-spacing: -0.02em;
}

.hero-sub {
  font-size: 14px;
  color: var(--gray-500);
  margin: 0;
  max-width: 400px;
  line-height: 1.6;
}

.status-footer {
  margin-top: auto;
  display: flex;
  gap: 32px;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 4px;

  .label {
    font-size: 11px;
    color: var(--gray-400);
    font-weight: 500;
  }

  .value {
    font-size: 15px;
    font-weight: 600;
    color: var(--gray-600);

    &.active {
      color: #22c55e;
    }
  }
}

/* 洛托姆装饰 */
.status-visual {
  position: absolute;
  right: 24px;
  bottom: 24px;
  width: 120px;
  height: 120px;
  pointer-events: none;
  opacity: 0.15;

  .rotom-silhouette {
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, var(--primary-color) 0%, rgba(255, 125, 0, 0.5) 100%);
    border-radius: 50%;
    animation: float 4s ease-in-out infinite;
  }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

/* Actions Card */
.actions-card {
  grid-column: span 12;
  @media (min-width: 768px) { grid-column: span 4; }
  display: flex; flex-direction: column;
}
.action-buttons { flex: 1; display: flex; flex-direction: column; gap: 10px; justify-content: center; }
.bento-btn {
  border: none; padding: 12px 16px; border-radius: 10px; cursor: pointer; display: flex; align-items: center; justify-content: space-between; font-weight: 500; transition: all 0.2s; font-size: 13px; width: 100%;
  .btn-content { display: flex; align-items: center; gap: 10px; }
  .icon { font-size: 14px; }
  &.primary { background: var(--pokedex-red); color: white; &:hover { background: color-mix(in srgb, var(--pokedex-red), white 10%); } &:disabled { background: #cbd5e1; cursor: not-allowed; } }
  &.secondary { background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; &:hover { background: #f1f5f9; border-color: #cbd5e1; } }
}

/* Model Card */
.model-card {
  grid-column: span 12; @media (min-width: 768px) { grid-column: span 6; }
}
.model-selector-area { display: flex; flex-direction: column; height: 100%; }
.model-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; }
.model-row .label { font-size: 13px; font-weight: 600; color: #64748b; }
.bento-select, .bento-input { width: 100%; max-width: 200px; text-align: right; font-size: 13px; font-family: 'JetBrains Mono', monospace; }
.select-option { display: flex; align-items: center; justify-content: flex-end; gap: 6px; font-size: 13px; }
.opt-icon { width: 14px; height: 14px; border-radius: 3px; }
.divider { height: 1px; background: #f1f5f9; margin: 2px 0; }

/* Providers Card */
.providers-card {
  grid-column: span 12; @media (min-width: 768px) { grid-column: span 6; } display: flex; flex-direction: column;
}
.card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.providers-scroll { flex: 1; overflow-y: auto; max-height: 180px; padding-right: 4px; }
.providers-scroll::-webkit-scrollbar { width: 4px; }
.providers-scroll::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 2px; }
.provider-row {
  display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; border-radius: 8px; cursor: pointer; transition: all 0.15s; border: 1px solid transparent; margin-bottom: 4px;
  background: var(--surface-color);
  &:hover { background: var(--surface-color-2); border-color: var(--border-color); box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
}
.p-left { display: flex; align-items: center; gap: 10px; }
.p-icon { width: 20px; height: 20px; border-radius: 5px; background: var(--surface-color); padding: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.p-name { font-weight: 600; color: var(--text-color); font-size: 13px; }
.status-pill {
  font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace;
  &.active { background: #dcfce7; color: #166534; }
  &.inactive { background: #f1f5f9; color: #94a3b8; }
}

/* Capabilities */
.capabilities-card { grid-column: span 12; }
.modules-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; }
.module-tile {
  background: var(--surface-color); border-radius: 12px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between; height: 90px;
  cursor: pointer; transition: all 0.2s ease; border: 1px solid var(--border-color); position: relative; overflow: hidden;

  .tile-bg { position: absolute; inset: 0; opacity: 0; background: linear-gradient(135deg, rgba(255,83,80,0.05), transparent); transition: opacity 0.2s; }
  .tile-content { position: relative; z-index: 1; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
  .tile-icon { font-size: 20px; color: var(--gray-500); transition: color 0.2s; }
  .tile-name { font-weight: 600; color: var(--gray-600); font-size: 12px; transition: color 0.2s; }
  .tile-status { font-size: 10px; color: var(--gray-400); font-weight: 700; margin-top: 2px; }
  .tile-bar { position: absolute; bottom: 0; left: 0; width: 100%; height: 3px; background: var(--gray-200); transition: background 0.2s; }

  &.active {
    border-color: rgba(255, 83, 80, 0.2); box-shadow: 0 4px 12px rgba(255, 83, 80, 0.05);
    .tile-bg { opacity: 1; }
    .tile-icon { color: var(--pokedex-red); }
    .tile-name { color: var(--text-color); }
    .tile-status { color: var(--pokedex-red); }
    .tile-bar { background: var(--pokedex-red); }
  }
  &:hover { transform: translateY(-1px); }
}

/* UI Card */
.ui-card { grid-column: span 12; }
.ui-options { display: flex; gap: 40px; align-items: center; flex-wrap: wrap; }
.ui-group { display: flex; flex-direction: column; gap: 8px; }
.group-label { font-size: 11px; font-weight: 600; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase; }
.theme-picker { display: flex; gap: 10px; }
.color-dot {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  cursor: pointer;
  border: 3px solid white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;

  &:hover {
    transform: scale(1.1);
  }

  &.selected {
    transform: scale(1.15);
    box-shadow: 0 0 0 3px rgba(255, 125, 0, 0.2), 0 4px 12px rgba(0, 0, 0, 0.15);
  }
}
.checkbox-group { display: flex; gap: 16px; }
.divider-v { width: 1px; height: 32px; background: #f1f5f9; }

/* Ant Design Overrides */
:deep(.ant-input), :deep(.ant-select-selector) { background: transparent !important; font-size: 13px; }
:deep(.ant-checkbox-wrapper) { font-size: 13px; }
:deep(.tech-modal .ant-modal-content) { border-radius: 16px; padding: 24px; }
.tech-input { border-radius: 8px; background: #f8fafc !important; border-color: #e2e8f0 !important; &:focus { background: white !important; border-color: var(--pokedex-red) !important; } }

/* 暗色模式 */
:root[data-theme='dark'] {
  .ambient-glow {
    mix-blend-mode: screen;
    opacity: 0.7;
  }

  .glow--orange {
    background: radial-gradient(circle, rgba(255, 125, 0, 0.25) 0%, rgba(255, 180, 100, 0.1) 40%, transparent 70%);
  }

  .glow--purple {
    background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(180, 150, 255, 0.08) 40%, transparent 70%);
  }

  .dot-grid {
    background-image:
      radial-gradient(circle, rgba(255, 125, 0, 0.06) 1px, transparent 1px),
      radial-gradient(circle, rgba(255, 125, 0, 0.03) 1px, transparent 1px);
  }

  .bento-card {
    background: rgba(40, 40, 40, 0.75);
    border-color: rgba(255, 255, 255, 0.08);

    &:hover {
      background: rgba(50, 50, 50, 0.9);
      border-color: rgba(255, 125, 0, 0.3);
    }
  }

  .status-indicator {
    background: rgba(255, 125, 0, 0.12);
    border-color: rgba(255, 125, 0, 0.2);

    &.online {
      background: rgba(34, 197, 94, 0.12);
      border-color: rgba(34, 197, 94, 0.25);
    }
  }

  .bento-btn.secondary {
    background: rgba(40, 40, 40, 0.8);
    color: var(--gray-300);
    border-color: rgba(255, 255, 255, 0.1);

    &:hover {
      background: rgba(50, 50, 50, 0.9);
      border-color: rgba(255, 255, 255, 0.15);
    }
  }

  .provider-row {
    background: rgba(30, 30, 30, 0.6);

    &:hover {
      background: rgba(40, 40, 40, 0.8);
      border-color: rgba(255, 255, 255, 0.1);
    }
  }

  .status-pill {
    &.active {
      background: rgba(34, 197, 94, 0.15);
      color: #4ade80;
    }
    &.inactive {
      background: rgba(255, 255, 255, 0.05);
      color: var(--gray-500);
    }
  }

  .module-tile {
    background: rgba(30, 30, 30, 0.6);
    border-color: rgba(255, 255, 255, 0.08);

    &.active {
      border-color: rgba(255, 125, 0, 0.3);
    }
  }

  .divider, .divider-v {
    background: rgba(255, 255, 255, 0.08);
  }

  .color-dot {
    border-color: rgba(40, 40, 40, 0.9);

    &.selected {
      box-shadow: 0 0 0 3px rgba(255, 125, 0, 0.3), 0 4px 12px rgba(0, 0, 0, 0.3);
    }
  }

  .tech-input {
    background: rgba(30, 30, 30, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;

    &:focus {
      background: rgba(40, 40, 40, 0.9) !important;
    }
  }
}
</style>
