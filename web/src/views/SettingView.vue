<template>
  <div class="setting-page">
    <div class="tech-bg"></div> <!-- Background Texture -->
    
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
            
            <!-- Status Card (Span 8) -->
            <div class="bento-card status-card" :class="{ online: backendOnline }">
              <div class="card-deco-corner top-left"></div>
              <div class="card-deco-corner top-right"></div>
              
              <div class="status-content">
                <div class="status-header">
                  <div class="status-indicator">
                    <span class="pulse-dot"></span>
                    <span class="status-text mono">{{ backendOnline ? 'SYSTEM_ONLINE' : 'SYSTEM_OFFLINE' }}</span>
                  </div>
                  <div class="version-badge mono">BUILD.{{ BUILD_SHA ? BUILD_SHA.slice(0, 7) : 'DEV' }}</div>
                </div>
                <div class="status-body">
                  <h2 class="hero-title">{{ backendOnline ? '洛托姆系统运转正常' : '与后端服务的连接已断开' }}</h2>
                  <p class="hero-sub">{{ backendOnline ? '所有神经元连接就绪，随时准备响应任务指令。' : '信号丢失。请检查后端服务状态或尝试重新连接。' }}</p>
                </div>
                <div class="status-footer">
                   <div class="metric">
                      <span class="label">API GATEWAY</span>
                      <span class="value mono">{{ backendReady ? 'READY' : 'PENDING' }}</span>
                      <div class="metric-bar" :class="{ active: backendReady }"></div>
                   </div>
                   <div class="metric">
                      <span class="label">LATENCY</span>
                      <span class="value mono">24ms</span>
                      <div class="metric-bar active"></div>
                   </div>
                </div>
              </div>
              <div class="status-visual">
                <!-- Tech Visual -->
                <div class="radar-circle"></div>
                <div class="radar-circle c2"></div>
                <div class="radar-grid"></div>
                <div class="radar-scan"></div>
              </div>
            </div>

            <!-- Quick Actions (Span 4) -->
            <div class="bento-card actions-card">
              <div class="card-label">SYSTEM_OPS</div>
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
              <div class="card-label">NEURAL_CORE (Model)</div>
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
                <div class="card-label">CONNECTIONS</div>
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
                       {{ providerConfigured(p) ? 'LINKED' : 'VOID' }}
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
              <div class="card-label">EXTENSIONS (Modules)</div>
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
                      <div class="tile-status mono">{{ configStore.config[mod.key] ? 'ON' : 'OFF' }}</div>
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
              <div class="card-label">INTERFACE</div>
              <div class="ui-options">
                <div class="ui-group">
                  <span class="group-label">Theme Color</span>
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
                  <span class="group-label">Density</span>
                  <a-segmented v-model:value="uiDensity" :options="densityOptions" @change="onUiDensityChange" size="small" class="tech-segment"/>
                </div>
                <div class="divider-v"></div>
                <div class="ui-group">
                  <span class="group-label">Navigation</span>
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
import { APP_NAME, APP_VERSION, BUILD_SHA } from '@/config/appMeta'
import { ApiError, apiFetch } from '@/api/http'
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
.setting-container { padding: 0; }
.bento-wrapper { max-width: 900px; margin: 0 auto; padding-bottom: 40px; position: relative; z-index: 1; }

.tech-bg {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background-color: #f4f6f9;
  background-image: 
    radial-gradient(rgba(0,0,0,0.06) 1px, transparent 1px);
  background-size: 20px 20px;
  z-index: 0;
  pointer-events: none;
}

.mono { font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; }

.bento-grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: 16px;
  grid-auto-rows: minmax(auto, auto);
}

.bento-card {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  border-radius: 16px;
  box-shadow: 
    0 4px 6px -1px rgba(0, 0, 0, 0.02), 
    0 2px 4px -1px rgba(0, 0, 0, 0.02),
    inset 0 1px 0 rgba(255,255,255,0.8);
  border: 1px solid rgba(0,0,0,0.06);
  padding: 20px;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  overflow: hidden;
  position: relative;
  
  &:hover {
    box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.06);
    border-color: rgba(0,0,0,0.1);
    transform: translateY(-2px);
  }
}

.card-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
  margin-bottom: 12px;
}

/* Status Card */
.status-card {
  grid-column: span 12;
  @media (min-width: 768px) { grid-column: span 8; }
  background: #0f172a; 
  color: white;
  display: flex;
  justify-content: space-between;
  min-height: 180px;
  position: relative;
  
  &.online {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(255,255,255,0.08);
    .pulse-dot { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
  }
  
  .card-deco-corner {
    position: absolute; width: 10px; height: 10px; border: 2px solid rgba(255,255,255,0.1);
    &.top-left { top: 12px; left: 12px; border-right: none; border-bottom: none; }
    &.top-right { top: 12px; right: 12px; border-left: none; border-bottom: none; }
  }
}

.status-content { flex: 1; display: flex; flex-direction: column; z-index: 2; }
.status-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.status-indicator { display: flex; align-items: center; background: rgba(255,255,255,0.08); padding: 3px 10px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.05); }
.pulse-dot { width: 6px; height: 6px; background: #64748b; border-radius: 50%; margin-right: 8px; }
.status-text { font-size: 11px; font-weight: 600; color: #cbd5e1; letter-spacing: 0.05em; }
.version-badge { font-size: 11px; opacity: 0.4; letter-spacing: 0.05em; }

.hero-title { font-size: 20px; margin: 0 0 6px; font-weight: 600; color: white; letter-spacing: -0.02em; }
.hero-sub { font-size: 13px; color: #94a3b8; margin: 0; max-width: 380px; line-height: 1.5; }

.status-footer { margin-top: auto; display: flex; gap: 24px; }
.metric { display: flex; flex-direction: column; min-width: 80px; }
.metric .label { font-size: 10px; text-transform: uppercase; color: #64748b; margin-bottom: 4px; letter-spacing: 0.05em; }
.metric .value { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.metric-bar { height: 3px; width: 100%; background: #334155; border-radius: 2px; overflow: hidden; position: relative; }
.metric-bar.active::after { content:''; position: absolute; left: 0; top: 0; height: 100%; width: 60%; background: #22c55e; animation: barAnim 2s infinite ease-in-out; }
@keyframes barAnim { 0% { width: 30%; } 50% { width: 80%; } 100% { width: 30%; } }

.status-visual {
  position: absolute; right: -30px; bottom: -30px; width: 160px; height: 160px; pointer-events: none;
  .radar-circle { position: absolute; inset: 0; border: 1px solid rgba(255,255,255,0.03); border-radius: 50%; &.c2 { inset: 30px; border-style: dashed; } }
  .radar-scan { position: absolute; width: 80px; height: 80px; top: 50%; left: 50%; background: linear-gradient(45deg, transparent 50%, rgba(34, 197, 94, 0.08) 100%); transform-origin: top left; animation: radarSpin 4s linear infinite; }
}
@keyframes radarSpin { to { transform: rotate(360deg); } }

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
  background: #fdfdfd;
  &:hover { background: #fff; border-color: #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
}
.p-left { display: flex; align-items: center; gap: 10px; }
.p-icon { width: 20px; height: 20px; border-radius: 5px; background: white; padding: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.p-name { font-weight: 600; color: #334155; font-size: 13px; }
.status-pill {
  font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace;
  &.active { background: #dcfce7; color: #166534; }
  &.inactive { background: #f1f5f9; color: #94a3b8; }
}

/* Capabilities */
.capabilities-card { grid-column: span 12; }
.modules-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; }
.module-tile {
  background: white; border-radius: 12px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between; height: 90px;
  cursor: pointer; transition: all 0.2s ease; border: 1px solid #f1f5f9; position: relative; overflow: hidden;
  
  .tile-bg { position: absolute; inset: 0; opacity: 0; background: linear-gradient(135deg, rgba(255,83,80,0.05), transparent); transition: opacity 0.2s; }
  .tile-content { position: relative; z-index: 1; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
  .tile-icon { font-size: 20px; color: #94a3b8; transition: color 0.2s; }
  .tile-name { font-weight: 600; color: #475569; font-size: 12px; transition: color 0.2s; }
  .tile-status { font-size: 10px; color: #cbd5e1; font-weight: 700; margin-top: 2px; }
  .tile-bar { position: absolute; bottom: 0; left: 0; width: 100%; height: 3px; background: #e2e8f0; transition: background 0.2s; }
  
  &.active {
    border-color: rgba(255, 83, 80, 0.2); box-shadow: 0 4px 12px rgba(255, 83, 80, 0.05);
    .tile-bg { opacity: 1; }
    .tile-icon { color: var(--pokedex-red); }
    .tile-name { color: #0f172a; }
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
.theme-picker { display: flex; gap: 8px; }
.color-dot { width: 20px; height: 20px; border-radius: 50%; cursor: pointer; border: 2px solid white; box-shadow: 0 0 0 1px #e2e8f0; transition: transform 0.2s; &.selected { box-shadow: 0 0 0 2px #0f172a; transform: scale(1.1); } }
.checkbox-group { display: flex; gap: 16px; }
.divider-v { width: 1px; height: 32px; background: #f1f5f9; }

/* Ant Design Overrides */
:deep(.ant-input), :deep(.ant-select-selector) { background: transparent !important; font-size: 13px; }
:deep(.ant-checkbox-wrapper) { font-size: 13px; }
:deep(.tech-modal .ant-modal-content) { border-radius: 16px; padding: 24px; }
.tech-input { border-radius: 8px; background: #f8fafc !important; border-color: #e2e8f0 !important; &:focus { background: white !important; border-color: var(--pokedex-red) !important; } }
</style>
