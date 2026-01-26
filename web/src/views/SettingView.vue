<template>
  <div class="setting-page">
    <HeaderComponent
      title="洛托姆终端"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '控制中心' }]"
      class="setting-header"
    >
      <template #actions>
        <a-space>
          <a-button class="glass-btn" @click="refreshAll" :loading="state.refreshing"> 刷新系统 </a-button>
          <a-button class="glass-btn action" :disabled="!backendOnline" @click="restartBackend">
            重启服务
          </a-button>
        </a-space>
      </template>
    </HeaderComponent>

    <div class="setting-container layout-container">
      <div class="setting-body ui-page">
        <div class="ui-container">
          
          <!-- 1. Rotom Status Card (Hero) -->
          <div class="rotom-card" :class="{ online: backendOnline, offline: !backendOnline }">
            <div class="rotom-visual">
              <!-- Rotom SVG -->
              <svg class="rotom-svg" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <!-- Body Shadow -->
                <ellipse cx="100" cy="180" rx="60" ry="10" fill="rgba(0,0,0,0.2)" class="shadow-pulse"/>
                <!-- Body -->
                <path d="M100 20 C 60 20, 30 60, 30 110 C 30 160, 60 180, 100 180 C 140 180, 170 160, 170 110 C 170 60, 140 20, 100 20 Z" 
                      :fill="backendOnline ? '#ff5350' : '#546e7a'" 
                      class="rotom-body"/>
                <!-- Spike -->
                <path d="M100 20 L 100 5" stroke="#ff5350" stroke-width="8" stroke-linecap="round" v-if="backendOnline"/>
                <circle cx="100" cy="5" r="6" fill="#7cd6fd" class="rotom-spark" v-if="backendOnline"/>

                <!-- Face Screen -->
                <template v-if="backendOnline">
                   <!-- Happy Eyes -->
                   <ellipse cx="75" cy="90" rx="18" ry="25" fill="white"/>
                   <ellipse cx="75" cy="90" rx="8" ry="12" fill="#304ffe"/>
                   <ellipse cx="125" cy="90" rx="18" ry="25" fill="white"/>
                   <ellipse cx="125" cy="90" rx="8" ry="12" fill="#304ffe"/>
                   <!-- Smile -->
                   <path d="M85 130 Q 100 140, 115 130" stroke="white" stroke-width="4" fill="none"/>
                </template>
                <template v-else>
                   <!-- Sleeping Eyes -->
                   <path d="M60 100 Q 75 110, 90 100" stroke="#cfd8dc" stroke-width="4" fill="none"/>
                   <path d="M110 100 Q 125 110, 140 100" stroke="#cfd8dc" stroke-width="4" fill="none"/>
                   <!-- Zzz -->
                   <text x="150" y="50" font-family="sans-serif" font-size="24" fill="#cfd8dc" class="zzz-anim">Zzz...</text>
                </template>
              </svg>
            </div>
            
            <div class="rotom-status-text">
              <h2 class="status-title">{{ backendOnline ? '洛托姆已就绪！' : '正在休眠中...' }}</h2>
              <p class="status-sub">
                {{ backendOnline ? '所有系统运转正常，随时可以开始冒险。' : '后端服务未连接。请检查服务是否启动。' }}
              </p>
              <div class="status-meta">
                <span class="meta-pill" :class="backendReady ? 'green' : 'orange'">
                  API: {{ backendReady ? 'Ready' : 'Waiting' }}
                </span>
                <span class="meta-pill">v{{ APP_VERSION }}</span>
              </div>
            </div>
          </div>

          <div class="dashboard-grid">
            
            <!-- Left Column: Core Functions -->
            <div class="col-left">
              
              <!-- 2. AI Brain (Model) -->
              <section class="dashboard-section">
                <h3 class="section-title"><span class="icon">🧠</span> 核心模型</h3>
                <div class="glass-card padded-card">
                  <a-form layout="vertical" class="clean-form">
                    <a-row :gutter="16">
                      <a-col :span="12">
                        <a-form-item label="供应商">
                          <a-select v-model:value="modelProvider" @change="onProviderChange" class="modern-select">
                            <a-select-option v-for="p in providerKeys" :key="p" :value="p">
                              <span class="provider-option">
                                <img class="provider-option-icon" :src="getProviderIcon(p)" :alt="p" />
                                <span>{{ modelCatalog[p]?.name || p }}</span>
                              </span>
                            </a-select-option>
                          </a-select>
                        </a-form-item>
                      </a-col>
                      <a-col :span="12">
                        <a-form-item label="模型名称">
                          <a-select v-if="providerModels.length" v-model:value="modelName" @change="onModelChange" class="modern-select">
                            <a-select-option v-for="m in providerModels" :key="m" :value="m">{{ m }}</a-select-option>
                          </a-select>
                          <a-input v-else v-model:value="modelName" placeholder="输入模型名称" class="modern-input" @blur="commitModelInput"/>
                        </a-form-item>
                      </a-col>
                    </a-row>
                  </a-form>
                </div>
              </section>

              <!-- 3. Capabilities -->
              <section class="dashboard-section">
                <h3 class="section-title"><span class="icon">⚡</span> 扩展能力</h3>
                <div class="modules-grid">
                  <div class="module-card glass-card" v-for="mod in modulesList" :key="mod.key">
                    <div class="module-icon"><component :is="mod.icon" /></div>
                    <div class="module-info">
                      <h4>{{ mod.label }}</h4>
                      <span>{{ mod.desc }}</span>
                    </div>
                    <a-switch 
                      :checked="Boolean(configStore.config[mod.key])" 
                      :loading="Boolean(featureState.saving[mod.key])"
                      @change="(v) => setBackendFeature(mod.key, v)"
                    />
                  </div>
                </div>
              </section>

            </div>

            <!-- Right Column: Config & System -->
            <div class="col-right">
              
              <!-- 4. Providers -->
              <section class="dashboard-section">
                <div class="section-header">
                   <h3 class="section-title"><span class="icon">🔌</span> 连接配置</h3>
                   <a-button size="small" type="text" @click="refreshProviders" :loading="providersState.loading">刷新</a-button>
                </div>
                
                <div class="providers-list">
                  <div v-for="p in providerList" :key="p" class="provider-item glass-card" :class="{ active: modelProvider === p }">
                    <div class="p-row-main" @click="toggleProviderEdit(p)">
                      <img class="p-icon" :src="getProviderIcon(p)" :alt="p" />
                      <div class="p-info">
                        <div class="p-name">{{ modelCatalog[p]?.name || p }}</div>
                        <div class="p-status-text" :class="providerConfigured(p) ? 'ok' : 'missing'">
                          {{ providerConfigured(p) ? '已配置' : '未配置' }}
                        </div>
                      </div>
                      <EditOutlined class="edit-icon" />
                    </div>
                    
                    <!-- Expandable Config Area -->
                    <div v-if="providersState.editingId === p" class="p-config-area">
                      <div class="input-group">
                        <label>Base URL</label>
                        <a-input v-model:value="providerForm[p].api_base" size="small" :placeholder="modelCatalog[p]?.base_url || 'Default'" />
                      </div>
                      <div class="input-group">
                        <label>API Key</label>
                        <a-input-password v-model:value="providerForm[p].api_key" size="small" placeholder="输入新 Key 以更新" />
                      </div>
                      <div class="p-actions">
                         <a-button size="small" type="primary" @click="saveProvider(p)" :loading="Boolean(providersState.saving[p])">保存</a-button>
                         <a-button size="small" @click="providersState.editingId = null">取消</a-button>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              <!-- 5. UI Preferences -->
              <section class="dashboard-section">
                <h3 class="section-title"><span class="icon">🎨</span> 界面偏好</h3>
                <div class="glass-card padded-card">
                   <div class="ui-row">
                      <span>布局密度</span>
                      <a-segmented v-model:value="uiDensity" :options="densityOptions" @change="onUiDensityChange" size="small"/>
                   </div>
                   <div class="ui-row">
                      <span>主题色</span>
                      <div class="theme-dots">
                         <div v-for="(p, key) in THEME_PRESETS" :key="key" 
                           class="theme-dot-btn" :style="{ background: p.primary }"
                           :class="{ active: themePreset === key }"
                           @click="onThemePresetChange(key)"></div>
                      </div>
                   </div>
                   <div class="ui-row">
                      <span>导航显示</span>
                      <div class="mini-switches">
                         <a-tooltip title="知识库"><a-switch size="small" :checked="uiVisibility.show_knowledge_base" @change="(v)=>setUiVisibility('show_knowledge_base',v)"/></a-tooltip>
                         <a-tooltip title="图谱"><a-switch size="small" :checked="uiVisibility.show_knowledge_graph" @change="(v)=>setUiVisibility('show_knowledge_graph',v)"/></a-tooltip>
                         <a-tooltip title="地图"><a-switch size="small" :checked="uiVisibility.show_map" @change="(v)=>setUiVisibility('show_map',v)"/></a-tooltip>
                      </div>
                   </div>
                </div>
              </section>

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
  EditOutlined,
  BookFilled,
  DeploymentUnitOutlined,
  CompassFilled,
  ApiFilled,
  SortAscendingOutlined,
  AudioFilled,
  TagFilled
} from '@ant-design/icons-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import { useConfigStore } from '@/stores/config'
import { DEFAULT_CONFIG } from '@/config/defaultConfig'
import { APP_VERSION } from '@/config/appMeta'
import { ApiError, apiFetch } from '@/api/http'
import { getUiDensity, setUiDensity } from '@/utils/uiDensity'
import { THEME_PRESETS, getThemePreset, setThemePreset } from '@/utils/themePreset'
import { notifyApiError } from '@/utils/notify'
import { getProviderIcon } from '@/utils/providerIcon'

const configStore = useConfigStore()

const state = reactive({
  refreshing: false
})

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const backendReady = computed(() => Boolean(configStore.config.backend?.ready))

const uiDensity = ref(getUiDensity())
const densityOptions = [{ label: '舒适', value: 'comfortable' }, { label: '紧凑', value: 'compact' }]
const themePreset = ref(getThemePreset())

const modelCatalog = computed(() => configStore.config.model_names || {})
const providerKeys = computed(() => Object.keys(modelCatalog.value || {}).filter((k) => k !== 'custom'))
const modelProvider = ref(configStore.config.model_provider)
const modelName = ref(configStore.config.model_name)
const providerModels = computed(() => modelCatalog.value?.[modelProvider.value]?.models || [])

const uiVisibility = computed(() => ({ ...(DEFAULT_CONFIG.ui || {}), ...(configStore.config?.ui || {}) }))

const modulesList = [
  { key: 'enable_knowledge_base', label: '知识库', desc: 'RAG 检索', icon: BookFilled },
  { key: 'enable_knowledge_graph', label: '知识图谱', desc: 'Graph RAG', icon: DeploymentUnitOutlined },
  { key: 'enable_web_search', label: '联网搜索', desc: 'Web Search', icon: CompassFilled },
  { key: 'enable_mcp', label: 'MCP', desc: '工具扩展', icon: ApiFilled },
  { key: 'enable_reranker', label: '重排序', desc: 'Rerank', icon: SortAscendingOutlined },
  { key: 'enable_asr', label: '语音识别', desc: 'ASR', icon: AudioFilled },
  { key: 'enable_ner_bert', label: '实体识别', desc: 'NER', icon: TagFilled },
]

// Feature toggles UI state (per-switch saving indicator).
const featureState = reactive({
  saving: {}
})

// Providers Logic
const providersState = reactive({
  loading: false,
  saving: {},
  status: {},
  editingId: null
})
const providerForm = reactive({})
const providerList = computed(() => {
  const fromCatalog = providerKeys.value || []
  const fromStatus = Object.keys(providersState.status || {})
  return Array.from(new Set([...fromCatalog, ...fromStatus])).filter((k) => k && k !== 'custom')
})

const ensureProviderForm = (p) => {
  if (!providerForm[p]) providerForm[p] = { api_base: '', api_key: '' }
  if (!providerForm[p].api_base) providerForm[p].api_base = providersState.status?.[p]?.api_base || modelCatalog.value?.[p]?.base_url || ''
}

const toggleProviderEdit = (p) => {
  if (providersState.editingId === p) providersState.editingId = null
  else {
    ensureProviderForm(p)
    providersState.editingId = p
  }
}

const providerConfigured = (provider) => {
  if (!backendOnline.value) return null
  const st = providersState.status?.[provider]
  if (st && typeof st.configured === 'boolean') return Boolean(st.configured)
  return null
}

// Actions
const setUiVisibility = (key, checked) => {
  configStore.patchLocal({ ui: { ...uiVisibility.value, [key]: Boolean(checked) } })
}
const onProviderChange = async (p) => {
  const models = modelCatalog.value?.[p]?.models || []
  const def = modelCatalog.value?.[p]?.default || models?.[0] || modelName.value || ''
  await configStore.setConfigValues({ model_provider: p, ...(def ? { model_name: def } : {}) })
}
const onModelChange = async (m) => await configStore.setConfigValue('model_name', m)
const commitModelInput = async () => { if (modelName.value) await configStore.setConfigValue('model_name', modelName.value) }
const onUiDensityChange = (v) => { uiDensity.value = setUiDensity(v) }
const onThemePresetChange = (v) => { themePreset.value = setThemePreset(v) }

const refreshAll = async () => {
  state.refreshing = true
  try { await configStore.refreshConfig(); await refreshProviders(); } finally { state.refreshing = false }
}
const restartBackend = async () => {
  if (!backendOnline.value) return
  try { await apiFetch('/restart', { method: 'POST', timeoutMs: 10000 }); message.success('已触发重启'); await refreshAll() } catch (e) { notifyApiError(e, { context: '重启', fallback: '失败' }) }
}
const setBackendFeature = async (key, checked) => {
  featureState.saving[key] = true
  try {
    const res = await apiFetch('/config', { method: 'PATCH', body: { [key]: Boolean(checked) }, timeoutMs: 10000 })
    configStore.patchLocal({ ...res, backend: { online: true, last_error: null, ...(res?.backend || {}) } })
    if (Boolean(res?.[key]) !== Boolean(checked)) message.warning('未生效')
    else message.success('已更新')
  } catch (e) { notifyApiError(e, { context: '更新', fallback: '失败' }) } 
  finally { featureState.saving[key] = false }
}
const refreshProviders = async () => {
  providersState.loading = true
  try {
    const res = await apiFetch('/providers', { method: 'GET', timeoutMs: 8000 })
    providersState.status = res?.providers || {}
    providerList.value.forEach(ensureProviderForm)
  } catch (e) { if (!(e instanceof ApiError && e.status === 404)) notifyApiError(e, { context: 'Providers', fallback: '获取失败' }) }
  finally { providersState.loading = false }
}
const saveProvider = async (provider) => {
  providersState.saving[provider] = true
  try {
    const body = { provider, ...(providerForm[provider].api_base ? { api_base: providerForm[provider].api_base } : {}), ...(providerForm[provider].api_key ? { api_key: providerForm[provider].api_key } : {}) }
    const res = await apiFetch('/providers', { method: 'PATCH', body, timeoutMs: 10000 })
    providersState.status = res?.providers || providersState.status
    providerForm[provider].api_key = ''
    providersState.editingId = null
    message.success('已保存')
  } catch (e) { notifyApiError(e, { context: '保存', fallback: '失败' }) }
  finally { providersState.saving[provider] = false }
}

// Init
watch(() => configStore.config.model_provider, (v) => (modelProvider.value = v))
watch(() => configStore.config.model_name, (v) => (modelName.value = v))
onMounted(() => { refreshProviders() })
</script>

<style scoped lang="less">
.setting-container { padding: 0; }

/* Rotom Card */
.rotom-card {
  display: flex;
  align-items: center;
  padding: 24px;
  border-radius: 24px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
  transition: all 0.5s ease;
  
  &.online {
    background: linear-gradient(135deg, #fff5f5 0%, #fff 100%);
    border: 1px solid rgba(255, 83, 80, 0.2);
    box-shadow: 0 10px 30px rgba(255, 83, 80, 0.1);
    .status-title { color: var(--pokedex-red); }
  }
  &.offline {
    background: linear-gradient(135deg, #f0f2f5 0%, #e0e0e0 100%);
    border: 1px solid rgba(0,0,0,0.05);
    .status-title { color: var(--gray-600); }
  }

  .rotom-visual {
    width: 100px;
    height: 100px;
    margin-right: 32px;
    flex-shrink: 0;
    
    .rotom-svg {
      width: 100%; height: 100%;
      filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
    }
    .rotom-body { transition: fill 0.5s ease; }
    .shadow-pulse { animation: pulseShadow 2s infinite; }
    .rotom-spark { animation: spark 1.5s infinite; }
    .zzz-anim { animation: floatUp 3s infinite; opacity: 0.6; }
  }

  .rotom-status-text {
    flex: 1;
    .status-title { margin: 0 0 8px; font-size: 24px; font-weight: 700; }
    .status-sub { margin: 0 0 16px; color: var(--gray-600); }
    .status-meta { display: flex; gap: 8px; }
    .meta-pill {
      background: rgba(0,0,0,0.05); padding: 4px 12px; border-radius: 99px; font-size: 12px; font-weight: 600; color: var(--gray-600);
      &.green { background: #e6f7ff; color: #1890ff; }
      &.orange { background: #fff7e6; color: #fa8c16; }
    }
  }
}

@keyframes pulseShadow { 0%, 100% { rx: 60; opacity: 0.2; } 50% { rx: 55; opacity: 0.1; } }
@keyframes spark { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
@keyframes floatUp { 0% { transform: translateY(0); opacity: 0.6; } 100% { transform: translateY(-20px); opacity: 0; } }

/* Grid Layout */
.dashboard-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
  
  @media (min-width: 900px) {
    grid-template-columns: 3fr 2fr; /* Left wider */
  }
}

.section-title {
  margin: 0 0 16px;
  font-size: 16px;
  font-weight: 700;
  color: var(--gray-800);
  display: flex;
  align-items: center;
  .icon { margin-right: 8px; font-size: 18px; }
}

.dashboard-section { margin-bottom: 24px; }

/* Modules */
.modules-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 16px;
}
.module-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 16px;
  
  .module-icon { 
    font-size: 24px; margin-bottom: 8px; 
    color: var(--pokedex-red);
    background: rgba(255, 83, 80, 0.1);
    width: 48px; height: 48px; border-radius: 12px;
    display: flex; justify-content: center; align-items: center;
  }
  .module-info { margin-bottom: 12px; h4 { font-size: 14px; margin: 0; } span { font-size: 11px; color: var(--gray-500); } }
}

/* Providers List */
.providers-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.provider-item {
  padding: 12px 16px;
  cursor: pointer;
  border: 1px solid transparent;
  &:hover { border-color: var(--pokedex-red); }
  
  .p-row-main { display: flex; align-items: center; }
  .p-icon { width: 24px; height: 24px; border-radius: 6px; margin-right: 12px; }
  .p-info { flex: 1; 
    .p-name { font-weight: 600; font-size: 14px; }
    .p-status-text { font-size: 11px; &.ok { color: #52c41a; } &.missing { color: var(--gray-400); } }
  }
  .edit-icon { color: var(--gray-400); }
  
  .p-config-area {
    margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(0,0,0,0.05);
    .input-group { margin-bottom: 8px; label { font-size: 11px; color: var(--gray-500); display: block; } }
    .p-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 8px; }
  }
}

/* UI Settings */
.ui-row {
  display: flex; justify-content: space-between; align-items: center; padding: 10px 0;
  border-bottom: 1px solid rgba(0,0,0,0.05);
  &:last-child { border-bottom: none; }
  span { font-size: 14px; font-weight: 500; }
}
.theme-dots { display: flex; gap: 6px; }
.theme-dot-btn { width: 18px; height: 18px; border-radius: 50%; cursor: pointer; border: 2px solid transparent; &.active { transform: scale(1.2); border-color: var(--text-color); } }
.mini-switches { display: flex; gap: 8px; }

/* Utilities */
.glass-card {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
  border-radius: 16px;
  transition: all 0.2s;
  &:hover { box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06); }
}
.padded-card { padding: 20px; }
.clean-form .ant-form-item { margin-bottom: 12px; }
.modern-select, .modern-input { width: 100%; border-radius: 8px; }
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; .section-title { margin: 0; } }
.glass-btn { background: white; border: 1px solid #eee; border-radius: 8px; &:hover { color: var(--pokedex-red); border-color: var(--pokedex-red); } }
.action-btn { background: var(--pokedex-red); border-color: var(--pokedex-red); border-radius: 8px; &:hover { background: #ff7875; border-color: #ff7875; } }
.provider-option { display: flex; align-items: center; gap: 8px; img { width: 16px; height: 16px; } }
</style>
