<template>
  <div class="setting-page">
    <HeaderComponent
      title="控制中心"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '设置' }]"
      class="setting-header"
    >
      <template #actions>
        <a-space>
          <a-button class="glass-btn" @click="refreshAll" :loading="state.refreshing"> 刷新 </a-button>
          <a-button type="primary" class="action-btn" :disabled="!backendOnline" @click="restartBackend">
            重启服务
          </a-button>
        </a-space>
      </template>
    </HeaderComponent>

    <div class="setting-container layout-container">
      <div class="setting-body ui-page">
        <div class="ui-container">
          
          <!-- Modern Tabs -->
          <a-tabs v-model:activeKey="activeTab" class="modern-tabs" :animated="false">
            
            <!-- Tab: Dashboard (Status) -->
            <a-tab-pane key="status" tab="仪表盘">
              <!-- Hero Status Card -->
              <div class="hero-status-card" :class="{ online: backendOnline, offline: !backendOnline }">
                <div class="status-icon-area">
                  <div class="status-pulse"></div>
                  <component :is="backendOnline ? CheckCircleFilled : CloseCircleFilled" class="status-icon" />
                </div>
                <div class="status-info">
                  <h2 class="status-title">{{ backendOnline ? '服务已连接' : '服务未连接' }}</h2>
                  <p class="status-sub">
                    {{ backendOnline ? '系统运行正常，所有模块已就绪' : '无法连接到后端服务，请检查网络或重启后端' }}
                  </p>
                  <div class="meta-tags">
                    <span class="meta-tag">{{ APP_NAME }} v{{ APP_VERSION }}</span>
                    <span v-if="BUILD_SHA" class="meta-tag">Build {{ BUILD_SHA.slice(0, 7) }}</span>
                  </div>
                </div>
              </div>

              <div class="section-grid">
                <div class="info-card glass-card">
                  <div class="card-icon"><CloudServerOutlined /></div>
                  <div class="card-content">
                    <h3>后端检查</h3>
                    <p class="card-value">{{ backendReady ? 'Ready' : 'Initializing...' }}</p>
                  </div>
                </div>
                <div class="info-card glass-card">
                  <div class="card-icon"><safety-certificate-outlined /></div>
                  <div class="card-content">
                    <h3>安全状态</h3>
                    <p class="card-value">已加密</p>
                  </div>
                </div>
              </div>
            </a-tab-pane>

            <!-- Tab: AI Models -->
            <a-tab-pane key="model" tab="模型与服务">
              <a-row :gutter="[24, 24]">
                <a-col :xs="24" :md="12">
                  <div class="glass-card padded-card">
                    <h3 class="card-title">当前模型</h3>
                    <a-form layout="vertical" class="clean-form">
                      <a-form-item label="供应商">
                        <a-select v-model:value="modelProvider" @change="onProviderChange" class="modern-select">
                          <a-select-option v-for="p in providerKeys" :key="p" :value="p">
                            <span class="provider-option">
                              <img class="provider-option-icon" :src="getProviderIcon(p)" :alt="p" />
                               <span>{{ modelCatalog[p]?.name || p }}</span>
                               <span class="spacer" />
                               <div class="status-dot-mini" :class="providerConfigured(p) ? 'green' : 'gray'"></div>
                             </span>
                           </a-select-option>
                         </a-select>
                      </a-form-item>
                      <a-form-item label="模型名称">
                        <a-select v-if="providerModels.length" v-model:value="modelName" @change="onModelChange" class="modern-select">
                          <a-select-option v-for="m in providerModels" :key="m" :value="m">
                            {{ m }}
                          </a-select-option>
                        </a-select>
                        <a-input
                          v-else
                          v-model:value="modelName"
                          placeholder="输入模型名称"
                          class="modern-input"
                          @pressEnter="commitModelInput"
                          @blur="commitModelInput"
                        />
                      </a-form-item>
                    </a-form>
                  </div>
                </a-col>

                <a-col :xs="24" :md="12">
                  <div class="glass-card padded-card">
                    <h3 class="card-title">快速操作</h3>
                    <div class="quick-actions">
                         <a-button block class="glass-btn" @click="activeTab = 'providers'">配置供应商 Key</a-button>
                         <a-button block class="glass-btn" @click="refreshProviders" :loading="providersState.loading">刷新服务列表</a-button>
                    </div>
                  </div>
                </a-col>
              </a-row>
            </a-tab-pane>

            <!-- Tab: Providers -->
            <a-tab-pane key="providers" tab="供应商配置">
                <div class="provider-grid">
                  <div v-for="p in providerList" :key="p" class="provider-card-modern glass-card">
                    <div class="provider-header">
                        <img class="p-icon" :src="getProviderIcon(p)" :alt="p" />
                        <div class="p-info">
                          <div class="p-name">{{ modelCatalog[p]?.name || p }}</div>
                          <div class="p-status">
                             <span class="status-dot-mini" :class="providerConfigured(p) ? 'green' : 'red'"></span>
                             {{ providerConfigured(p) ? '已配置' : '未配置' }}
                          </div>
                        </div>
                    </div>
                    
                    <div class="provider-body">
                      <div class="input-group">
                        <label>Base URL</label>
                        <a-input
                          v-model:value="providerForm[p].api_base"
                          :placeholder="modelCatalog[p]?.base_url || 'https://...'"
                          class="modern-input small"
                        />
                      </div>
                      <div class="input-group">
                        <label>API Key</label>
                        <div v-if="providersState.status?.[p]?.configured && !providersState.editingKey?.[p]" class="key-mask" @click="enableEditKey(p)">
                           <span class="mask-dots">••••••••••••••</span>
                           <EditOutlined />
                        </div>
                        <a-input-password
                          v-else
                          v-model:value="providerForm[p].api_key"
                          class="modern-input small"
                          placeholder="输入 Key"
                        />
                      </div>
                    </div>

                    <div class="provider-footer">
                        <a-button
                          type="text"
                          size="small"
                          class="action-text"
                          :disabled="!backendOnline || !hasProviderChanges(p)"
                          :loading="Boolean(providersState.saving?.[p])"
                          @click="saveProvider(p)"
                        >
                          保存
                        </a-button>
                        <a-button
                          type="text"
                          danger
                          size="small"
                          class="action-text"
                          :disabled="providersState.status?.[p]?.source !== 'file'"
                          :loading="Boolean(providersState.saving?.[p])"
                          @click="clearProvider(p)"
                        >
                          清空
                        </a-button>
                    </div>
                  </div>
                </div>
            </a-tab-pane>

            <!-- Tab: Capabilities (Modules) -->
            <a-tab-pane key="capabilities" tab="功能模块">
              <div class="modules-grid">
                <!-- Knowledge Base -->
                <div class="module-card glass-card">
                  <div class="module-icon"><BookFilled /></div>
                  <div class="module-info">
                    <h4>知识库</h4>
                    <span>RAG 检索增强</span>
                  </div>
                  <a-switch
                    :checked="Boolean(configStore.config.enable_knowledge_base)"
                    :loading="Boolean(featureState.saving.enable_knowledge_base)"
                    @change="(v) => setBackendFeature('enable_knowledge_base', v)"
                  />
                </div>

                <!-- Knowledge Graph -->
                <div class="module-card glass-card">
                  <div class="module-icon"><DeploymentUnitOutlined /></div>
                  <div class="module-info">
                    <h4>知识图谱</h4>
                    <span>Graph RAG</span>
                  </div>
                  <a-switch
                    :checked="Boolean(configStore.config.enable_knowledge_graph)"
                    :loading="Boolean(featureState.saving.enable_knowledge_graph)"
                    @change="(v) => setBackendFeature('enable_knowledge_graph', v)"
                  />
                </div>

                <!-- Web Search -->
                <div class="module-card glass-card">
                  <div class="module-icon"><CompassFilled /></div>
                  <div class="module-info">
                    <h4>联网搜索</h4>
                    <span>Web Search</span>
                  </div>
                  <a-switch
                    :checked="Boolean(configStore.config.enable_web_search)"
                    :loading="Boolean(featureState.saving.enable_web_search)"
                    @change="(v) => setBackendFeature('enable_web_search', v)"
                  />
                </div>

                <!-- MCP -->
                <div class="module-card glass-card">
                  <div class="module-icon"><ApiFilled /></div>
                  <div class="module-info">
                    <h4>MCP 协议</h4>
                    <span>工具扩展</span>
                  </div>
                  <a-switch
                    :checked="Boolean(configStore.config.enable_mcp)"
                    :loading="Boolean(featureState.saving.enable_mcp)"
                    @change="(v) => setBackendFeature('enable_mcp', v)"
                  />
                </div>
                
                 <!-- Reranker -->
                <div class="module-card glass-card">
                  <div class="module-icon"><SortAscendingOutlined /></div>
                  <div class="module-info">
                    <h4>重排序</h4>
                    <span>Rerank Model</span>
                  </div>
                  <a-switch
                    :checked="Boolean(configStore.config.enable_reranker)"
                    :loading="Boolean(featureState.saving.enable_reranker)"
                    @change="(v) => setBackendFeature('enable_reranker', v)"
                  />
                </div>
              </div>
            </a-tab-pane>

            <!-- Tab: UI -->
            <a-tab-pane key="ui" tab="界面偏好">
               <a-row :gutter="[24, 24]">
                 <a-col :xs="24" :md="12">
                    <div class="glass-card padded-card">
                      <h3 class="card-title">外观</h3>
                      <div class="ui-setting-row">
                        <span>布局密度</span>
                        <a-segmented
                          v-model:value="uiDensity"
                          :options="densityOptions"
                          @change="onUiDensityChange"
                        />
                      </div>
                      <div class="ui-setting-row">
                        <span>主题色</span>
                        <div class="theme-dots">
                           <div 
                             v-for="(p, key) in THEME_PRESETS" 
                             :key="key" 
                             class="theme-dot-btn"
                             :style="{ background: p.primary }"
                             :class="{ active: themePreset === key }"
                             @click="onThemePresetChange(key)"
                             :title="p.label"
                           ></div>
                        </div>
                      </div>
                    </div>
                 </a-col>
                 <a-col :xs="24" :md="12">
                    <div class="glass-card padded-card">
                      <h3 class="card-title">导航可见性</h3>
                      <div class="switches-list">
                        <div class="switch-row">
                          <span>知识库入口</span>
                          <a-switch :checked="uiVisibility.show_knowledge_base" @change="(v) => setUiVisibility('show_knowledge_base', v)" />
                        </div>
                        <div class="switch-row">
                          <span>图谱入口</span>
                          <a-switch :checked="uiVisibility.show_knowledge_graph" @change="(v) => setUiVisibility('show_knowledge_graph', v)" />
                        </div>
                        <div class="switch-row">
                          <span>地图入口</span>
                          <a-switch :checked="uiVisibility.show_map" @change="(v) => setUiVisibility('show_map', v)" />
                        </div>
                      </div>
                    </div>
                 </a-col>
               </a-row>
            </a-tab-pane>
          </a-tabs>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { message } from 'ant-design-vue'
import { 
  CheckCircleFilled, 
  CloseCircleFilled, 
  CloudServerOutlined, 
  SafetyCertificateOutlined,
  EditOutlined,
  BookFilled,
  DeploymentUnitOutlined,
  CompassFilled,
  ApiFilled,
  SortAscendingOutlined
} from '@ant-design/icons-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import { useConfigStore } from '@/stores/config'
import { DEFAULT_CONFIG } from '@/config/defaultConfig'
import { APP_NAME, APP_VERSION, BUILD_SHA, BUILD_TIME } from '@/config/appMeta'
import { ApiError, apiFetch } from '@/api/http'
import { getUiDensity, setUiDensity } from '@/utils/uiDensity'
import { THEME_PRESETS, getThemePreset, setThemePreset } from '@/utils/themePreset'
import { notifyApiError } from '@/utils/notify'
import { getProviderIcon } from '@/utils/providerIcon'

const configStore = useConfigStore()

const state = reactive({
  refreshing: false
})

const activeTab = ref('status')

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const backendReady = computed(() => Boolean(configStore.config.backend?.ready))

const uiDensity = ref(getUiDensity())
const densityOptions = [
  { label: '舒适', value: 'comfortable' },
  { label: '紧凑', value: 'compact' }
]

const themePreset = ref(getThemePreset())

const modelCatalog = computed(() => configStore.config.model_names || {})
const providerKeys = computed(() =>
  Object.keys(modelCatalog.value || {}).filter((k) => k !== 'custom')
)

const modelProvider = ref(configStore.config.model_provider)
const modelName = ref(configStore.config.model_name)

const uiVisibility = computed(() => ({
  ...(DEFAULT_CONFIG.ui || {}),
  ...(configStore.config?.ui || {})
}))

const setUiVisibility = (key, checked) => {
  const next = { ...uiVisibility.value, [key]: Boolean(checked) }
  configStore.patchLocal({ ui: next })
}

const resetUiVisibility = () => {
  configStore.patchLocal({ ui: { ...(DEFAULT_CONFIG.ui || {}) } })
  message.success('已重置界面展示开关')
}

watch(
  () => configStore.config.model_provider,
  (v) => (modelProvider.value = v)
)
watch(
  () => configStore.config.model_name,
  (v) => (modelName.value = v)
)

const providerModels = computed(() => modelCatalog.value?.[modelProvider.value]?.models || [])

const providerConfigured = (provider) => {
  if (!backendOnline.value) return null
  const st = providersState.status?.[provider]
  if (st && typeof st.configured === 'boolean') return Boolean(st.configured)
  return null
}

const normalizeApiBase = (v) => String(v || '').trim().replace(/\/+$/, '')
const hasProviderChanges = (provider) => {
  ensureProviderForm(provider)
  const form = providerForm[provider] || {}
  const st = providersState.status?.[provider] || {}
  const baseChanged = normalizeApiBase(form.api_base) && normalizeApiBase(form.api_base) !== normalizeApiBase(st.api_base)
  const keyChanged = Boolean(String(form.api_key || '').trim())
  return baseChanged || keyChanged
}

const providersState = reactive({
  loading: false,
  saving: {},
  status: {},
  editingKey: {}
})

const providerForm = reactive({})
const providerList = computed(() => {
  const fromCatalog = providerKeys.value || []
  const fromStatus = Object.keys(providersState.status || {})
  const merged = Array.from(new Set([...fromCatalog, ...fromStatus])).filter((k) => k && k !== 'custom')
  return merged
})

const ensureProviderForm = (p) => {
  if (!providerForm[p]) {
    providerForm[p] = { api_base: '', api_key: '' }
  }
  if (!providerForm[p].api_base) {
    providerForm[p].api_base =
      providersState.status?.[p]?.api_base || modelCatalog.value?.[p]?.base_url || ''
  }
}

const onProviderChange = async (p) => {
  const models = modelCatalog.value?.[p]?.models || []
  const def = modelCatalog.value?.[p]?.default || models?.[0] || modelName.value || ''
  await configStore.setConfigValues({ model_provider: p, ...(def ? { model_name: def } : {}) })
}

const onModelChange = async (m) => {
  await configStore.setConfigValue('model_name', m)
}

const commitModelInput = async () => {
  if (!modelName.value) return
  await configStore.setConfigValue('model_name', modelName.value)
}

const refreshAll = async () => {
  state.refreshing = true
  try {
    await configStore.refreshConfig()
  } finally {
    state.refreshing = false
  }
}

const restartBackend = async () => {
  if (!backendOnline.value) return
  try {
    await apiFetch('/restart', { method: 'POST', timeoutMs: 10000 })
    message.success('已触发后端重启')
    await refreshAll()
  } catch (e) {
    notifyApiError(e, { context: '后端重启', fallback: '后端重启失败' })
  }
}

const featureState = reactive({
  saving: {}
})

const setBackendFeature = async (key, checked) => {
  featureState.saving[key] = true
  const prev = Boolean(configStore.config?.[key])
  // Optimistic UI
  configStore.patchLocal({ [key]: Boolean(checked) })
  try {
    const res = await apiFetch('/config', {
      method: 'PATCH',
      body: { [key]: Boolean(checked) },
      timeoutMs: 10000
    })
    configStore.patchLocal({
      ...res,
      backend: { online: true, last_error: null, ...(res?.backend || {}) }
    })
    const effective = Boolean(res?.[key])
    if (effective !== Boolean(checked)) {
      message.warning('后端未应用该开关')
    } else {
      message.success('已更新开关')
    }
  } catch (e) {
    // Revert optimistic change.
    configStore.patchLocal({ [key]: prev })
    notifyApiError(e, { context: '更新配置', fallback: '更新失败' })
  } finally {
    featureState.saving[key] = false
  }
}
	
const refreshProviders = async () => {
  providersState.loading = true
  try {
    const res = await apiFetch('/providers', { method: 'GET', timeoutMs: 8000 })
    providersState.status = res?.providers || {}
    providerList.value.forEach((p) => ensureProviderForm(p))
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      return
    }
    notifyApiError(e, { context: 'Provider 配置', fallback: '获取 Provider 状态失败' })
  } finally {
    providersState.loading = false
  }
}

const enableEditKey = (provider) => {
  providersState.editingKey[provider] = true
  ensureProviderForm(provider)
  providerForm[provider].api_key = ''
}

const saveProvider = async (provider) => {
  ensureProviderForm(provider)
  const form = providerForm[provider] || {}
  const body = {
    provider,
    ...(form.api_base ? { api_base: form.api_base } : {}),
    ...(form.api_key ? { api_key: form.api_key } : {})
  }

  providersState.saving[provider] = true
  try {
    const res = await apiFetch('/providers', { method: 'PATCH', body, timeoutMs: 10000 })
    providersState.status = res?.providers || providersState.status
    providerForm[provider].api_key = '' 
    providersState.editingKey[provider] = false
    message.success('已保存 Provider 配置')
  } catch (e) {
    notifyApiError(e, { context: 'Provider 配置', fallback: '保存失败' })
  } finally {
    providersState.saving[provider] = false
  }
}

const clearProvider = async (provider) => {
  ensureProviderForm(provider)
  providersState.saving[provider] = true
  try {
    const res = await apiFetch('/providers', {
      method: 'PATCH',
      body: { provider, api_key: '', api_base: '' },
      timeoutMs: 10000
    })
    providersState.status = res?.providers || providersState.status
    providerForm[provider].api_key = ''
    providerForm[provider].api_base = modelCatalog.value?.[provider]?.base_url || ''
    providersState.editingKey[provider] = false
    message.success('已清空 Provider 配置')
  } catch (e) {
    notifyApiError(e, { context: 'Provider 配置', fallback: '清空失败' })
  } finally {
    providersState.saving[provider] = false
  }
}

const onUiDensityChange = (v) => {
  uiDensity.value = setUiDensity(v)
  message.success(`已切换界面密度`)
}

const onThemePresetChange = (v) => {
  themePreset.value = setThemePreset(v)
  message.success(`已切换主题色`)
}

watch(
  () => activeTab.value,
  (tab) => {
    if ((tab === 'providers' || tab === 'model') && backendOnline.value && !providersState.loading) {
      refreshProviders()
    }
  }
)

watch(
  () => backendOnline.value,
  (on) => {
    if (!on) return
    if (providersState.loading) return
    if (Object.keys(providersState.status || {}).length) return
    refreshProviders()
  }
)

onMounted(() => {
  providerList.value.forEach((p) => ensureProviderForm(p))
})
</script>

<style scoped lang="less">
.setting-container {
  padding: 0;
}

/* Glassmorphism Utilities */
.glass-card {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
  border-radius: 20px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  overflow: hidden;
  
  &:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
  }
}

.glass-btn {
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid rgba(0,0,0,0.1);
  border-radius: 8px;
  
  &:hover {
    background: #fff;
    border-color: var(--pokedex-red);
    color: var(--pokedex-red);
  }
}

.action-btn {
  background: var(--pokedex-red);
  border-color: var(--pokedex-red);
  box-shadow: 0 4px 12px rgba(255, 83, 80, 0.3);
  border-radius: 8px;
  
  &:hover {
    background: color-mix(in srgb, var(--pokedex-red), white 10%);
    transform: translateY(-1px);
  }
}

/* Tabs Styling */
.modern-tabs :deep(.ant-tabs-nav) {
  margin-bottom: 24px;
}
.modern-tabs :deep(.ant-tabs-tab) {
  padding: 10px 20px;
  border-radius: 99px;
  font-size: 15px;
  transition: all 0.3s ease;
  margin: 0 4px;
}
.modern-tabs :deep(.ant-tabs-tab-active) {
  background: white;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.modern-tabs :deep(.ant-tabs-ink-bar) {
  display: none; /* Hide standard underline */
}

/* Hero Status Card */
.hero-status-card {
  display: flex;
  align-items: center;
  padding: 32px;
  border-radius: 24px;
  margin-bottom: 24px;
  color: #fff;
  position: relative;
  overflow: hidden;
  
  &.online {
    background: linear-gradient(135deg, #388e3c, #66bb6a);
    box-shadow: 0 10px 30px rgba(56, 142, 60, 0.3);
  }
  &.offline {
    background: linear-gradient(135deg, #d32f2f, #ef5350);
    box-shadow: 0 10px 30px rgba(211, 47, 47, 0.3);
  }

  .status-icon-area {
    position: relative;
    margin-right: 24px;
    font-size: 48px;
    display: flex;
  }
  
  .status-info {
    z-index: 1;
  }
  .status-title {
    font-size: 24px;
    margin: 0;
    font-weight: 700;
    color: #fff;
  }
  .status-sub {
    margin: 4px 0 12px;
    opacity: 0.9;
    font-size: 15px;
  }
  .meta-tags {
    display: flex;
    gap: 8px;
  }
  .meta-tag {
    background: rgba(255,255,255,0.2);
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
  }
}

/* Section Grid */
.section-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 24px;
}

.info-card {
  display: flex;
  align-items: center;
  padding: 20px;
  
  .card-icon {
    font-size: 24px;
    color: var(--gray-500);
    margin-right: 16px;
    background: var(--surface-color-2);
    padding: 12px;
    border-radius: 12px;
  }
  
  h3 {
    margin: 0;
    font-size: 14px;
    color: var(--gray-600);
  }
  .card-value {
    margin: 4px 0 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-color);
  }
}

/* Providers Grid */
.provider-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.provider-card-modern {
  padding: 20px;
  display: flex;
  flex-direction: column;
  
  .provider-header {
    display: flex;
    align-items: center;
    margin-bottom: 16px;
    .p-icon {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      margin-right: 12px;
    }
    .p-info {
      flex: 1;
    }
    .p-name {
      font-weight: 600;
      font-size: 15px;
    }
    .p-status {
      font-size: 12px;
      color: var(--gray-500);
      display: flex;
      align-items: center;
      gap: 4px;
    }
  }
  
  .provider-body {
    flex: 1;
  }
  
  .input-group {
    margin-bottom: 12px;
    label {
      display: block;
      font-size: 12px;
      color: var(--gray-500);
      margin-bottom: 4px;
    }
  }
  
  .provider-footer {
    margin-top: 16px;
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
}

.key-mask {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--surface-color-2);
  padding: 6px 12px;
  border-radius: 8px;
  cursor: pointer;
  color: var(--gray-600);
  &:hover {
    color: var(--pokedex-red);
    background: rgba(255, 83, 80, 0.05);
  }
  .mask-dots {
    letter-spacing: 2px;
    font-size: 10px;
  }
}

/* Modules Grid */
.modules-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

.module-card {
  display: flex;
  align-items: center;
  padding: 20px;
  
  .module-icon {
    font-size: 20px;
    color: var(--pokedex-red);
    background: rgba(255, 83, 80, 0.1);
    width: 40px;
    height: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 10px;
    margin-right: 16px;
  }
  
  .module-info {
    flex: 1;
    h4 {
      margin: 0;
      font-size: 15px;
      font-weight: 600;
    }
    span {
      font-size: 12px;
      color: var(--gray-500);
    }
  }
}

.modern-input {
  border-radius: 8px;
  background: rgba(255,255,255,0.5);
  &.small { font-size: 13px; }
  &:focus {
    background: #fff;
  }
}

.padded-card {
  padding: 24px;
}

.card-title {
  margin-top: 0;
  margin-bottom: 20px;
  font-size: 16px;
  font-weight: 600;
  color: var(--gray-800);
}

.status-dot-mini {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  &.green { background: #52c41a; }
  &.red { background: #ff4d4f; }
  &.gray { background: #d9d9d9; }
}

.provider-option {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}
.provider-option-icon {
  width: 16px; height: 16px; border-radius: 4px; object-fit: contain;
}
.spacer { flex: 1; }

.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.ui-setting-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid rgba(0,0,0,0.05);
  &:last-child { border-bottom: none; }
}

.theme-dots {
  display: flex;
  gap: 8px;
}
.theme-dot-btn {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.2s;
  &.active {
    transform: scale(1.2);
    border-color: var(--text-color);
  }
}

.switches-list {
  display: flex;
  flex-direction: column;
}
.switch-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
}
</style>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { message } from 'ant-design-vue'
import HeaderComponent from '@/components/HeaderComponent.vue'
import StatusTag from '@/components/StatusTag.vue'
import { useConfigStore } from '@/stores/config'
import { DEFAULT_CONFIG } from '@/config/defaultConfig'
import { APP_NAME, APP_VERSION, BUILD_SHA, BUILD_TIME } from '@/config/appMeta'
import { ApiError, apiFetch } from '@/api/http'
import { getUiDensity, setUiDensity } from '@/utils/uiDensity'
import { THEME_PRESETS, getThemePreset, setThemePreset } from '@/utils/themePreset'
import { notifyApiError } from '@/utils/notify'
import { getProviderIcon } from '@/utils/providerIcon'

const configStore = useConfigStore()

const state = reactive({
  refreshing: false
})

const activeTab = ref('status')

const backendOnline = computed(() => Boolean(configStore.config.backend?.online))
const backendReady = computed(() => Boolean(configStore.config.backend?.ready))

const uiDensity = ref(getUiDensity())
const densityOptions = [
  { label: '舒适', value: 'comfortable' },
  { label: '紧凑', value: 'compact' }
]

const themePreset = ref(getThemePreset())

const modelCatalog = computed(() => configStore.config.model_names || {})
const providerKeys = computed(() =>
  Object.keys(modelCatalog.value || {}).filter((k) => k !== 'custom')
)

const modelProvider = ref(configStore.config.model_provider)
const modelName = ref(configStore.config.model_name)

const uiVisibility = computed(() => ({
  ...(DEFAULT_CONFIG.ui || {}),
  ...(configStore.config?.ui || {})
}))

const setUiVisibility = (key, checked) => {
  const next = { ...uiVisibility.value, [key]: Boolean(checked) }
  configStore.patchLocal({ ui: next })
}

const resetUiVisibility = () => {
  configStore.patchLocal({ ui: { ...(DEFAULT_CONFIG.ui || {}) } })
  message.success('已重置界面展示开关')
}

watch(
  () => configStore.config.model_provider,
  (v) => (modelProvider.value = v)
)
watch(
  () => configStore.config.model_name,
  (v) => (modelName.value = v)
)

const providerModels = computed(() => modelCatalog.value?.[modelProvider.value]?.models || [])

const providerConfigured = (provider) => {
  if (!backendOnline.value) return null
  const st = providersState.status?.[provider]
  if (st && typeof st.configured === 'boolean') return Boolean(st.configured)
  return null
}

const providerDotStatus = (provider) => {
  const configured = providerConfigured(provider)
  if (configured === null) return 'info'
  return configured ? 'online' : 'offline'
}

const providerDotLabel = (provider) => {
  const configured = providerConfigured(provider)
  if (configured === null) return backendOnline.value ? '未知' : '断开连接'
  return configured ? '可用' : '不可用'
}

const providerSourceText = (provider) => {
  const s = providersState.status?.[provider]?.source
  if (s === 'file') return '本地'
  if (s === 'env') return 'ENV'
  return '默认'
}

const providerSourceColor = (provider) => {
  const s = providersState.status?.[provider]?.source
  if (s === 'file') return 'blue'
  if (s === 'env') return 'geekblue'
  return 'default'
}

const normalizeApiBase = (v) => String(v || '').trim().replace(/\/+$/, '')
const hasProviderChanges = (provider) => {
  ensureProviderForm(provider)
  const form = providerForm[provider] || {}
  const st = providersState.status?.[provider] || {}
  const baseChanged = normalizeApiBase(form.api_base) && normalizeApiBase(form.api_base) !== normalizeApiBase(st.api_base)
  const keyChanged = Boolean(String(form.api_key || '').trim())
  return baseChanged || keyChanged
}

const providersState = reactive({
  loading: false,
  saving: {},
  status: {},
  editingKey: {}
})

const providerForm = reactive({})
const providerList = computed(() => {
  const fromCatalog = providerKeys.value || []
  const fromStatus = Object.keys(providersState.status || {})
  const merged = Array.from(new Set([...fromCatalog, ...fromStatus])).filter((k) => k && k !== 'custom')
  return merged
})

const ensureProviderForm = (p) => {
  if (!providerForm[p]) {
    providerForm[p] = { api_base: '', api_key: '' }
  }
  if (!providerForm[p].api_base) {
    providerForm[p].api_base =
      providersState.status?.[p]?.api_base || modelCatalog.value?.[p]?.base_url || ''
  }
}

const onProviderChange = async (p) => {
  const models = modelCatalog.value?.[p]?.models || []
  const def = modelCatalog.value?.[p]?.default || models?.[0] || modelName.value || ''
  await configStore.setConfigValues({ model_provider: p, ...(def ? { model_name: def } : {}) })
}

const onModelChange = async (m) => {
  await configStore.setConfigValue('model_name', m)
}

const commitModelInput = async () => {
  if (!modelName.value) return
  await configStore.setConfigValue('model_name', modelName.value)
}

const refreshAll = async () => {
  state.refreshing = true
  try {
    await configStore.refreshConfig()
  } finally {
    state.refreshing = false
  }
}

const restartBackend = async () => {
  if (!backendOnline.value) return
  try {
    await apiFetch('/restart', { method: 'POST', timeoutMs: 10000 })
    message.success('已触发后端重启/刷新（best-effort）')
    await refreshAll()
  } catch (e) {
    notifyApiError(e, { context: '后端重启', fallback: '后端重启失败' })
  }
}

const featureState = reactive({
  saving: {}
})

const setBackendFeature = async (key, checked) => {
  featureState.saving[key] = true
  const prev = Boolean(configStore.config?.[key])
  // Optimistic UI: flip immediately, then confirm with backend response.
  configStore.patchLocal({ [key]: Boolean(checked) })
  try {
    const res = await apiFetch('/config', {
      method: 'PATCH',
      body: { [key]: Boolean(checked) },
      timeoutMs: 10000
    })
    configStore.patchLocal({
      ...res,
      backend: { online: true, last_error: null, ...(res?.backend || {}) }
    })
    const effective = Boolean(res?.[key])
    if (effective !== Boolean(checked)) {
      message.warning('后端未应用该开关（可能后端版本较旧或未支持该能力）')
    } else {
      message.success('已更新后端开关')
    }
  } catch (e) {
    // Revert optimistic change.
    configStore.patchLocal({ [key]: prev })
    notifyApiError(e, { context: '后端开关', fallback: '更新失败' })
  } finally {
    featureState.saving[key] = false
  }
}
	
const refreshProviders = async () => {
  providersState.loading = true
  try {
    const res = await apiFetch('/providers', { method: 'GET', timeoutMs: 8000 })
    providersState.status = res?.providers || {}
    providerList.value.forEach((p) => ensureProviderForm(p))
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      message.warning('后端暂不支持 Provider 配置接口（/providers）。请停止后端并重新启动到最新代码。')
      return
    }
    notifyApiError(e, { context: 'Provider 配置', fallback: '获取 Provider 状态失败' })
  } finally {
    providersState.loading = false
  }
}

const enableEditKey = (provider) => {
  providersState.editingKey[provider] = true
  ensureProviderForm(provider)
  providerForm[provider].api_key = ''
}

const saveProvider = async (provider) => {
  ensureProviderForm(provider)
  const form = providerForm[provider] || {}
  const body = {
    provider,
    ...(form.api_base ? { api_base: form.api_base } : {}),
    ...(form.api_key ? { api_key: form.api_key } : {})
  }

  providersState.saving[provider] = true
  try {
    const res = await apiFetch('/providers', { method: 'PATCH', body, timeoutMs: 10000 })
    providersState.status = res?.providers || providersState.status
    providerForm[provider].api_key = '' // never keep key in memory after save
    providersState.editingKey[provider] = false
    message.success('已保存 Provider 配置')
  } catch (e) {
    notifyApiError(e, { context: 'Provider 配置', fallback: '保存失败' })
  } finally {
    providersState.saving[provider] = false
  }
}

const clearProvider = async (provider) => {
  ensureProviderForm(provider)
  providersState.saving[provider] = true
  try {
    const res = await apiFetch('/providers', {
      method: 'PATCH',
      body: { provider, api_key: '', api_base: '' },
      timeoutMs: 10000
    })
    providersState.status = res?.providers || providersState.status
    providerForm[provider].api_key = ''
    providerForm[provider].api_base = modelCatalog.value?.[provider]?.base_url || ''
    providersState.editingKey[provider] = false
    message.success('已清空 Provider 配置')
  } catch (e) {
    notifyApiError(e, { context: 'Provider 配置', fallback: '清空失败' })
  } finally {
    providersState.saving[provider] = false
  }
}

const onUiDensityChange = (v) => {
  uiDensity.value = setUiDensity(v)
  message.success(`已切换界面密度：${uiDensity.value === 'compact' ? '紧凑' : '舒适'}`)
}

const onThemePresetChange = (v) => {
  themePreset.value = setThemePreset(v)
  message.success(`已切换主题色：${THEME_PRESETS[themePreset.value]?.label || themePreset.value}`)
}

watch(
  () => activeTab.value,
  (tab) => {
    if ((tab === 'providers' || tab === 'model') && backendOnline.value && !providersState.loading) {
      // Lazy load provider status when the tab is opened.
      refreshProviders()
    }
  }
)

watch(
  () => backendOnline.value,
  (on) => {
    if (!on) return
    if (providersState.loading) return
    if (Object.keys(providersState.status || {}).length) return
    refreshProviders()
  }
)

onMounted(() => {
  // Ensure provider form has defaults even when backend is offline.
  providerList.value.forEach((p) => ensureProviderForm(p))
})
</script>

<style scoped>
.setting-container {
  padding: 0;
}

.setting-tabs :deep(.ant-tabs-nav) {
  margin: 0 0 12px;
}

.preset-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  border: 1px solid var(--border-color);
  margin-right: 8px;
}

.provider-card :deep(.ant-card-head-title) {
  width: 100%;
}

.provider-title {
  display: flex;
  align-items: center;
  gap: 10px;
}

.provider-icon {
  width: 20px;
  height: 20px;
  border-radius: 6px;
  background: var(--surface-color-2);
  object-fit: contain;
}

.provider-name {
  font-weight: 600;
  flex: 1;
}

.provider-source {
  user-select: none;
}

.provider-docs {
  font-size: 12px;
  color: var(--gray-600);
  text-decoration: none;
  user-select: none;
}

.provider-docs:hover {
  color: var(--main-500);
  text-decoration: underline;
}

.provider-status {
  margin-left: auto;
}

.provider-option {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}

.provider-option-icon {
  width: 16px;
  height: 16px;
  border-radius: 4px;
  background: var(--surface-color-2);
  object-fit: contain;
}

.provider-option-spacer {
  flex: 1;
}

.troubleshoot-list {
  padding-left: 18px;
  margin: 0;
}

.troubleshoot-list li {
  margin: 6px 0;
}

</style>
