<template>
  <div class="setting-page">
    <HeaderComponent
      title="设置"
      description="连接状态、模型选择与后端能力概览"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '设置' }]"
      class="setting-header"
    >
      <template #actions>
        <a-space>
          <a-button @click="refreshAll" :loading="state.refreshing"> 刷新状态 </a-button>
          <a-button type="primary" :disabled="!backendOnline" @click="restartBackend">
            重新加载后端
          </a-button>
        </a-space>
      </template>
    </HeaderComponent>

    <div class="setting-container layout-container">
      <div class="setting-body ui-page">
        <div class="ui-container">
          <a-tabs v-model:activeKey="activeTab" class="setting-tabs">
            <a-tab-pane key="status" tab="状态">
              <a-row :gutter="[16, 16]">
                <a-col :xs="24" :md="12">
                  <a-card title="连接状态" :bordered="false">
                    <div class="kv">
                      <span class="k">Backend</span>
                      <span class="v">
                        <StatusTag :status="backendOnline ? 'online' : 'offline'" />
                        <StatusTag :status="backendReady ? 'ready' : 'not_ready'" />
                      </span>
                    </div>
                    <div class="kv" v-if="configStore.config.backend?.last_error">
                      <span class="k">Last error</span>
                      <span class="v muted">{{ configStore.config.backend?.last_error }}</span>
                    </div>
                    <div class="kv" v-if="configStore.config.backend?.checks">
                      <span class="k">Checks</span>
                      <span class="v muted">见 /readyz</span>
                    </div>
                  </a-card>
                </a-col>

                <a-col :xs="24" :md="12">
                  <a-card title="版本信息" :bordered="false">
                    <div class="kv">
                      <span class="k">前端</span>
                      <span class="v">{{ APP_NAME }} v{{ APP_VERSION }}</span>
                    </div>
                    <div class="kv" v-if="BUILD_SHA">
                      <span class="k">Commit</span>
                      <span class="v muted">{{ BUILD_SHA.slice(0, 7) }}</span>
                    </div>
                    <div class="kv" v-if="BUILD_TIME">
                      <span class="k">Build</span>
                      <span class="v muted">{{ BUILD_TIME }}</span>
                    </div>
                  </a-card>
                </a-col>
              </a-row>
            </a-tab-pane>

            <a-tab-pane key="providers" tab="供应商">
              <a-card title="Provider 配置" :bordered="false" style="margin-bottom: 16px">
                <a-space wrap>
                  <a-button @click="refreshProviders" :loading="providersState.loading"
                    >刷新 Provider 状态</a-button
                  >
                </a-space>
                <a-alert
                  style="margin-top: 12px"
                  type="info"
                  show-icon
                  message="在这里配置各供应商的 API Key / Base URL"
                  description="保存后会写入后端本地配置（resources/save/config/provider_secrets.json），前端不会回显明文 Key。"
                />
              </a-card>

                <a-row :gutter="[16, 16]">
                  <a-col v-for="p in providerList" :key="p" :xs="24" :md="12" :lg="8">
                    <a-card :bordered="false" class="provider-card">
                    <template #title>
                        <div class="provider-title">
                          <img class="provider-icon" :src="getProviderIcon(p)" :alt="p" />
                          <span class="provider-name">{{ modelCatalog[p]?.name || p }}</span>
                          <a-tag class="provider-source" :color="providerSourceColor(p)">
                            {{ providerSourceText(p) }}
                          </a-tag>
                          <a
                            v-if="modelCatalog[p]?.url"
                            class="provider-docs"
                            :href="modelCatalog[p].url"
                            target="_blank"
                            rel="noreferrer"
                            @click.stop
                          >
                            Docs
                          </a>
                          <StatusTag
                            class="provider-status"
                            variant="dot"
                            :status="providerDotStatus(p)"
                            :label="providerDotLabel(p)"
                          />
                        </div>
                    </template>

                    <a-form layout="vertical">
                      <a-form-item label="API Base（可选）">
                        <a-input
                          v-model:value="providerForm[p].api_base"
                          :placeholder="modelCatalog[p]?.base_url || 'https://.../v1'"
                        />
                      </a-form-item>
                      <a-form-item label="API Key（可选）">
                        <div v-if="providersState.status?.[p]?.configured && !providersState.editingKey?.[p]">
                          <a-space wrap>
                            <a-tag color="green">已配置</a-tag>
                            <span class="muted">{{ providersState.status?.[p]?.api_key_masked || '***' }}</span>
                            <a-button size="small" @click="enableEditKey(p)">更换 Key</a-button>
                          </a-space>
                          <div class="muted" style="margin-top: 6px">
                            出于安全考虑，前端不会回显明文 Key；如需更换，请点击“更换 Key”。
                          </div>
                        </div>
                        <a-input-password
                          v-else
                          v-model:value="providerForm[p].api_key"
                          autocomplete="new-password"
                          :placeholder="providersState.status?.[p]?.configured ? '输入新的 Key（留空则不修改）' : '输入 Key'"
                        />
                      </a-form-item>
                      <a-space wrap>
                        <a-button
                          type="primary"
                          :disabled="!backendOnline || !hasProviderChanges(p)"
                          :loading="Boolean(providersState.saving?.[p])"
                          @click="saveProvider(p)"
                        >
                          保存
                        </a-button>
                        <a-button
                          danger
                          :disabled="providersState.status?.[p]?.source !== 'file'"
                          :loading="Boolean(providersState.saving?.[p])"
                          @click="clearProvider(p)"
                        >
                          清空
                        </a-button>
                      </a-space>
                    </a-form>
                  </a-card>
                </a-col>
              </a-row>
            </a-tab-pane>

            <a-tab-pane key="model" tab="模型">
              <a-row :gutter="[16, 16]">
                <a-col :xs="24" :md="12">
                  <a-card title="模型选择" :bordered="false">
                    <a-form layout="vertical">
                      <a-form-item label="Provider">
                        <a-select v-model:value="modelProvider" @change="onProviderChange">
                          <a-select-option v-for="p in providerKeys" :key="p" :value="p">
                            <span class="provider-option">
                              <img class="provider-option-icon" :src="getProviderIcon(p)" :alt="p" />
                               <span>{{ modelCatalog[p]?.name || p }}</span>
                               <span class="provider-option-spacer" />
                               <StatusTag
                                 variant="dot"
                                 :status="providerDotStatus(p)"
                                 :label="providerDotLabel(p)"
                               />
                             </span>
                           </a-select-option>
                         </a-select>
                      </a-form-item>
                      <a-form-item label="Model">
                        <a-select v-if="providerModels.length" v-model:value="modelName" @change="onModelChange">
                          <a-select-option v-for="m in providerModels" :key="m" :value="m">
                            {{ m }}
                          </a-select-option>
                        </a-select>
                        <a-input
                          v-else
                          v-model:value="modelName"
                          placeholder="输入模型名称（如 gpt-4o-mini）"
                          @pressEnter="commitModelInput"
                          @blur="commitModelInput"
                        />
                      </a-form-item>
                      <a-alert
                        type="info"
                        show-icon
                        message="提示：这里选择的是“使用哪一个模型名”并会随请求发送给后端。API Key / Base URL 可在「供应商」页配置。"
                      />
                    </a-form>
                  </a-card>
                </a-col>

                <a-col :xs="24" :md="12">
                  <a-card title="使用建议" :bordered="false">
                    <a-collapse ghost>
                      <a-collapse-panel key="troubleshoot" header="快速排查">
                        <ol class="muted troubleshoot-list">
                          <li>后端是否 Online</li>
                          <li>是否选择了正确的 Provider / Model</li>
                          <li>后端环境变量中是否配置了对应的 API Key</li>
                        </ol>
                      </a-collapse-panel>
                    </a-collapse>
                  </a-card>
                </a-col>
              </a-row>
            </a-tab-pane>

            <a-tab-pane key="capabilities" tab="能力">
              <a-card title="后端功能开关" :bordered="false">
                <a-space wrap>
                  <a-space>
                    <span class="muted">知识库</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_knowledge_base)"
                      :loading="Boolean(featureState.saving.enable_knowledge_base)"
                      @change="(v) => setBackendFeature('enable_knowledge_base', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">知识图谱</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_knowledge_graph)"
                      :loading="Boolean(featureState.saving.enable_knowledge_graph)"
                      @change="(v) => setBackendFeature('enable_knowledge_graph', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">联网搜索</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_web_search)"
                      :loading="Boolean(featureState.saving.enable_web_search)"
                      @change="(v) => setBackendFeature('enable_web_search', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">MCP</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_mcp)"
                      :loading="Boolean(featureState.saving.enable_mcp)"
                      @change="(v) => setBackendFeature('enable_mcp', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">Reranker</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_reranker)"
                      :loading="Boolean(featureState.saving.enable_reranker)"
                      @change="(v) => setBackendFeature('enable_reranker', v)"
                    />
                  </a-space>
                </a-space>

                <a-divider style="margin: 12px 0" />
                <a-space wrap>
                  <a-space>
                    <span class="muted">ASR</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_asr)"
                      :loading="Boolean(featureState.saving.enable_asr)"
                      @change="(v) => setBackendFeature('enable_asr', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">NER BERT</span>
                    <a-switch
                      :checked="Boolean(configStore.config.enable_ner_bert)"
                      :loading="Boolean(featureState.saving.enable_ner_bert)"
                      @change="(v) => setBackendFeature('enable_ner_bert', v)"
                    />
                  </a-space>
                </a-space>

                <a-alert
                  style="margin-top: 12px"
                  type="info"
                  show-icon
                  message="说明"
                  description="这些开关会写入后端本地配置（resources/save/config/ui_config.json），不修改 .env。部分能力需要额外服务/依赖：知识库需要 Milvus；图谱需要 Neo4j；联网搜索需要 tavily_api_key；NER BERT 需要 torch/transformers。"
                />
              </a-card>
            </a-tab-pane>

            <a-tab-pane key="ui" tab="界面">
              <a-card title="界面展示（前端）" :bordered="false">
                <a-space wrap style="margin-bottom: 12px">
                  <span class="muted">界面密度</span>
                  <a-segmented
                    v-model:value="uiDensity"
                    :options="densityOptions"
                    @change="onUiDensityChange"
                  />
                </a-space>
                <a-space wrap style="margin-bottom: 12px">
                  <span class="muted">主题色</span>
                  <a-select
                    v-model:value="themePreset"
                    style="width: 180px"
                    @change="onThemePresetChange"
                  >
                    <a-select-option v-for="(p, key) in THEME_PRESETS" :key="key" :value="key">
                      <span class="preset-dot" :style="{ background: p.primary }"></span>
                      {{ p.label }}
                    </a-select-option>
                  </a-select>
                </a-space>
                <a-divider style="margin: 12px 0" />
                <a-space wrap>
                  <a-space>
                    <span class="muted">知识库</span>
                    <a-switch
                      :checked="uiVisibility.show_knowledge_base"
                      @change="(v) => setUiVisibility('show_knowledge_base', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">知识图谱</span>
                    <a-switch
                      :checked="uiVisibility.show_knowledge_graph"
                      @change="(v) => setUiVisibility('show_knowledge_graph', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">联网搜索</span>
                    <a-switch
                      :checked="uiVisibility.show_web_search"
                      @change="(v) => setUiVisibility('show_web_search', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">MCP</span>
                    <a-switch
                      :checked="uiVisibility.show_mcp"
                      @change="(v) => setUiVisibility('show_mcp', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">工具</span>
                    <a-switch
                      :checked="uiVisibility.show_tools"
                      @change="(v) => setUiVisibility('show_tools', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">智能体</span>
                    <a-switch
                      :checked="uiVisibility.show_agents"
                      @change="(v) => setUiVisibility('show_agents', v)"
                    />
                  </a-space>
                  <a-space>
                    <span class="muted">地图</span>
                    <a-switch
                      :checked="uiVisibility.show_map"
                      @change="(v) => setUiVisibility('show_map', v)"
                    />
                  </a-space>
                  <a-button @click="resetUiVisibility">重置为默认</a-button>
                </a-space>
                <div class="muted" style="margin-top: 10px">
                  这些开关只影响前端导航与入口显示；后端能力请在「能力」页配置。
                </div>
              </a-card>
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
  if (configured === null) return backendOnline.value ? '未知' : '后端离线'
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
