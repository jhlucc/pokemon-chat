<template>
  <div class="setting-page">
    <HeaderComponent title="设置" description="连接状态、模型选择与后端能力概览" class="setting-header">
      <template #actions>
        <a-space>
          <a-button @click="refreshAll" :loading="state.refreshing">
            刷新状态
          </a-button>
          <a-button type="primary" :disabled="!backendOnline" @click="restartBackend">
            重新加载后端
          </a-button>
        </a-space>
      </template>
    </HeaderComponent>

    <div class="setting-container layout-container">
      <a-row :gutter="[16, 16]" style="padding: 0 24px 24px;">
        <a-col :xs="24" :md="12">
          <a-card title="连接状态" :bordered="false">
            <div class="kv">
              <span class="k">Backend</span>
              <span class="v">
                <a-tag :color="backendOnline ? 'green' : 'red'">{{ backendOnline ? 'Online' : 'Offline' }}</a-tag>
                <a-tag :color="backendReady ? 'green' : 'orange'">{{ backendReady ? 'Ready' : 'Not Ready' }}</a-tag>
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
          <a-card title="模型" :bordered="false">
            <a-form layout="vertical">
              <a-form-item label="Provider">
                <a-select v-model:value="modelProvider" @change="onProviderChange">
                  <a-select-option v-for="p in providerKeys" :key="p" :value="p">
                    {{ modelCatalog[p]?.name || p }}
                  </a-select-option>
                </a-select>
              </a-form-item>
              <a-form-item label="Model">
                <a-select v-model:value="modelName" @change="onModelChange">
                  <a-select-option v-for="m in providerModels" :key="m" :value="m">
                    {{ m }}
                  </a-select-option>
                </a-select>
              </a-form-item>
              <a-alert
                type="info"
                show-icon
                message="提示：模型与 API Key 属于后端配置（.env / docker 环境变量）。这里选择的是“使用哪一个模型名”并会随请求发送给后端。"
              />
            </a-form>
          </a-card>
        </a-col>

        <a-col :xs="24" :md="24">
          <a-card title="后端能力（只读）" :bordered="false">
            <a-space wrap>
              <a-tag :color="configStore.config.enable_knowledge_base ? 'green' : 'default'">知识库</a-tag>
              <a-tag :color="configStore.config.enable_knowledge_graph ? 'green' : 'default'">知识图谱</a-tag>
              <a-tag :color="configStore.config.enable_web_search ? 'green' : 'default'">联网搜索</a-tag>
              <a-tag :color="configStore.config.enable_mcp ? 'green' : 'default'">MCP</a-tag>
              <a-tag :color="configStore.config.enable_reranker ? 'green' : 'default'">Reranker</a-tag>
            </a-space>
            <div class="muted" style="margin-top: 10px;">
              这些开关由后端环境变量控制（见 `.env` / `docker-compose.yml`）。前端不依赖后端也能渲染，但功能调用需要后端在线且对应能力开启。
            </div>
          </a-card>
        </a-col>

        <a-col :xs="24" :md="24">
          <a-card title="本地配置" :bordered="false">
            <a-space wrap>
              <a-button danger @click="resetLocalConfig">
                重置本地配置
              </a-button>
            </a-space>
            <div class="muted" style="margin-top: 10px;">
              本地配置存储在浏览器 localStorage（用于离线可用/页面正常显示）。
            </div>
          </a-card>
        </a-col>
      </a-row>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';
import { message } from 'ant-design-vue';
import HeaderComponent from '@/components/HeaderComponent.vue';
import { useConfigStore } from '@/stores/config';
import { apiFetch } from '@/api/http';

const configStore = useConfigStore();

const state = reactive({
  refreshing: false,
});

const backendOnline = computed(() => Boolean(configStore.config.backend?.online));
const backendReady = computed(() => Boolean(configStore.config.backend?.ready));

const modelCatalog = computed(() => configStore.config.model_names || {});
const providerKeys = computed(() => Object.keys(modelCatalog.value || {}).filter((k) => k !== 'custom'));

const modelProvider = ref(configStore.config.model_provider);
const modelName = ref(configStore.config.model_name);

watch(
  () => configStore.config.model_provider,
  (v) => (modelProvider.value = v)
);
watch(
  () => configStore.config.model_name,
  (v) => (modelName.value = v)
);

const providerModels = computed(() => modelCatalog.value?.[modelProvider.value]?.models || []);

const onProviderChange = async (p) => {
  await configStore.setConfigValue('model_provider', p);
  const def = modelCatalog.value?.[p]?.default || providerModels.value?.[0] || '';
  if (def) await configStore.setConfigValue('model_name', def);
};

const onModelChange = async (m) => {
  await configStore.setConfigValue('model_name', m);
};

const refreshAll = async () => {
  state.refreshing = true;
  try {
    await configStore.refreshConfig();
  } finally {
    state.refreshing = false;
  }
};

const restartBackend = async () => {
  if (!backendOnline.value) return;
  try {
    await apiFetch('/api/restart', { method: 'POST', timeoutMs: 10000 });
    message.success('已触发后端重启/刷新（best-effort）');
    await refreshAll();
  } catch (e) {
    message.error(e?.message || '后端重启失败');
  }
};

const resetLocalConfig = () => {
  try {
    localStorage.removeItem('pokemon_chat_config_v1');
    message.success('已重置本地配置，刷新页面生效');
  } catch (e) {
    message.error('重置失败');
  }
};
</script>

<style scoped>
.kv {
  display: flex;
  justify-content: space-between;
  margin: 8px 0;
}
.k {
  color: var(--gray-700);
}
.v {
  font-weight: 600;
}
.muted {
  color: var(--gray-600);
}
</style>
