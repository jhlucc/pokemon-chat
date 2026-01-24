<template>
  <div class="tools-container layout-container">
    <HeaderComponent
      title="工具箱"
      description="这里展示了各种可用的工具"
    >
    </HeaderComponent>
    <div class="tools-grid">
      <template v-if="state.loadingTools">
        <div v-for="n in 6" :key="n" class="tool-card ui-card tool-card--skeleton">
          <a-skeleton active :title="false" :paragraph="{ rows: 3 }" />
        </div>
      </template>
      <template v-else>
        <div v-if="tools.length === 0" class="tools-empty ui-muted">暂无工具</div>
        <div v-for="tool in tools" :key="tool.name" class="tool-card ui-card" @click="navigateToTool(tool.url)">
          <div class="tool-header">
            <h3>{{ tool.title }}</h3>
          </div>
          <div class="tool-info">
            <p>{{ tool.description }}</p>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue';
import { useRouter } from 'vue-router';
import HeaderComponent from '@/components/HeaderComponent.vue';
import { apiFetch } from '@/api/http'

const router = useRouter();
const tools = ref([]);

const state = reactive({
  loadingTools: true,
})

const getTools = () => {
  state.loadingTools = true
  apiFetch('/tools/', { method: 'GET', timeoutMs: 5000 })
    .then((data) => {
      tools.value = Array.isArray(data) ? data : (data?.tools || [])
    })
    .catch(() => {
      // Offline fallback so the page still renders.
      tools.value = [
        { name: 'file-chunking', title: '文件分块', description: '离线模式：仅展示', url: '/tools/file-chunking' },
        { name: 'pdf2txt', title: 'PDF 转文本', description: '离线模式：仅展示', url: '/tools/pdf2txt' },
        { name: 'agent', title: '智能体', description: '离线模式：仅展示', url: '/tools/agent' },
      ]
    })
    .finally(() => {
      state.loadingTools = false
    })
};

const navigateToTool = (toolUrl) => {
  router.push(toolUrl);
};

onMounted(() => {
  getTools();
});
</script>

<style scoped lang="less">
.tools-container {
  padding: 0;
}

.tools-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  padding: 20px;

  .tool-card {
    display: flex;
    flex-direction: column;
    padding: 20px;
    cursor: pointer;

    &:hover {
      // ui-card handles hover
    }

    .tool-header {
      display: flex;
      align-items: center;
      margin-bottom: 15px;
      font-size: 15px;

      .tool-icon {
        margin-right: 10px;
      }

      h3 {
        margin: 0;
      }
    }

    .tool-info {
      flex-grow: 1;

      p {
        margin: 0;
        color: var(--gray-700);
      }
    }
  }

  .tool-card--skeleton {
    cursor: default;
  }
}

.tools-empty {
  padding: 8px;
}
</style>
