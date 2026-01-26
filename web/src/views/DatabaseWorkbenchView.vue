<template>
  <div class="database-workbench layout-container">
    <HeaderComponent
      title="知识库工作台"
      description="把文件解析/切块并写入知识库索引（原工具箱能力已合并到这里）"
      :breadcrumbs="[{ label: '首页', to: '/' }, { label: '知识库', to: '/database' }, { label: '工作台' }]"
    >
      <template #actions>
        <a-button @click="goDatabase">返回知识库</a-button>
      </template>
    </HeaderComponent>

    <div class="ui-page">
      <div class="ui-container">
        <DatabaseRagWorkbench />
      </div>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import HeaderComponent from '@/components/HeaderComponent.vue'
import DatabaseRagWorkbench from '@/components/database/DatabaseRagWorkbench.vue'

const router = useRouter()
const goDatabase = () => router.push('/database')
</script>

<style scoped lang="less">
.database-workbench {
  padding: 0;
}
</style>

