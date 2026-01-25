<template>
  <div class="not-found">
    <a-result status="404" title="404" sub-title="页面不存在，或已被移动。">
      <template #extra>
        <a-space>
          <a-button @click="goBack">返回</a-button>
          <a-button type="primary" @click="goHome">回到首页</a-button>
        </a-space>
      </template>
    </a-result>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()
const goHome = () => router.push('/')
const goBack = () => router.back()
</script>

<style scoped>
.not-found {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  min-height: 70vh;
}
</style>
