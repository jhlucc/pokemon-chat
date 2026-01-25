<template>
  <slot v-if="!error" />
  <div v-else class="error-boundary">
    <a-result status="500" title="页面出错了" :sub-title="subtitle">
      <template #extra>
        <a-space>
          <a-button @click="reset">重试</a-button>
          <a-button type="primary" @click="reload">刷新页面</a-button>
        </a-space>
      </template>
    </a-result>
  </div>
</template>

<script setup lang="ts">
import { computed, onErrorCaptured, ref } from 'vue'
import { ApiError } from '@/api/http'
import { trackError } from '@/utils/telemetry'

const error = ref<unknown>(null)

const subtitle = computed(() => {
  const err = error.value
  if (!err) return ''
  if (err instanceof ApiError) {
    return err.requestId ? `${err.message}（RID: ${err.requestId}）` : err.message
  }
  if (err instanceof Error) return err.message || 'Unknown error'
  return typeof err === 'string' ? err : 'Unknown error'
})

onErrorCaptured((err) => {
  error.value = err
  trackError(err, { type: 'errorCaptured' })
  // Stop propagation: we render a friendly fallback page instead.
  return false
})

const reset = () => {
  error.value = null
}

const reload = () => {
  window.location.reload()
}
</script>

<style scoped>
.error-boundary {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  min-height: 70vh;
}
</style>
