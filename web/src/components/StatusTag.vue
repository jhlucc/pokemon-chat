<script setup lang="ts">
import { computed } from 'vue'

type StatusKey = 'online' | 'offline' | 'ready' | 'not_ready' | 'info'

const props = defineProps<{
  status: StatusKey
  label?: string
}>()

const color = computed(() => {
  switch (props.status) {
    case 'online':
      return 'green'
    case 'offline':
      return 'red'
    case 'ready':
      return 'green'
    case 'not_ready':
      return 'orange'
    default:
      return 'default'
  }
})

const text = computed(() => {
  if (props.label) return props.label
  switch (props.status) {
    case 'online':
      return 'Online'
    case 'offline':
      return 'Offline'
    case 'ready':
      return 'Ready'
    case 'not_ready':
      return 'Not Ready'
    default:
      return 'Info'
  }
})
</script>

<template>
  <a-tag class="status-tag" :color="color">{{ text }}</a-tag>
</template>

<style scoped>
.status-tag {
  font-weight: 650;
  letter-spacing: 0.2px;
  user-select: none;
}
</style>
