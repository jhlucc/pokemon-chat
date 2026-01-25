<script setup lang="ts">
import { computed } from 'vue'

type StatusKey = 'online' | 'offline' | 'ready' | 'not_ready' | 'info'
type Variant = 'tag' | 'dot'

const props = defineProps<{
  status: StatusKey
  label?: string
  variant?: Variant
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

const dotColor = computed(() => {
  // Dot is binary-focused: usable=green, unusable=red.
  switch (props.status) {
    case 'online':
    case 'ready':
      return 'green'
    case 'offline':
    case 'not_ready':
      return 'red'
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
  <span
    v-if="props.variant === 'dot'"
    class="status-dot"
    :class="`status-dot--${dotColor}`"
    :title="text"
    aria-hidden="true"
  />
  <a-tag v-else class="status-tag" :color="color">{{ text }}</a-tag>
</template>

<style scoped>
.status-tag {
  font-weight: 650;
  letter-spacing: 0.2px;
  user-select: none;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  border: 1px solid color-mix(in srgb, var(--border-color) 70%, transparent);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
}

.status-dot--green {
  background: var(--success-color);
}

.status-dot--red {
  background: var(--danger-500);
}

.status-dot--default {
  background: var(--gray-400);
}
</style>
