<template>
  <div class="glass-panel" :class="{ 'glass-panel--compact': compact }">
    <div v-if="title || $slots.header" class="glass-panel__header">
      <span v-if="title" class="glass-panel__title">{{ title }}</span>
      <slot name="header" />
      <span v-if="badge" class="glass-panel__badge">{{ badge }}</span>
      <slot name="header-actions" />
    </div>
    <div class="glass-panel__body">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, default: '' },
  badge: { type: [String, Number], default: '' },
  compact: { type: Boolean, default: false }
})
</script>

<style scoped lang="less">
.glass-panel {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.5);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03), 0 4px 12px color-mix(in srgb, var(--primary-color) 3%, transparent);

  &--compact {
    padding: 16px;
    border-radius: 16px;
  }
}

.glass-panel__header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color);
}

.glass-panel__title {
  font-size: 16px;
  font-weight: 650;
  color: var(--text-color);
}

.glass-panel__badge {
  font-size: 12px;
  padding: 2px 10px;
  background: color-mix(in srgb, var(--primary-color) 10%, transparent);
  color: var(--primary-color);
  border-radius: 100px;
  font-weight: 500;
  margin-left: auto;
}

.glass-panel__body {
  /* Content container */
}

:root[data-theme='dark'] {
  .glass-panel {
    background: rgba(40, 40, 40, 0.85);
    border-color: rgba(255, 255, 255, 0.08);
  }
}
</style>
