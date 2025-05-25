<template>
  <div class="header-container">
    <div class="header-content">
      <div class="header-title">
        <h1>{{ title }}</h1>
        <slot name="description">
          <p v-if="description">{{ description }}</p>
        </slot>
      </div>
      <div class="header-actions" v-if="$slots.actions">
        <slot name="actions"></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  }
});
</script>

<style scoped lang="less">
.header-container {
  background-color: var(--header-bg);
  backdrop-filter: blur(10px);
  padding: 16px 24px;
  border-bottom: 1px solid var(--header-border);
  position: sticky;
  top: 0;
  z-index: 1000;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-title {
  font-size: 14px;
  color: var(--header-desc);

  h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 500;
    color: var(--header-title);
  }

  p {
    margin: 8px 0 0;
  }
}

.header-actions {
  display: flex;
  gap: 8px;
}
</style>

<style lang="less">
/* 这里是非 scoped，定义主题色变量 */
:root {
  --header-bg: rgba(255, 255, 255, 0.8);
  --header-border: #f0f0f0;
  --header-title: rgba(0, 0, 0, 0.85);
  --header-desc: rgba(0, 0, 0, 0.45);
}

body.dark-theme {
  --header-bg: rgba(30, 30, 30, 0.8);
  --header-border: #333333;
  --header-title: rgba(255, 255, 255, 0.85);
  --header-desc: rgba(255, 255, 255, 0.45);
}
</style>
