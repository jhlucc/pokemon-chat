<template>
  <div class="header-container" :class="{ compact: compact, sticky: sticky }">
    <div class="header-content">
      <div class="header-title">
        <div
          v-if="$slots.breadcrumbs || (breadcrumbs && breadcrumbs.length > 0)"
          class="header-breadcrumb"
        >
          <slot name="breadcrumbs">
            <a-breadcrumb>
              <a-breadcrumb-item v-for="(bc, idx) in breadcrumbs" :key="idx">
                <router-link v-if="bc?.to" :to="bc.to">{{ bc.label }}</router-link>
                <span v-else>{{ bc.label }}</span>
              </a-breadcrumb-item>
            </a-breadcrumb>
          </slot>
        </div>

        <div class="header-title-row">
          <h1>{{ title }}</h1>
          <slot name="extra"></slot>
        </div>

        <slot name="description">
          <p v-if="description" class="header-description">{{ description }}</p>
        </slot>
      </div>
      <div class="header-actions" v-if="$slots.actions">
        <slot name="actions"></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  },
  breadcrumbs: {
    type: Array,
    default: () => []
  },
  compact: {
    type: Boolean,
    default: false
  },
  sticky: {
    type: Boolean,
    default: true
  }
})
</script>

<style scoped lang="less">
.header-container {
  /* 完全透明化：让暖色光晕透出 */
  background: transparent !important;
  background-color: transparent !important;
  padding: 16px 24px;
  /* 去掉底部边框，更加通透 */
  border-bottom: none;

  &.sticky {
    position: sticky;
    top: 0;
    z-index: 1000;
  }

  &.compact {
    padding: 10px 16px;
  }
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.header-title {
  font-size: 14px;
  color: var(--header-desc);
  min-width: 0;

  h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 500;
    color: var(--header-title);
  }

  .header-breadcrumb :deep(.ant-breadcrumb) {
    font-size: 12px;
    color: var(--header-desc);
  }

  .header-title-row {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .header-title-row h1 {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .header-description {
    margin: 8px 0 0;
  }
}

.header-actions {
  display: flex;
  gap: 8px;
  flex: 0 0 auto;
  flex-wrap: wrap;
}

@media (max-width: 520px) {
  .header-container {
    padding: 12px 14px;
  }

  .header-content {
    flex-direction: column;
    align-items: stretch;
  }

  .header-title .header-title-row {
    flex-wrap: wrap;
    row-gap: 8px;
  }

  .header-actions {
    justify-content: flex-start;
  }
}
</style>
