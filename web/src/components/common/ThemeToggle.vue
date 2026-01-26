<template>
  <a-dropdown placement="bottomRight" trigger="click">
    <button
      class="theme-toggle"
      :title="currentLabel"
      aria-label="切换主题"
    >
      <transition name="icon-swap" mode="out-in">
        <BulbOutlined v-if="resolvedTheme === 'light'" key="light" class="theme-icon" />
        <BulbFilled v-else key="dark" class="theme-icon" />
      </transition>
    </button>
    <template #overlay>
      <a-menu class="theme-menu">
        <a-menu-item
          v-for="option in themeOptions"
          :key="option.value"
          :class="{ 'is-active': theme === option.value }"
          @click="setTheme(option.value)"
        >
          <div class="theme-option">
            <span class="option-icon">{{ option.icon }}</span>
            <span class="option-label">{{ option.label }}</span>
            <CheckOutlined v-if="theme === option.value" class="check-icon" />
          </div>
        </a-menu-item>
      </a-menu>
    </template>
  </a-dropdown>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { BulbOutlined, BulbFilled, CheckOutlined } from '@ant-design/icons-vue'
import { useTheme, themeOptions, type Theme } from '@/composables/useTheme'

const { theme, resolvedTheme, setTheme } = useTheme()

const currentLabel = computed(() => {
  const option = themeOptions.find((o) => o.value === theme.value)
  return option?.label || '主题'
})
</script>

<style scoped lang="less">
.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: var(--gray-600);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-default);

  &:hover {
    background: var(--hover-bg);
    color: var(--primary-color);
  }

  &:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
  }
}

.theme-icon {
  font-size: 18px;
}

.theme-menu {
  min-width: 160px;

  .theme-option {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-1) 0;

    .option-icon {
      font-size: 16px;
    }

    .option-label {
      flex: 1;
    }

    .check-icon {
      color: var(--primary-color);
      font-size: 12px;
    }
  }

  :deep(.ant-dropdown-menu-item) {
    &.is-active {
      background: var(--main-5);
      color: var(--primary-color);
    }
  }
}

/* Icon swap transition */
.icon-swap-enter-active,
.icon-swap-leave-active {
  transition: all var(--duration-fast) var(--ease-default);
}

.icon-swap-enter-from {
  opacity: 0;
  transform: rotate(-90deg) scale(0.8);
}

.icon-swap-leave-to {
  opacity: 0;
  transform: rotate(90deg) scale(0.8);
}
</style>
