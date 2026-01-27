<template>
  <a-dropdown>
    <a-tooltip
      placement="bottom"
      :title="`${modelProvider || '-'} / ${modelName || '-'}`"
    >
      <div
        class="model-select"
        @click.prevent
        role="button"
        tabindex="0"
        aria-label="选择模型"
        @keydown.enter.prevent="onKeyboardActivate"
        @keydown.space.prevent="onKeyboardActivate"
      >
        <img
          class="model-select-icon"
          :src="getProviderIcon(modelProvider)"
          :alt="modelProvider || 'provider'"
        />
        <span class="text">{{ modelProvider || '-' }}/{{ modelName || '-' }}</span>
        <DownOutlined class="model-select-caret" />
      </div>
    </a-tooltip>
    <template #overlay>
      <a-menu class="scrollable-menu">
        <a-menu-item-group
          v-for="(item, key) in filteredModelKeys"
          :key="key"
          :title="modelNames[item]?.name"
        >
          <a-menu-item
            v-for="(model, idx) in modelNames[item]?.models"
            :key="`${item}-${idx}`"
            @click="emit('select-model', item, model)"
          >
            <span class="model-menu-item">
              <img class="model-menu-icon" :src="getProviderIcon(item)" :alt="item" />
              <span class="model-menu-text">{{ model }}</span>
            </span>
          </a-menu-item>
        </a-menu-item-group>
        <a-menu-item-group v-if="customModels.length > 0" title="自定义模型">
          <a-menu-item
            v-for="(model, idx) in customModels"
            :key="`custom-${idx}`"
            @click="emit('select-model', 'custom', model.custom_id)"
          >
            custom/{{ model.custom_id }}
          </a-menu-item>
        </a-menu-item-group>
      </a-menu>
    </template>
  </a-dropdown>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { DownOutlined } from '@ant-design/icons-vue'
import { getProviderIcon } from '@/utils/providerIcon'
import type { ModelProvider, CustomModel } from '@/types/chat'

const props = defineProps<{
  modelProvider?: string
  modelName?: string
  modelNames: Record<string, ModelProvider>
  customModels: CustomModel[]
}>()

const emit = defineEmits<{
  (e: 'select-model', provider: string, name: string): void
}>()

// 过滤掉 custom 键
const filteredModelKeys = computed(() => {
  return Object.keys(props.modelNames || {}).filter((k) => k !== 'custom')
})

function onKeyboardActivate(event: KeyboardEvent) {
  ;(event.currentTarget as HTMLElement | null)?.click()
}
</script>

<style scoped lang="less">
.model-select {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: 6px var(--space-4);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-color);
  background: color-mix(in srgb, var(--glass-bg) 70%, transparent);
  backdrop-filter: blur(var(--blur-sm));
  color: var(--text-color);
  cursor: pointer;
  transition: all var(--duration-base) ease;
  max-width: 300px;
  box-shadow: var(--shadow-xs);

  &:hover {
    background: var(--surface-color);
    border-color: var(--primary-color);
    box-shadow: var(--shadow-sm);
    transform: translateY(-1px);
  }

  .text {
    font-weight: 500;
    font-size: var(--font-size-sm);
  }

  .model-select-icon {
    width: 18px;
    height: 18px;
    border-radius: var(--radius-xs);
    object-fit: contain;
  }

  .model-select-caret {
    font-size: 10px;
    color: var(--gray-500);
  }
}

.scrollable-menu {
  max-height: 300px;
  overflow-y: auto;

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: transparent;
  }

  &::-webkit-scrollbar-thumb {
    background: var(--gray-400);
    border-radius: 3px;
  }
}

.model-menu-item {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  max-width: 100%;
}

.model-menu-icon {
  width: 16px;
  height: 16px;
  border-radius: var(--radius-xs);
  background: var(--surface-color-2);
  object-fit: contain;
  flex: 0 0 auto;
}

.model-menu-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
