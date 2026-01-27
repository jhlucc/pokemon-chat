<template>
  <div
    v-if="visible"
    class="options-panel options-panel--right swing-in-top-fwd"
    ref="panelRef"
  >
    <div class="options-row" @click="emit('toggle-stream')">
      流式输出
      <div @click.stop>
        <a-switch :checked="stream" @change="emit('toggle-stream')" />
      </div>
    </div>
    <div class="options-row" @click="emit('toggle-summary-title')">
      总结对话标题
      <div @click.stop>
        <a-switch :checked="summaryTitle" @change="emit('toggle-summary-title')" />
      </div>
    </div>
    <div class="options-row">
      最大历史轮数
      <a-input-number
        :value="historyRound"
        :min="1"
        :max="50"
        @change="(val: number | string | null) => emit('update:history-round', Number(val ?? 1))"
      />
    </div>
    <div class="options-row">
      字体大小
      <a-select
        :value="fontSize"
        style="width: 100px"
        placeholder="选择字体大小"
        @change="(val: string) => emit('update:font-size', val as FontSize)"
      >
        <a-select-option value="smaller">更小</a-select-option>
        <a-select-option value="default">默认</a-select-option>
        <a-select-option value="larger">更大</a-select-option>
      </a-select>
    </div>
    <div class="options-row" @click="emit('toggle-wide-screen')">
      宽屏模式
      <div @click.stop>
        <a-switch :checked="wideScreen" @change="emit('toggle-wide-screen')" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { onClickOutside } from '@vueuse/core'
import type { FontSize } from '@/types/chat'

defineProps<{
  visible: boolean
  stream: boolean
  summaryTitle: boolean
  historyRound: number
  fontSize: FontSize
  wideScreen: boolean
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'toggle-stream'): void
  (e: 'toggle-summary-title'): void
  (e: 'update:history-round', value: number): void
  (e: 'update:font-size', value: FontSize): void
  (e: 'toggle-wide-screen'): void
}>()

const panelRef = ref<HTMLElement | null>(null)

// 点击外部关闭
onClickOutside(panelRef, () => {
  setTimeout(() => emit('close'), 30)
})
</script>

<style scoped lang="less">
.options-panel {
  position: absolute;
  margin-top: var(--space-2);
  background-color: color-mix(in srgb, var(--surface-color) 95%, transparent);
  backdrop-filter: blur(var(--blur-md));
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-md);
  border-radius: var(--radius-md);
  padding: var(--space-4);
  z-index: 11;
  width: 300px;
  transition: transform var(--duration-base) ease, opacity var(--duration-base) ease;

  .options-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background-color var(--duration-base);
    font-weight: 500;
    color: var(--gray-700);

    &:hover {
      background-color: var(--hover-bg);
      color: var(--primary-color);
    }

    .anticon {
      margin-right: var(--space-2);
    }
  }
}

.options-panel--right {
  top: 100%;
  right: 0;
}

/* Animation */
.swing-in-top-fwd {
  animation: swing-in-top-fwd 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
}

@keyframes swing-in-top-fwd {
  0% {
    transform: rotateX(-100deg);
    transform-origin: top;
    opacity: 0;
  }
  100% {
    transform: rotateX(0deg);
    transform-origin: top;
    opacity: 1;
  }
}
</style>
