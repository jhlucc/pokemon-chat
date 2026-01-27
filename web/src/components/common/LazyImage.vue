<template>
  <picture v-if="webpSrc || srcset">
    <source v-if="webpSrc" :srcset="webpSrc" type="image/webp" />
    <source v-if="srcset" :srcset="srcset" :sizes="sizes" />
    <img
      ref="imgRef"
      :src="currentSrc"
      :alt="alt"
      :class="imgClass"
      :style="imgStyle"
      :loading="lazy ? 'lazy' : 'eager'"
      :decoding="lazy ? 'async' : 'auto'"
      @load="onLoad"
      @error="onError"
    />
  </picture>
  <img
    v-else
    ref="imgRef"
    :src="currentSrc"
    :alt="alt"
    :class="imgClass"
    :style="imgStyle"
    :loading="lazy ? 'lazy' : 'eager'"
    :decoding="lazy ? 'async' : 'auto'"
    @load="onLoad"
    @error="onError"
  />
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

interface Props {
  /** 图片源 */
  src: string
  /** 替代文本 */
  alt?: string
  /** WebP 版本源 */
  webpSrc?: string
  /** 响应式 srcset */
  srcset?: string
  /** 响应式 sizes */
  sizes?: string
  /** 占位图 */
  placeholder?: string
  /** 加载失败时的回退图 */
  fallback?: string
  /** 是否懒加载 */
  lazy?: boolean
  /** 自定义类名 */
  imgClass?: string | string[] | Record<string, boolean>
  /** 自定义样式 */
  imgStyle?: string | Record<string, string>
}

const props = withDefaults(defineProps<Props>(), {
  alt: '',
  lazy: true,
  placeholder: '',
  fallback: ''
})

const emit = defineEmits<{
  (e: 'load', event: Event): void
  (e: 'error', event: Event): void
}>()

const imgRef = ref<HTMLImageElement | null>(null)
const isLoaded = ref(false)
const hasError = ref(false)

const currentSrc = computed(() => {
  if (hasError.value && props.fallback) {
    return props.fallback
  }
  if (!isLoaded.value && props.placeholder) {
    return props.placeholder
  }
  return props.src
})

function onLoad(event: Event) {
  isLoaded.value = true
  emit('load', event)
}

function onError(event: Event) {
  hasError.value = true
  emit('error', event)
}

// 暴露方法
defineExpose({
  isLoaded,
  hasError,
  imgRef
})
</script>

<style scoped>
img {
  max-width: 100%;
  height: auto;
}
</style>
