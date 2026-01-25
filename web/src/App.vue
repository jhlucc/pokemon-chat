<template>
  <AppErrorBoundary>
    <a-config-provider :theme="antdThemeConfig" :component-size="antdComponentSize">
      <router-view />
    </a-config-provider>
  </AppErrorBoundary>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { usePreferredDark } from '@vueuse/core'
import { theme as antdTheme } from 'ant-design-vue'

import AppErrorBoundary from '@/components/AppErrorBoundary.vue'
import { THEME_MODE_CHANGED_EVENT, getSavedThemeMode, themeConfig } from '@/assets/theme'
import { getUiDensity } from '@/utils/uiDensity'
import { getThemePreset, getThemePresetToken } from '@/utils/themePreset'

const preferredDark = usePreferredDark()

const themeMode = ref(getSavedThemeMode())
const appliedIsDark = computed(
  () => themeMode.value === 'dark' || (themeMode.value === 'system' && preferredDark.value)
)

const uiDensity = ref(getUiDensity())
const themePreset = ref(getThemePreset())

const antdComponentSize = computed(() => (uiDensity.value === 'compact' ? 'small' : 'middle'))
const antdThemeConfig = computed(() => ({
  ...themeConfig,
  token: {
    ...(themeConfig.token || {}),
    ...getThemePresetToken(themePreset.value)
  },
  ...(appliedIsDark.value ? { algorithm: antdTheme.darkAlgorithm } : {})
}))

const syncThemeMode = () => {
  themeMode.value = getSavedThemeMode()
}
const syncUiDensity = () => {
  uiDensity.value = getUiDensity()
}
const syncThemePreset = () => {
  themePreset.value = getThemePreset()
}

onMounted(() => {
  window.addEventListener(THEME_MODE_CHANGED_EVENT, syncThemeMode)
  window.addEventListener('ui-density-changed', syncUiDensity)
  window.addEventListener('ui-theme-preset-changed', syncThemePreset)
})

onUnmounted(() => {
  window.removeEventListener(THEME_MODE_CHANGED_EVENT, syncThemeMode)
  window.removeEventListener('ui-density-changed', syncUiDensity)
  window.removeEventListener('ui-theme-preset-changed', syncThemePreset)
})
</script>
