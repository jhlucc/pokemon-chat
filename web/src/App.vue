<template>
  <AppErrorBoundary>
    <router-view />
  </AppErrorBoundary>
</template>

<script setup lang="ts">
import AppErrorBoundary from '@/components/AppErrorBoundary.vue'
</script>
