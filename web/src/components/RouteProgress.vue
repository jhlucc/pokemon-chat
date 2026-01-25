<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const visible = ref(false)
const percent = ref(0)
const reducedMotion = ref(false)

let motionMql: MediaQueryList | null = null
let removeMotionListener: null | (() => void) = null

let tickTimer: number | null = null
let doneTimer: number | null = null
let removeBefore: null | (() => void) = null
let removeAfter: null | (() => void) = null

function clearTimers(): void {
  if (tickTimer) window.clearInterval(tickTimer)
  if (doneTimer) window.clearTimeout(doneTimer)
  tickTimer = null
  doneTimer = null
}

function start(): void {
  clearTimers()
  visible.value = true
  percent.value = 12

  // Respect reduced motion: keep UI snappy without the fake incremental animation.
  if (reducedMotion.value) {
    percent.value = 70
    return
  }

  // Fake progress: quickly approach ~85% while async chunks load.
  tickTimer = window.setInterval(() => {
    if (!visible.value) return
    if (percent.value >= 85) return
    percent.value = Math.min(85, percent.value + Math.max(1, (90 - percent.value) * 0.06))
  }, 120)
}

function done(): void {
  clearTimers()
  percent.value = 100
  const hideDelay = reducedMotion.value ? 120 : 220
  doneTimer = window.setTimeout(() => {
    visible.value = false
    percent.value = 0
  }, hideDelay)
}

onMounted(() => {
  try {
    motionMql = window.matchMedia?.('(prefers-reduced-motion: reduce)') || null
    const syncMotion = () => (reducedMotion.value = Boolean(motionMql?.matches))
    syncMotion()
    // Safari < 14 uses addListener/removeListener.
    const legacyMql = motionMql as MediaQueryList & {
      addListener?: (listener: () => void) => void
      removeListener?: (listener: () => void) => void
    }
    if (motionMql?.addEventListener) {
      motionMql.addEventListener('change', syncMotion)
      removeMotionListener = () => motionMql?.removeEventListener('change', syncMotion)
    } else if (legacyMql?.addListener) {
      legacyMql.addListener(syncMotion)
      removeMotionListener = () => legacyMql?.removeListener?.(syncMotion)
    }
  } catch {
    // ignore
  }

  removeBefore = router.beforeEach((_to, _from, next) => {
    start()
    next()
  })
  removeAfter = router.afterEach(() => done())
  router.onError(() => done())
})

onUnmounted(() => {
  clearTimers()

  try {
    removeMotionListener?.()
  } catch {
    // ignore
  }
  removeMotionListener = null
  motionMql = null

  try {
    removeBefore?.()
  } catch {
    // ignore
  }
  try {
    removeAfter?.()
  } catch {
    // ignore
  }
})
</script>

<template>
  <div
    class="route-progress"
    :class="{ 'is-visible': visible }"
    :style="{ width: `${percent}%` }"
    aria-hidden="true"
  />
</template>

<style scoped>
.route-progress {
  position: fixed;
  top: 0;
  left: 0;
  height: 2px;
  width: 0%;
  background: linear-gradient(90deg, var(--primary-color), var(--primary-light-color));
  box-shadow: 0 0 10px color-mix(in srgb, var(--primary-color) 35%, transparent);
  opacity: 0;
  z-index: 3000;
  transition: width 0.18s ease, opacity 0.18s ease;
  will-change: width, opacity;
  pointer-events: none;
}

.route-progress.is-visible {
  opacity: 1;
}
</style>
