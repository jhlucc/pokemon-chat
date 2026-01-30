<template>
  <div class="db-layout">
    <div class="ambient-glow glow--orange"></div>
    <div class="ambient-glow glow--purple"></div>
    <div class="dot-grid"></div>
    <div class="db-layout__content">
      <slot />
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped lang="less">
.db-layout {
  position: relative;
  min-height: 100vh;
  background: transparent;
  overflow-x: hidden;
}

.ambient-glow {
  position: fixed;
  border-radius: 50%;
  filter: blur(100px);
  pointer-events: none;
  z-index: 0;
  mix-blend-mode: normal;
  animation: glow-drift 20s ease-in-out infinite;
}

.glow--orange {
  width: 700px;
  height: 700px;
  top: -25%;
  left: -20%;
  background: radial-gradient(circle, rgba(255, 125, 0, 0.35) 0%, rgba(255, 180, 100, 0.15) 40%, transparent 70%);
}

.glow--purple {
  width: 550px;
  height: 550px;
  bottom: -15%;
  right: -15%;
  background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, rgba(180, 150, 255, 0.1) 40%, transparent 70%);
  animation-delay: -7s;
}

@keyframes glow-drift {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(20px, -15px) scale(1.03); }
  66% { transform: translate(-15px, 15px) scale(0.97); }
}

.dot-grid {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    radial-gradient(circle, rgba(255, 125, 0, 0.05) 1px, transparent 1px),
    radial-gradient(circle, rgba(255, 125, 0, 0.025) 1px, transparent 1px);
  background-size: 24px 24px, 96px 96px;
  pointer-events: none;
  z-index: 0;
}

.db-layout__content {
  position: relative;
  z-index: 1;
}

:root[data-theme='dark'] {
  .ambient-glow { mix-blend-mode: screen; opacity: 0.7; }
  .glow--orange { background: radial-gradient(circle, rgba(255, 125, 0, 0.25) 0%, rgba(255, 180, 100, 0.1) 40%, transparent 70%); }
  .glow--purple { background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, rgba(180, 150, 255, 0.08) 40%, transparent 70%); }
  .dot-grid { background-image: radial-gradient(circle, rgba(255, 125, 0, 0.06) 1px, transparent 1px), radial-gradient(circle, rgba(255, 125, 0, 0.03) 1px, transparent 1px); }
}
</style>
