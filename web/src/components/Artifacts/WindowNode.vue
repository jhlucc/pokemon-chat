<template>
  <div class="window-node" :class="{ selected: selected }">
    <div class="traffic-lights">
       <div class="traffic-light red"></div>
       <div class="traffic-light yellow"></div>
       <div class="traffic-light green"></div>
    </div>
    <div class="node-content">
        <div class="node-title">{{ label }}</div>
        <div class="node-sub" v-if="data.status">{{ data.status }}</div>
    </div>
    <Handle type="target" :position="Position.Top" />
    <Handle type="source" :position="Position.Bottom" />
  </div>
</template>

<script setup>
import { Handle, Position } from '@vue-flow/core'

defineProps({
  label: String,
  data: {
      type: Object,
      default: () => ({})
  },
  selected: Boolean
});
</script>

<style scoped>
.window-node {
    background: var(--surface-card, #fff);
    border: 1px solid var(--border-color, #e2e8f0);
    border-radius: 8px;
    padding: 8px;
    min-width: 140px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.2s;
}

.window-node.selected {
    border-color: var(--primary-color, #f97316);
    box-shadow: 0 0 0 2px var(--primary-bg-light, #fff7ed);
}

.traffic-lights {
    display: flex;
    gap: 4px;
    margin-bottom: 8px;
}

.traffic-light {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.red { background: #ef4444; }
.yellow { background: #f59e0b; }
.green { background: #10b981; }

.node-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-color, #1e293b);
    text-align: center;
}

.node-sub {
    font-size: 11px;
    color: var(--subtext-color, #64748b);
    text-align: center;
    margin-top: 2px;
}
</style>
