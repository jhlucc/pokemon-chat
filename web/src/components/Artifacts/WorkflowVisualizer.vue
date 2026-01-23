<template>
  <div class="workflow-visualizer">
    <VueFlow v-model="elements" :fit-view-on-init="true">
        <Background pattern-color="#aaa" gap="8" />
        <Controls />
        
        <!-- Custom Nodes could go here -->
    </VueFlow>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { VueFlow } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'

import '@vue-flow/core/dist/style.css'
import '@vue-flow/core/dist/theme-default.css'
import '@vue-flow/controls/dist/style.css'

const props = defineProps({
    steps: {
        type: Array,
        default: () => []
    }
})

// Mock initial data or transform props
const elements = ref([
  { id: '1', type: 'input', label: 'User Query', position: { x: 250, y: 5 }, class: 'light' },
  { id: '2', label: 'Retrieval', position: { x: 100, y: 100 }, class: 'light' },
  { id: '3', label: 'Rerank', position: { x: 400, y: 100 }, class: 'light' },
  { id: '4', label: 'Generation', position: { x: 250, y: 200 }, class: 'light' },
  { id: 'e1-2', source: '1', target: '2', animated: true },
  { id: 'e1-3', source: '1', target: '3', animated: true },
  { id: 'e2-4', source: '2', target: '4' },
  { id: 'e3-4', source: '3', target: '4' },
])

</script>

<style scoped>
.workflow-visualizer {
    width: 100%;
    height: 100%;
    background: white;
}
</style>
