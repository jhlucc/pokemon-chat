<template>
  <div class="workflow-visualizer">
    <VueFlow 
        v-model="elements" 
        :fit-view-on-init="true"
        :node-types="nodeTypes"
    >
        <Background pattern-color="#ccc" gap="16" size="1" />
        <Controls />
    </VueFlow>
  </div>
</template>

<script setup>
import { ref, markRaw } from 'vue'
import { VueFlow } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import WindowNode from './WindowNode.vue'

import '@vue-flow/core/dist/style.css'
import '@vue-flow/core/dist/theme-default.css'
import '@vue-flow/controls/dist/style.css'

// Register custom node type
const nodeTypes = {
  'window-node': markRaw(WindowNode),
}

const props = defineProps({
    steps: {
        type: Array,
        default: () => []
    }
})

// Use window-node type
const elements = ref([
  { id: '1', type: 'window-node', label: 'User Query', position: { x: 250, y: 5 }, data: { status: 'Received' } },
  { id: '2', type: 'window-node', label: 'Retrieval', position: { x: 100, y: 120 }, data: { status: 'Processing' } },
  { id: '3', type: 'window-node', label: 'Rerank', position: { x: 400, y: 120 }, data: { status: 'Pending' } },
  { id: '4', type: 'window-node', label: 'Generation', position: { x: 250, y: 240 }, data: { status: 'Waiting' } },
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
    /* Transparent to let parent grid show, or specific grid */
    background: transparent;
}
</style>
