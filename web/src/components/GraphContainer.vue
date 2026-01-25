<template>
  <div class="graph-container" ref="container"></div>
</template>

<script setup>
import { Graph } from '@antv/g6'
import { onMounted, onUnmounted, watch, ref } from 'vue'

const props = defineProps({
  graphData: {
    type: Object,
    required: true,
    default: () => ({ nodes: [], edges: [] })
  }
})

const container = ref(null)
let graphInstance = null
let resizeObserver = null

const cssVar = (name, fallback) => {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
    return v || fallback
  } catch {
    return fallback
  }
}

const initGraph = () => {
  const surface = cssVar('--surface-color', '#fff')
  const text = cssVar('--text-color', '#000')
  const border = cssVar('--border-color', '#e6e6e6')
  graphInstance = new Graph({
    container: container.value,
    width: container.value.offsetWidth,
    height: container.value.offsetHeight,
    autoFit: true,
    autoResize: true,
    layout: {
      type: 'd3-force',
      preventOverlap: true,
      kr: 20,
      collide: {
        strength: 1.0
      }
    },
    node: {
      type: 'circle',
      style: {
        labelText: (d) => d.data.label,
        size: 70
      },
      palette: {
        field: 'label',
        color: 'tableau'
      }
    },
    edge: {
      type: 'line',
      style: {
        labelText: (d) => d.data.label,
        labelFill: text,
        labelBackground: surface,
        stroke: border,
        endArrow: true
      }
    },
    behaviors: ['drag-element', 'zoom-canvas', 'drag-canvas']
  })
}

const renderGraph = () => {
  if (!graphInstance) initGraph()

  const formattedData = {
    nodes: props.graphData.nodes.map((node) => ({
      id: node.id,
      data: { label: node.name }
    })),
    edges: props.graphData.edges.map((edge) => ({
      source: edge.source_id,
      target: edge.target_id,
      data: { label: edge.type }
    }))
  }

  graphInstance.setData(formattedData)
  graphInstance.render()
}

onMounted(() => {
  renderGraph()

  // Resize based on the actual container size (modal open/close, viewport resize, etc).
  if (typeof ResizeObserver !== 'undefined' && container.value) {
    resizeObserver = new ResizeObserver(() => {
      if (!graphInstance || !container.value) return
      try {
        graphInstance.resize?.(container.value.offsetWidth, container.value.offsetHeight)
        graphInstance.render?.()
      } catch {
        // ignore
      }
    })
    resizeObserver.observe(container.value)
  }
})

watch(() => props.graphData, renderGraph, { deep: true })

onUnmounted(() => {
  try {
    resizeObserver?.disconnect?.()
  } catch {
    // ignore
  }
  resizeObserver = null

  if (graphInstance) {
    try {
      graphInstance.destroy()
    } catch {
      // ignore
    }
    graphInstance = null
  }
})
</script>

<style scoped>
.graph-container {
  background: var(--surface-color-2);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  width: 100%;
  height: 600px;
  overflow: hidden;
}
</style>
