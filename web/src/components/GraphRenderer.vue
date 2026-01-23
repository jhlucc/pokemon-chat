<template>
  <div class="graph-container" ref="containerRef"></div>
</template>

<script setup>
import { onMounted, ref, watch, onUnmounted } from 'vue';
import { Graph } from '@antv/g6';

const props = defineProps({
  data: {
    type: [Object, Array],
    required: true
  }
});

const containerRef = ref(null);
let graph = null;

const initGraph = () => {
  if (!containerRef.value) return;

  // Convert raw data (List[Dict]) to G6 format (nodes/edges)
  // Heuristic: Iterate keys in the list items.
  // If value has 'elementId' or 'id', treat as Node.
  // If value is a simple value, treat as Property.
  
  const nodes = new Map();
  const edges = new Map();
  
  const rawList = Array.isArray(props.data) ? props.data : [props.data];
  
  rawList.forEach(row => {
      Object.values(row).forEach(item => {
          // Detect Node (heuristic: has 'elementId' or labels/properties)
          // LangChain graph result might be messy.
          // Assuming we get something that we can parse or at least visualize as entities.
          // If it's a simple string, make a node for it.
          
          if (typeof item === 'object' && item !== null) {
             // Likely a Node object from Neo4j driver
             const id = item.elementId || item.id || JSON.stringify(item);
             if (!nodes.has(id)) {
                 nodes.set(id, {
                     id,
                     label: item.properties?.name || item.name || item.title || id.substring(0, 10),
                     ...item
                 });
             }
          } else {
             // Value
             const id = String(item);
             if (!nodes.has(id)) {
                 nodes.set(id, {
                     id,
                     label: id,
                     isValue: true
                 })
             }
          }
      });
  });
  
  // Create edges between items in the same row?
  // Heuristic: Connect first item to others?
  rawList.forEach(row => {
      const items = Object.values(row);
      if (items.length > 1) {
          const source = items[0];
          const sourceId = typeof source === 'object' ? (source.elementId || source.id) : String(source);
          
          for (let i = 1; i < items.length; i++) {
              const target = items[i];
              const targetId = typeof target === 'object' ? (target.elementId || target.id) : String(target);
              
              const edgeId = `${sourceId}-${targetId}`;
              if (!edges.has(edgeId) && sourceId !== targetId) {
                  edges.set(edgeId, {
                      source: sourceId,
                      target: targetId,
                  });
              }
          }
      }
  });

  const data = {
    nodes: Array.from(nodes.values()),
    edges: Array.from(edges.values()),
  };
  
  if (data.nodes.length === 0) return;

  graph = new Graph({
    container: containerRef.value,
    width: containerRef.value.scrollWidth || 600,
    height: 400,
    fitView: true,
    modes: {
      default: ['drag-canvas', 'zoom-canvas', 'drag-node'],
    },
    layout: {
      type: 'force',
      preventOverlap: true,
      linkDistance: 100,
    },
    defaultNode: {
      size: 30,
      style: {
        fill: '#C6E5FF',
        stroke: '#5B8FF9',
      },
      labelCfg: {
        position: 'bottom',
      },
    },
    defaultEdge: {
      style: {
        endArrow: true,
      },
    },
  });

  graph.data(data);
  graph.render();
};

onMounted(() => {
  initGraph();
});

watch(() => props.data, () => {
    if (graph) {
        graph.destroy();
    }
    initGraph();
});

onUnmounted(() => {
    if (graph) graph.destroy();
});
</script>

<style scoped>
.graph-container {
  width: 100%;
  height: 400px;
  background: #f9f9f9;
  border-radius: 8px;
  border: 1px solid #eee;
}
</style>
