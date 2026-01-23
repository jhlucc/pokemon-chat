<template>
  <div class="artifacts-view">
    <CodePreview 
        v-if="type === 'code'" 
        :content="content" 
        :language="language" 
    />
    <PdfPreview
        v-else-if="type === 'pdf'"
        :source="content"
        :initial-page="page"
    />
    <div v-else-if="type ==='html'" class="html-preview">
        <!-- Secure iframe for HTML content -->
        <iframe :srcdoc="content" sandbox="allow-scripts" frameborder="0"></iframe>
    </div>
    <WorkflowVisualizer 
        v-else-if="type === 'workflow'"
    />
    <div v-else class="empty-artifact">
        <a-empty description="Select an artifact to view" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import CodePreview from './CodePreview.vue'
import PdfPreview from './PdfPreview.vue'
import WorkflowVisualizer from './WorkflowVisualizer.vue'

const props = defineProps({
    artifact: {
        type: Object,
        default: () => ({})
    }
})

const type = computed(() => props.artifact?.type || 'none')
const content = computed(() => props.artifact?.content || '')
const language = computed(() => props.artifact?.language || 'text')
// Optional extra properties
const page = computed(() => props.artifact?.page || 1)

</script>

<style scoped>
.artifacts-view {
    width: 100%;
    height: 100%;
    overflow: hidden;
}
.empty-artifact {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
}
.html-preview {
    width: 100%;
    height: 100%;
    iframe {
        width: 100%;
        height: 100%;
        border: none;
        background: white;
    }
}
</style>
