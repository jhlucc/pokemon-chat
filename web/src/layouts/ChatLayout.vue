<template>
  <div class="chat-layout">
    <splitpanes class="default-theme" @resize="onResize">
      <!-- Main Chat Pane -->
      <pane :min-size="30" :size="chatPaneSize">
        <div class="chat-pane-content">
           <slot name="chat"></slot>
        </div>
      </pane>

      <!-- Artifacts Pane (Optional) -->
      <pane v-if="showArtifacts" :size="100 - chatPaneSize" :min-size="20">
         <div class="artifacts-pane-content">
             <div class="artifacts-header">
                 <span class="title">Artifacts</span>
                 <a-button type="text" size="small" @click="$emit('close-artifacts')">
                     <CloseOutlined />
                 </a-button>
             </div>
             <div class="artifacts-body">
                 <slot name="artifacts"></slot>
             </div>
         </div>
      </pane>
    </splitpanes>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'
import { CloseOutlined } from '@ant-design/icons-vue'

const props = defineProps({
    showArtifacts: {
        type: Boolean,
        default: false
    }
})

 defineEmits(['close-artifacts'])

const chatPaneSize = ref(100)

const onResize = (val) => {
    // Save preference if needed
}

// Watch showArtifacts to adjust sizes automatically
import { watch } from 'vue'
watch(() => props.showArtifacts, (newVal) => {
    if(newVal) {
        chatPaneSize.value = 60 // Default split 60/40
    } else {
        chatPaneSize.value = 100
    }
}, { immediate: true })

</script>

<style lang="less" scoped>
.chat-layout {
    width: 100%;
    height: 100%;
}

.chat-pane-content {
    height: 100%;
    width: 100%;
    overflow: hidden;
    position: relative;
    /* Grid background from body shows through */
    background: transparent; 
}

.artifacts-pane-content {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    background: var(--surface-card); /* Solid background for artifacts to be readable */
    border-left: 1px solid var(--border-color);
    box-shadow: var(--shadow-xl); /* Lift it up */
}

.artifacts-header {
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--surface-card); /* Match card */
    
    .title {
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: 13px;
        color: var(--text-color);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
}

.artifacts-body {
    flex: 1;
    overflow: hidden;
    position: relative;
    background: var(--surface-card);
}

/* Splitpanes Theme Overrides */
:deep(.splitpanes__splitter) {
    background-color: transparent;
    border-left: 1px solid var(--border-color);
    width: 6px;
    position: relative;
    
    &:hover {
        background-color: rgba(0,0,0,0.02);
    }
    
    /* Handle grip */
    &:after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 2px;
        height: 24px;
        background-color: var(--border-color);
        border-radius: 1px;
    }
}
</style>
