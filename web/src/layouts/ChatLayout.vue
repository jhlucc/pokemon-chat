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
    background: var(--background-color);
}

.artifacts-pane-content {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    background: var(--surface-card);
    border-left: 1px solid var(--border-color);
}

.artifacts-header {
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--surface-overlay);
    backdrop-filter: blur(8px);
    
    .title {
        font-weight: 600;
        color: var(--text-color);
    }
}

.artifacts-body {
    flex: 1;
    overflow: hidden;
    position: relative;
}

/* Splitpanes Theme Overrides */
:deep(.splitpanes__splitter) {
    background-color: var(--border-color);
    width: 4px;
    position: relative;
    
    &:before, &:after {
        background-color: var(--subtext-color);
        opacity: 0.3;
    }
    
    &:hover {
        background-color: var(--primary-light-color);
    }
}
</style>
