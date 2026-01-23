<template>
  <div class="pdf-preview">
      <div class="pdf-toolbar">
          <div class="page-controls">
              <a-button size="small" :disabled="page <= 1" @click="page--">
                  <LeftOutlined />
              </a-button>
              <span class="page-info">{{ page }} / {{ pages }}</span>
              <a-button size="small" :disabled="page >= pages" @click="page++">
                  <RightOutlined />
              </a-button>
          </div>
          <div class="zoom-controls">
              <a-button size="small" @click="scale -= 0.1"><MinusOutlined /></a-button>
              <span class="zoom-info">{{ Math.round(scale * 100) }}%</span>
              <a-button size="small" @click="scale += 0.1"><PlusOutlined /></a-button>
          </div>
      </div>
      
      <div class="pdf-container">
          <VuePdfEmbed
            ref="pdfRef"
            :source="source"
            :page="page"
            :scale="scale"
            @loaded="onLoaded"
          />
      </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import VuePdfEmbed from 'vue-pdf-embed'
import { LeftOutlined, RightOutlined, PlusOutlined, MinusOutlined } from '@ant-design/icons-vue'

const props = defineProps({
    source: {
        type: [String, Object], // URL or Base64
        required: true
    },
    initialPage: {
        type: Number,
        default: 1
    }
})

const pdfRef = ref(null)
const page = ref(props.initialPage)
const pages = ref(0)
const scale = ref(1.0)

const onLoaded = (pdf) => {
    pages.value = pdf.numPages
}

watch(() => props.initialPage, (val) => {
    if(val) page.value = val
})

</script>

<style scoped lang="less">
.pdf-preview {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--gray-100);
}

.pdf-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: white;
    border-bottom: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
    z-index: 10;
    
    .page-controls, .zoom-controls {
        display: flex;
        align-items: center;
        gap: 8px;
        
        span {
            font-size: 13px;
            min-width: 40px;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }
    }
}

.pdf-container {
    flex: 1;
    overflow: auto;
    padding: 20px;
    display: flex;
    justify-content: center;
}
</style>
