<template>
  <transition name="slide-fade">
    <div class="detail-card" v-if="visible">
      <div class="info-card">
        <div class="card-header">
          <span class="title">{{ title }}</span>
          <a-button type="text" size="small" @click="$emit('close')" class="close-btn">
             <CloseOutlined />
          </a-button>
        </div>

        <div class="card-content">
          <template v-if="item">
            <a-descriptions :column="1" size="small" :bordered="false" class="custom-desc">
              <template v-if="type === 'node'">
                <a-descriptions-item label="名称">
                    <span class="value-text">{{ item.data?.label }}</span>
                </a-descriptions-item>
                <a-descriptions-item label="ID">
                    <span class="mono-text">{{ item.id }}</span>
                </a-descriptions-item>

                <!-- 原始属性 -->
                <template v-if="item.data?.original?.properties">
                  <a-descriptions-item
                    v-for="(value, key) in item.data.original.properties"
                    :key="key"
                    :label="key"
                  >
                   <span class="value-text">{{ value }}</span>
                  </a-descriptions-item>
                </template>

                <!-- 标签 -->
                <a-descriptions-item label="标签" v-if="item.data?.original?.labels">
                  <div class="tags-container">
                    <a-tag v-for="tag in item.data.original.labels" :key="tag" color="orange" :bordered="false">{{
                      tag
                    }}</a-tag>
                  </div>
                </a-descriptions-item>
                
                <a-descriptions-item label="度" v-if="item.data?.degree !== undefined">
                    <a-tag>{{ item.data.degree }} Links</a-tag>
                </a-descriptions-item>
              </template>

              <template v-else-if="type === 'edge'">
                <a-descriptions-item label="类型">
                    <a-tag color="blue">{{ item.data?.label }}</a-tag>
                </a-descriptions-item>
                <a-descriptions-item label="源节点">
                    <span class="mono-text">{{ item.source }}</span>
                </a-descriptions-item>
                <a-descriptions-item label="目标节点">
                    <span class="mono-text">{{ item.target }}</span>
                </a-descriptions-item>
              </template>
            </a-descriptions>
          </template>
        </div>
      </div>
    </div>
  </transition>
</template>

<script setup>
import { computed } from 'vue'
import { CloseOutlined } from '@ant-design/icons-vue'

const props = defineProps({
  visible: Boolean,
  item: Object,
  type: String // 'node' | 'edge'
})

defineEmits(['close'])

const title = computed(() => {
  return props.type === 'node' ? '节点详情' : '关系详情'
})
</script>

<style scoped lang="less">
.detail-card {
  position: absolute;
  top: 80px;
  right: 24px;
  width: 340px;
  max-height: calc(100% - 100px);
  overflow-y: auto;
  z-index: 100;
  pointer-events: auto;

  .info-card {
    background: var(--surface-overlay);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-xl);
    border-radius: 16px;
    border: 1px solid var(--border-color);
    overflow: hidden;
    padding: 0;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
    background: rgba(255, 255, 255, 0.5);
    
    .title {
        font-weight: 600;
        font-size: 15px;
        color: var(--text-color);
    }
    
    .close-btn {
        color: var(--subtext-color);
        &:hover {
            color: var(--text-color);
            background: rgba(0,0,0,0.05);
        }
    }
  }
  
  .card-content {
      padding: 20px;
      
      .value-text {
          color: var(--text-color);
          font-weight: 500;
      }
      
      .mono-text {
          font-family: var(--font-mono);
          font-size: 12px;
          color: var(--subtext-color);
          background: var(--slate-100);
          padding: 2px 6px;
          border-radius: 4px;
      }
  }

  .tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
}

[data-theme='dark'] .detail-card .card-header {
    background: rgba(30, 41, 59, 0.5);
}

[data-theme='dark'] .mono-text {
    background: var(--slate-800);
}

/* Ant Design Overrides for cleaner look */
:deep(.ant-descriptions-item-label) {
    color: var(--subtext-color) !important;
    font-size: 13px !important;
    width: 80px;
}
:deep(.ant-descriptions-item-content) {
    color: var(--text-color) !important;
    font-size: 13px !important;
}

/* Transitions */
.slide-fade-enter-active {
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.slide-fade-leave-active {
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  transform: translateX(30px);
  opacity: 0;
}
</style>
