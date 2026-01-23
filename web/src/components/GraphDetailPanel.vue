<template>
  <transition name="slide-fade">
    <div class="detail-card" v-if="visible">
      <div class="info-card">
        <div class="card-header">
          <span class="title">{{ title }}</span>
          <CloseOutlined @click="$emit('close')" class="close-icon" />
        </div>

        <div class="card-content">
          <template v-if="item">
            <a-descriptions :column="1" size="small" :bordered="false" class="custom-desc">
              <template v-if="type === 'node'">
                <a-descriptions-item label="名称">{{ item.data?.label }}</a-descriptions-item>
                <a-descriptions-item label="ID">{{ item.id }}</a-descriptions-item>

                <!-- 原始属性 -->
                <template v-if="item.data?.original?.properties">
                  <a-descriptions-item
                    v-for="(value, key) in item.data.original.properties"
                    :key="key"
                    :label="key"
                  >
                    {{ value }}
                  </a-descriptions-item>
                </template>

                <!-- 标签 -->
                <a-descriptions-item label="标签" v-if="item.data?.original?.labels">
                  <div class="tags-container">
                    <a-tag v-for="tag in item.data.original.labels" :key="tag" color="blue">{{
                      tag
                    }}</a-tag>
                  </div>
                </a-descriptions-item>
                
                <a-descriptions-item label="度" v-if="item.data?.degree !== undefined">
                    {{ item.data.degree }}
                </a-descriptions-item>
              </template>

              <template v-else-if="type === 'edge'">
                <a-descriptions-item label="类型">{{ item.data?.label }}</a-descriptions-item>
                <a-descriptions-item label="源节点">{{ item.source }}</a-descriptions-item>
                <a-descriptions-item label="目标节点">{{ item.target }}</a-descriptions-item>
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
  width: 320px;
  max-height: calc(100% - 100px);
  overflow-y: auto;
  z-index: 100;
  pointer-events: auto;

  .info-card {
    background: var(--surface-overlay); // Use our new variable
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--shadow-lg);
    border-radius: 16px;
    border: 1px solid var(--border-color);
    overflow: hidden;
    padding: 0;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    background: rgba(255, 255, 255, 0.5);
    
    .title {
        font-weight: 600;
        font-size: 14px;
        color: var(--text-color);
    }

    .close-icon {
      cursor: pointer;
      color: var(--subtext-color);
      transition: color 0.2s;
      font-size: 14px;

      &:hover {
        color: var(--text-color);
      }
    }
  }
  
  .card-content {
      padding: 12px 16px;
  }

  .tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
}

/* Ant Design Overrides for cleaner look */
:deep(.ant-descriptions-item-label) {
    color: var(--subtext-color) !important;
    font-size: 13px !important;
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
  transform: translateX(20px);
  opacity: 0;
}
</style>
