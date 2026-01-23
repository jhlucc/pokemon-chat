<template>
  <div class="code-preview">
      <div class="code-actions">
          <span class="lang-badge">{{ language }}</span>
          <a-button size="small" type="text" @click="copyCode">
              <CopyOutlined /> Copy
          </a-button>
      </div>
      <div class="code-content">
        <pre><code>{{ content }}</code></pre>
      </div>
  </div>
</template>

<script setup>
import { message } from 'ant-design-vue';
import { CopyOutlined } from '@ant-design/icons-vue';

const props = defineProps({
    content: String,
    language: String
})

const copyCode = () => {
    navigator.clipboard.writeText(props.content).then(() => {
        message.success("Copied to clipboard")
    })
}
</script>

<style scoped lang="less">
.code-preview {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #1e1e1e; /* VS Code-ish dark */
    color: #d4d4d4;
}

.code-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: #252526;
    border-bottom: 1px solid #333;
    
    .lang-badge {
        font-size: 12px;
        color: #9cdcfe;
        text-transform: uppercase;
    }
    
    button {
        color: #cccccc;
        &:hover { color: white; }
    }
}

.code-content {
    flex: 1;
    overflow: auto;
    padding: 16px;
    
    pre {
        margin: 0;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 14px;
        line-height: 1.5;
    }
}
</style>
