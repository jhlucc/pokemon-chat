<template>
  <div class="markdown-body" v-html="renderedContent"></div>
</template>

<script setup>
import { computed } from 'vue'
import { Marked } from 'marked'
import { markedHighlight } from 'marked-highlight'
import hljs from 'highlight.js'
import 'highlight.js/styles/atom-one-light.css'

const props = defineProps({
  content: {
    type: String,
    default: ''
  }
})

// Configure Marked with Highlight.js
// 实例化一次即可，无需在每次渲染时重复创建
const marked = new Marked(
  markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
      const language = hljs.getLanguage(lang) ? lang : 'plaintext'
      return hljs.highlight(code, { language }).value
    }
  })
)

marked.setOptions({
  breaks: true, // Enable GFM line breaks
  gfm: true
})

const renderedContent = computed(() => {
  return marked.parse(props.content || '')
})
</script>

<style lang="less" scoped>
/* ===== Markdown Content Styles (Clean & Lightweight) ===== */
.markdown-body {
  /* Inherit text properties to blend seamlessly */
  font-size: inherit;
  line-height: inherit;
  color: inherit;
  
  /* Remove default margins to respect container padding */
  margin: 0;
  padding: 0;
  
  /* Ensure it behaves like a text block */
  display: inline-block;
  width: auto;
  min-width: 0;
  max-width: 100%;

  /* Paragraphs */
  :deep(p) {
    margin: 0 0 8px 0; /* Tighter paragraph spacing */
  }
  :deep(p:last-child) {
    margin-bottom: 0;
  }

  /* Lists */
  :deep(ul), :deep(ol) {
    padding-left: 20px;
    margin: 0 0 8px 0;
  }
  :deep(ul:last-child), :deep(ol:last-child) {
    margin-bottom: 0;
  }
  :deep(li) {
    margin-bottom: 4px;
  }

  /* Code Blocks */
  :deep(pre) {
    margin: 8px 0;
    padding: 12px;
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
    overflow-x: auto;
    font-family: var(--font-family-mono);
    font-size: 0.9em;
  }
  :deep(pre:first-child) { margin-top: 0; }
  :deep(pre:last-child) { margin-bottom: 0; }

  /* Inline Code */
  :deep(code) {
    font-family: var(--font-family-mono);
    background: rgba(0, 0, 0, 0.05);
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 0.9em;
  }
  :deep(pre code) {
    background: transparent;
    padding: 0;
  }

  /* Links */
  :deep(a) {
    color: var(--primary-color);
    text-decoration: none;
    &:hover { text-decoration: underline; }
  }

  /* Headings */
  :deep(h1), :deep(h2), :deep(h3), :deep(h4), :deep(h5), :deep(h6) {
    font-weight: 600;
    margin: 16px 0 8px 0;
    line-height: 1.4;
  }
  :deep(h1:first-child), :deep(h2:first-child), :deep(h3:first-child) {
    margin-top: 0;
  }
}
</style>
