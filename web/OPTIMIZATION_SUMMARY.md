# Pokemon-Chat 前端优化总结

## 优化概览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段一 | 样式系统标准化 | ✅ 完成 |
| 阶段二 | 组件重构与 Composables | ✅ 完成 |
| 阶段三 | 性能优化 | ✅ 完成 |
| 阶段四 | UX 增强与无障碍 | ✅ 完成 |

---

## 阶段一：样式系统标准化

### 新建文件

#### `src/assets/tokens.css`
集中管理设计令牌，消除硬编码值：

```css
:root {
  /* 渐变色令牌 */
  --gradient-primary: linear-gradient(135deg, var(--pokedex-red), #ff8a88);
  --gradient-primary-hover: linear-gradient(135deg, var(--primary-light-color), #ffb3b0);

  /* 消息气泡专用令牌 */
  --message-user-text: #ffffff;
  --message-user-shadow: 0 4px 12px rgba(255, 83, 80, 0.2);
  --message-user-radius: var(--radius-md) var(--radius-md) var(--radius-xs) var(--radius-md);

  /* 工具栏令牌 */
  --toolbar-btn-size: 30px;
  --toolbar-gap: var(--space-1);

  /* 模糊效果令牌 */
  --blur-sm: 8px;
  --blur-md: 12px;
  --blur-lg: 20px;
}
```

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `main.css` | 导入 tokens.css，清理 legacy 变量 |
| `ChatComponent.vue` | 使用 `var(--glass-bg)`, `var(--radius-*)` 等变量 |
| `MessageComponent.vue` | 使用 `var(--gradient-primary)`, `var(--message-user-text)` |
| `ConversationList.vue` | 使用 `color-mix()` 替代 `--main-*` 变量 |
| `ChatCapabilityBar.vue` | 统一使用设计令牌 |

### 变量迁移对照表

| 旧变量 | 新变量/方法 |
|--------|------------|
| `--main-5` | `color-mix(in srgb, var(--primary-color) 5%, transparent)` |
| `--main-10` | `color-mix(in srgb, var(--primary-color) 10%, transparent)` |
| `--main-400` | `var(--primary-light-color)` |
| `--main-500` | `var(--primary-color)` |
| `--main-600` | `var(--primary-light-color)` |
| `rgba(255,255,255,0.7)` | `var(--glass-bg)` |

---

## 阶段二：组件重构与 Composables

### 类型定义 (`src/types/chat.ts`)

```typescript
// 核心类型
export interface Message { ... }
export interface Conversation { ... }
export interface ChatMeta { ... }

// 组件 Props 类型
export interface MessageComponentProps { ... }
export interface ChatComponentProps { ... }
```

### Composables

| Composable | 功能 | 文件 |
|------------|------|------|
| `useChat` | 聊天核心逻辑（发送/接收/流式处理） | `src/composables/useChat.ts` |
| `useChatMeta` | 元数据管理与持久化 | `src/composables/useChatMeta.ts` |
| `useChatScroll` | 滚动行为管理 | `src/composables/useChatScroll.ts` |
| `useMessageUpdate` | 消息 CRUD 操作 | `src/composables/useMessageUpdate.ts` |
| `useRetry` | 指数退避重试 | `src/composables/useRetry.ts` |
| `useOnlineStatus` | 网络状态检测 | `src/composables/useOnlineStatus.ts` |

### 使用示例

```typescript
import { useChatMeta } from '@/composables/useChatMeta'
import { useChatScroll } from '@/composables/useChatScroll'

const { meta, toggleAgent, toggleWebSearch } = useChatMeta()
const { scrollToBottom, showScrollToBottom } = useChatScroll(containerRef)
```

### 新增子组件

| 组件 | 路径 | 功能 |
|------|------|------|
| ChatHeader | `src/components/chat/ChatHeader.vue` | 头部导航 |
| ModelSelector | `src/components/chat/ModelSelector.vue` | 模型选择下拉 |
| ChatOptionsPanel | `src/components/chat/ChatOptionsPanel.vue` | 选项面板 |
| MessageToolbar | `src/components/message/MessageToolbar.vue` | 消息工具栏 |
| ReasoningBox | `src/components/message/ReasoningBox.vue` | 推理过程展示 |
| LazyImage | `src/components/common/LazyImage.vue` | 懒加载图片 |

---

## 阶段三：性能优化

### Vite 配置优化 (`vite.config.js`)

```javascript
build: {
  cssCodeSplit: true,        // CSS 代码分割
  assetsInlineLimit: 4096,   // 小于 4KB 的资源内联
  rollupOptions: {
    output: {
      manualChunks(id) {
        // 智能分包策略
        if (id.includes('vue')) return 'vue-vendor'
        if (id.includes('ant-design-vue')) return 'antd-vendor'
        if (id.includes('codemirror')) return 'codemirror-core'
        if (id.includes('md-editor-v3')) return 'markdown-vendor'
        if (id.includes('echarts')) return 'viz-vendor'
      }
    }
  }
}
```

### Store 请求优化 (`stores/config.js`)

- **缓存 TTL**: 30 秒内重复请求直接返回缓存
- **请求去重**: 并发请求复用同一个 Promise
- **错误处理**: 优雅降级，保持 UI 可用

```javascript
// 缓存检查
if (!force && now - lastRefreshTime < CACHE_TTL) {
  return config.value
}

// 请求去重
if (pendingRefresh) {
  return pendingRefresh
}
```

### 图片优化

1. **懒加载**: 所有非首屏图片添加 `loading="lazy"`
2. **异步解码**: 添加 `decoding="async"`
3. **优化脚本**: `scripts/optimize-images.sh`

#### 手动压缩建议

| 图片 | 原始大小 | 目标大小 | 方法 |
|------|---------|---------|------|
| home.jpg | 7.8 MB | < 200 KB | 调整尺寸至 1920x1080，质量 75% |
| logo.png | 460 KB | < 50 KB | 调整尺寸至 256x256，PNG8 格式 |

---

## 阶段四：UX 增强与无障碍

### 错误重试机制 (`useRetry.ts`)

```typescript
const { execute, isRetrying, canRetry } = useRetry({
  maxRetries: 3,
  initialDelay: 1000,
  backoffMultiplier: 2,
  jitter: true
})

// 使用
const result = await execute(async () => {
  return await fetchData()
})
```

### 网络状态检测 (`useOnlineStatus.ts`)

```typescript
const { isOnline, offlineMessage, refresh } = useOnlineStatus({
  pingInterval: 30000,
  pingUrl: '/api/health'
})

// 监听离线状态
watch(isOnline, (online) => {
  if (!online) {
    showOfflineAlert()
  }
})
```

### ARIA 无障碍改进

| 组件 | 添加的属性 |
|------|-----------|
| ChatComponent | `role="log"`, `aria-live="polite"`, `aria-label="聊天消息"` |
| MessageComponent | `role="article"`, `aria-label="用户/助手消息"` |
| ConversationList | `role="listbox"`, `aria-label="对话列表"` |
| conversation-item | `role="option"`, `aria-selected` |

---

## 新建文件清单

```
web/src/
├── assets/
│   └── tokens.css                    # 设计令牌
├── types/
│   └── chat.ts                       # TypeScript 类型定义
├── composables/
│   ├── useChat.ts                    # 聊天核心逻辑
│   ├── useChatMeta.ts                # 元数据管理
│   ├── useChatScroll.ts              # 滚动管理
│   ├── useMessageUpdate.ts           # 消息更新
│   ├── useRetry.ts                   # 重试机制
│   └── useOnlineStatus.ts            # 在线状态
├── components/
│   ├── chat/
│   │   ├── ChatHeader.vue            # 头部组件
│   │   ├── ChatOptionsPanel.vue      # 选项面板
│   │   └── ModelSelector.vue         # 模型选择
│   ├── message/
│   │   ├── MessageToolbar.vue        # 消息工具栏
│   │   └── ReasoningBox.vue          # 推理展示
│   └── common/
│       └── LazyImage.vue             # 懒加载图片
└── scripts/
    └── optimize-images.sh            # 图片优化脚本
```

---

## 后续建议

### 短期

1. **运行图片优化脚本**: 压缩 home.jpg 和 logo.png
2. **集成 Lighthouse CI**: 持续监控性能指标
3. **添加 E2E 测试**: 验证重构后的功能

### 中期

1. **完整整合子组件**: 将 ChatComponent 和 MessageComponent 重构为使用新子组件
2. **虚拟滚动**: 对长对话列表使用虚拟滚动
3. **服务端渲染**: 考虑 Nuxt.js 提升首屏性能

### 长期

1. **微前端**: 拆分为独立可部署的模块
2. **PWA**: 添加离线支持和安装能力
3. **国际化**: i18n 支持

---

## 测试清单

- [ ] 暗色模式切换正常
- [ ] 移动端响应式布局正常
- [ ] 消息发送/接收流程正常
- [ ] 流式输出正常
- [ ] 错误重试功能正常
- [ ] 键盘导航可用
- [ ] 屏幕阅读器友好
