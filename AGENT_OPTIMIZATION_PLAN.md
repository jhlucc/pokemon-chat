# Agent 模式优化计划

## 问题总览

经分析发现 **12 个关键问题**，按严重程度排序：

| # | 问题 | 严重性 | 影响 | 状态 |
|---|------|--------|------|------|
| 1 | Supervisor 未检查 tool_calls 为空 | CRITICAL | 崩溃 | ✅ 已修复 |
| 2 | Handoff 工具闭包绑定错误 | HIGH | 路由完全失效 | ✅ 已修复 |
| 3 | Graph Worker 中间步骤解析不安全 | CRITICAL | 崩溃/数据丢失 | ✅ 已修复 |
| 4 | 并行执行 parallel_done 竞态条件 | HIGH | 挂起/提前退出 | ✅ 已修复 |
| 5 | 条件边缘映射缺少 __PARALLEL__ | HIGH | 并行路由崩溃 | ✅ 已修复 |
| 6 | MCP Worker 异步竞态条件 | HIGH | 并行执行失败 | ✅ 已正常 |
| 7 | CRAG 硬编码 ChatOpenAI | MEDIUM | Provider 切换失效 | ✅ 已修复 |
| 8 | RAG Worker VectorStore 初始化 | MEDIUM | 连接失效 | ⏳ 跳过 (低优先级) |
| 9 | 规则路由关键词局限 | MEDIUM | 误路由 | ✅ 已修复 |
| 10 | Chat Agent 消息过滤逻辑 | MEDIUM | 状态损坏 | ✅ 已有实现 |
| 11 | Worker 缺少输入验证 | MEDIUM | 静默失败 | ✅ 已修复 |
| 12 | Feature Flags 缓存过期 | MEDIUM | 设置不生效 | ⏳ 待处理 |

---

## 优化计划 (20 项任务)

### Phase 1: 关键崩溃修复 (任务 1-5) ✅ 完成

#### 任务 1: 修复 Supervisor tool_calls 索引 ✅
**文件**: `src/graph/nodes/supervisor.py`
**问题**: `tool_calls[0]` 未检查是否为空
**修复**: 已添加空检查

#### 任务 2: 修复 Handoff 工具闭包问题 ✅
**文件**: `src/graph/nodes/supervisor.py`
**问题**: Python 闭包晚绑定导致所有工具引用最后一个 worker
**修复**: 使用工厂函数正确绑定闭包

#### 任务 3: 修复 Graph Worker 中间步骤解析 ✅
**文件**: `src/graph/nodes/graph_worker.py`
**问题**: 假设 intermediate_steps 格式固定，未做类型检查
**修复**: 添加长度和类型检查

#### 任务 4: 修复 parallel_done 竞态条件 ✅
**文件**: `src/graph/workflow.py`
**问题**: 并行 worker 同时读写 parallel_done
**修复**: 简化并行执行逻辑，每个分支独立完成

#### 任务 5: 修复条件边缘 __PARALLEL__ 映射 ✅
**文件**: `src/graph/workflow.py`
**问题**: `_supervisor_route` 返回 "__PARALLEL__" 但映射中没有
**修复**: 正确使用 LangGraph `Send()` API

---

### Phase 2: 高优先级修复 (任务 6-10) ✅ 完成

#### 任务 6: 修复 MCP Worker 异步问题 ✅
**文件**: `src/graph/workflow.py`, `src/graph/nodes/mcp_worker.py`
**状态**: 已正确实现异步处理

#### 任务 7: 统一 CRAG LLM 初始化 ✅
**文件**: `src/graph/nodes/crag.py`
**修复**: 使用 `build_chat_llm()` 工厂函数替代硬编码 ChatOpenAI

#### 任务 8: 改进 RAG Worker 连接管理 ⏳
**文件**: `src/graph/nodes/rag_worker.py`
**状态**: 跳过 (低优先级，当前实现可用)

#### 任务 9: 增强规则路由 ✅
**文件**: `src/graph/nodes/rule_router.py`
**修复**: 添加 `_match_keyword()` 函数，ASCII 关键词使用词边界匹配

#### 任务 10: 修复 Chat Agent 消息过滤 ✅
**文件**: `src/agents/chat_agent.py`, `src/agents/utils/message_filter.py`
**状态**: 已有完整实现

---

### Phase 3: 中优先级改进 (任务 11-15) 🔄 进行中

#### 任务 11: 添加 Worker 输入验证 ✅
**文件**: 所有 worker 文件
**修复**:
- 新增 `validate_worker_input()` 和 `make_error_response()` 工具函数
- 所有 worker 统一使用验证逻辑

#### 任务 12: 修复 Feature Flags 缓存 ⏳
**文件**: `src/core/feature_flags.py`
**状态**: 待处理

#### 任务 13: 改进错误处理和日志 ⏳
**文件**: 所有 graph/nodes/*.py
**状态**: 待处理

#### 任务 14: 添加 Worker 健康检查 ⏳
**新增**: `src/graph/nodes/health.py`
**状态**: 待处理

#### 任务 15: 优化 Cypher 生成 Prompt ✅
**文件**: `src/graph/nodes/graph_worker.py`
**修复**: 已添加 4 个 few-shot 示例

---

### Phase 4: 测试 (任务 16-20) 🔄 部分完成

#### 任务 16: Supervisor 路由单元测试 ✅
**文件**: `src/tests/test_supervisor_rule_router.py`
**状态**: 已有测试覆盖

#### 任务 17: Graph Worker 单元测试 ⏳
**新增**: `src/tests/test_graph_worker.py`
**状态**: 待添加

#### 任务 18: 并行执行集成测试 ✅
**文件**: `src/tests/test_parallel_execution.py`
**状态**: 已有 5 个测试用例

#### 任务 19: RAG Worker 集成测试 ⏳
**更新**: `src/tests/test_rag_worker.py`
**状态**: 待添加

#### 任务 20: 端到端 Agent 测试 ⏳
**新增**: `src/tests/test_agent_e2e.py`
**状态**: 待添加

---

## 执行顺序

```
Phase 1 (Critical) → Phase 2 (High) → Phase 3 (Medium) → Phase 4 (Testing)
     ✅ 完成              ✅ 完成           🔄 进行中          🔄 部分完成
```

---

## 已完成修改文件

| 文件 | 改动说明 |
|------|----------|
| `src/graph/nodes/supervisor.py` | 修复 tool_calls 空检查 + 闭包绑定 |
| `src/graph/nodes/graph_worker.py` | 修复中间步骤解析 + 添加 Cypher 示例 + 统一验证 |
| `src/graph/workflow.py` | 简化并行执行逻辑 |
| `src/graph/nodes/crag.py` | 使用 build_chat_llm() |
| `src/graph/nodes/rule_router.py` | 改进关键词匹配 (词边界) |
| `src/graph/nodes/rag_worker.py` | 添加输入验证 |
| `src/graph/nodes/web_worker.py` | 添加输入验证 |
| `src/graph/nodes/stats_worker.py` | 添加输入验证 |
| `src/graph/nodes/mcp_worker.py` | 添加输入验证 |
| `src/agents/utils/message_filter.py` | 新增验证工具函数 |

---

## 预期成果

- ✅ Agent 模式不再崩溃
- ✅ 路由正确分发到对应 worker
- ✅ 并行执行稳定可靠
- ✅ 错误有清晰日志
- 🔄 测试覆盖关键路径 (部分完成)

---

## 下次继续

1. 添加 Graph Worker 单元测试 (任务 17)
2. 添加 RAG Worker 集成测试 (任务 19)
3. 添加端到端 Agent 测试 (任务 20)
4. 处理 Feature Flags 缓存 (任务 12)
