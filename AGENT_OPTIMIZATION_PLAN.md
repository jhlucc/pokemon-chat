# Agent 模式优化计划

## 问题总览

经分析发现 **12 个关键问题**，按严重程度排序：

| # | 问题 | 严重性 | 影响 |
|---|------|--------|------|
| 1 | Supervisor 未检查 tool_calls 为空 | CRITICAL | 崩溃 |
| 2 | Handoff 工具闭包绑定错误 | HIGH | 路由完全失效 |
| 3 | Graph Worker 中间步骤解析不安全 | CRITICAL | 崩溃/数据丢失 |
| 4 | 并行执行 parallel_done 竞态条件 | HIGH | 挂起/提前退出 |
| 5 | 条件边缘映射缺少 __PARALLEL__ | HIGH | 并行路由崩溃 |
| 6 | MCP Worker 异步竞态条件 | HIGH | 并行执行失败 |
| 7 | CRAG 硬编码 ChatOpenAI | MEDIUM | Provider 切换失效 |
| 8 | RAG Worker VectorStore 初始化 | MEDIUM | 连接失效 |
| 9 | 规则路由关键词局限 | MEDIUM | 误路由 |
| 10 | Chat Agent 消息过滤逻辑 | MEDIUM | 状态损坏 |
| 11 | Worker 缺少输入验证 | MEDIUM | 静默失败 |
| 12 | Feature Flags 缓存过期 | MEDIUM | 设置不生效 |

---

## 优化计划 (20 项任务)

### Phase 1: 关键崩溃修复 (任务 1-5)

#### 任务 1: 修复 Supervisor tool_calls 索引
**文件**: `src/graph/nodes/supervisor.py`
**问题**: `tool_calls[0]` 未检查是否为空
**修复**:
```python
if not tool_calls or len(tool_calls) == 0:
    return {"next": "FINISH"}
```

#### 任务 2: 修复 Handoff 工具闭包问题
**文件**: `src/graph/nodes/supervisor.py`
**问题**: Python 闭包晚绑定导致所有工具引用最后一个 worker
**修复**: 使用默认参数捕获当前值
```python
def _handoff(reason: str = "", _worker=worker) -> str:
    return f"routed to {_worker}"
```

#### 任务 3: 修复 Graph Worker 中间步骤解析
**文件**: `src/graph/nodes/graph_worker.py`
**问题**: 假设 intermediate_steps 格式固定，未做类型检查
**修复**: 添加长度和类型检查

#### 任务 4: 修复 parallel_done 竞态条件
**文件**: `src/graph/workflow.py`
**问题**: 并行 worker 同时读写 parallel_done
**修复**: 使用 `max()` reducer 或 atomic counter

#### 任务 5: 修复条件边缘 __PARALLEL__ 映射
**文件**: `src/graph/workflow.py`
**问题**: `_supervisor_route` 返回 "__PARALLEL__" 但映射中没有
**修复**: 重构路由逻辑，正确使用 `Send()`

---

### Phase 2: 高优先级修复 (任务 6-10)

#### 任务 6: 修复 MCP Worker 异步问题
**文件**: `src/graph/workflow.py`, `src/graph/nodes/mcp_worker.py`
**修复**: 统一同步/异步处理

#### 任务 7: 统一 CRAG LLM 初始化
**文件**: `src/graph/nodes/crag.py`
**修复**: 使用 `build_chat_llm()` 工厂

#### 任务 8: 改进 RAG Worker 连接管理
**文件**: `src/graph/nodes/rag_worker.py`
**修复**: 延迟初始化 + 连接验证

#### 任务 9: 增强规则路由
**文件**: `src/graph/nodes/rule_router.py`
**修复**: 使用分词 + 词边界匹配

#### 任务 10: 修复 Chat Agent 消息过滤
**文件**: `src/agents/chat_agent.py`
**修复**: 更健壮的消息提取逻辑

---

### Phase 3: 中优先级改进 (任务 11-15)

#### 任务 11: 添加 Worker 输入验证
**文件**: 所有 worker 文件
**修复**: 统一验证 messages 非空 + content 有效

#### 任务 12: 修复 Feature Flags 缓存
**文件**: `src/core/feature_flags.py`
**修复**: 添加 TTL 或手动刷新机制

#### 任务 13: 改进错误处理和日志
**文件**: 所有 graph/nodes/*.py
**修复**: 统一错误格式 + 详细日志

#### 任务 14: 添加 Worker 健康检查
**新增**: `src/graph/nodes/health.py`
**功能**: 检查各服务连接状态

#### 任务 15: 优化 Cypher 生成 Prompt
**文件**: `src/graph/nodes/graph_worker.py`
**改进**: 更多 few-shot 示例 + 错误处理

---

### Phase 4: 测试 (任务 16-20)

#### 任务 16: Supervisor 路由单元测试
**新增**: `src/tests/test_supervisor.py`
**覆盖**: 空 tool_calls、闭包绑定、单/多 worker 路由

#### 任务 17: Graph Worker 单元测试
**新增**: `src/tests/test_graph_worker.py`
**覆盖**: Cypher 生成、中间步骤解析、Neo4j 连接失败

#### 任务 18: 并行执行集成测试
**新增**: `src/tests/test_parallel_execution.py`
**覆盖**: 多 worker 并行、竞态条件、完成检测

#### 任务 19: RAG Worker 集成测试
**更新**: `src/tests/test_rag_worker.py`
**覆盖**: 向量搜索、重排序、连接重试

#### 任务 20: 端到端 Agent 测试
**新增**: `src/tests/test_agent_e2e.py`
**覆盖**: 完整对话流程、多轮交互、错误恢复

---

## 执行顺序

```
Phase 1 (Critical) → Phase 2 (High) → Phase 4 (Testing) → Phase 3 (Medium)
     任务 1-5              任务 6-10         任务 16-20          任务 11-15
```

建议先修复崩溃问题，再添加测试验证，最后做改进优化。

---

## 预期成果

- Agent 模式不再崩溃
- 路由正确分发到对应 worker
- 并行执行稳定可靠
- 错误有清晰日志
- 测试覆盖关键路径
