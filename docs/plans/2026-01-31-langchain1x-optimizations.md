# LangChain 1.x Optimizations (3 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade LangChain/LangGraph deps, migrate legacy JSON parsing to structured output, and improve retrieval latency.

**Architecture:** Keep the existing FastAPI + LangGraph architecture, but (1) align versions to the latest compatible LangChain 1.x/LangGraph 1.x, (2) use `with_structured_output(...)` for routing/guardrails, and (3) parallelize retrieval sub-steps when multiple tools are enabled.

**Tech Stack:** Python 3.11, FastAPI, LangChain 1.x, LangGraph 1.x, pytest

---

## Task 1: Dependency Upgrades (LangChain/LangGraph 1.x)

**Files:**
- Modify: `requirements.txt`

**Step 1: Update versions**
- Bump `langchain` to the latest 1.2.x patch.
- Bump `langgraph` to the latest 1.0.x patch.
- Allow `langgraph-checkpoint-sqlite` v3+ (still compatible with `from langgraph.checkpoint.sqlite import SqliteSaver`).

**Step 2: Update local venv deps**

Run:
```bash
.venv/bin/python -m pip install -r requirements.txt
```

**Step 3: Verify**

Run:
```bash
.venv/bin/python -m pytest -q
```
Expected: PASS

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): bump langchain/langgraph within 1.x"
```

---

## Task 2: Structured Output Migration (Agent Routing / Guardrails)

**Files:**
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_chat_agent_routing.py`

**Step 1: Write failing tests**
- Add unit tests that validate:
  - Guardrail routes to `end_with_block` when status is `block`
  - Supervisor routing only accepts known workers (invalid output -> safe fallback)

Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_chat_agent_routing.py
```
Expected: FAIL (until implementation is updated)

**Step 2: Implement minimal migration**
- Replace `JsonOutputParser()` usage with `with_structured_output(...)`
- Prefer `ChatPromptTemplate.from_messages` + `MessagesPlaceholder` for routing decisions
- Keep behavior-compatible fallbacks on parsing errors

**Step 3: Verify**

Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_chat_agent_routing.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit**

```bash
git add src/agents/chat_agent.py src/tests/test_chat_agent_routing.py
git commit -m "refactor(agent): use structured outputs for guardrail and routing"
```

---

## Task 3: Performance Improvements (Parallel Retrieval)

**Files:**
- Modify: `src/knowledge/core/retriever.py`
- Test: `src/tests/test_retriever_parallel.py`

**Step 1: Write failing test**
- Create a unit test with stubbed `query_*` functions that:
  - sleep for a short time
  - asserts overall runtime is closer to max(sleeps) than sum(sleeps)
  - preserves result shape

Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_retriever_parallel.py
```
Expected: FAIL (until implementation is parallel)

**Step 2: Implement**
- Use a small `ThreadPoolExecutor` inside `Retriever.retrieval(...)` when multiple sub-steps are enabled.
- Keep ordering + response shape stable.
- Preserve offline-safe behavior and feature flags.

**Step 3: Verify**

Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_retriever_parallel.py
.venv/bin/python -m pytest -q
```
Expected: PASS

**Step 4: Commit**

```bash
git add src/knowledge/core/retriever.py src/tests/test_retriever_parallel.py
git commit -m "perf(retriever): parallelize tool retrieval sub-steps"
```

