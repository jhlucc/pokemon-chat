# Agent Intelligence (20 Tasks) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `/chat/agent/*` feel noticeably “smarter” by reducing hallucinations first, then improving clarifying questions, routing accuracy, and finally performance—using LangChain/LangGraph 1.x best practices (structured outputs, deterministic routing, tool/data-first).

**Architecture:** Add a deterministic “Pokemon Data + Intent Router” layer in front of LLM routing. When a question is answerable from local data (`resources/data/raw_data/pokemon_detail.json`), answer without an LLM. Otherwise, route to the right worker/agent (pokedex/stats/trainer/graph/web) with safer fallbacks and better clarification prompts.

**Tech Stack:** Python 3.11, FastAPI, LangChain 1.x, LangGraph 1.x, pytest

**User priority:** (3) fewer hallucinations → (2) better clarifying questions → (1) more accurate routing → (4) faster.

**Note (Corridor):** `corridor` MCP server is not available in this environment, so we cannot run Corridor analysis. We’ll compensate with stricter TDD + offline-safe tests.

---

### Task 1: Add Pokemon data loader (local truth source)

**Files:**
- Create: `src/agents/pokemon_data.py`
- Test: `src/tests/test_pokemon_data.py`

**Step 1: Write failing tests**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_pokemon_data.py
```
Expected: FAIL (module missing / behavior missing)

**Step 2: Implement minimal loader**
- Load `resources/data/raw_data/pokemon_detail.json`
- Provide `get_by_cn_name()`, `get_by_id()`, `iter_all()`
- Keep it cached (in-process) and offline-safe

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_pokemon_data.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/pokemon_data.py src/tests/test_pokemon_data.py
git commit -m "feat(pokemon): add local pokemon data loader"
git push
```

---

### Task 2: Add name normalization + alias lookup (CN/EN/JP)

**Files:**
- Modify: `src/agents/pokemon_data.py`
- Test: `src/tests/test_pokemon_data.py`

**Step 1: Write failing tests**
- `resolve_name("Pikachu") -> "皮卡丘"`
- `resolve_name("ピカチュウ") -> "皮卡丘"`

**Step 2: Implement**
- Build alias index from `english_name` / `japanese_name` fields
- Normalize whitespace/case/punctuation

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_pokemon_data.py
```

**Step 4: Commit + push**
```bash
git add src/agents/pokemon_data.py src/tests/test_pokemon_data.py
git commit -m "feat(pokemon): resolve pokemon names via aliases"
git push
```

---

### Task 3: Add entity extraction from user text (robust matching)

**Files:**
- Create: `src/agents/pokemon_entities.py`
- Test: `src/tests/test_pokemon_entities.py`

**Step 1: Write failing tests**
- Extract `["皮卡丘"]` from “皮卡丘的属性是什么？”
- Extract `["皮卡丘"]` from “Pikachu ability?”

**Step 2: Implement**
- Longest-match scan over known names/aliases
- Return canonical CN names, de-duped in appearance order

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_pokemon_entities.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/pokemon_entities.py src/tests/test_pokemon_entities.py
git commit -m "feat(pokemon): extract pokemon entities from text"
git push
```

---

### Task 4: Add deterministic Pokedex “facts formatter” (no LLM)

**Files:**
- Create: `src/agents/pokemon_facts.py`
- Test: `src/tests/test_pokemon_facts.py`

**Step 1: Write failing tests**
- Given `"皮卡丘"` record, format includes `属性: 电`
- Evolution chain formatting uses dataset `进化`

**Step 2: Implement**
- `format_basic_facts(record) -> str`
- `format_evolution(record) -> str`
- `format_type_matchups(record) -> str` (from `属性相性` when present)

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_pokemon_facts.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/pokemon_facts.py src/tests/test_pokemon_facts.py
git commit -m "feat(pokemon): add deterministic pokedex fact formatting"
git push
```

---

### Task 5: Upgrade PokedexAgent tools to use local dataset (stop hardcoding)

**Files:**
- Modify: `src/agents/pokedex_agent.py`
- Test: `src/tests/test_pokedex_agent_tools.py`

**Step 1: Write failing tests**
- `search_pokedex("电")` includes `"皮卡丘"`
- `get_evolution_chain("皮卡丘")` returns a chain containing `"皮丘"`

**Step 2: Implement**
- Replace hardcoded dicts with `PokemonData`
- Keep tool outputs stable and readable

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_pokedex_agent_tools.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/pokedex_agent.py src/tests/test_pokedex_agent_tools.py
git commit -m "refactor(pokedex): back tools by local dataset"
git push
```

---

### Task 6: Add intent model + rule-based classifier (routing without LLM)

**Files:**
- Create: `src/agents/intent.py`
- Test: `src/tests/test_intent.py`

**Step 1: Write failing tests**
- “皮卡丘 属性” → intent=`POKEDEX_FACTS`
- “给我一套队伍” → intent=`TEAM_BUILDING`
- “最新 活动” → intent=`WEB_SEARCH`

**Step 2: Implement**
- `Intent` enum + `IntentDecision` (pydantic)
- `classify_intent(text, entities) -> IntentDecision`

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_intent.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/intent.py src/tests/test_intent.py
git commit -m "feat(agent): add rule-based intent classifier"
git push
```

---

### Task 7: Add clarification question generator (missing slots)

**Files:**
- Modify: `src/agents/intent.py`
- Test: `src/tests/test_intent.py`

**Step 1: Write failing tests**
- Query “属性相性怎么查？” with no entity returns `needs_clarification=True` and asks “想问哪只宝可梦？”

**Step 2: Implement**
- `clarify(decision) -> str | None`
- Set `decision.needs_clarification`

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_intent.py
```

**Step 4: Commit + push**
```bash
git add src/agents/intent.py src/tests/test_intent.py
git commit -m "feat(agent): ask targeted clarifying questions for missing info"
git push
```

---

### Task 8: Integrate deterministic router node into PokemonKGChatAgent graph

**Files:**
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_chat_agent_intent_router.py`

**Step 1: Write failing tests**
- “皮卡丘 属性” routes to `facts_answerer` (no LLM)
- “你好” routes to `chat`

**Step 2: Implement**
- Add node `intent_router`
- Add edges: `START -> guardrail -> intent_router -> (facts_answerer | supervisor | chat)`

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_chat_agent_intent_router.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/chat_agent.py src/tests/test_chat_agent_intent_router.py
git commit -m "feat(chat_agent): add deterministic intent router node"
git push
```

---

### Task 9: Add facts_answerer node (answer from dataset, no hallucination)

**Files:**
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_chat_agent_facts_answerer.py`

**Step 1: Write failing tests**
- “皮卡丘 身高体重” returns `0.4m` and `6.0kg` (from dataset)

**Step 2: Implement**
- Node reads last user message → entities → record → format via `pokemon_facts`
- If no match, ask clarification instead of guessing

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_chat_agent_facts_answerer.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/chat_agent.py src/tests/test_chat_agent_facts_answerer.py
git commit -m "feat(chat_agent): answer pokedex facts deterministically"
git push
```

---

### Task 10: Replace LLM-based guardrail with hybrid (rules first, LLM fallback)

**Files:**
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_guardrail_rules.py`

**Step 1: Write failing tests**
- “写个Python爬虫” blocks without calling LLM
- “皮卡丘” passes without calling LLM

**Step 2: Implement**
- Use entity extraction + keyword heuristics
- Only call LLM guardrail when uncertain

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_guardrail_rules.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/chat_agent.py src/tests/test_guardrail_rules.py
git commit -m "refactor(guardrail): rules-first guardrail with LLM fallback"
git push
```

---

### Task 11: Standardize agent LLM creation via `build_chat_llm`

**Files:**
- Modify: `src/agents/chat_agent.py`
- Modify: `src/agents/base.py`
- Test: `src/tests/test_llm_factory_integration.py`

**Step 1: Write failing tests**
- Under pytest, `_default_llm()` does not read UI overrides from disk (already required behavior)
- `PokemonKGChatAgent` uses `build_chat_llm` so it can be patched consistently

**Step 2: Implement**
- BaseAgent `_default_llm` delegates to `build_chat_llm`
- Chat agent uses factory too (keeps temperature settings)

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_llm_factory_integration.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/base.py src/agents/chat_agent.py src/tests/test_llm_factory_integration.py
git commit -m "refactor(llm): use centralized llm factory across agents"
git push
```

---

### Task 12: Add loop/step guard to PokemonKGChatAgent execution

**Files:**
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_chat_agent_recursion_limit.py`

**Step 1: Write failing test**
- Ensure `query()` passes a `recursion_limit` in config (prevents runaway loops)

**Step 2: Implement**
- Add `recursion_limit` (e.g. 25) to graph streaming config

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_chat_agent_recursion_limit.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/chat_agent.py src/tests/test_chat_agent_recursion_limit.py
git commit -m "feat(chat_agent): add recursion limit safety guard"
git push
```

---

### Task 13: Make semantic long-term memory middleware feature-flagged + lazy

**Files:**
- Modify: `src/agents/middleware/long_term_memory.py`
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_long_term_memory_feature_flag.py`

**Step 1: Write failing test**
- When feature flag off, `LongTermMemoryMiddleware` is not imported/constructed

**Step 2: Implement**
- Gate by `feature_enabled("enable_long_term_memory")`
- Avoid heavy imports at import-time; lazy-init inside middleware methods

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_long_term_memory_feature_flag.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/middleware/long_term_memory.py src/agents/chat_agent.py src/tests/test_long_term_memory_feature_flag.py
git commit -m "perf(memory): lazy-init long-term memory behind feature flag"
git push
```

---

### Task 14: Web search gating (only when user asks for “latest/current”)

**Files:**
- Create: `src/agents/web_gating.py`
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_web_gating.py`

**Step 1: Write failing tests**
- “宝可梦最新活动” => `should_web_search=True`
- “皮卡丘属性” => `should_web_search=False`

**Step 2: Implement**
- Simple keyword heuristic + year/version detection
- Use in routing (prefer web_searcher only when needed)

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_web_gating.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/web_gating.py src/agents/chat_agent.py src/tests/test_web_gating.py
git commit -m "feat(web): gate web search to time-sensitive queries"
git push
```

---

### Task 15: Improve DeepAgent to use real PokemonData (less toy knowledge)

**Files:**
- Modify: `src/agents/deep_agent/graph.py`
- Test: `src/tests/test_deep_agent.py`

**Step 1: Write failing tests**
- Deep research “皮卡丘” includes at least one fact derived from dataset (type/ability/etc)

**Step 2: Implement**
- Replace `POKEMON_KNOWLEDGE` with `PokemonData` + `pokemon_facts`

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_deep_agent.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/deep_agent/graph.py src/tests/test_deep_agent.py
git commit -m "refactor(deep_agent): ground research in local pokemon dataset"
git push
```

---

### Task 16: Improve TrainerAgent suggestions using PokemonData typing

**Files:**
- Modify: `src/agents/trainer_agent.py`
- Test: `src/tests/test_trainer_agent_tools.py`

**Step 1: Write failing tests**
- `counter_team(["火"])` includes recommending `水` (still)
- `type_coverage([...])` handles unknown types gracefully

**Step 2: Implement**
- Validate types against a canonical list
- Keep deterministic text outputs (no LLM dependency)

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_trainer_agent_tools.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/trainer_agent.py src/tests/test_trainer_agent_tools.py
git commit -m "refactor(trainer): harden tools and typing with local data"
git push
```

---

### Task 17: Add rule-based pre-router to Supervisor workflow (avoid LLM misroutes)

**Files:**
- Create: `src/graph/nodes/rule_router.py`
- Modify: `src/graph/nodes/supervisor.py`
- Test: `src/tests/test_supervisor_rule_router.py`

**Step 1: Write failing tests**
- “皮卡丘 属性” routes to `stats_worker` or `rag_worker`? (define expected)
- “最新活动” routes to `web_worker` when allowed

**Step 2: Implement**
- Fast heuristic routing when intent obvious
- Fallback to LLM supervisor otherwise

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_supervisor_rule_router.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/graph/nodes/rule_router.py src/graph/nodes/supervisor.py src/tests/test_supervisor_rule_router.py
git commit -m "feat(supervisor): rules-first routing with LLM fallback"
git push
```

---

### Task 18: Make StatsWorker non-placeholder (deterministic answers for type matchups)

**Files:**
- Modify: `src/graph/nodes/stats_worker.py`
- Test: `src/tests/test_stats_worker.py`

**Step 1: Write failing tests**
- Query “电打水克制吗” returns includes “效果拔群” / multiplier

**Step 2: Implement**
- Use a deterministic type chart helper (shared with stats agent)
- Only use LLM for explanation phrasing, not for the numeric result

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_stats_worker.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/graph/nodes/stats_worker.py src/tests/test_stats_worker.py
git commit -m "feat(stats): make stats worker answer type matchups deterministically"
git push
```

---

### Task 19: Token-level streaming for PokemonKGChatAgent (LangGraph events)

**Files:**
- Modify: `src/agents/chat_agent.py`
- Test: `src/tests/test_chat_agent_streaming.py`

**Step 1: Write failing tests**
- Ensure `query()` uses `astream_events` and yields partial tokens

**Step 2: Implement**
- Switch from `astream(..., stream_mode=\"values\")` to `astream_events(..., version=\"v1\")`
- Preserve dict status chunks compatibility with `/chat/agent`

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_chat_agent_streaming.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/agents/chat_agent.py src/tests/test_chat_agent_streaming.py
git commit -m "perf(chat_agent): stream tokens via LangGraph astream_events"
git push
```

---

### Task 20: End-to-end offline test for deterministic “facts” path in `/chat/agent/chat_agent`

**Files:**
- Test: `src/tests/test_agent_facts_e2e.py`

**Step 1: Write failing test**
- Patch LLM to raise on invocation
- Ask “皮卡丘 属性” and assert response still contains “电”

**Step 2: Implement**
- Only if needed: small refactors to ensure facts path never touches LLM

**Step 3: Verify**
Run:
```bash
.venv/bin/python -m pytest -q src/tests/test_agent_facts_e2e.py
.venv/bin/python -m pytest -q
```

**Step 4: Commit + push**
```bash
git add src/tests/test_agent_facts_e2e.py
git commit -m "test(agent): ensure deterministic facts path is offline-safe"
git push
```

