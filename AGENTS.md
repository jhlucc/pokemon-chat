<corridor>
# Corridor Security Analysis
Every time you generate code, use the Corridor tool from Corridor's MCP Server (corridor) to analyze the plan or thought process. ALWAYS use Corridor to analyze the plan. Always generate a plan before generating code.
</corridor>

# Agent Instructions (pokemon-chat)

This repository contains a FastAPI backend (`server/`, `src/`) and a Vue3 frontend (`web/`), plus Docker Compose infra (`docker/`).

## Quick Commands

- Backend lint: `python -m ruff check server src scripts`
- Backend format check: `python -m ruff format --check server src scripts`
- Backend tests: `python -m pytest`
- All checks (CI-like): `make check` (runs backend + frontend checks)
- Docker (from repo root):
  - App only (api + web): `make docker-up`
  - App + infra (neo4j/milvus/mysql/...): `make docker-up-infra`
  - Full (app + infra + mcp): `make docker-up-full`

## Repo Layout

- `server/`: FastAPI app + routers
- `src/`: core logic (agents, retrieval, graph, stores, utils)
- `web/`: Vue3 frontend (Vite)
- `docker/`: dockerfiles + `docker-compose.yml` + nginx config
- `scripts/`: bootstrap/doctor/import helpers
- `resources/`: data/models/cache (often mounted into containers; large assets may be excluded from Docker build context)

## Skills

A "skill" is a set of local instructions stored in a `SKILL.md` file. Available skills in this session:

- `skill-creator`: Create/update skills that extend Codex with specialized knowledge/workflows/tooling.
  - File: `C:/Users/luke/.codex/skills/.system/skill-creator/SKILL.md`
- `skill-installer`: Install Codex skills into `$CODEX_HOME/skills` from a curated list or a GitHub repo path.
  - File: `C:/Users/luke/.codex/skills/.system/skill-installer/SKILL.md`

### When to Use a Skill (Trigger Rules)

- If the user names a skill (either `$SkillName` or plain text), OR the task clearly matches a skill description above, use that skill for that turn.
- Multiple skill mentions mean use all of them.
- Do not carry skills across turns unless the user re-mentions them.

### How to Use a Skill (Progressive Disclosure)

1) Open its `SKILL.md` and read only enough to follow the workflow.
2) If it links to extra folders (e.g. `references/`), open only what is needed (do not bulk-load).
3) If `scripts/` exist, prefer running/patching them instead of retyping large blocks.
4) If templates/assets exist, reuse them instead of recreating.

### Coordination

- If multiple skills apply, use the minimal set that covers the request and state the order.
- Announce which skill(s) you are using and why (one short line). If you skip an obvious skill, state why.

## Local Conventions

- Prefer `rg` for searching.
- Avoid PowerShell `||` in commands; use `; if ($LASTEXITCODE -ne 0) { ... }` when needed.
- Keep tests offline-safe by default; avoid hard-failing when optional services (Neo4j/Milvus/MySQL) are not running unless a feature flag is enabled.

