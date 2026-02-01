from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import traceback
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src import executor
from src.core.feature_flags import feature_enabled
from src.core.settings import settings
from src.knowledge.cache.cache import get_semantic_cache
from src.knowledge.memory.agentic import get_agentic_memory
from src.runtime import get_asr_client, get_retriever
from src.utils.logger import LogManager, request_id_var

logger = LogManager()

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

chat = APIRouter(prefix="/chat")

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------


def convert_messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """
    把 LangChain Message 转成 {role, content}，确保:
      - HumanMessage  → "user"
      - AIMessage     → "assistant"
      - SystemMessage → "system"
    """
    # Lazy import: keeps server startup cheaper.
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    result: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "user"  # fallback

        result.append({"role": role, "content": msg.content})
    return result


def make_chunk(meta: dict[str, Any], content: str | None = None, **kwargs) -> bytes:
    """统一的 SSE / chunk 打包函数（返回 bytes 行）"""

    def convert(obj):
        # Lazy import: keeps server startup cheaper.
        from langchain_core.messages import BaseMessage

        if isinstance(obj, BaseMessage):
            return {"role": getattr(obj, "_type", "user"), "content": obj.content}
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    payload = {
        "request_id": request_id_var.get(),
        "response": content,
        "meta": meta,
        **kwargs,
    }
    return json.dumps(payload, ensure_ascii=False, default=convert).encode("utf-8") + b"\n"


def need_retrieve(meta: dict[str, Any]) -> bool:
    return meta.get("use_web") or meta.get("use_graph") or meta.get("db_id") or meta.get("mcp_id")


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------


@chat.get("/")
async def chat_get():
    return "Chat Get!"


@chat.post("/")
async def chat_post(
    query: str = Body(...),
    meta: dict[str, Any] | None = Body(None),
    history: list[dict[str, Any]] | None = Body(None),
    thread_id: str | None = Body(None),
):
    """主聊天接口，支持 **流式** 返回"""
    meta = dict(meta or {})
    history = list(history or [])

    # 1. Resolve model preference (lazy).
    # We only instantiate/select the model if we actually need an LLM call.
    # This keeps simple Pokédex queries offline-safe (no API key required).
    from server.runtime_config import load_ui_overrides
    from src.knowledge.core.history_chat import HistoryManager

    overrides = load_ui_overrides()
    model_provider = meta.get("model_provider") or overrides.get("model_provider") or "siliconflow"
    model_name = meta.get("model_name") or overrides.get("model_name") or settings.llm.model_name
    meta["model_provider"] = model_provider
    meta["model_name"] = model_name

    # Get user preferences and inject into system prompt
    user_id = thread_id or meta.get("thread_id") or "default_user"
    memory = get_agentic_memory()
    preference_injection = memory.get_system_prompt_injection(user_id)

    base_system_prompt = meta.get("system_prompt", "")
    enhanced_system_prompt = base_system_prompt + preference_injection if preference_injection else base_system_prompt
    prompt_sha = (
        hashlib.sha256(enhanced_system_prompt.encode("utf-8")).hexdigest()[:12] if enhanced_system_prompt else ""
    )
    cache_meta = {"model_provider": model_provider, "model_name": model_name, "system_prompt_sha": prompt_sha}

    history_manager = HistoryManager(system_prompt=enhanced_system_prompt)
    # Load client-provided history into the prompt context (system prompt is managed server-side).
    for item in history:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            history_manager.add_user(content)
        elif role == "assistant":
            history_manager.add_ai(content)

    logger.debug(f"Received query: {query} with meta: {meta}")

    # Log conversation turn for memory extraction
    memory.add_conversation_turn(user_id, "user", query)

    # ---------------------------------------------------------------------------
    async def generate_response():
        modified_query = query
        refs = None

        # Only safe for single-turn, non-retrieval queries (avoids cross-history poisoning).
        safe_cache = (not history) and (not need_retrieve(meta))

        # 0. Local deterministic Pokédex shortcut ------------------------------
        # Prevent obvious hallucinations (e.g. “喷火龙进化成什么”) by answering
        # simple fact queries directly from the bundled dataset.
        try:
            from src.agents.intent import Intent, classify_intent

            decision = classify_intent(modified_query)
            # Only shortcut explicit "facts" queries to avoid hijacking open-ended chats.
            if decision.intent == Intent.POKEDEX_FACTS and (decision.confidence >= 0.8 or decision.needs_clarification):
                if decision.needs_clarification and decision.clarification_question:
                    content = decision.clarification_question
                else:
                    from src.agents.pokemon_data import get_pokemon_data
                    from src.agents.pokemon_facts import format_basic_facts, format_evolution

                    entities = decision.entities
                    if not entities:
                        content = "你想查询哪只宝可梦？请告诉我宝可梦名字。"
                    else:
                        name = entities[0]
                        data = get_pokemon_data()
                        rec = data.get_by_cn_name(name)
                        if not rec:
                            content = f"我没找到宝可梦：{name}。你可以换个名字试试吗？"
                        else:
                            content = format_basic_facts(rec) + "\n\n" + format_evolution(rec, data=data)
                            # Helpful hint for common confusion: "what does X evolve into?"
                            # If the evolution chain ends at the queried Pokemon, it has no further evolution.
                            if "进化" in modified_query and content.strip().endswith(name):
                                content += "（最终阶段）"

                logger.debug(f"Final response (local facts): {content}")

                # Mirror the /chat streaming protocol: send content, then a final chunk.
                history_manager.add_user(modified_query)
                updated_history = history_manager.update_ai(content)
                history_serializable = convert_messages_to_dicts(updated_history)

                # Cache the deterministic response (best-effort).
                cache = get_semantic_cache()
                if content and safe_cache:
                    cache.set(query, content, meta=cache_meta)

                # Log assistant response into agentic memory (best-effort).
                memory.add_conversation_turn(user_id, "assistant", content)

                refs = {
                    "query": query,
                    "history": convert_messages_to_dicts(history_manager.history.messages),
                    "meta": meta,
                    "model_name": "local_dataset",
                    "entities": decision.entities,
                    "knowledge_base": {
                        "results": [],
                        "all_results": [],
                        "rw_query": query,
                        "message": "Answer来自本地数据集（无 LLM 调用）",
                    },
                    "graph_base": {"results": {"nodes": [], "edges": []}},
                    "web_search": {"results": [], "message": "Web search is disabled"},
                    "mysql_mcp": {"answer": "", "coords": []},
                }

                yield make_chunk(meta, content=content, status="loading")
                yield make_chunk(meta, status="finished", history=history_serializable, refs=refs)
                return

        except Exception as e:
            logger.warning(f"Local pokedex shortcut failed (fallback to LLM): {e}")

        # 1. Semantic Cache ----------------------------------------------------
        cache = get_semantic_cache()
        if safe_cache:
            cached_response = cache.get(query, meta=cache_meta)
            if cached_response:
                yield make_chunk(meta, content=cached_response, status="finished")
                return

        # 2. 检索阶段 -------------------------------------------------------------
        if meta and need_retrieve(meta):
            yield make_chunk(meta, status="searching")
            try:
                from asyncio import to_thread

                retriever = get_retriever()
                modified_query, refs = await to_thread(
                    retriever,  # 同步函数
                    modified_query,
                    history,
                    meta,
                )
            except Exception as e:
                logger.error(f"Retriever error: {e}\n{traceback.format_exc()}")
                yield make_chunk(meta, message=f"Retriever error: {e}", status="error")
                return
            yield make_chunk(meta, status="generating")

        # 3. Select model (only when needed) -----------------------------------
        from src.models import select_model

        model = select_model(model_provider=model_provider, model_name=model_name)
        if model is None:
            yield make_chunk(meta, message="没有可用的模型（请检查模型提供方/模型名/密钥配置）", status="error")
            return
        # Expose the actual resolved model name to the frontend/logs.
        meta["server_model_name"] = getattr(model, "model_name", "") or model_name

        # 构造 Prompt ----
        messages = history_manager.get_history_with_msg(modified_query, max_rounds=meta.get("history_round"))
        history_manager.add_user(modified_query)  # 把原始用户查询加入历史
        formatted_messages = convert_messages_to_dicts(messages)

        content = ""
        reasoning_content = ""
        try:
            for delta in model.predict(formatted_messages, stream=True):
                # 一些模型将「思考过程」放在 reasoning_content
                if not delta.content and hasattr(delta, "reasoning_content"):
                    reasoning_content += delta.reasoning_content or ""
                    yield make_chunk(meta, reasoning_content=reasoning_content, status="reasoning")
                    continue

                content += delta.content or ""

                yield make_chunk(meta, content=delta.content, status="loading")

            logger.debug(f"Final response: {content}")
            logger.debug(f"Final reasoning response: {reasoning_content}")

            # 4. 更新历史，发送最终块
            updated_history = history_manager.update_ai(content)
            history_serializable = convert_messages_to_dicts(updated_history)

            # 5. Cache the response (avoid cross-history poisoning)
            if content and safe_cache:
                cache.set(query, content, meta=cache_meta)

            # 6. Log AI response and periodically extract preferences
            memory.add_conversation_turn(user_id, "assistant", content)

            # Extract preferences every 5 conversation turns (configurable)
            from random import random

            if random() < 0.2:  # 20% chance to trigger extraction (async-friendly)
                try:
                    memory.extract_and_update_preferences(user_id)
                except Exception as e:
                    logger.warning(f"Preference extraction failed: {e}")
            # print(updated_history)
            # print(history_serializable)

            # logger.debug(
            #     f"Output refs dump (for frontend):\n{json.dumps(refs, indent=2, ensure_ascii=False, default=str)}")
            if refs is None:
                refs = {
                    "query": query,
                    "history": convert_messages_to_dicts(history_manager.history.messages),
                    "meta": meta,
                    "model_name": model.model_name,
                    "entities": [],
                    "knowledge_base": {
                        "results": [],
                        "all_results": [],
                        "rw_query": query,
                        "message": "知识库未启用、或未指定知识库、或知识库不存在",
                    },
                    "graph_base": {"results": {"nodes": [], "edges": []}},
                    "web_search": {"results": [], "message": "Web search is disabled"},
                    "mysql_mcp": {"answer": "", "coords": []},
                }

            # 最后一条流式返回
            yield make_chunk(meta, status="finished", history=history_serializable, refs=refs)

        except Exception as e:
            logger.error(f"Model error: {e}\n{traceback.format_exc()}")
            yield make_chunk(meta, message=f"Model error: {e}", status="error")

    return StreamingResponse(
        generate_response(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat.post("/call")
async def call(query: str = Body(...), meta: dict[str, Any] | None = Body(None)):
    """同步调用完整模型"""
    from src.models import select_model

    meta = dict(meta or {})
    model = select_model(model_provider=meta.get("model_provider"), model_name=meta.get("model_name"))
    if model is None:
        raise HTTPException(status_code=500, detail="模型不可用")

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(executor, model.predict, query)
    logger.debug(f"query: {query}, response: {response.content}")
    return {"response": response.content}


@chat.post("/call_lite")
async def call_lite(query: str = Body(...), meta: dict[str, Any] | None = Body(None)):
    """使用 Lite 模型，同步调用"""
    from src.models import select_model

    async def _predict_async(q):
        meta_d = dict(meta or {})
        model_provider = meta_d.get("model_provider", "siliconflow")
        model_name = meta_d.get("model_name", settings.llm.model_name)
        model = select_model(model_provider=model_provider, model_name=model_name)
        if model is None:
            raise HTTPException(status_code=500, detail="Lite 模型不可用")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, model.predict, q)

    response = await _predict_async(query)
    logger.debug(f"query: {query}, response: {response.content}")
    return {"response": response.content}


@chat.get("/agent")
async def get_agent_list():
    from src.agents.manager import agent_manager

    infos = agent_manager.list_agents()
    return {
        "agents": [
            ({**info, "name": name} if isinstance(info, dict) else {"name": name, "info": info})
            for name, info in infos.items()
        ]
    }


@chat.post("/agent/{agent_name}")
async def chat_agent(
    agent_name: str,
    query: str = Body(...),
    history: list[dict[str, Any]] | None = Body(None),
    cfg: dict[str, Any] | None = Body(None),
    meta: dict[str, Any] | None = Body(None),
):
    history = list(history or [])
    cfg = dict(cfg or {})
    meta = dict(meta or {})
    from src.agents.manager import agent_manager
    from src.knowledge.core.history_chat import HistoryManager

    try:
        agent = agent_manager.get_agent(agent_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found") from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # 给历史消息补 id，防止前端缺字段
    for msg in history:
        msg.setdefault("id", str(uuid.uuid4()))

    # 本次请求 id / 线程 id
    request_id = cfg.get("request_id", str(uuid.uuid4()))
    thread_id = cfg.get("thread_id", request_id)
    meta.update({"query": query, "agent_name": agent_name, "thread_id": thread_id})

    # 会话历史管理（仅用来更新助手消息）
    history_mgr = HistoryManager()
    history_mgr.add_user(query)

    # ----------   关键：统一的 chunk 打包   ----------
    def make_agent_chunk(
        *,
        content: str = "",
        status: str = "",
        msg_id: str,
        error: str | None = None,
        history_resp=None,
        refs=None,
        data: dict[str, Any] | None = None,
    ):
        """
        前端必须要拿到 msg.id，所以这里统一塞一个 msg 字段
        """
        payload = {
            "request_id": request_id,
            "response": content,
            "status": status,
            "meta": meta,
            "msg": {"id": msg_id, "type": "assistant"},  # ★ 重点
        }
        if error:
            payload["error"] = error
        if history_resp:
            for m in history_resp:
                m.setdefault("id", str(uuid.uuid4()))
            payload["history"] = history_resp
        if refs:
            payload["refs"] = refs
        if data:
            payload["data"] = data
        return json.dumps(payload, ensure_ascii=False).encode() + b"\n"

    # ----------   streaming   ----------
    async def streamer():
        # 生成一个本轮 assistant 消息的固定 id
        requested_msg_id = cfg.get("msg_id") or cfg.get("cur_res_id")
        if isinstance(requested_msg_id, str) and requested_msg_id.strip():
            cur_msg_id = requested_msg_id.strip()
        else:
            cur_msg_id = str(uuid.uuid4())

        # ① init
        yield make_agent_chunk(status="init", msg_id=cur_msg_id)

        # ② token-by-token
        final_answer = ""
        try:
            async for part in agent.query(query, meta=meta, history=history):
                if isinstance(part, str):
                    # Token
                    final_answer += part
                    yield make_agent_chunk(content=part, status="loading", msg_id=cur_msg_id)
                elif isinstance(part, dict):
                    # Structured Metadata (e.g. status update)
                    if "status" in part:
                        yield make_agent_chunk(
                            # Keep status compatible with ChatComponent UI:
                            # - do not emit arbitrary status strings (would be treated as error)
                            # - keep "init" until we have actual tokens to render
                            status="init",
                            msg_id=cur_msg_id,
                            data=part,
                        )

            # ③ finished
            updated = history_mgr.update_ai(final_answer)
            history_serializ = convert_messages_to_dicts(updated)

            refs = {
                "query": query,
                "history": history_serializ,
                "meta": meta,
                "model_name": "",  # 需要的话自行补充
                "entities": [],
                "knowledge_base": {
                    "results": [],
                    "all_results": [],
                    "rw_query": query,
                    "message": "知识库未启用、或未指定知识库、或知识库不存在",
                },
                "graph_base": {"results": {"nodes": [], "edges": []}},
                "web_search": {"results": [], "message": "Web search is disabled"},
                "mysql_mcp": {"answer": "", "coords": []},
            }

            yield make_agent_chunk(
                # Keep protocol aligned with /chat/: don't repeat the full answer in the final chunk.
                content="",
                status="finished",
                msg_id=cur_msg_id,
                history_resp=history_serializ,
                refs=refs,
            )

        except Exception as e:
            logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
            yield make_agent_chunk(status="error", msg_id=cur_msg_id, error=str(e))

    return StreamingResponse(
        streamer(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat.get("/models")
async def get_chat_models(model_provider: str):
    from src.models import select_model

    model = select_model(model_provider=model_provider)
    return {"models": model.get_models()}


@chat.post("/models/update")
async def update_chat_models(model_provider: str, model_names: list[str]):
    # config.model_names[model_provider]["models"] = model_names
    # config._save_models_to_file()
    # return {"models": config.model_names[model_provider]["models"]}
    raise HTTPException(status_code=501, detail="Configuration update not supported in this version")


@chat.post("/asr/")
async def asr_upload(file: UploadFile = File(...)):
    if not feature_enabled("enable_asr"):
        raise HTTPException(status_code=503, detail="ASR is disabled (enable_asr=false).")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg not found (required for audio conversion).")

    # Guardrails: keep this endpoint safe by default.
    max_bytes = int(os.getenv("ASR_MAX_UPLOAD_BYTES") or str(15 * 1024 * 1024))
    audio_bytes = await file.read(max_bytes + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Audio too large (>{max_bytes} bytes).")

    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
    }
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported audio content-type: {file.content_type}")

    input_suffix = ".ogg"
    if file.filename:
        _, ext = os.path.splitext(file.filename)
        ext = (ext or "").lower()
        if ext in {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".webm", ".mp4"}:
            input_suffix = ext

    with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as temp_input:
        temp_input.write(audio_bytes)
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
        output_path = temp_output.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                input_path,
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                output_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        with open(output_path, "rb") as f:
            wav_bytes = f.read()
        result_text = get_asr_client().transcribe(wav_bytes)
        return {"text": result_text}

    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        logger.error(f"音频转码失败: {e}. stderr={stderr}")
        raise HTTPException(status_code=500, detail="Audio conversion failed") from e

    finally:
        for p in (locals().get("input_path"), locals().get("output_path")):
            if not p:
                continue
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as cleanup_err:  # noqa: BLE001
                logger.warning(f"Temp file cleanup failed: {cleanup_err}")
