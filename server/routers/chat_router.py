from __future__ import annotations

import json
import asyncio
import traceback
import uuid
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File
from src import executor, config
from src.utils.logger import LogManager
from src.runtime import get_retriever, get_whisper_client
logger = LogManager()
import subprocess
import tempfile

# Semantic Cache
from src.knowledge.cache.semantic_cache import get_semantic_cache

# Agentic Memory
from src.knowledge.memory.agentic_memory import get_agentic_memory

chat = APIRouter(prefix="/chat")

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def convert_messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    把 LangChain Message 转成 {role, content}，确保:
      - HumanMessage  → "user"
      - AIMessage     → "assistant"
      - SystemMessage → "system"
    """
    # Lazy import: keeps server startup cheaper.
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    result: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "user"          # fallback

        result.append({"role": role, "content": msg.content})
    return result

def make_chunk(meta: Dict[str, Any],
               content: Optional[str] = None,
               **kwargs) -> bytes:
    """统一的 SSE / chunk 打包函数（返回 bytes 行）"""
    def convert(obj):
        # Lazy import: keeps server startup cheaper.
        from langchain_core.messages import BaseMessage

        if isinstance(obj, BaseMessage):
            return {"role": getattr(obj, "_type", "user"), "content": obj.content}
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    payload = {
        "response": content,
        "meta": meta,
        **kwargs,
    }
    return json.dumps(payload, ensure_ascii=False, default=convert).encode("utf-8") + b"\n"



def need_retrieve(meta: Dict[str, Any]) -> bool:
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
        meta: Dict[str, Any] = Body({}),
        history: Optional[List[Dict[str, Any]]] = Body(None),
        thread_id: Optional[str] = Body(None),
):
    """主聊天接口，支持 **流式** 返回"""

    # 1. 选择模型 ----------------------------------------------------------------
    from src.models import select_model
    from src.knowledge.core.history_chat import HistoryManager

    model = select_model()
    if model is None:
        raise HTTPException(status_code=500, detail="没有可用的模型，请检查模型配置")

    meta["server_model_name"] = model.model_name
    
    # Get user preferences and inject into system prompt
    user_id = thread_id or "default_user"
    memory = get_agentic_memory()
    preference_injection = memory.get_system_prompt_injection(user_id)
    
    base_system_prompt = meta.get("system_prompt", "")
    enhanced_system_prompt = base_system_prompt + preference_injection if preference_injection else base_system_prompt
    
    history_manager = HistoryManager(system_prompt=enhanced_system_prompt)

    logger.debug(f"Received query: {query} with meta: {meta}")
    
    # Log conversation turn for memory extraction
    memory.add_conversation_turn(user_id, "user", query)

    # ---------------------------------------------------------------------------
    async def generate_response():
        modified_query = query
        refs = None
        
        # 1. Check Semantic Cache FIRST ----------------------------------------
        cache = get_semantic_cache()
        cached_response = cache.get(query)
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
                    history_manager.history.messages,
                    meta
                )
            except Exception as e:
                logger.error(f"Retriever error: {e}\n{traceback.format_exc()}")
                yield make_chunk(meta,
                                 message=f"Retriever error: {e}",
                                 status="error")
                return
            yield make_chunk(meta, status="generating")

        # 构造 Prompt ----
        messages = history_manager.get_history_with_msg(
            modified_query,
            max_rounds=meta.get("history_round")
        )
        history_manager.add_user(modified_query)                    # 把原始用户查询加入历史
        formatted_messages = convert_messages_to_dicts(messages)

        content = ""
        reasoning_content = ""
        try:
            for delta in model.predict(formatted_messages, stream=True):
                # 一些模型将「思考过程」放在 reasoning_content
                if not delta.content and hasattr(delta, "reasoning_content"):
                    reasoning_content += delta.reasoning_content or ""
                    yield make_chunk(meta,
                                     reasoning_content=reasoning_content,
                                     status="reasoning")
                    continue

                content += delta.content or ""

                yield make_chunk(meta, content=delta.content, status="loading")

            logger.debug(f"Final response: {content}")
            logger.debug(f"Final reasoning response: {reasoning_content}")

            # 4. 更新历史，发送最终块
            updated_history = history_manager.update_ai(content)
            history_serializable = convert_messages_to_dicts(updated_history)
            
            # 5. Cache the response (only cache non-retrieval queries for now)
            if content and not need_retrieve(meta):
                cache.set(query, content)
            
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
                        "message": "知识库未启用、或未指定知识库、或知识库不存在"
                    },
                    "graph_base": {
                        "results": {
                            "nodes": [],
                            "edges": []
                        }
                    },
                    "web_search": {
                        "results": [],
                        "message": "Web search is disabled"
                    },
                    "mysql_mcp": {
                        "answer": "",
                        "coords": []
                    },
                }

            # 最后一条流式返回
            yield make_chunk(
                meta,
                status="finished",
                history=history_serializable,
                refs=refs
            )


        except Exception as e:
            logger.error(f"Model error: {e}\n{traceback.format_exc()}")
            yield make_chunk(meta,
                             message=f"Model error: {e}",
                             status="error")

    return StreamingResponse(
        generate_response(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat.post("/call")
async def call(query: str = Body(...), meta: Dict[str, Any] = Body({})):
    """同步调用完整模型"""
    from src.models import select_model

    model = select_model(model_provider=meta.get("model_provider"), model_name=meta.get("model_name"))
    if model is None:
        raise HTTPException(status_code=500, detail="模型不可用")

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, model.predict, query)
    logger.debug(f"query: {query}, response: {response.content}")
    return {"response": response.content}


@chat.post("/call_lite")
async def call_lite(query: str = Body(...), meta: Dict[str, Any] = Body({})):
    """使用 Lite 模型，同步调用"""
    from src.models import select_model

    async def _predict_async(q):
        model_provider = meta.get("model_provider", config.model_provider_lite)
        model_name = meta.get("model_name", config.model_name_lite)
        model = select_model(model_provider=model_provider, model_name=model_name)
        if model is None:
            raise HTTPException(status_code=500, detail="Lite 模型不可用")
        loop = asyncio.get_event_loop()
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
    history: List[Dict[str, Any]] = Body([]),
    cfg: Dict[str, Any] = Body({}),
    meta: Dict[str, Any] = Body({})
):
    from src.agents.manager import agent_manager
    from src.knowledge.core.history_chat import HistoryManager

    try:
        agent = agent_manager.get_agent(agent_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 给历史消息补 id，防止前端缺字段
    for msg in history:
        msg.setdefault("id", str(uuid.uuid4()))

    # 本次请求 id / 线程 id
    request_id = cfg.get("request_id", str(uuid.uuid4()))
    thread_id  = cfg.get("thread_id",  request_id)
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
        msg_id: str,
        error: str | None = None,
        history_resp=None,
        refs=None,
        data: Dict[str, Any] | None = None
    ):
        """
        前端必须要拿到 msg.id，所以这里统一塞一个 msg 字段
        """
        payload = {
            "request_id": request_id,
            "response":  content,
            "status":    status,
            "meta":      meta,
            "msg":       {"id": msg_id, "type": "assistant"}     # ★ 重点
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
                        yield make_agent_chunk(
                            content=part,
                            status="loading",
                            msg_id=cur_msg_id
                        )
                    elif isinstance(part, dict):
                         # Structured Metadata (e.g. status update)
                         if "status" in part:
                             yield make_agent_chunk(
                                 status=part.get("status_text", "thinking"),
                                 msg_id=cur_msg_id,
                                 data=part
                             )

            # ③ finished
            updated           = history_mgr.update_ai(final_answer)
            history_serializ  = convert_messages_to_dicts(updated)

            refs = {
                "query": query,
                "history": history_serializ,
                "meta": meta,
                "model_name": "",        # 需要的话自行补充
                "entities": [],
                "knowledge_base": {
                    "results": [], "all_results": [], "rw_query": query,
                    "message": "知识库未启用、或未指定知识库、或知识库不存在"
                },
                "graph_base":  {"results": {"nodes": [], "edges": []}},
                "web_search":  {"results": [], "message": "Web search is disabled"},
                "mysql_mcp":   {"answer": "", "coords": []}
            }

            yield make_agent_chunk(
                content=final_answer,
                status="finished",
                msg_id=cur_msg_id,
                history_resp=history_serializ,
                refs=refs
            )

        except Exception as e:
            logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
            yield make_agent_chunk(
                status="error",
                msg_id=cur_msg_id,
                error=str(e)
            )

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
async def update_chat_models(model_provider: str, model_names: List[str]):
    config.model_names[model_provider]["models"] = model_names
    config._save_models_to_file()
    return {"models": config.model_names[model_provider]["models"]}

import os
@chat.post("/asr/")
async def asr_upload(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_input:
        temp_input.write(audio_bytes)
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
        output_path = temp_output.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        with open(output_path, "rb") as f:
            wav_bytes = f.read()
        result_text = get_whisper_client().transcribe(wav_bytes)
        return {"text": result_text}

    except subprocess.CalledProcessError as e:
        print("音频转码失败", e)
        return {"text": ""}

    finally:
        try:
            os.remove(input_path)
            os.remove(output_path)
        except Exception as cleanup_err:
            print("临时文件删除失败", cleanup_err)
