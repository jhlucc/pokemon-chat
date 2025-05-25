# server/routers/tool_router.py

import os
import json
import uuid
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

#  chunk_file & parse_file
from rag.core.indexing import chunk_file, parse_file

#

from src.agents.manager import agent_manager
# —— 日志
from src.utils.logger import LogManager
logger = LogManager()

router = APIRouter(tags=["tools","agent"])

#
# —— 1./tools前缀：工具列表+文件分块+PDF→文本
#
tools_router = APIRouter(prefix="/tools", tags=["tools"])

class Tool(BaseModel):
    name: str
    title: str
    description: str
    url: str
    method: Optional[str] = "POST"
    params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@tools_router.get("/", response_model=List[Tool])
async def list_tools():
    """
    列出所有可用工具，包括文件分块、PDF 转文本，以及已注册的 agent
    """
    tools: List[Tool] = [
        Tool(
            name="file-chunking",
            title="文件分块",
            description="给定一个本地文件路径，用 chunk_file 切分成若干 Document。",
            url="/tools/file-chunking",
            method="POST",
            params={"file":"/path/to/file.pdf","chunk_size":1000,"chunk_overlap":100}
        ),
        Tool(
            name="pdf2txt",
            title="PDF 转文本",
            description="给定一个本地 PDF 路径，跑 OCR/解析，返回纯文本。",
            url="/tools/pdf2txt",
            method="POST",
            params={"file":"/path/to/file.pdf"}
        ),
        Tool(
            name="agent",
            title="智能体",
            description="智能体演练平台",
            url="/tools/agent",
        )

    ]

    # for agent_cls in agent_manager.agents.values():
    #     try:
    #         agent = agent_cls()  # ✅ 实例化
    #         info = agent.get_info()
    #         tools.append(
    #             Tool(
    #                 name=info["name"],
    #                 title=info["name"],
    #                 description=info.get("description", ""),
    #                 url=f"/chat/agent/{info['name']}",
    #                 method="POST",
    #                 metadata=agent.config_schema.to_dict() if hasattr(agent, "config_schema") else {}
    #             )
    #         )
    #     except Exception as e:
    #         logger.error(f"加载 agent {agent_cls} 失败: {e}")

    return tools

class FileChunkPayload(BaseModel):
    file: str
    chunk_size: int = 1000
    chunk_overlap: int = 100

@tools_router.post("/file-chunking")
async def file_chunking(payload: FileChunkPayload):
    try:
        docs = chunk_file(
            payload.file,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            do_ocr=False
        )
        # Document.page_content + metadata
        return {"chunks": [
            {"text": d.page_content, "meta": d.metadata}
            for d in docs
        ]}
    except Exception as e:
        logger.error(f"file-chunking 出错: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

class PDFPayload(BaseModel):
    file: str
@tools_router.post("/pdf2txt")
async def pdf_to_text(payload: PDFPayload):
    try:
        real_path = payload.file  # 前端已经传的是绝对路径
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"文件不存在: {real_path}")

        text = parse_file(real_path, do_ocr=True)
        return {"text": text}
    except Exception as e:
        logger.error(f"pdf2txt 出错: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


# # 2. /agent 前缀：流式调用QA_Agent
# agent_router = APIRouter(prefix="/agent", tags=["agent"])
# qa_agent = PokemonKGChatAgent()
#
# def _make_chunk(request_id: str,
#                 content: str = "",
#                 status: str = "",
#                 error: str = None):
#     payload = {
#         "request_id": request_id,
#         "response": content,
#         "status": status,
#     }
#     if error:
#         payload["error"] = error
#     return json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
#
# @agent_router.post("/{agent_name}")
# async def run_agent(
#     agent_name: str,
#     query: str = Body(...),
#     meta: dict = Body({}),
#     history: List[dict] = Body(None),
# ):
#     """
#     POST /agent/{agent_name}
#     {
#       "query": "用户的问题",
#       "meta": { … },
#       "history": [ … ]
#     }
#     """
#     request_id = str(uuid.uuid4())
#
#     async def streamer():
#         # 1) 初始化
#         yield _make_chunk(request_id, status="init")
#
#         # 2) 调用agent
#         try:
#             # 假设 qa_agent.query 返回 async generator of str
#             async for piece in qa_agent.query(query, meta=meta, history=history):
#                 yield _make_chunk(request_id, content=piece, status="loading")
#         except Exception as e:
#             logger.error(f"Agent 调用失败: {e}\n{traceback.format_exc()}")
#             yield _make_chunk(request_id, status="error", error=str(e))
#             return
#
#         # 3) 完成
#         yield _make_chunk(request_id, status="finished")
#
#     return StreamingResponse(streamer(), media_type="application/json")

#
# —— 3. 挂载子路由
#
router.include_router(tools_router)
# router.include_router(agent_router)
