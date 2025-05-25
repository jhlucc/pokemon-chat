import os
import asyncio
import traceback
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Body, Query, Form
from fastapi.responses import JSONResponse
from pymilvus import Collection, MilvusException
from src.utils.logger import LogManager
from src import config
from src.stores import KnowledgeBase
from src.utils import hashstr
from pydantic import BaseModel
from src.stores.graphbase import GraphDatabase   # ← 你的 GraphDatabase 类
kgdb = GraphDatabase()                                # 单例，启动时自动连接 Neo4j
logger = LogManager()
data = APIRouter(prefix="/data")

# 单例 或者 在模块顶层初始化一次
kb = KnowledgeBase(
    milvus_uri=config.get("milvus_uri"),
    embedding_config={"enable_knowledge_base": True, **config}
)

@data.get("/")
async def list_databases():
    """列出所有知识库"""
    try:
        rows = kb.list_databases()
        return {"databases": rows}
    except Exception as e:
        logger.error(f"list_databases failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/")
async def create_database(
    database_name: str = Body(...),
    description: str = Body(...),
    dimension: Optional[int] = Body(None)
):
    """创建一个新的知识库 Collection"""
    try:
        info = kb.create_database(database_name, description, dimension)
        return JSONResponse(info)
    except Exception as e:
        logger.error(f"create_database failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.delete("/")
async def delete_database(db_id: str = Query(...)):
    """删除一个 Collection"""
    try:
        kb.delete_database(db_id)
        return {"message": "删除成功"}
    except Exception as e:
        logger.error(f"delete_database failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.get("/info")
async def get_database_info(db_id: str = Query(...)):
    try:
        db = kb.db_manager.get_database(db_id)
        if not db:
            raise HTTPException(404, "数据库不存在")
        db["collection_info"] = kb.get_collection_info(db_id)
        db["files"] = kb.db_manager.get_files_by_database(db_id)
        return db
    except Exception as e:
        logger.error(f"get_database_info failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


class FileChunkPayload(BaseModel):
    file: str
    chunk_size: int = 1000
    chunk_overlap: int = 100
    do_ocr: bool = False

@data.post("/file-to-chunk")
async def file_to_chunk(payload: FileChunkPayload):
    from rag.core.indexing import chunk_file
    try:
        docs = chunk_file(
            payload.file,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            do_ocr=payload.do_ocr
        )
        return {
            "chunks": [{"text": d.page_content, "meta": d.metadata} for d in docs]
        }
    except Exception as e:
        logger.error(f"file-to-chunk failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))
@data.post("/add-by-chunks")
async def add_by_chunks(
    db_id: str = Body(...),
    file_chunks: Dict[str, Dict[str, Any]] = Body(...)
):
    try:
        # 补：确保 Milvus 中有该 Collection（防止前端绕过 /data POST 创建数据库）
        if not kb.client.has_collection(db_id):
            dim = kb.db_manager.get_database(db_id)['dimension']
            kb.add_collection(db_id, dim)

        for file_id, file_data in file_chunks.items():
            # 添加文件记录
            kb.db_manager.add_file(
                db_id=db_id,
                file_id=file_id,
                filename=file_data.get("filename"),
                path="uploaded-via-chunks",
                file_type="custom",
                status="processing"
            )

            chunks = [
                {**chunk["meta"], "text": chunk["text"], "file_id": file_id}
                for chunk in file_data.get("nodes", [])
            ]
            texts = [chunk["text"] for chunk in file_data.get("nodes", [])]
            kb._insert_vectors(db_id, file_id, texts, chunks)
            kb.db_manager.update_file_status(file_id, 'done')

        return {"status": "success", "message": f"共导入 {len(file_chunks)} 个文件"}
    except Exception as e:
        logger.error(f"add-by-chunks failed: {e}\n{traceback.format_exc()}")
        return {"status": "failed", "message": str(e)}


@data.post("/ingest/file")
async def ingest_file(
    db_id: str = Body(...),
    path: str = Body(...),
    do_ocr: bool = Body(False),
    chunk_size: int = Body(1000),
    chunk_overlap: int = Body(100),
    ocr_threshold: float = Body(0.3)
):
    """把服务器路径下的单个文件导入到指定库"""
    try:
        file_id = kb.ingest_file(
            db_id=db_id,
            path=path,
            do_ocr=do_ocr,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_det_threshold=ocr_threshold
        )
        return {"file_id": file_id, "status": "success"}
    except Exception as e:
        logger.error(f"ingest_file failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/ingest/dir")
async def ingest_directory(
    db_id: str = Body(...),
    folder: str = Body(...),
    suffixes: Optional[List[str]] = Body(None)
):
    """把服务器目录下所有支持后缀的文件批量导入"""
    try:
        ids = kb.ingest_directory(db_id, folder, suffixes)
        return {"file_ids": ids, "status": "success"}
    except Exception as e:
        logger.error(f"ingest_directory failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.get("/search")
async def search_kb(
    query: str = Query(...),
    db_id: str = Query(...),
    distance_threshold: float = Query(None),
    rerank: bool = Query(True),
    top_k: int = Query(None)
):
    """向量检索接口"""
    try:
        res = kb.search(
            query=query,
            db_id=db_id,
            distance_threshold=distance_threshold,
            rerank=rerank,
            top_k=top_k
        )
        return res
    except Exception as e:
        logger.error(f"search_kb failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db_id: Optional[str] = Form(None)
):
    """前端上传文件到后端，再由你自己调 ingest_file 导入"""
    if not file.filename:
        raise HTTPException(400, "No file")
    # 存到临时 uploads 目录
    upload_dir = kb.work_dir
    if db_id:
        _, upload_dir = kb._ensure_directories(db_id)
    os.makedirs(upload_dir, exist_ok=True)

    name, ext = os.path.splitext(file.filename)
    fname = f"{name}_{hashstr(name, 4, True)}{ext}"
    path = os.path.join(upload_dir, fname)
    with open(path, "wb") as buf:
        buf.write(await file.read())
    return {"file_path": path, "db_id": db_id}

from pymilvus import Collection, connections

@data.delete("/document")
async def delete_document(db_id: str = Body(...), file_id: str = Body(...)):
    try:
        dim = kb.db_manager.get_database(db_id)['dimension']

        # 若 Collection 不存在则先建
        if not kb.client.has_collection(db_id):
            kb.add_collection(db_id, dim)

        # ✅ 手动建立连接
        connections.connect(alias="default", uri=config.get("milvus_uri", "http://localhost:19530"))
        collection = Collection(db_id)

        try:
            collection.load()
        except MilvusException as e:
            if "index doesn't exist" in str(e):
                collection.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "L2",
                        "params": {"nlist": 1024}
                    }
                )
                collection.load()
            else:
                raise e

        # 查询对应向量 ID
        rows = kb.client.query(
            collection_name=db_id,
            filter=f'file_id == "{file_id}"',
            output_fields=["id"]
        )
        ids = [r["id"] for r in rows]

        if ids:
            kb.client.delete(collection_name=db_id, ids=ids)
        kb.db_manager.delete_file(file_id)

        return {"message": "删除成功"}
    except Exception as e:
        logger.error(f"delete_document failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@data.get("/debug/list-uploads")
async def list_uploaded_files(db_id: str = Query(...)):
    _, upload_dir = kb._ensure_directories(db_id)
    if not os.path.exists(upload_dir):
        return {"error": "Upload folder not found"}
    files = os.listdir(upload_dir)
    return {"files": files}
from neo4j import GraphDatabase

# graph
@data.get("/graph")
async def get_graph_info():
    """
    前端进入图数据库页面时的“概览”接口
    """
    try:
        info = kgdb.get_graph_info() or {}
        return info
    except Exception as e:
        logger.error(f"get_graph_info failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@data.get("/graph/nodes")
async def get_graph_nodes(kgdb_name: str = Query("neo4j"), num: int = Query(100)):
    """
    抽样返回 num 条节点+关系，用于首页随机展示
    """
    try:
        raw = kgdb.get_sample_nodes(kgdb_name, num)
        result = kgdb.format_general_results(raw)
        return {"result": result}
    except Exception as e:
        logger.error(f"get_graph_nodes failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


def dedup_subgraph(subgraph: dict) -> dict:
    name_to_node = {}
    id_map = {}

    # 去重节点（以 name 为唯一标识）
    for node in subgraph.get("nodes", []):
        name = node["name"]
        if name not in name_to_node:
            name_to_node[name] = node
        id_map[node["id"]] = name_to_node[name]["id"]

    # 处理边：将 source_id/target_id 映射到唯一 id，并避免自环与重复
    new_edges = []
    seen = set()
    for edge in subgraph.get("edges", []):
        src = id_map.get(edge["source_id"])
        tgt = id_map.get(edge["target_id"])
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt, edge["type"])
        if key not in seen:
            seen.add(key)
            new_edges.append({
                "source_id": src,
                "target_id": tgt,
                "type": edge["type"]
            })

    return {
        "nodes": list(name_to_node.values()),
        "edges": new_edges
    }

@data.get("/graph/node")
async def query_graph_node(entity_name: str = Query(...)):
    try:
        raw = kgdb.query_node(entity_name)
        cleaned = dedup_subgraph(raw)
        return {"result": cleaned}
    except Exception as e:
        logger.error(f"query_graph_node failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

