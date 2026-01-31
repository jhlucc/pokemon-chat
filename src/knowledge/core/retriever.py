from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.feature_flags import feature_enabled
from src.core.llm_factory import build_chat_llm
from src.core.settings import settings
from src.knowledge.core.operators import HyDEOperator
from src.knowledge.core.prompts import keywords_prompt_template, knowbase_qa_template, rewritten_query_prompt_template
from src.models.reranker_model import RerankerWrapper
from src.runtime import get_kb, get_kg_agent, get_mcp_client
from src.utils.logger import get_logger

_log = get_logger(__name__)


_DEFAULT_MCP_TIMEOUT_S = 15.0


class Retriever:
    def __init__(self):
        self._load_models()
        self._kg_agent = None
        self.default_distance_threshold = settings.kb_config.default_distance_threshold
        self.top_k = settings.kb_config.default_top_k
        # self._mcp_client = MCPClient()  # 全局复用一条连接

    def _get_kb(self):
        return get_kb()

    def _get_kg_agent(self):
        if not feature_enabled("enable_knowledge_graph"):
            return None
        if self._kg_agent is None:
            self._kg_agent = get_kg_agent()
        return self._kg_agent

    def _get_llm(self, meta: dict[str, Any] | None = None):
        """
        Return an OpenAI-compatible LLM for internal retrieval steps.

        Prefer the request-selected model (meta['model_provider'/'model_name']) when available
        so retriever-side rewrites/NER stay aligned with the chat model.
        """
        meta = meta or {}
        return build_chat_llm(
            model_provider=meta.get("model_provider"),
            model_name=meta.get("model_name"),
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    def _load_models(self):
        # Defaults so callers can safely access attributes behind feature flags.
        self.reranker = None
        self.web_searcher = None

        if feature_enabled("enable_reranker"):
            # 使用 settings 默认配置初始化
            self.reranker = RerankerWrapper()

        if feature_enabled("enable_web_search"):
            from src.agents.tools.websearch.websearcher import LiteBaseSearcher

            self.web_searcher = LiteBaseSearcher()

    def retrieval(self, query: str, history: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
        refs = {"query": query, "history": history, "meta": meta}
        refs["model_name"] = settings.llm.model_name
        refs["entities"] = self.reco_entities(query, history, refs)

        # Parallelize independent retrieval steps when multiple tools are requested.
        # This keeps end-to-end latency closer to max(step_latency) than sum(step_latency).
        requested = sum(
            1
            for v in (
                bool(meta.get("db_id")),
                bool(meta.get("use_graph")),
                bool(meta.get("use_web")),
                bool(meta.get("mcp_id")),
            )
            if v
        )

        if requested >= 2:
            with ThreadPoolExecutor(max_workers=4) as ex:
                fut_kb = ex.submit(self.query_knowledgebase, query, history, refs)
                fut_graph = ex.submit(self.query_graph, query, history, refs)
                fut_web = ex.submit(self.query_web, query, history, refs)
                fut_mcp = ex.submit(self.query_mysql_mcp, query, history, refs)

                refs["knowledge_base"] = fut_kb.result()
                refs["graph_base"] = fut_graph.result()  # 图谱
                refs["web_search"] = fut_web.result()
                refs["mysql_mcp"] = fut_mcp.result()
        else:
            refs["knowledge_base"] = self.query_knowledgebase(query, history, refs)
            refs["graph_base"] = self.query_graph(query, history, refs)  # 图谱
            refs["web_search"] = self.query_web(query, history, refs)
            refs["mysql_mcp"] = self.query_mysql_mcp(query, history, refs)
        return refs

    async def _call_mcp(self, query: str, *, timeout_s: float) -> dict:
        client = get_mcp_client()
        try:
            answer, _ = await asyncio.wait_for(client.ask(query), timeout=timeout_s)
            return {"answer": answer}
        except asyncio.TimeoutError:
            _log.error(f"MCP 查询超时: {timeout_s}s")
            return {"answer": ""}

    def restart(self):
        """所有需要重启的模型"""
        self._load_models()

    def query_mysql_mcp(self, query: str, history: list[dict[str, Any]], refs: dict[str, Any]) -> dict[str, Any]:
        meta = refs["meta"]
        mcp_id = meta.get("mcp_id")  # 按钮亮时 = 'default'
        if not mcp_id or not feature_enabled("enable_mcp"):
            return {"answer": ""}

        try:
            timeout_s = float(meta.get("mcp_timeout_s", _DEFAULT_MCP_TIMEOUT_S))

            # A) 如果当前线程已有事件循环（典型：FastAPI 路由函数）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Avoid deadlock: run the coroutine in a fresh thread/event-loop.
                result: dict = {"answer": ""}
                err: Exception | None = None

                def _runner():
                    nonlocal result, err
                    try:
                        result = asyncio.run(self._call_mcp(query, timeout_s=timeout_s))
                    except Exception as e:
                        err = e

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                t.join(timeout=timeout_s + 1.0)
                if t.is_alive():
                    _log.error(f"MCP 查询超时(线程): {timeout_s}s")
                    return {"answer": ""}
                if err:
                    raise err
                return result

            # B) 普通同步环境：自己开一个 loop
            return asyncio.run(self._call_mcp(query, timeout_s=timeout_s))

        except Exception as e:
            _log.error(f"MCP 查询失败: {e}")
            return {"answer": ""}

    def clean_kb_text(self, kb_res, max_len=100):
        seen = set()
        cleaned = []
        for r in kb_res:
            text = r.get("entity", {}).get("text", "").strip().replace("\n", " ")
            if text not in seen:
                seen.add(text)
                short_text = text[:max_len] + "..." if len(text) > max_len else text
                cleaned.append(f"{r.get('id', 'N/A')}: {short_text}")
        return cleaned

    def construct_query(self, query, refs, meta):
        _log.debug(f"{refs=}")
        if not refs or len(refs) == 0:
            return query

        external_parts = []

        # 加入图谱 answer（自然语言形式）
        graph_answer = refs.get("graph_base", {}).get("answer")
        if isinstance(graph_answer, str) and graph_answer.strip():
            external_parts.append("图谱回答:\n" + graph_answer.strip())

        # 加入知识库结果（结构化块）
        kb_res = refs.get("knowledge_base", {}).get("results", [])
        # print(kb_res)
        if kb_res:
            kb_text = "\n".join(self.clean_kb_text(kb_res))
            external_parts.append("知识库信息:\n" + kb_text)

        # 加入网络搜索结果
        web_res = refs.get("web_search", {}).get("results", [])
        if web_res:

            def _web_item_to_text(item: Any) -> str:
                if item is None:
                    return ""
                if isinstance(item, dict):
                    title = (item.get("title") or "").strip()
                    content = (item.get("content") or item.get("content_snippet") or item.get("snippet") or "").strip()
                else:
                    title = str(getattr(item, "title", "") or "").strip()
                    content = str(
                        getattr(item, "content", None) or getattr(item, "content_snippet", None) or ""
                    ).strip()
                if not (title or content):
                    return ""
                return f"{title}:\n{content}" if title else content

            web_text = "\n".join(t for t in (_web_item_to_text(r) for r in web_res) if t)
            external_parts.append("网络搜索信息:\n" + web_text)
        mcp_tes = refs.get("mysql_mcp", {}).get("answer", [])
        if mcp_tes:
            external_parts.append("MySQL_MCP:\n" + mcp_tes.strip())
        # print(external_parts)
        # 构造最终查询
        if external_parts:
            external = "\n\n".join(external_parts)
            query = knowbase_qa_template.format(external=external, query=query)

        return query

    def query_classification(self, query):
        """判断是否需要查询
        - 对于完全基于用户给定信息的任务，称之为"足够""sufficient"，不需要检索；
        - 否则，称之为"不足""insufficient"，可能需要检索，
        """
        raise NotImplementedError

    def query_graph(self, query, history, refs):
        empty = {"nodes": [], "edges": []}
        if not (refs["meta"].get("use_graph") and feature_enabled("enable_knowledge_graph")):
            # Frontend expects `results` for visualization.
            return {"answer": None, "results": empty, "subgraph": empty}

        try:
            agent = self._get_kg_agent()
            if agent is None:
                return {"answer": None, "results": empty, "subgraph": empty}

            # 调用 KGQueryAgent.query，传hops参数
            result = agent.query(query, hops=refs["meta"].get("graphHops", 2))
            # Normalize shape for the frontend: `graph_base.results.{nodes,edges}`
            if isinstance(result, dict):
                subgraph = result.get("results") or result.get("subgraph") or empty
                return {**result, "results": subgraph, "subgraph": subgraph}
            return {"answer": None, "results": empty, "subgraph": empty, "message": "Invalid graph response"}
        except Exception as e:
            _log.error(f"Graph query error: {e}")
            return {"answer": None, "results": empty, "subgraph": empty, "message": str(e)}

    def query_knowledgebase(self, query, history, refs):
        response = {
            "results": [],
            "all_results": [],
            "rw_query": query,
            "message": "",
        }

        meta = refs["meta"]
        db_id = meta.get("db_id")
        if not db_id or not feature_enabled("enable_knowledge_base"):
            response["message"] = "知识库未启用、或未指定知识库、或知识库不存在"
            return response

        rw_query = self.rewrite_query(query, history, refs)
        response["rw_query"] = rw_query

        try:
            kb = self._get_kb()
            kb_res = kb.search(
                query=rw_query,
                db_id=db_id,
                distance_threshold=meta.get("distanceThreshold", self.default_distance_threshold),
                rerank=True,
                top_k=meta.get("topK", self.top_k),
            )
            response["results"] = kb_res["results"]
            response["all_results"] = kb_res["all_results"]
        except Exception as e:
            response["message"] = f"检索出错: {e}"
        return response

    def query_web(self, query, history, refs):
        """查询网络：直接同步调用 WebSearcher"""
        if not (refs["meta"].get("use_web") and feature_enabled("enable_web_search")):
            return {"results": [], "message": "Web search is disabled"}

        try:
            if self.web_searcher is None:
                # Feature might be toggled on at runtime; lazy init here.
                from src.agents.tools.websearch.websearcher import LiteBaseSearcher

                self.web_searcher = LiteBaseSearcher()
            raw_results = self.web_searcher.search(query, top_k=5)
        except Exception as e:
            _log.error(f"Web search error: {e}")
            return {"results": [], "message": f"Web search error: {e}"}

        # Frontend expects each result shape: {title, url, content, score}.
        results: list[dict[str, Any]] = []
        for item in raw_results or []:
            if item is None:
                continue
            if isinstance(item, dict):
                title = (item.get("title") or "").strip()
                url = (item.get("url") or item.get("link") or "").strip()
                content = (item.get("content") or item.get("content_snippet") or item.get("snippet") or "").strip()
                score = item.get("score", 1.0)
            else:
                title = str(getattr(item, "title", "") or "").strip()
                url = str(getattr(item, "url", "") or "").strip()
                content = str(getattr(item, "content", None) or getattr(item, "content_snippet", None) or "").strip()
                score = getattr(item, "score", 1.0)
            try:
                score_f = float(score) if score is not None else 0.0
            except Exception:
                score_f = 0.0
            results.append({"title": title, "url": url, "content": content, "score": score_f})

        return {"results": results}

    def rewrite_query(self, query, history, refs):
        """重写查询"""
        model = self._get_llm(refs.get("meta"))
        if refs["meta"].get("mode") == "search":  # 如果是搜索模式，就使用 meta 的配置，否则就使用全局的配置
            rewrite_query_span = refs["meta"].get("use_rewrite_query", "off")
        else:
            # 暂时默认为 off 或从 settings 读取如果存在
            rewrite_query_span = "off"  # settings.features.use_rewrite_query? 暂无该配置，默认为 off

        if rewrite_query_span == "off":
            rewritten_query = query
        else:
            history_query = [entry["content"] for entry in history if entry["role"] == "user"] if history else ""
            rewritten_query_prompt = rewritten_query_prompt_template.format(history=history_query, query=query)
            rewritten_query = model.invoke(rewritten_query_prompt).content

        if rewrite_query_span == "hyde":
            res = HyDEOperator.call(model_callable=model.invoke, query=query, context_str=history_query)
            rewritten_query = res.content

        return rewritten_query

    def reco_entities(self, query, history, refs):
        """识别句子中的实体"""
        query = refs.get("rewritten_query", query)
        model = self._get_llm(refs.get("meta"))

        entities = []
        if refs["meta"].get("use_graph"):
            entity_extraction_prompt = keywords_prompt_template.format(text=query)
            entities = model.invoke(entity_extraction_prompt).content.split("<->")
            # entities = [entity for entity in entities if all(char.isalnum() or char in "汉字" for char in entity)]

        return entities

    def _extract_relationship_info(self, relationship, source_name=None, target_name=None, node_dict=None):
        """
        提取关系信息并返回格式化的节点和边信息
        """
        rel_id = relationship.element_id
        nodes = relationship.nodes
        if len(nodes) != 2:
            return None, None

        source, target = nodes
        source_id = source.element_id
        target_id = target.element_id

        source_name = node_dict[source_id]["name"] if source_name is None else source_name
        target_name = node_dict[target_id]["name"] if target_name is None else target_name

        relationship_type = relationship._properties.get("type", "unknown")
        if relationship_type == "unknown":
            relationship_type = relationship.type

        edge_info = {
            "id": rel_id,
            "type": relationship_type,
            "source_id": source_id,
            "target_id": target_id,
            "source_name": source_name,
            "target_name": target_name,
        }

        node_info = [
            {"id": source_id, "name": source_name},
            {"id": target_id, "name": target_name},
        ]

        return node_info, edge_info

    def format_general_results(self, results):
        formatted_results = {"nodes": [], "edges": []}

        for item in results:
            relationship = item[1]
            source_name = item[0]._properties.get("name", "unknown")
            target_name = item[2]._properties.get("name", "unknown") if len(item) > 2 else "unknown"

            node_info, edge_info = self._extract_relationship_info(relationship, source_name, target_name)
            if node_info is None or edge_info is None:
                continue

            for node in node_info:
                if node["id"] not in [n["id"] for n in formatted_results["nodes"]]:
                    formatted_results["nodes"].append(node)

            formatted_results["edges"].append(edge_info)

        return formatted_results

    def format_query_results(self, results):
        _log.debug(f"Graph Query Results: {results}")
        formatted_results = {"nodes": [], "edges": []}
        node_dict = {}

        for item in results:
            # 检查数据格式
            if len(item) < 2 or not isinstance(item[1], list):
                continue

            node_dict[item[0].element_id] = dict(id=item[0].element_id, name=item[0]._properties.get("name", "Unknown"))
            node_dict[item[2].element_id] = dict(id=item[2].element_id, name=item[2]._properties.get("name", "Unknown"))

            # 处理关系列表中的每个关系
            for _i, relationship in enumerate(item[1]):
                try:
                    # 提取关系信息
                    node_info, edge_info = self._extract_relationship_info(relationship, node_dict=node_dict)
                    if node_info is None or edge_info is None:
                        continue

                    # 添加边
                    formatted_results["edges"].append(edge_info)
                except Exception as e:
                    _log.error(f"处理关系时出错: {e}, 关系: {relationship}")
                    continue

        # 将节点字典转换为列表
        formatted_results["nodes"] = list(node_dict.values())

        return formatted_results

    def __call__(self, query, history, meta):
        refs = self.retrieval(query, history, meta)
        query = self.construct_query(query, refs, meta)
        return query, refs


if __name__ == "__main__":
    from pprint import pprint

    retriever = Retriever()
    query = "皮卡丘在哪里能抓到？"
    history = []
    meta = {
        "use_graph": False,
        "use_web": False,
        "db_id": "kb_66c4fe27",
        "history_round": 4,
        "mode": "search",  # 启用 query rewrite
        "use_rewrite_query": "off",  # off / on / hyde
    }

    # 调用 __call__
    constructed_query, refs = retriever(query, history, meta)

    print("\n==== 拼装后 Query ====\n")
    print(constructed_query)
    print("\n==== 拼装长度 ====\n")
    print(len(constructed_query))

    print("\n==== refs 内容 ====\n")
    pprint(refs)
