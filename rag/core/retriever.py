
from src import config
from src.models.reranker_model import RerankerWrapper
from src.utils.logger import LogManager
from src.models import select_model
from rag.core.prompts import *
from src.stores import  KnowledgeBase
from rag.core.operators import HyDEOperator
from src.mcp.client_core import MCPClient
import asyncio, inspect,json
knowledge_base = KnowledgeBase()

_log = LogManager()

def get_kg_agent():
    from src.agents.kg_agent import KGQueryAgent
    return KGQueryAgent()
class Retriever:

    def __init__(self):
        self._load_models()
        self.kg_agent = get_kg_agent()
        self.default_distance_threshold = config.get("default_distance_threshold", 1.6)
        self.top_k = config.get("default_top_k", 10)
        # self._mcp_client = MCPClient()  # 全局复用一条连接
    def _load_models(self):
        if config.enable_reranker:
            self.reranker = RerankerWrapper(config)

        if config.enable_web_search:
            from api.websearch.websearcher import LiteBaseSearcher, TavilyBasicSearcher
            self.web_searcher = LiteBaseSearcher()

    def retrieval(self, query, history, meta):
        refs = {"query": query, "history": history, "meta": meta}
        refs["model_name"] = config.model_name
        refs["entities"] = self.reco_entities(query, history, refs)
        refs["knowledge_base"] = self.query_knowledgebase(query, history, refs)
        refs["graph_base"] = self.query_graph(query, history, refs) #图谱
        refs["web_search"] = self.query_web(query, history, refs)
        refs["mysql_mcp"]=self.query_mysql_mcp(query, history, refs)
        return refs

    async def _call_mcp(self, query: str) -> dict:
        client = MCPClient()
        answer,_ = await  client.ask(query)

        return {"answer": answer}
    def restart(self):
        """所有需要重启的模型"""
        self._load_models()

    def query_mysql_mcp(self, query, history, refs):
        meta = refs["meta"]
        mcp_id = meta.get("mcp_id")  # 按钮亮时 = 'default'
        if not mcp_id:
            return {"answer": "" }

        try:
            # A) 如果当前线程已有事件循环（典型：FastAPI 路由函数）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._call_mcp(query), loop)
                return future.result()

            # B) 普通同步环境：自己开一个 loop
            return asyncio.run(self._call_mcp(query))

        except Exception as e:
            _log.error(f"MCP 查询失败: {e}")
            return {"answer": ""}
    def clean_kb_text(self,kb_res, max_len=100):
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
            web_text = "\n".join(f"{r['title']}:\n{r['content']}" for r in web_res)
            external_parts.append("网络搜索信息:\n" + web_text)
        mcp_tes=refs.get("mysql_mcp", {}).get("answer", [])
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
        if refs["meta"].get("use_graph"):
            # 调用 KGQueryAgent.query，传hops参数
            result = self.kg_agent.query(
                query,
                hops=refs["meta"].get("graphHops", 2)
            )
            # result = {"answer": "...", "subgraph": {nodes{},edges}}
            return result
        return {"answer": None, "subgraph": None}

    def query_knowledgebase(self, query, history, refs):
        response = {
            "results": [],
            "all_results": [],
            "rw_query": query,
            "message": "",
        }

        meta = refs["meta"]
        db_id = meta.get("db_id")
        if not db_id or not config.enable_knowledge_base:
            response["message"] = "知识库未启用、或未指定知识库、或知识库不存在"
            return response

        rw_query = self.rewrite_query(query, history, refs)
        response["rw_query"] = rw_query

        try:
            kb_res = knowledge_base.search(
                query=rw_query,
                db_id=db_id,
                distance_threshold=meta.get("distanceThreshold", self.default_distance_threshold),
                rerank=True,
                top_k=meta.get("topK", self.top_k)
            )
            response["results"] = kb_res["results"]
            response["all_results"] = kb_res["all_results"]
        except Exception as e:
            response["message"] = f"检索出错: {e}"
        return response

    def query_web(self, query, history, refs):
        """查询网络：直接同步调用 WebSearcher"""
        if not (refs["meta"].get("use_web") and config.enable_web_search):
            return {"results": [], "message": "Web search is disabled"}

        try:
            search_results = self.web_searcher.search(query, top_k=5)
        except Exception as e:
            _log.error(f"Web search error: {e}")
            return {"results": [], "message": f"Web search error: {e}"}

        return {"results": search_results}

    def rewrite_query(self, query, history, refs):
        """重写查询"""
        model_provider = config.model_provider
        model_name = config.model_name
        model = select_model(model_provider=model_provider, model_name=model_name)
        if refs["meta"].get("mode") == "search":  # 如果是搜索模式，就使用 meta 的配置，否则就使用全局的配置
            rewrite_query_span = refs["meta"].get("use_rewrite_query", "off")
        else:
            rewrite_query_span = config.use_rewrite_query

        if rewrite_query_span == "off":
            rewritten_query = query
        else:

            history_query = [entry["content"] for entry in history if entry["role"] == "user"] if history else ""
            rewritten_query_prompt = rewritten_query_prompt_template.format(history=history_query, query=query)
            rewritten_query = model.predict(rewritten_query_prompt).content

        if rewrite_query_span == "hyde":
            res = HyDEOperator.call(model_callable=model.predict, query=query, context_str=history_query)
            rewritten_query = res.content

        return rewritten_query

    def reco_entities(self, query, history, refs):
        """识别句子中的实体"""
        query = refs.get("rewritten_query", query)
        model_provider = config.model_provider
        model_name = config.model_name
        model = select_model(model_provider=model_provider, model_name=model_name)

        entities = []
        if refs["meta"].get("use_graph"):
            entity_extraction_prompt = keywords_prompt_template.format(text=query)
            entities = model.predict(entity_extraction_prompt).content.split("<->")
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
            for i, relationship in enumerate(item[1]):
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
