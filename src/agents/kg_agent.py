import logging
import os
import pickle
import traceback
import warnings
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from py2neo import Graph
from pydantic import BaseModel, Field

from src.core.feature_flags import feature_enabled
from src.core.settings import settings
from src.ner.ner_model import _BERT_AVAILABLE, get_ner_result_simple, rule_find, tfidf_alignment
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")

_log = get_logger(__name__)


# ========== 子图提取器 ==========
class GraphSubgraphExtractor:
    """给定实体名，抽取 n 跳子图并格式化为 {nodes, edges}"""

    def __init__(self, graph: Graph):
        self.graph = graph

    # ---------- Cypher 查询 ----------
    def _query(self, entity: str, hops: int, limit: int = 200):
        cypher = f"""
        MATCH (n {{name:$name}})-[r*1..{hops}]-(m)
        RETURN n AS n, r, m AS m
        LIMIT $limit
        """
        return self.graph.run(cypher, name=entity, limit=limit).data()

    # ---------- 单条关系→节点/边 ----------
    @staticmethod
    def _extract_relationship(rel, node_dict):
        try:
            # 用 id(rel) 替代不可用的 rel.element_id
            rel_id = id(rel)

            # 获取起止节点（py2neo 的风格）
            source = rel.start_node
            target = rel.end_node
            sid = id(source)
            tid = id(target)

            # 节点缓存（注意不要用 element_id）
            for node in (source, target):
                nid = id(node)
                if nid not in node_dict:
                    node_dict[nid] = {"id": nid, "name": node.get("name", "Unknown")}

            # 构造边
            return {
                "id": rel_id,
                "type": rel.get("type", rel.__class__.__name__),  # 先尝试属性中有无 type，否则使用关系类型名
                "source_id": sid,
                "target_id": tid,
                "source_name": node_dict[sid]["name"],
                "target_name": node_dict[tid]["name"],
            }

        except Exception as e:
            logging.error(f"关系解析失败: {e}\n{traceback.format_exc()}")
            return None

    # ---------- 原 GraphDatabase 风格的格式化 ----------
    @classmethod
    def _format(cls, raw: list[dict]) -> dict:
        node_dict, edge_dict = {}, {}
        for row in raw:
            n1, rels, n2 = row["n"], row["r"], row["m"]
            for node in (n1, n2):
                nid = node.identity  # or id(node)
                if nid not in node_dict:
                    node_dict[nid] = {"id": nid, "name": node.get("name", "Unknown")}
            for rel in rels:
                edge_info = cls._extract_relationship(rel, node_dict)
                if edge_info:
                    edge_dict[edge_info["id"]] = edge_info
        return {"nodes": list(node_dict.values()), "edges": list(edge_dict.values())}

    # ---------- 对外接口 ----------
    def get_subgraph(self, entity: str, hops: int = 2, limit: int = 200):
        raw = self._query(entity, hops, limit)
        return None if not raw else self._format(raw)


class EntityRecognizer:
    """Entity recognition helper for KG subgraph extraction.

    Prefers BERT NER when enabled and resources are present; falls back to
    AC automaton + TF-IDF alignment (no torch/transformers required).
    """

    def __init__(self):
        self._rule = None
        self._tfidf_r = None

        self._use_bert = bool(feature_enabled("enable_ner_bert") and _BERT_AVAILABLE)
        self._bert_loaded = False
        self._bert_model = None
        self._bert_tokenizer = None
        self._bert_device = None
        self._idx2tag = None
        self._bert_get_ner_result = None

    def _ensure_simple(self):
        if self._rule is not None and self._tfidf_r is not None:
            return
        try:
            self._rule = rule_find()
            self._tfidf_r = tfidf_alignment()
        except Exception as e:
            _log.warning(f"NER simple init failed, will return empty entities: {e}")
            self._rule = None
            self._tfidf_r = None

    def _ensure_bert(self):
        if not self._use_bert or self._bert_loaded:
            return

        # BERT NER still needs rule/tfidf for merging/alignment.
        self._ensure_simple()
        if self._rule is None or self._tfidf_r is None:
            self._use_bert = False
            return

        try:
            import torch

            from src.ner.ner_model import Bert_Model, BertTokenizer, get_ner_result

            ner_tag_path = str(settings.paths.ner_tag_path)
            pt_path = str(settings.paths.cache_berta_model)
            model_dir = str(settings.paths.model_roberta_path)

            if not (os.path.exists(ner_tag_path) and os.path.exists(pt_path) and os.path.exists(model_dir)):
                raise FileNotFoundError("NER BERT resources missing")

            with open(ner_tag_path, "rb") as f:
                tag2idx = pickle.load(f)
            idx2tag = list(tag2idx)

            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            tokenizer = BertTokenizer.from_pretrained(model_dir, cache_dir=model_dir)
            model = Bert_Model(model_dir, hidden_size=128, tag_num=len(tag2idx), bi=True)
            model.load_state_dict(torch.load(pt_path, map_location=device))
            model = model.to(device)

            self._bert_model = model
            self._bert_tokenizer = tokenizer
            self._bert_device = device
            self._idx2tag = idx2tag
            self._bert_get_ner_result = get_ner_result
            self._bert_loaded = True
        except Exception as e:
            _log.warning(f"NER BERT init failed, fallback to simple NER: {e}")
            self._use_bert = False
            self._bert_loaded = False

    def ner(self, question: str) -> dict:
        # Try BERT first
        if self._use_bert:
            self._ensure_bert()
            if self._bert_loaded:
                try:
                    return self._bert_get_ner_result(
                        self._bert_model,
                        self._bert_tokenizer,
                        question,
                        self._rule,
                        self._tfidf_r,
                        self._bert_device,
                        self._idx2tag,
                    )
                except Exception as e:
                    _log.warning(f"NER BERT run failed, fallback to simple NER: {e}")
                    self._use_bert = False

        # Fallback: rule + tfidf
        try:
            self._ensure_simple()
            if self._rule is None or self._tfidf_r is None:
                return {}
            return get_ner_result_simple(question, rule=self._rule, tfidf_r=self._tfidf_r)
        except Exception as e:
            _log.warning(f"NER fallback failed: {e}")
            return {}


class KGQueryAgent:
    """宝可梦知识图谱查询代理"""

    def __init__(self, llm=None):
        """
        初始化查询代理
        :param llm: 可选的语言模型实例，默认使用ChatOpenAI
        """
        # Neo4j: docker-compose.yml uses `NEO4J_AUTH=none` by default, so auth may be empty.
        if settings.database.neo4j_password:
            self.graph = Graph(settings.database.neo4j_uri, auth=settings.database.neo4j_auth)
        else:
            self.graph = Graph(settings.database.neo4j_uri)
        self.llm = llm or self._default_llm()
        self.ner = EntityRecognizer()
        self.subgraph_extractor = GraphSubgraphExtractor(self.graph)
        self.tools = self._init_tools()
        self.agent = self._create_agent()

    def _default_llm(self):
        """默认语言模型配置"""
        return ChatOpenAI(
            model=settings.llm.model_name, base_url=settings.llm.api_base, api_key=settings.llm.api_key, temperature=0
        )

    def _create_agent(self):
        """创建React代理"""
        return create_react_agent(
            self.llm,
            tools=self.tools,
            prompt="当用户询问关于宝可梦、人物、城镇、地区、属性的相关信息时，你将使用这些函数来查询neo4j数据库中的数据",
        )

    def query(
        self, question: str, hops: int = 4, stream: bool = False
    ) -> Iterator[dict[str, Any] | Any] | dict[str, Any] | Any:
        """
        执行知识图谱查询
        :param question: 自然语言问题
        :param stream: 是否使用流式输出
        :return: 查询结果字典
        """
        input_message = {"messages": [HumanMessage(content=question)]}

        if stream:
            return self.agent.stream(input_message, stream_mode="updates")
        try:
            llm_ans = self.agent.invoke(input_message)
        except Exception as e:
            _log.error(f"KG agent invoke failed: {e}")
            return {"answer": f"图谱查询失败: {e}", "subgraph": {"nodes": [], "edges": []}}
        # === 提取最终回答文本 ===
        final_answer = ""
        try:
            for msg in llm_ans.get("messages", [])[::-1]:  # 从后往前找
                if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                    final_answer = msg.content.strip()
                    break
        except Exception:
            final_answer = "抱歉，未能从图谱回答中提取结果"

        try:
            ner_result = self.ner.ner(question)
            flat_entities = [ent for ents in ner_result.values() for ent in ents]
            # print(flat_entities)
        except Exception:
            flat_entities = []

        try:
            subgraph_json = (
                self.subgraph_extractor.get_subgraph(flat_entities[0], hops=hops)
                if flat_entities
                else {"nodes": [], "edges": []}
            )
        except Exception as e:
            _log.warning(f"Subgraph extraction failed: {e}")
            subgraph_json = {"nodes": [], "edges": []}
        # print(subgraph_json)
        final_answer = "有图谱中的内容可知：" + final_answer
        return {"answer": final_answer, "subgraph": subgraph_json}

    def _init_tools(self) -> list:
        """初始化语义分组查询工具 (32 → 8 工具)"""

        # ---- Schema 定义 ----
        class PokemonInfoQuery(BaseModel):
            pokemon: str = Field(description="宝可梦名称")
            attributes: list[str] = Field(
                default=["all"],
                description="要查询的属性列表，可选: name, english_name, ability, hidden_ability, "
                "height, weight, evolution_level, attr_ability, types, evolution, 或 'all'",
            )

        class PersonInfoQuery(BaseModel):
            person: str = Field(description="人物名称")
            attributes: list[str] = Field(
                default=["all"],
                description="可选: gender, english_name, japanese_name, hometown, town, 或 'all'",
            )

        class PersonRelationQuery(BaseModel):
            person: str = Field(description="人物名称")
            relation: str = Field(
                description="关系类型: challengers, partners, enemies, relatives, pokemons",
            )

        class LocationQuery(BaseModel):
            name: str = Field(description="城镇或地区名称")
            query_type: str = Field(
                description="查询类型: town_region, region_towns, region_people, town_people, town_pokemons",
            )

        class CrossEntityQuery(BaseModel):
            entity: str = Field(description="实体名称 (宝可梦/人物)")
            query_type: str = Field(
                description="查询类型: pokemon_owners, pokemon_towns, person_pokemons",
            )

        class CountQuery(BaseModel):
            entity: str = Field(description="实体名称")
            count_type: str = Field(
                description="统计类型: region_towns, town_pokemons, person_pokemons, pokemon_types",
            )

        class SubgraphQuery(BaseModel):
            entity: str = Field(description="实体名称")
            hops: int = Field(2, description="跳数 1~4")

        class Entity(BaseModel):
            question: str

        # ---- 内部查询辅助 ----
        def execute_query(sql: str, result_key: str, not_found_msg: str):
            try:
                result = self.graph.run(sql).data()
                if result:
                    val = result[0].get(result_key)
                    return {result_key: val}
                return {"message": not_found_msg}
            except Exception as e:
                return {"error": f"查询失败: {str(e)}", "sql": sql}

        # ---- 工具实现 ----

        @tool(args_schema=PokemonInfoQuery)
        def query_pokemon(pokemon: str, attributes: list[str] | None = None) -> dict:
            """查询宝可梦的各种属性：名字、特性、身高、体重、进化、属性等。一次可查多个属性。"""
            attr_map = {
                "name": ("p.name", "name"),
                "english_name": ("p.english_name", "english_name"),
                "ability": ("p.ability", "ability"),
                "hidden_ability": ("p.hidden_ability", "hidden_ability"),
                "height": ("p.height", "height"),
                "weight": ("p.weight", "weight"),
                "evolution_level": ("p.evolution_level", "evolution_level"),
                "attr_ability": ("p.attr_ability", "attr_ability"),
            }
            if attributes is None or "all" in attributes:
                attributes = list(attr_map.keys()) + ["types", "evolution"]

            result = {}

            # 简单属性：一条 Cypher 查出
            simple = [a for a in attributes if a in attr_map]
            if simple:
                returns = ", ".join(f"{attr_map[a][0]} AS {attr_map[a][1]}" for a in simple)
                sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN {returns};"
                data = self.graph.run(sql).data()
                if data:
                    result.update(data[0])

            # 关系属性：types
            if "types" in attributes:
                sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE p.name = '{pokemon}' RETURN COLLECT(i.name) AS types;"
                data = self.graph.run(sql).data()
                if data:
                    result["types"] = data[0].get("types", [])

            # 关系属性：evolution
            if "evolution" in attributes:
                sql = f"MATCH (p1:Pokémon)-[:evolves_into]->(p2:Pokémon) WHERE p1.name = '{pokemon}' RETURN p2.name AS evolution;"
                data = self.graph.run(sql).data()
                if data:
                    result["evolution"] = data[0].get("evolution")

            return result if result else {"message": f"未找到宝可梦: {pokemon}"}

        @tool(args_schema=PersonInfoQuery)
        def query_person(person: str, attributes: list[str] | None = None) -> dict:
            """查询人物基本信息：性别、英文名、日文名、家乡等。"""
            attr_map = {
                "gender": ("per.gender", "gender"),
                "english_name": ("per.english_name", "english_name"),
                "japanese_name": ("per.japanese_name", "japanese_name"),
            }
            if attributes is None or "all" in attributes:
                attributes = list(attr_map.keys()) + ["hometown", "town"]

            result = {}
            simple = [a for a in attributes if a in attr_map]
            if simple:
                returns = ", ".join(f"{attr_map[a][0]} AS {attr_map[a][1]}" for a in simple)
                sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN {returns};"
                data = self.graph.run(sql).data()
                if data:
                    result.update(data[0])

            if "hometown" in attributes:
                r = execute_query(
                    f"MATCH (per:Person)-[:come_from]->(r:Region) WHERE per.name = '{person}' RETURN r.name AS region;",
                    "region",
                    "",
                )
                if "region" in r:
                    result["hometown_region"] = r["region"]

            if "town" in attributes:
                r = execute_query(
                    f"MATCH (per:Person)-[:come_from]->(t:Town) WHERE per.name = '{person}' RETURN t.name AS town;",
                    "town",
                    "",
                )
                if "town" in r:
                    result["hometown_town"] = r["town"]

            return result if result else {"message": f"未找到人物: {person}"}

        @tool(args_schema=PersonRelationQuery)
        def query_person_relations(person: str, relation: str) -> dict:
            """查询人物关系：挑战者、伙伴、敌人、亲戚、拥有的宝可梦。"""
            rel_map = {
                "challengers": ("[:challenge]", "Person", "challengers"),
                "partners": ("[:partner]", "Person", "partners"),
                "enemies": ("[:hostility]", "Person", "enemies"),
                "relatives": ("[:relative]", "Person", "relatives"),
                "pokemons": ("[:has_pokemon]", "Pokémon", "pokemons"),
            }
            if relation not in rel_map:
                return {"error": f"不支持的关系类型: {relation}，可选: {list(rel_map.keys())}"}

            edge, target_label, key = rel_map[relation]
            sql = f"MATCH (per:Person)-{edge}->(t:{target_label}) WHERE per.name = '{person}' RETURN COLLECT(t.name) AS {key};"
            return execute_query(sql, key, f"未找到 {person} 的{relation}")

        @tool(args_schema=LocationQuery)
        def query_location(name: str, query_type: str) -> dict:
            """查询城镇/地区信息：所属地区、地区城镇、地区人物、城镇人物、城镇宝可梦。"""
            queries = {
                "town_region": (
                    f"MATCH (t:Town)-[:located_in]->(r:Region) WHERE t.name = '{name}' RETURN r.name AS region;",
                    "region",
                ),
                "region_towns": (
                    f"MATCH (r:Region)<-[:located_in]-(t:Town) WHERE r.name = '{name}' RETURN COLLECT(t.name) AS towns;",
                    "towns",
                ),
                "region_people": (
                    f"MATCH (per:Person)-[:come_from]->(r:Region) WHERE r.name = '{name}' RETURN COLLECT(per.name) AS people;",
                    "people",
                ),
                "town_people": (
                    f"MATCH (t:Town)-[:has_celebrity]->(per:Person) WHERE t.name = '{name}' RETURN COLLECT(per.name) AS people;",
                    "people",
                ),
                "town_pokemons": (
                    f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE t.name = '{name}' RETURN COLLECT(p.name) AS pokemons;",
                    "pokemons",
                ),
            }
            if query_type not in queries:
                return {"error": f"不支持的查询类型: {query_type}，可选: {list(queries.keys())}"}

            sql, key = queries[query_type]
            return execute_query(sql, key, f"未找到: {name}")

        @tool(args_schema=CrossEntityQuery)
        def query_cross_entity(entity: str, query_type: str) -> dict:
            """查询跨实体关系：宝可梦的拥有者、宝可梦出现的城镇、人物拥有的宝可梦。"""
            queries = {
                "pokemon_owners": (
                    f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE p.name = '{entity}' RETURN COLLECT(per.name) AS owners;",
                    "owners",
                ),
                "pokemon_towns": (
                    f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE p.name = '{entity}' RETURN COLLECT(t.name) AS towns;",
                    "towns",
                ),
                "person_pokemons": (
                    f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE per.name = '{entity}' RETURN COLLECT(p.name) AS pokemons;",
                    "pokemons",
                ),
            }
            if query_type not in queries:
                return {"error": f"不支持的查询类型: {query_type}，可选: {list(queries.keys())}"}

            sql, key = queries[query_type]
            return execute_query(sql, key, f"未找到: {entity}")

        @tool(args_schema=CountQuery)
        def count_entities(entity: str, count_type: str) -> dict:
            """统计查询：地区城镇数、城镇宝可梦数、人物宝可梦数、宝可梦属性数。"""
            queries = {
                "region_towns": f"MATCH (r:Region)<-[:located_in]-(t:Town) WHERE r.name = '{entity}' RETURN COUNT(t) AS count;",
                "town_pokemons": f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE t.name = '{entity}' RETURN COUNT(p) AS count;",
                "person_pokemons": f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE per.name = '{entity}' RETURN COUNT(p) AS count;",
                "pokemon_types": f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE p.name = '{entity}' RETURN COUNT(i) AS count;",
            }
            if count_type not in queries:
                return {"error": f"不支持的统计类型: {count_type}，可选: {list(queries.keys())}"}

            return execute_query(queries[count_type], "count", f"未找到: {entity}")

        @tool(args_schema=SubgraphQuery)
        def get_entity_subgraph(entity: str, hops: int = 2):
            """返回实体的 n 跳子图 JSON（{nodes, edges}）"""
            data = self.subgraph_extractor.get_subgraph(entity, hops=hops)
            if not data:
                return {"message": f"未找到实体 {entity} 的子图"}
            return data

        @tool(args_schema=Entity)
        def get_entity(question: str):
            """对用户输入进行实体匹配，后续查询参数需在返回实体中选择"""
            return self.ner.ner(question)

        # 返回 8 个语义分组工具 (原 32 个)
        return [
            query_pokemon,  # 替代 10 个独立工具
            query_person,  # 替代 3 个独立工具
            query_person_relations,  # 替代 4 个独立工具
            query_location,  # 替代 5 个独立工具
            query_cross_entity,  # 替代 3 个独立工具
            count_entities,  # 替代 4 个独立工具
            get_entity_subgraph,  # 保持不变
            get_entity,  # 保持不变
        ]

    def _execute_query(self, sql: str, result_key: str, not_found_msg: str) -> dict:
        """执行Neo4j查询"""
        try:
            result = self.graph.run(sql).data()
            if result:
                return {result_key: result[0][result_key]}
            return {"message": not_found_msg}
        except Exception as e:
            return {"error": f"查询失败: {str(e)}"}


# 使用示例
if __name__ == "__main__":
    agent = KGQueryAgent()
    # 默认 2 跳
    res = agent.query("皮卡丘在哪里能抓到？")
    print(res)

    # 指定 4 跳
    # res = agent.query("火恐龙相关人物有哪些？", hops=4)
