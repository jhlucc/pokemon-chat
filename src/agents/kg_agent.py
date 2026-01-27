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
        """初始化所有查询工具"""

        # 定义查询模型
        class PokemonQuery(BaseModel):
            pokemon: str

        class PersonQuery(BaseModel):
            person: str

        class TownQuery(BaseModel):
            town: str

        class RegionQuery(BaseModel):
            region: str

        class IdentityQuery(BaseModel):
            identity: str

        class Entity(BaseModel):
            question: str

        class SubgraphQuery(BaseModel):
            entity: str = Field(..., description="实体名称")
            hops: int = Field(2, description="跳数 1~4 等")

        @tool(args_schema=SubgraphQuery)
        def get_entity_subgraph(entity: str, hops: int = 2):
            """返回实体的 n 跳子图 JSON（{nodes, edges}）"""
            data = self.subgraph_extractor.get_subgraph(entity, hops=hops)
            if not data:
                return {"message": f"未找到实体 {entity} 的子图"}
            return data

        # 宝可梦相关查询
        @tool(args_schema=PokemonQuery)
        def get_pokemon_chinese_name(pokemon: str):
            """查询宝可梦的中文名"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.name AS chinese_name;"
            return execute_query(sql, "chinese_name", f"未找到宝可梦: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_english_name(pokemon: str):
            """查询宝可梦的英文名"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.english_name AS english_name;"
            return execute_query(sql, "english_name", f"未找到宝可梦的英文名: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_ability(pokemon: str):
            """查询宝可梦的特性"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.ability AS ability;"
            return execute_query(sql, "ability", f"未找到宝可梦特性: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_hidden_ability(pokemon: str):
            """查询宝可梦的隐藏特性"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.hidden_ability AS hidden_ability;"
            return execute_query(sql, "hidden_ability", f"未找到宝可梦隐藏特性: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_height(pokemon: str):
            """查询宝可梦的身高"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.height AS height;"
            return execute_query(sql, "height", f"未找到宝可梦身高: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_weight(pokemon: str):
            """查询宝可梦的体重"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.weight AS weight;"
            return execute_query(sql, "weight", f"未找到宝可梦体重: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_evolution_level(pokemon: str):
            """查询宝可梦的进化等级"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.evolution_level AS evolution_level;"
            return execute_query(sql, "evolution_level", f"未找到宝可梦进化等级: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_attributes(pokemon: str):
            """查询宝可梦的属性抗性"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.attr_ability AS attr_ability;"
            return execute_query(sql, "attr_ability", f"未找到宝可梦属性抗性: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_evolution(pokemon: str):
            """查询宝可梦的进化形态"""
            sql = f"MATCH (p1:Pokémon)-[:evolves_into]->(p2:Pokémon) WHERE p1.name = '{pokemon}' RETURN p2.name AS evolution_name;"
            return execute_query(sql, "evolution_name", f"未找到宝可梦进化形态: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_types(pokemon: str):
            """查询宝可梦的属性"""
            sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE p.name = '{pokemon}' RETURN COLLECT(i.name) AS types;"
            return execute_query(sql, "types", f"未找到宝可梦属性: {pokemon}")

        # 人物相关查询
        @tool(args_schema=PersonQuery)
        def get_person_gender(person: str):
            """查询人物性别"""
            sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN per.gender AS gender;"
            return execute_query(sql, "gender", f"未找到人物性别: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_english_name(person: str):
            """查询人物英文名"""
            sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN per.english_name AS english_name;"
            return execute_query(sql, "english_name", f"未找到人物英文名: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_japanese_name(person: str):
            """查询人物日本名"""
            sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN per.japanese_name AS japanese_name;"
            return execute_query(sql, "japanese_name", f"未找到人物日本名: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_challengers(person: str):
            """查询人物的挑战者"""
            sql = f"MATCH (per1:Person)-[:challenge]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS challengers;"
            return execute_query(sql, "challengers", f"未找到人物挑战者: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_partners(person: str):
            """查询人物的伙伴"""
            sql = f"MATCH (per1:Person)-[:partner]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS partners;"
            return execute_query(sql, "partners", f"未找到人物伙伴: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_enemies(person: str):
            """查询人物的敌对者"""
            sql = f"MATCH (per1:Person)-[:hostility]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS enemies;"
            return execute_query(sql, "enemies", f"未找到人物敌对者: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_relatives(person: str):
            """查询人物的亲戚"""
            sql = f"MATCH (per1:Person)-[:relative]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS relatives;"
            return execute_query(sql, "relatives", f"未找到人物亲戚: {person}")

        # 地区、城镇相关查询
        @tool(args_schema=TownQuery)
        def get_town_region(town: str):
            """查询城镇所在的地区"""
            sql = f"MATCH (t:Town)-[:located_in]->(r:Region) WHERE t.name = '{town}' RETURN r.name AS region;"
            return execute_query(sql, "region", f"未找到城镇所在地区: {town}")

        @tool(args_schema=RegionQuery)
        def get_region_towns(region: str):
            """查询地区的城镇"""
            sql = f"MATCH (r:Region)<-[:located_in]-(t:Town) WHERE r.name = '{region}' RETURN COLLECT(t.name) AS towns;"
            return execute_query(sql, "towns", f"未找到地区城镇: {region}")

        @tool(args_schema=PersonQuery)
        def get_person_hometown(person: str):
            """查询人物来自哪个地区"""
            sql = f"MATCH (per:Person)-[:come_from]->(r:Region) WHERE per.name = '{person}' RETURN r.name AS region;"
            return execute_query(sql, "region", f"未找到人物家乡: {person}")

        @tool(args_schema=RegionQuery)
        def get_region_people(region: str):
            """查询地区有哪些人物"""
            sql = f"MATCH (per:Person)-[:come_from]->(r:Region) WHERE r.name = '{region}' RETURN COLLECT(per.name) AS people;"
            return execute_query(sql, "people", f"未找到地区人物: {region}")

        # 人物与宝可梦关系查询
        @tool(args_schema=PersonQuery)
        def get_person_pokemons(person: str):
            """查询人物拥有哪些宝可梦"""
            sql = f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE per.name = '{person}' RETURN COLLECT(p.name) AS pokemons;"
            return execute_query(sql, "pokemons", f"未找到人物拥有的宝可梦: {person}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_owners(pokemon: str):
            """查询拥有某个宝可梦的人物"""
            sql = f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE p.name = '{pokemon}' RETURN COLLECT(per.name) AS owners;"
            return execute_query(sql, "owners", f"未找到宝可梦的拥有者: {pokemon}")

        # 城镇与宝可梦关系查询
        @tool(args_schema=TownQuery)
        def get_town_pokemons(town: str):
            """查询城镇有哪些宝可梦"""
            sql = f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE t.name = '{town}' RETURN COLLECT(p.name) AS pokemons;"
            return execute_query(sql, "pokemons", f"未找到城镇的宝可梦: {town}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_towns(pokemon: str):
            """查询哪些城镇有某个宝可梦"""
            sql = f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE p.name = '{pokemon}' RETURN COLLECT(t.name) AS towns;"
            return execute_query(sql, "towns", f"未找到宝可梦出现的城镇: {pokemon}")

        # 城镇与人物关系查询
        @tool(args_schema=TownQuery)
        def get_town_people(town: str):
            """查询城镇有哪些人物"""
            sql = f"MATCH (t:Town)-[:has_celebrity]->(per:Person) WHERE t.name = '{town}' RETURN COLLECT(per.name) AS people;"
            return execute_query(sql, "people", f"未找到城镇人物: {town}")

        @tool(args_schema=PersonQuery)
        def get_person_town(person: str):
            """查询人物来自哪个城镇"""
            sql = f"MATCH (per:Person)-[:come_from]->(t:Town) WHERE per.name = '{person}' RETURN t.name AS town;"
            return execute_query(sql, "town", f"未找到人物所在城镇: {person}")

        # 统计查询
        @tool(args_schema=RegionQuery)
        def count_region_towns(region: str):
            """查询某个地区有多少城镇"""
            sql = f"MATCH (r:Region)<-[:located_in]-(t:Town) WHERE r.name = '{region}' RETURN COUNT(t) AS count;"
            return execute_query(sql, "count", f"未找到地区城镇数量: {region}")

        @tool(args_schema=TownQuery)
        def count_town_pokemons(town: str):
            """查询某个城镇有多少宝可梦"""
            sql = f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE t.name = '{town}' RETURN COUNT(p) AS count;"
            return execute_query(sql, "count", f"未找到城镇宝可梦数量: {town}")

        @tool(args_schema=PersonQuery)
        def count_person_pokemons(person: str):
            """查询人物拥有多少宝可梦"""
            sql = (
                f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE per.name = '{person}' RETURN COUNT(p) AS count;"
            )
            return execute_query(sql, "count", f"未找到人物宝可梦数量: {person}")

        @tool(args_schema=PokemonQuery)
        def count_pokemon_types(pokemon: str):
            """查询宝可梦有多少种属性"""
            sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE p.name = '{pokemon}' RETURN COUNT(i) AS count;"
            return execute_query(sql, "count", f"未找到宝可梦属性数量: {pokemon}")

        # 属性相关查询
        @tool(args_schema=IdentityQuery)
        def get_pokemons_by_type(identity: str):
            """查询某个属性的宝可梦有哪些"""
            sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE i.name = '{identity}' RETURN COLLECT(p.name) AS pokemons;"
            return execute_query(sql, "pokemons", f"未找到该属性的宝可梦: {identity}")

        # 实体匹配
        @tool(args_schema=Entity)
        def get_entity(question: str):
            """你必须调用这个工具，且只调用一次，对于用户的输入进行实体匹配，且后续查询的参数需在返回的实体中选择"""
            return self.ner.ner(question)

        def execute_query(sql: str, result_key: str, not_found_msg: str):
            """执行Neo4j查询并返回格式化结果"""
            try:
                result = self.graph.run(sql).data()
                if result:
                    # 处理单条结果和列表结果
                    if isinstance(result[0].get(result_key), list):
                        return {result_key: result[0][result_key]}
                    return {result_key: result[0][result_key]}
                return {"message": not_found_msg}
            except Exception as e:
                return {"error": f"查询失败: {str(e)}", "sql": sql}

        return [get_entity_subgraph] + [
            get_pokemon_chinese_name,
            get_pokemon_english_name,
            get_pokemon_ability,
            get_pokemon_hidden_ability,
            get_pokemon_height,
            get_pokemon_weight,
            get_pokemon_evolution_level,
            get_pokemon_attributes,
            get_pokemon_evolution,
            get_pokemon_types,
            get_person_gender,
            get_person_english_name,
            get_person_japanese_name,
            get_person_challengers,
            get_person_partners,
            get_person_enemies,
            get_person_relatives,
            get_town_region,
            get_region_towns,
            get_person_hometown,
            get_region_people,
            get_person_pokemons,
            get_pokemon_owners,
            get_town_pokemons,
            get_pokemon_towns,
            get_town_people,
            get_person_town,
            count_region_towns,
            count_town_pokemons,
            count_person_pokemons,
            count_pokemon_types,
            get_pokemons_by_type,
            get_entity,
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
