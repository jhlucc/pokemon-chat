import pickle
import warnings
from pathlib import Path
from typing import cast, Dict, Any

import pandas as pd
from langchain_graphrag.indexing import IndexerArtifacts
from langchain_graphrag.query.global_search import GlobalSearch
from langchain_graphrag.query.global_search.community_weight_calculator import CommunityWeightCalculator
from langchain_graphrag.query.global_search.key_points_aggregator import (
    KeyPointsAggregator, KeyPointsAggregatorPromptBuilder, KeyPointsContextBuilder
)
from langchain_graphrag.query.global_search.key_points_generator import (
    CommunityReportContextBuilder, KeyPointsGenerator, KeyPointsGeneratorPromptBuilder
)
from langchain_graphrag.types.graphs.community import CommunityLevel
from langchain_graphrag.utils import TiktokenCounter
from langchain_openai import ChatOpenAI
from configs import *

warnings.filterwarnings("ignore")


class GraphRAG:
    def __init__(
            self,
            artifacts_path: str = ARTIFACTS_DATA,
            openai_base_url: str = MODEL_API_BASE,
            openai_api_key: str = MODEL_API_KEY,
            model_name: str = MODEL_NAME,
            community_level: int = 0
    ):
        """
        初始化GraphRAG系统
        
        参数:
            artifacts_path: 知识图谱数据路径

            community_level: 社区级别阈值
        """
        self.artifacts_path = artifacts_path
        self.community_level = community_level
        self.artifacts = None
        self.global_search = None
        self.local_search = None
        self.llm = None
        self.model_name = model_name
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key

        self._initialize()

    def _initialize(self):
        """初始化系统组件"""
        # 1. 加载知识图谱数据
        self.artifacts = self._load_artifacts(self.artifacts_path)

        # 2. 初始化LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.openai_base_url,
            api_key=self.openai_api_key
        )

        # 3. 初始化全局搜索
        self._init_global_search()

        # 4. 初始化本地搜索（可选）
        # self._init_local_search()

    def _load_artifacts(self, path: str) -> IndexerArtifacts:
        """加载知识图谱数据"""
        path = Path(path)
        entities = pd.read_parquet(path / "entities.parquet")
        relationships = pd.read_parquet(path / "relationships.parquet")
        text_units = pd.read_parquet(path / "text_units.parquet")
        communities_reports = pd.read_parquet(path / "communities_reports.parquet")

        # 加载pickle文件
        def load_pickle(file_path):
            if file_path.exists():
                with file_path.open("rb") as fp:
                    return pickle.load(fp)
            return None

        merged_graph = load_pickle(path / "merged-graph.pickle")
        summarized_graph = load_pickle(path / "summarized-graph.pickle")
        communities = load_pickle(path / "community_info.pickle")

        return IndexerArtifacts(
            entities=entities,
            relationships=relationships,
            text_units=text_units,
            communities_reports=communities_reports,
            merged_graph=merged_graph,
            summarized_graph=summarized_graph,
            communities=communities,
        )

    def _init_global_search(self):
        """初始化全局搜索组件"""
        # 社区报告上下文构建器
        report_context_builder = CommunityReportContextBuilder(
            community_level=cast(CommunityLevel, self.community_level),
            weight_calculator=CommunityWeightCalculator(),
            artifacts=self.artifacts,
            token_counter=TiktokenCounter(),
        )

        # 关键点生成器
        kp_generator = KeyPointsGenerator(
            llm=self.llm,
            prompt_builder=KeyPointsGeneratorPromptBuilder(
                show_references=False,
                repeat_instructions=True
            ),
            context_builder=report_context_builder,
        )

        # 关键点聚合器
        kp_aggregator = KeyPointsAggregator(
            llm=self.llm,
            prompt_builder=KeyPointsAggregatorPromptBuilder(
                show_references=False,
                repeat_instructions=True,
            ),
            context_builder=KeyPointsContextBuilder(
                token_counter=TiktokenCounter(),
            ),
            output_raw=True,
        )

        # 全局搜索实例
        self.global_search = GlobalSearch(
            kp_generator=kp_generator,
            kp_aggregator=kp_aggregator,
            generation_chain_config={"tags": ["kp-generation"]},
            aggregation_chain_config={"tags": ["kp-aggregation"]},
        )

    def query(self, question: str) -> str:
        """
        执行查询
        参数:
            question: 用户提问
        返回:
            回答结果
        """
        if not self.global_search:
            raise ValueError("Global search not initialized")

        return self.global_search.invoke(question)


# 使用示例
if __name__ == "__main__":
    # 初始化GraphRAG系统
    graph_rag = GraphRAG(
    )
    # 执行查询
    question = "介绍一下恭平是谁？"
    response = graph_rag.query(question)
    print(response.content)
