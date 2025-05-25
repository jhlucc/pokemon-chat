import pickle
import re
import warnings
from copy import deepcopy
from pathlib import Path

import networkx as nx
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma as ChromaVectorStore
from langchain_graphrag.indexing import TextUnitExtractor
from langchain_graphrag.indexing.artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from langchain_graphrag.indexing.graph_clustering.leiden_community_detector import HierarchicalLeidenCommunityDetector
from langchain_graphrag.indexing.graph_generation import EntityRelationshipExtractor, GraphsMerger, \
    EntityRelationshipDescriptionSummarizer, GraphGenerator
from langchain_graphrag.indexing.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter,
)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from configs import *

warnings.filterwarnings("ignore")

# 读取pdf到一个Document对象中
# def pdf_to_doc(pdf_path):
#     pdf = PdfReader(pdf_path)
#     text = ""
#     for page in pdf.pages:
#         text += page.extract_text()

#     doc = Document(page_content=text)
#     return doc

doc1 = TextLoader(GRAPHRAG_RAW_DATA,
                  encoding="gb18030").load()
# doc2 = TextLoader(r"F:\bigmodel\meet-Pok-mon\4.KGqa\Pokemon-KGQA\RAG\data\精灵之黑暗崛起.txt", encoding="utf-8").load()
docs = []
docs.append(doc1[0])
# docs.append(doc2[0])

# 递归地将文本分割成块
spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# 将分割后的文本块封装为结构化数据（如DataFrame）
text_unit_extractor = TextUnitExtractor(text_splitter=spliter)
textunit_df = text_unit_extractor.run(docs)

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=MODEL_API_BASE,
    api_key=MODEL_API_KEY
)


def sanitize_graph(graph: nx.Graph) -> nx.Graph:
    """
    清理图，删除以下节点及其对应的边：
    1. 包含英文字符的节点。
    2. 包含数字的节点。
    :param graph: 输入的图
    :return: 清理后的图
    """

    # 创建图的副本，避免直接修改原图
    def contains_english(s: str) -> bool:
        """
        判断字符串是否包含英文字符。
        :param s: 输入字符串
        :return: 如果包含英文字符，返回 True；否则返回 False
        """
        return bool(re.search(r'[a-zA-Z]', s))

    def contains_digit(s: str) -> bool:
        """
        判断字符串是否包含数字。
        :param s: 输入字符串
        :return: 如果包含数字，返回 True；否则返回 False
        """
        return bool(re.search(r'\d', s))

    sanitized_graph = graph.copy()

    # 找出包含英文字符或数字的节点
    nodes_to_remove = [
        node for node in sanitized_graph.nodes
        if contains_english(str(node)) or contains_digit(str(node))
    ]

    # 删除这些节点及其对应的边
    sanitized_graph.remove_nodes_from(nodes_to_remove)

    return sanitized_graph


# 生成图谱
graph_generator = GraphGenerator(
    er_extractor=EntityRelationshipExtractor.build_default(llm=llm),
    graphs_merger=GraphsMerger(),  # 每个文本块对应一个子图，所以要合并
    graph_sanitizer=sanitize_graph,
    er_description_summarizer=EntityRelationshipDescriptionSummarizer.build_default(llm=llm),
)


# merged_graph, summarized_graph = graph_generator.run(textunit_df)

def safe_run_graph_generator(textunit_df, cache_dir="graph_progress"):
    """
    支持断点续跑的图谱生成流程（包含sanitize_graph步骤）
    返回: (merged_graph, summarized_graph)
    """
    # 初始化缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # 检查是否有完整缓存
    if (cache_path / "FINAL_DONE.flag").exists():
        with open(cache_path / "step3_merged_clean.pkl", "rb") as f:
            merged_graph = pickle.load(f)
        with open(cache_path / "step4_summarized.pkl", "rb") as f:
            summarized_graph = pickle.load(f)
        return merged_graph, summarized_graph

    # === 第一步：实体关系提取 ===
    if not (cache_path / "step1_er_results.pkl").exists():
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
        def _extract_entities():
            er_extractor = EntityRelationshipExtractor.build_default(llm=llm)
            return er_extractor.invoke(textunit_df)

        er_results = _extract_entities()
        with open(cache_path / "step1_er_results.pkl", "wb") as f:
            pickle.dump(er_results, f)
    else:
        with open(cache_path / "step1_er_results.pkl", "rb") as f:
            er_results = pickle.load(f)

    # === 第二步：合并子图 ===
    if not (cache_path / "step2_merged_raw.pkl").exists():
        merger = GraphsMerger()
        merged_raw = merger(er_results)  # 原始合并图（未清理）
        with open(cache_path / "step2_merged_raw.pkl", "wb") as f:
            pickle.dump(merged_raw, f)
    else:
        with open(cache_path / "step2_merged_raw.pkl", "rb") as f:
            merged_raw = pickle.load(f)

    # === 第三步：清理图谱（关键补充） ===
    if not (cache_path / "step3_merged_clean.pkl").exists():
        merged_graph = sanitize_graph(merged_raw)  # 应用清理函数
        with open(cache_path / "step3_merged_clean.pkl", "wb") as f:
            pickle.dump(merged_graph, f)
    else:
        with open(cache_path / "step3_merged_clean.pkl", "rb") as f:
            merged_graph = pickle.load(f)

    # === 第四步：生成摘要图 ===
    if not (cache_path / "step4_summarized.pkl").exists():
        @retry(stop=stop_after_attempt(3))
        def _summarize():
            summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=llm)
            return summarizer.invoke(deepcopy(merged_graph))

        summarized_graph = _summarize()
        with open(cache_path / "step4_summarized.pkl", "wb") as f:
            pickle.dump(summarized_graph, f)
    else:
        with open(cache_path / "step4_summarized.pkl", "rb") as f:
            summarized_graph = pickle.load(f)

    # 标记完成
    with open(cache_path / "FINAL_DONE.flag", "w") as f:
        f.write("")

    return merged_graph, summarized_graph


merged_graph, summarized_graph = safe_run_graph_generator(textunit_df=textunit_df)
# 社区检测
community_detector = HierarchicalLeidenCommunityDetector(max_cluster_size=10, use_lcc=True)
community_detection_result = community_detector.run(merged_graph)

output_dir = Path()
output_dir.mkdir(parents=True, exist_ok=True)

vector_store_dir = output_dir / "vector_stores"
artifacts_dir = output_dir / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

entities_vector_store = ChromaVectorStore(
    collection_name="entity-embedding",
    persist_directory=str(vector_store_dir),
    embedding_function=OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=MODEL_API_BASE,
        openai_api_key=MODEL_API_KEY,
        chunk_size=32
    ),
)

# 实体报告
entities_artifacts_generator = EntitiesArtifactsGenerator(
    entities_vector_store=entities_vector_store
)
df_entities = entities_artifacts_generator.run(
    community_detection_result,
    summarized_graph,
)

# 关系报告
relationships_artifacts_generator = RelationshipsArtifactsGenerator()
df_relationships = relationships_artifacts_generator.run(summarized_graph)

# 社区报告
report_generator = CommunityReportGenerator.build_default(
    llm=llm,
    chain_config={"tags": ["community-report"]},
)
report_writer = CommunityReportWriter()  # 格式化报告输出

communities_report_artifacts_generator = CommunitiesReportsArtifactsGenerator(
    report_generator=report_generator,
    report_writer=report_writer,
)

df_communities_reports = communities_report_artifacts_generator.run(
    community_detection_result,
    summarized_graph,
)

# 文本片段报告
text_units_artifacts_generator = TextUnitsArtifactsGenerator()
df_text_units = text_units_artifacts_generator.run(textunit_df, df_entities, df_relationships)

# indexer = SimpleIndexer(
#     text_unit_extractor=text_unit_extractor,
#     graph_generator=graph_generator,
#     community_detector=community_detector,
#     entities_artifacts_generator=entities_artifacts_generator,
#     relationships_artifacts_generator=relationships_artifacts_generator,
#     text_units_artifacts_generator=text_units_artifacts_generator,
#     communities_report_artifacts_generator=communities_report_artifacts_generator,
# )

# 保存 merged_graph
with open(artifacts_dir / "merged-graph.pickle", "wb") as f:
    pickle.dump(merged_graph, f)
# 保存 summarized_graph
with open(artifacts_dir / "summarized-graph.pickle", "wb") as f:
    pickle.dump(summarized_graph, f)
# 保存 community
with open(artifacts_dir / "community_info.pickle", "wb") as f:
    pickle.dump(community_detection_result, f)

df_entities.to_parquet(artifacts_dir / "entities.parquet")
df_relationships.to_parquet(artifacts_dir / "relationships.parquet")
df_communities_reports.to_parquet(artifacts_dir / "communities_reports.parquet")
df_text_units.to_parquet(artifacts_dir / "text_units.parquet")
