"""
Pokemon-Chat 统一配置

使用 Pydantic v2 BaseSettings 管理所有配置
替代旧的 configs/settings.py
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Tuple
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录 (Path 对象)
_BASE_DIR_PATH = Path(__file__).parent.parent.parent.resolve()


class PathSettings(BaseSettings):
    """路径配置"""
    model_config = SettingsConfigDict(extra="ignore")
    
    @computed_field
    @property
    def base_dir(self) -> Path:
        return _BASE_DIR_PATH
    
    @computed_field
    @property
    def model_reranker_path(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "models" / "bge-reranker-v2-m3"
    
    @computed_field
    @property
    def model_roberta_path(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "models" / "chinese-roberta-wwm-ext"
    
    @computed_field
    @property
    def model_embedding_path(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "models" / "bge-large-zh-v1.5"
    
    @computed_field
    @property
    def model_ocr_path(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "models" / "ocr"
    
    @computed_field
    @property
    def cache_berta_model(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "cache" / "roberta" / "best_roberta.pt"
    
    @computed_field
    @property
    def ner_tag_path(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "ner_data" / "tag2idx.npy"
    
    @computed_field
    @property
    def log_dir(self) -> Path:
        return _BASE_DIR_PATH / "logs"
    
    @computed_field
    @property
    def save_yaml_path(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "save"
    
    # Data paths
    @computed_field
    @property
    def json_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "json_data"
    
    @computed_field
    @property
    def entity_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "entity_data"
    
    @computed_field
    @property
    def ner_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "ner_data"
    
    @computed_field
    @property
    def raw_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "raw_data"
    
    @computed_field
    @property
    def relations_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "relations_data"
    
    @computed_field
    @property
    def graphrag_raw_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data" / "graph_data" / "精灵之沙暴天王.txt"
    
    @computed_field
    @property
    def artifacts_data(self) -> Path:
        return _BASE_DIR_PATH / "rag" / "artifacts"
    
    @computed_field
    @property
    def data_parser_data(self) -> Path:
        return _BASE_DIR_PATH / "resources" / "data_parser"


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    
    @property
    def neo4j_auth(self) -> Tuple[str, str]:
        return (self.neo4j_username, self.neo4j_password)
    
    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "langgraph"
    
    # Milvus
    milvus_uri: str = "http://localhost:19530"


class TavilySettings(BaseSettings):
    """Tavily 搜索配置"""
    model_config = SettingsConfigDict(env_prefix="tavily_", extra="ignore")
    
    api_key: str = ""


class EmbeddingSettings(BaseSettings):
    """Embedding 配置"""
    model_config = SettingsConfigDict(env_prefix="embedding_", extra="ignore")
    
    api_key: str = ""
    api_base: str = "https://api.siliconflow.cn/v1/embeddings"
    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024


class RerankerSettings(BaseSettings):
    """Reranker 配置"""
    model_config = SettingsConfigDict(env_prefix="reranker_", extra="ignore")
    
    enabled: bool = True
    api_key: str = ""
    api_base: str = "https://api.siliconflow.cn/v1"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 10
    threshold: float = 0.1


class LLMSettings(BaseSettings):
    """LLM 配置"""
    model_config = SettingsConfigDict(env_prefix="llm_", extra="ignore")
    
    api_key: str = ""
    api_base: str = "https://api.siliconflow.cn/v1"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 4096


class FeatureSettings(BaseSettings):
    """功能开关"""
    model_config = SettingsConfigDict(extra="ignore")
    
    enable_knowledge_base: bool = Field(default=False, alias="ENABLE_KNOWLEDGE_BASE")
    enable_knowledge_graph: bool = Field(default=False, alias="ENABLE_KNOWLEDGE_GRAPH")
    enable_web_search: bool = Field(default=False, alias="ENABLE_WEB_SEARCH")
    enable_mcp: bool = Field(default=False, alias="ENABLE_MCP")
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")


class KnowledgeBaseConfig(BaseSettings):
    """知识库配置（兼容旧 CONFIG 字典）"""
    model_config = SettingsConfigDict(extra="ignore")
    
    milvus_uri: str = "http://localhost:19530"
    default_distance_threshold: float = 0.5
    default_rerank_threshold: float = 0.1
    default_max_query_count: int = 20
    default_top_k: int = 10
    embed_model: str = "siliconflow/BAAI/bge-m3"
    reranker_key: str = "siliconflow/bge-reranker-v2-m3"


class Settings(BaseSettings):
    """主配置类"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # 子配置
    paths: PathSettings = Field(default_factory=PathSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    kb_config: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)



@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()


# 全局配置实例
settings = get_settings()


# ============ 兼容旧代码的导出 ============
# 路径 (str)
BASE_DIR = str(settings.paths.base_dir)
MODEL_RERANKER_PATH = str(settings.paths.model_reranker_path)
MODEL_ROBERTA_PATH = str(settings.paths.model_roberta_path)
MODEL_EMBEDDING_PATH = str(settings.paths.model_embedding_path)
MODEL_OCR_PATH = str(settings.paths.model_ocr_path)
CACHE_BERTA_MODEL = str(settings.paths.cache_berta_model)
NER_TAG_PATH = str(settings.paths.ner_tag_path)
LOG_DIR = str(settings.paths.log_dir)
SAVE_YAML_PATH = str(settings.paths.save_yaml_path)
JSON_DATA = str(settings.paths.json_data)
ENTITY_DATA = str(settings.paths.entity_data)
NER_DATA = str(settings.paths.ner_data)
RAW_DATA = str(settings.paths.raw_data)
RELATIONS_DATA = str(settings.paths.relations_data)
GRAPHRAG_RAW_DATA = str(settings.paths.graphrag_raw_data)
ARTIFACTS_DATA = str(settings.paths.artifacts_data)
DATA_PARSER_DATA = str(settings.paths.data_parser_data)

# API 配置
MODEL_API_KEY = settings.llm.model_api_key or settings.api_keys.siliconflow_api_key
MODEL_API_BASE = settings.llm.model_api_base
MODEL_NAME = settings.llm.model_name
TAVILY_API_KEY = settings.api_keys.tavily_api_key

# Embedding 配置
EMBEDDING_MODEL = settings.embedding.model
EMBEDDING_MODEL_DIM = settings.embedding.dimension

# 数据库配置
NEO4J_URI = settings.database.neo4j_uri
NEO4J_AUTH = settings.database.neo4j_auth

# 知识库配置（兼容旧 CONFIG 字典）
CONFIG = {
    "milvus_uri": settings.kb_config.milvus_uri,
    "default_distance_threshold": settings.kb_config.default_distance_threshold,
    "default_rerank_threshold": settings.kb_config.default_rerank_threshold,
    "default_max_query_count": settings.kb_config.default_max_query_count,
    "default_top_k": settings.kb_config.default_top_k,
    "enable_knowledge_base": settings.features.enable_knowledge_base,
    "embed_model": settings.kb_config.embed_model,
    "reranker_key": settings.kb_config.reranker_key,
    "model_name": settings.reranker.model,
    "enable_reranker": settings.features.enable_reranker,
    "MODEL_RERANKER_PATH": MODEL_RERANKER_PATH,
}

# Embedding 模型信息
EMBED_MODEL_INFO = {
    "siliconflow/BAAI/bge-m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "url": settings.embedding.siliconflow_url,
        "api_key": "SILICONFLOW_API_KEY",
    },
    "openai/bge-m3-pro": {
        "name": "bge-m3-pro",
        "dimension": 1024,
    },
    "ollama/bge-m3:latest": {
        "name": "bge-m3:latest",
        "dimension": 1024,
        "url": settings.embedding.ollama_url,
    },
    "dashscope/text-embedding-v3": {
        "name": "text-embedding-v3",
        "dimension": 1024,
        "url": settings.embedding.dashscope_url,
    },
}


if __name__ == "__main__":
    print("=== Settings Test ===")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Reranker Provider: {settings.reranker.provider}")
    print(f"Reranker Model: {settings.reranker.model}")
    print(f"Embedding Provider: {settings.embedding.provider}")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"LOG_DIR: {LOG_DIR}")
