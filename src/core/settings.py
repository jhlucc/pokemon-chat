"""
Pokemon-Chat 统一配置
使用 Pydantic v2 BaseSettings 管理所有配置
"""
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Tuple
from pydantic import AliasChoices, Field, computed_field
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
    def resources_dir(self) -> Path:
        return _BASE_DIR_PATH / "resources"

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.resources_dir / "data"

    @computed_field
    @property
    def cache_dir(self) -> Path:
        return self.resources_dir / "cache"

    @computed_field
    @property
    def model_reranker_path(self) -> Path:
        return self.resources_dir / "models" / "bge-reranker-v2-m3"

    @computed_field
    @property
    def model_roberta_path(self) -> Path:
        return self.resources_dir / "models" / "chinese-roberta-wwm-ext"

    @computed_field
    @property
    def model_embedding_path(self) -> Path:
        return self.resources_dir / "models" / "bge-large-zh-v1.5"

    @computed_field
    @property
    def model_ocr_path(self) -> Path:
        return self.resources_dir / "models" / "ocr"

    @computed_field
    @property
    def cache_berta_model(self) -> Path:
        return self.cache_dir / "roberta" / "best_roberta.pt"

    @computed_field
    @property
    def ner_tag_path(self) -> Path:
        return self.data_dir / "ner_data" / "tag2idx.npy"

    @computed_field
    @property
    def log_dir(self) -> Path:
        return _BASE_DIR_PATH / "logs"

    @computed_field
    @property
    def save_yaml_path(self) -> Path:
        return self.resources_dir / "save"

    # Data paths
    @computed_field
    @property
    def json_data(self) -> Path:
        return self.data_dir / "json_data"

    @computed_field
    @property
    def entity_data(self) -> Path:
        return self.data_dir / "entity_data"

    @computed_field
    @property
    def ner_data(self) -> Path:
        return self.data_dir / "ner_data"

    @computed_field
    @property
    def raw_data(self) -> Path:
        return self.data_dir / "raw_data"

    @computed_field
    @property
    def relations_data(self) -> Path:
        return self.data_dir / "relations_data"

    @computed_field
    @property
    def graphrag_raw_data(self) -> Path:
        return self.data_dir / "graph_data" / "精灵之沙暴天王.txt"

    @computed_field
    @property
    def artifacts_data(self) -> Path:
        return _BASE_DIR_PATH / "src" / "knowledge" / "artifacts"

    @computed_field
    @property
    def data_parser_data(self) -> Path:
        return _BASE_DIR_PATH / "src" / "plugins"


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
    milvus_collection_name: str = "pokemon_knowledge"


class TavilySettings(BaseSettings):
    """Tavily 搜索配置"""
    model_config = SettingsConfigDict(env_prefix="tavily_", extra="ignore")

    api_key: str = ""


class CorsSettings(BaseSettings):
    """CORS 配置"""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Comma-separated list, or "*" (default).
    allow_origins: str = Field(
        default="*",
        validation_alias=AliasChoices("cors_allow_origins", "CORS_ALLOW_ORIGINS"),
    )
    allow_credentials: bool = Field(
        default=True,
        validation_alias=AliasChoices("cors_allow_credentials", "CORS_ALLOW_CREDENTIALS"),
    )


class ToolSettings(BaseSettings):
    """第三方工具配置"""
    # Accept both `tool_openweather_api_key` (recommended) and legacy `OPENWEATHER_API_KEY`.
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # OpenWeather
    openweather_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("tool_openweather_api_key", "OPENWEATHER_API_KEY"),
    )


class EmbeddingSettings(BaseSettings):
    """Embedding 配置"""
    model_config = SettingsConfigDict(env_prefix="embedding_", extra="ignore")

    provider: str = "siliconflow"  # 默认提供商
    api_key: str = ""
    api_base: str = "https://api.siliconflow.cn/v1/embeddings"
    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024


class RerankerSettings(BaseSettings):
    """Reranker 配置"""
    model_config = SettingsConfigDict(env_prefix="reranker_", extra="ignore")

    provider: str = "siliconflow"
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

    # Accept both lower-case (docker/.env, .env.template) and legacy UPPER_SNAKE_CASE keys.
    enable_knowledge_base: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_knowledge_base", "ENABLE_KNOWLEDGE_BASE"),
    )
    enable_knowledge_graph: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_knowledge_graph", "ENABLE_KNOWLEDGE_GRAPH"),
    )
    # Safety: allow executing potentially dangerous graph queries (GraphCypherQAChain).
    # Default is False; enable only in trusted environments.
    allow_dangerous_graph_requests: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow_dangerous_graph_requests", "ALLOW_DANGEROUS_GRAPH_REQUESTS"),
    )
    enable_web_search: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_web_search", "ENABLE_WEB_SEARCH"),
    )
    enable_mcp: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_mcp", "ENABLE_MCP"),
    )
    enable_reranker: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_reranker", "ENABLE_RERANKER"),
    )
    enable_ner_bert: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_ner_bert", "ENABLE_NER_BERT"),
    )
    enable_asr: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_asr", "ENABLE_ASR"),
    )


class AsrSettings(BaseSettings):
    """ASR 配置"""
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    funasr_url: str = Field(
        default="ws://localhost:10095",
        validation_alias=AliasChoices("funasr_url", "FUNASR_URL"),
    )


class AgentSettings(BaseSettings):
    """Agent 高级功能配置"""
    model_config = SettingsConfigDict(extra="ignore")

    # Checkpointer 类型: memory, sqlite
    checkpointer_type: str = Field(
        default="memory",
        validation_alias=AliasChoices("checkpointer_type", "CHECKPOINTER_TYPE"),
    )
    # 对话最大消息数
    conversation_max_messages: int = Field(
        default=50,
        validation_alias=AliasChoices("conversation_max_messages", "CONVERSATION_MAX_MESSAGES"),
    )
    # 时间旅行功能
    enable_time_travel: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_time_travel", "ENABLE_TIME_TRAVEL"),
    )
    # 中断/审批功能
    enable_interrupts: bool = Field(
        default=False,
        validation_alias=AliasChoices("enable_interrupts", "ENABLE_INTERRUPTS"),
    )


class KnowledgeBaseConfig(BaseSettings):
    """知识库配置"""
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
        # Use an absolute path so `.env` is found regardless of current working directory.
        env_file=str(_BASE_DIR_PATH / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
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
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    kb_config: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    asr: AsrSettings = Field(default_factory=AsrSettings)

    def get_api_key(self, provider: str) -> str:
        """根据 provider 获取 API Key"""
        provider = provider.lower()
        key = ""
        if provider == "siliconflow":
            # 优先使用 embedding/reranker/llm 配置中的 key
            key = self.embedding.api_key or self.reranker.api_key or self.llm.api_key
            if not key:
                key = os.getenv("SILICONFLOW_API_KEY", "")
        elif provider == "dashscope":
            key = os.getenv("DASHSCOPE_API_KEY", "")
        elif provider == "jina":
            key = os.getenv("JINA_API_KEY", "")
        elif provider == "cohere":
            key = os.getenv("COHERE_API_KEY", "")
        elif provider == "openai":
            key = self.llm.api_key or os.getenv("OPENAI_API_KEY", "")

        return key



@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例"""
    # Keep tests deterministic/offline-safe: do not implicitly load `.env`.
    if "pytest" in sys.modules:
        return Settings(_env_file=None)
    return Settings()


# 全局配置实例
settings = get_settings()


if __name__ == "__main__":
    print("=== Settings Test ===")
    print(f"Base Dir: {settings.paths.base_dir}")
    print(f"Reranker Model: {settings.reranker.model_name}")
    print(f"Embedding Model: {settings.embedding.model_name}")
    print(f"Neo4j URI: {settings.database.neo4j_uri}")
    print(f"Log Dir: {settings.paths.log_dir}")
    print(f"LLM Model: {settings.llm.model_name}")
