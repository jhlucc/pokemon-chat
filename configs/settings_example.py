import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# model
MODEL_RERANKER_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'bge-reranker-v2-m3')
MODEL_ROBERTA_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'chinese-roberta-wwm-ext')
MODEL_EMBEDDING_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'bge-large-zh-v1.5')
MODEL_OCR_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'ocr')
CACHE_BERTA_MODEL = os.path.join(BASE_DIR, 'resources', 'cache', 'roberta', 'best_roberta.pt')
EMBEDDING_MODEL = 'bge-m3-pro'
EMBEDDING_MODEL_DIM = 1024
NER_TAG_PATH = os.path.join(BASE_DIR, "resources", "data", "ner_data", "tag2idx.npy")
# api
MODEL_API_KEY = 'sk-r8xrhfzRc3MLUVfdAa80B470321747cA40256D9BEa23Cb'
MODEL_API_BASE = 'http://139.224.116.116:3000/v1'
MODEL_NAME = 'moonshot-v1-32k'
TAVILY_API_KEY = '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'
SerperAPI = 'https://google.serper.dev/search'

# data
JSON_DATA=os.path.join(BASE_DIR, 'resources', 'data','json_data')
ENTITY_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'entity_data')

NER_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'ner_data')

RAW_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'raw_data')

RELATIONS_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'relations_data')

GRAPHRAG_RAW_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'graph_data', '精灵之沙暴天王.txt')

ARTIFACTS_DATA = os.path.join(BASE_DIR, 'rag', 'artifacts')

DATA_PARSER_DATA = os.path.join(BASE_DIR, 'resources', 'data_parser')

LOG_DIR = os.path.join(BASE_DIR, 'logs')

EMBED_MODEL_INFO = {
    "local/bge-large-zh-v1.5": {
        "name": "bge-large-zh-v1.5",
        "dimension": 1024,
        "local_path": "/data/Langagent/resources/models/bge-large-zh-v1.5",
    },
    "ollama/bge-m3:latest": {
        "name": "bge-m3:latest",
        "dimension": 1024,
        "url": "http://localhost:11434/api/embed"
    },

}

# knowledage base
CONFIG = {
    # Milvus
    "milvus_uri": "http://localhost:19530",
    "default_distance_threshold": 0.5,
    "default_rerank_threshold": 0.1,
    "default_max_query_count": 20,
    "default_top_k": 10,
    "enable_knowledge_base": True,
    "embed_model": "openai/bge-m3-pro",
    "reranker_key": "siliconflow/bge-reranker-v2-m3",
    "model_name": "BAAI/bge-reranker-v2-m3",
    'enable_reranker': True,
    'MODEL_RERANKER_PATH': MODEL_RERANKER_PATH

}
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "tczaslw278")  # 替换为实际认证信息


SAVE_YAML_PATH= os.path.join(BASE_DIR, 'resources', 'save')

