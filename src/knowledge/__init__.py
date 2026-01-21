"""
Knowledge Module - 知识库相关功能

提供 LightRAG 知识图谱检索能力。
"""
from .graphrag import PokemonLightRAG, get_lightrag_instance

__all__ = [
    "PokemonLightRAG",
    "get_lightrag_instance",
]
