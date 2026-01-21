"""
GraphRAG Module

基于 LightRAG 实现的知识图谱 RAG 系统。
"""
from .lightrag_wrapper import (
    PokemonLightRAG,
    get_lightrag_instance,
)

__all__ = [
    "PokemonLightRAG",
    "get_lightrag_instance",
]
