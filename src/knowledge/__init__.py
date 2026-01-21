"""
Knowledge Module

知识库相关功能模块。
"""
from .graphrag import PokemonLightRAG, get_lightrag_instance

# Alias for backward compatibility
GraphRAG = PokemonLightRAG

__all__ = [
    "PokemonLightRAG",
    "get_lightrag_instance",
    "GraphRAG",
]
