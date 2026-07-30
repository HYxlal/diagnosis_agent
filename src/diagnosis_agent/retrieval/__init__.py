"""检索模块

对外暴露：
- HybridRetriever：两段式检索（Neo4j 召回 + Embedding 精排 + Chroma 兜底）
- Neo4jFaultRetriever：Neo4j 结构化召回
- SemanticReranker：Embedding 精排
- ChromaVectorRetriever / document_to_record / create_chroma_retriever：Chroma 语义检索
- FaultRetriever：检索器抽象基类
- FieldMapper：字段映射层
- cypher_builder：Cypher 构建器（本项目自实现，脱离 fault_knowledge_graph）
"""

from .base import FaultRetriever
from .cypher_builder import QueryCondition, build_query, build_count_query
from .field_mapper import FieldMapper
from .hybrid_retriever import HybridRetriever, create_hybrid_retriever
from .neo4j_retriever import Neo4jFaultRetriever
from .reranker import SemanticReranker
from .langchain_retrievers import (
    ChromaVectorRetriever,
    Neo4jGraphRetriever,
    document_to_record,
    create_chroma_retriever,
    create_neo4j_retriever,
)

__all__ = [
    "FaultRetriever",
    "FieldMapper",
    "QueryCondition",
    "build_query",
    "build_count_query",
    "HybridRetriever",
    "create_hybrid_retriever",
    "Neo4jFaultRetriever",
    "SemanticReranker",
    "ChromaVectorRetriever",
    "Neo4jGraphRetriever",
    "document_to_record",
    "create_chroma_retriever",
    "create_neo4j_retriever",
]
