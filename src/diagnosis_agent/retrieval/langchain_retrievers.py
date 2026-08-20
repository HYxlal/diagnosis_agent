"""基于 LangChain BaseRetriever 的 Chroma 向量检索器

统一检索器接口，支持与 LangChain 生态无缝集成。
Neo4j 知识图谱检索已迁移到 neo4j_retriever / hybrid_retriever 模块。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..models.incident import COLUMN_EN_TO_CN, IncidentRecord
from ..config import get_settings
from ..storage.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# 缓存已创建的 ChromaVectorStore 实例，避免重复初始化
_store_cache: dict[str, ChromaVectorStore] = {}


class ChromaVectorRetriever(BaseRetriever):
    """基于 ChromaVectorStore 的向量检索器

    继承 BaseRetriever，获得标准接口和 LangChain 生态支持。
    """

    store: ChromaVectorStore
    top_k: int = 5
    score_threshold: float = 0.6

    def _filter_and_wrap_results(self, results: list) -> list[Document]:
        """统一阈值过滤 + SearchResult → Document 转换逻辑

        Chroma 返回的是 cosine distance（值越小越相似），统一在此处做 <= 阈值判断。
        所有上层调用方无需再写 score 判断，彻底避免方向写反的重复错误。
        """
        docs = []
        for result in results:
            # 阈值判断全局唯一入口：cosine distance <= 阈值 → 通过
            if result.score <= self.score_threshold:
                metadata = result.metadata.copy()
                metadata["id"] = result.id
                metadata["score"] = result.score
                docs.append(Document(
                    page_content=result.content,
                    metadata=metadata,
                ))
        return docs

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """语义检索（同步）"""
        try:
            results = self.store.search(
                query=query,
                top_k=self.top_k,
            )
            filtered = self._filter_and_wrap_results(results)
            logger.info(
                f"语义检索: query='{query[:50]}...', "
                f"返回 {len(filtered)}/{len(results)} 条 (阈值={self.score_threshold})"
            )
            return filtered
        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            return []

    def search_with_filters(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        drive_code: Optional[str] = None,
        dtc_code: Optional[str] = None,
        top_k: int = 5,
    ) -> list[Document]:
        """带过滤条件的语义检索"""
        k = top_k or self.top_k

        filters = {}
        if vehicle_type:
            filters["vehicle_type"] = vehicle_type
        if drive_code:
            filters["drive_code"] = drive_code
        if dtc_code:
            filters["dtc_code"] = dtc_code

        try:
            results = self.store.search(
                query=query,
                top_k=k,
                filters=filters if filters else None,
            )
            return self._filter_and_wrap_results(results)
        except Exception as e:
            logger.error(f"带过滤条件的语义检索失败: {e}")
            return []


def document_to_record(doc: Document) -> IncidentRecord:
    """将 LangChain Document 转换为 IncidentRecord"""
    data = {}
    for en_col, cn_col in COLUMN_EN_TO_CN.items():
        value = doc.metadata.get(en_col, "")
        if value:
            data[en_col] = value
        else:
            value = doc.metadata.get(cn_col, "")
            if value:
                data[en_col] = value

    if not data.get("problem_description"):
        data["problem_description"] = doc.page_content[:500]

    return IncidentRecord(**data)


def create_chroma_retriever() -> ChromaVectorRetriever:
    """创建 Chroma 向量检索器"""
    settings = get_settings()
    persist_dir = settings.vector_store.persist_dir
    collection_name = settings.vector_store.collection_name

    # 缓存键：persist_dir + collection_name，避免不同集合共享同一 store
    cache_key = f"{persist_dir}:{collection_name}"
    if cache_key in _store_cache:
        store = _store_cache[cache_key]
    else:
        from ..utils.llm_factory import create_embedding
        embedding = create_embedding()
        store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding=embedding,
        )
        _store_cache[cache_key] = store

    return ChromaVectorRetriever(
        store=store,
        top_k=settings.retrieval.semantic.top_k,
        score_threshold=settings.retrieval.semantic.score_threshold,
    )


