"""基于 LangChain BaseRetriever 的检索器实现

统一检索器接口，支持与 LangChain 生态无缝集成。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..models.incident import INCIDENT_COLUMNS, COLUMN_EN_TO_CN, IncidentRecord
from ..config import get_settings
from ..storage.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# 缓存已创建的 ChromaVectorStore 实例，避免重复初始化
_store_cache: dict[str, ChromaVectorStore] = {}


class ChromaVectorRetriever(BaseRetriever):
    """基于 ChromaVectorStore 的向量检索器

    继承 BaseRetriever，获得标准接口和 LangChain 生态支持。
    。
    """

    store: ChromaVectorStore
    top_k: int = 5
    score_threshold: float = 0.3

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """语义检索（同步）"""
        try:
            results = self.store.search(
                query=query,
                top_k=self.top_k,
            )

            filtered = []
            for result in results:
                if result.score >= self.score_threshold:
                    metadata = result.metadata.copy()
                    metadata["id"] = result.id
                    metadata["score"] = result.score
                    doc = Document(
                        page_content=result.content,
                        metadata=metadata,
                    )
                    filtered.append(doc)

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

            filtered = []
            for result in results:
                if result.score >= self.score_threshold:
                    metadata = result.metadata.copy()
                    metadata["id"] = result.id
                    metadata["score"] = result.score
                    doc = Document(
                        page_content=result.content,
                        metadata=metadata,
                    )
                    filtered.append(doc)

            return filtered
        except Exception as e:
            logger.error(f"带过滤条件的语义检索失败: {e}")
            return []


class BM25KeywordRetriever(BaseRetriever):
    """基于 BM25 的关键词检索器

    补充语义检索，提高精确匹配能力。
    """

    documents: list[Document]
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """关键词检索（同步）"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba

            if not self.documents:
                return []

            # 使用 jieba 分词
            tokenized_corpus = [
                list(jieba.cut(doc.page_content)) for doc in self.documents
            ]
            tokenized_query = list(jieba.cut(query))

            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)

            # 获取 top_k 结果
            results = []
            for idx in scores.argsort()[::-1][:self.top_k]:
                if scores[idx] > 0:
                    doc = self.documents[idx]
                    doc.metadata["score"] = float(scores[idx])
                    results.append(doc)

            logger.info(
                f"BM25 检索: query='{query[:50]}...', "
                f"返回 {len(results)} 条"
            )

            return results
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []


class Neo4jGraphRetriever(BaseRetriever):
    """基于 Neo4j 知识图谱的检索器

    预留接口，用于后续集成知识图谱加权检索。
    """

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """知识图谱检索（同步）"""
        try:
            logger.info(f"Neo4j知识图谱检索: query='{query[:50]}...'")
            return []
        except Exception as e:
            logger.error(f"Neo4j知识图谱检索失败: {e}")
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

    # 检查缓存
    if persist_dir in _store_cache:
        store = _store_cache[persist_dir]
    else:
        store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name="incidents",
            embedding_model=settings.embedding.model,
            api_key=settings.embedding.api_key or None,
            api_base=settings.embedding.api_base or None,
        )
        _store_cache[persist_dir] = store

    return ChromaVectorRetriever(
        store=store,
        top_k=settings.retrieval.semantic.top_k,
        score_threshold=settings.retrieval.semantic.score_threshold,
    )


def create_bm25_retriever(documents: list[Document]) -> BM25KeywordRetriever:
    """创建 BM25 关键词检索器"""
    return BM25KeywordRetriever(documents=documents)


def create_neo4j_retriever() -> Neo4jGraphRetriever:
    """创建 Neo4j 知识图谱检索器"""
    return Neo4jGraphRetriever()