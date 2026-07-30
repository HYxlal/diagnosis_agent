"""ChromaDB 向量存储实现"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from ..models.incident import INCIDENT_COLUMNS, IncidentRecord
from .vector_store import SearchResult, VectorStoreAdapter

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStoreAdapter):
    """基于 ChromaDB 的向量存储实现

    特性：
    - 持久化到本地磁盘
    - 支持语义检索
    - 支持 metadata 过滤
    - 自动使用 OpenAI Embedding（如配置了 API key）
      否则回退到 ChromaDB 默认 embedding
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "incidents",
        embedding_model: str = "text-embedding-v2",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # 创建持久化目录
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # 初始化客户端
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 配置 embedding
        self._embedding_fn = self._create_embedding_fn(api_key, api_base)

        # 创建或获取集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def embedding_function(self) -> Optional[Any]:
        """返回当前使用的 embedding 函数（供 reranker 等模块复用）"""
        return self._embedding_fn

        logger.info(
            f"ChromaVectorStore 初始化: dir={persist_dir}, "
            f"collection={collection_name}, count={self._collection.count()}"
        )

    def _create_embedding_fn(self, api_key: Optional[str], api_base: Optional[str]):
        """创建 embedding 函数

        优先使用通义千问/OpenAI 兼容 embedding，否则使用 ChromaDB 默认 embedding。
        """
        if api_key:
            try:
                from chromadb.utils.embedding_functions import (
                    OpenAIEmbeddingFunction,
                )
                logger.info(f"使用 embedding 模型: {self.embedding_model}")
                return OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=self.embedding_model,
                    api_base=api_base,
                )
            except Exception as e:
                logger.warning(f"Embedding 初始化失败，使用默认: {e}")

        # ChromaDB 默认 embedding (all-MiniLM-L6-v2)
        logger.info("使用 ChromaDB 默认 embedding (all-MiniLM-L6-v2)")
        return None

    def add_records(self, records: list[IncidentRecord]) -> int:
        """批量添加故障记录"""
        if not records:
            return 0

        ids = []
        documents = []
        metadatas = []

        for record in records:
            # 生成唯一 ID（移除了 incident_id，使用 UUID）
            record_id = f"REC-{uuid.uuid4().hex[:12]}"
            doc_text = record.to_searchable_text()

            meta = record.to_dict()
            # 确保所有 metadata 值都是字符串
            for k, v in meta.items():
                meta[k] = str(v) if v is not None else ""

            ids.append(record_id)
            documents.append(doc_text)
            metadatas.append(meta)

        # ChromaDB 批量添加（embedding API 限制 batch_size <= 25）
        batch_size = 25
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            self._collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            added += len(batch_ids)

        logger.info(f"成功添加 {added} 条记录到 ChromaDB")
        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """语义检索"""
        if self._collection.count() == 0:
            return []

        # 构建 ChromaDB where 条件
        where = None
        if filters:
            where = {k: v for k, v in filters.items() if v is not None}

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count()),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            logger.error(f"ChromaDB 查询失败: {e}")
            return []

        search_results: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return []

        for i in range(len(results["ids"][0])):
            rid = results["ids"][0][i]
            doc = results["documents"][0][i] if results["documents"] else ""
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0.0

            # ChromaDB cosine distance -> similarity
            score = max(0.0, 1.0 - dist)

            # 重建 IncidentRecord
            record = IncidentRecord.from_dict(meta)

            search_results.append(SearchResult(
                id=rid,
                content=doc,
                score=score,
                metadata=meta,
                record=record,
            ))

        return search_results

    def get_by_id(self, record_id: str) -> Optional[IncidentRecord]:
        """根据ID获取记录"""
        try:
            result = self._collection.get(ids=[record_id])
            if result["ids"] and result["metadatas"]:
                meta = result["metadatas"][0]
                return IncidentRecord.from_dict(meta)
        except Exception as e:
            logger.warning(f"获取记录失败 {record_id}: {e}")
        return None

    def filter(
        self,
        filters: dict[str, Any],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """纯 metadata 精确过滤（不走 embedding 路径）

        使用 ChromaDB 原生 collection.get(where=...) 做纯 metadata 过滤，
        不调用 embedding 函数，彻底消除 query 占位符问题。

        Args:
            filters: 元数据过滤条件（key=value 的 AND 组合）
            top_k: 返回结果数

        Returns:
            SearchResult 列表，score 固定为 1.0
        """
        if not filters:
            return []

        # 构建 where 条件，过滤掉 None 值
        where = {k: v for k, v in filters.items() if v is not None}
        if not where:
            return []

        try:
            results = self._collection.get(
                where=where,
                limit=top_k,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error(f"ChromaDB filter 查询失败: {e}")
            return []

        search_results: list[SearchResult] = []
        if not results["ids"]:
            return []

        for i in range(len(results["ids"])):
            rid = results["ids"][i]
            doc = results["documents"][i] if results["documents"] else ""
            meta = results["metadatas"][i] if results["metadatas"] else {}

            # 重建 IncidentRecord
            record = IncidentRecord.from_dict(meta)

            search_results.append(SearchResult(
                id=rid,
                content=doc,
                score=1.0,  # 固定 1.0，表示 100% 匹配过滤条件
                metadata=meta,
                record=record,
            ))

        logger.info(
            f"ChromaDB filter: where={where}, 返回 {len(search_results)} 条"
        )

        return search_results

    def count(self) -> int:
        """返回记录总数"""
        return self._collection.count()

    def clear(self) -> None:
        """清空集合"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("已清空 ChromaDB 集合")

    def persist(self) -> None:
        """持久化（ChromaDB PersistentClient 自动持久化，此处为接口兼容）"""
        pass
