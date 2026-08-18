"""ChromaDB 向量存储实现

基于 langchain_chroma.Chroma，对齐 LangChain 标准接口。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from langchain_core.embeddings import Embeddings

from ..models.incident import IncidentRecord
from .vector_store import SearchResult, VectorStoreAdapter

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStoreAdapter):
    """基于 langchain_chroma.Chroma 的向量存储实现

    内部使用 langchain_chroma.Chroma（LangChain 标准 VectorStore），
    对外保持 SearchResult + IncidentRecord 的业务接口不变。

    特性：
    - 持久化到本地磁盘
    - 支持语义检索
    - 支持 metadata 过滤
    - embedding 函数通过 provider 参数切换
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "incidents",
        embedding_model: str = "text-embedding-v2",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
    ):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._embedding = embedding or self._create_default_embedding(
            api_key, api_base
        )

        from langchain_chroma import Chroma

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embedding,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaVectorStore 初始化: dir={persist_dir}, "
            f"collection={collection_name}, count={self._store._collection.count()}"
        )

    @property
    def embedding_function(self) -> Optional[Any]:
        """返回当前使用的 embedding 函数（供 reranker 等模块复用）"""
        return self._embedding

    @property
    def _collection(self):
        """底层 chromadb collection（供 filter() 和 get_by_id() 直接用）"""
        return self._store._collection

    def _create_default_embedding(
        self, api_key: Optional[str], api_base: Optional[str]
    ) -> Embeddings:
        """创建默认 embedding 函数"""
        if api_key:
            try:
                from ..utils.embedding_wrapper import create_embedding_fn
                return create_embedding_fn(
                    model=self.embedding_model,
                    api_key=api_key,
                    api_base=api_base or "",
                )
            except Exception as e:
                logger.warning(f"Embedding 初始化失败，使用默认: {e}")

        # 无 API key 时回退到 ChromaDB 默认 all-MiniLM-L6-v2
        logger.info("使用 ChromaDB 默认 embedding (all-MiniLM-L6-v2)")
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return _ChromaDefaultEmbeddings()

    def add_records(self, records: list[IncidentRecord]) -> int:
        """批量添加故障记录"""
        if not records:
            return 0

        ids = []
        texts = []
        metadatas = []

        for record in records:
            record_id = f"REC-{uuid.uuid4().hex[:12]}"
            doc_text = record.to_searchable_text()
            meta = record.to_dict()
            for k, v in meta.items():
                meta[k] = str(v) if v is not None else ""
            meta["id"] = record_id  # 存到 metadata 中以便检索时返回

            ids.append(record_id)
            texts.append(doc_text)
            metadatas.append(meta)

        batch_size = 10
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            try:
                self._store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
                added += len(batch_ids)
            except Exception as e:
                logger.error(f"Chroma 批量添加失败 (batch {i}): {e}")
                continue

        logger.info(f"成功添加 {added} 条记录到 ChromaDB")
        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """语义检索

        使用 langchain_chroma 的 similarity_search_with_score，
        确保 embedding 函数通过 LangChain 标准接口传递。
        """
        if self._collection.count() == 0:
            return []

        where = None
        if filters:
            where = {k: v for k, v in filters.items() if v is not None}

        try:
            results = self._store.similarity_search_with_score(
                query=query,
                k=min(top_k, self._collection.count()),
                filter=where,
            )
        except Exception as e:
            logger.error(f"ChromaDB 查询失败: {e}")
            return []

        search_results: list[SearchResult] = []
        for doc, score in results:
            meta = doc.metadata or {}
            rid = meta.get("id", "")
            record = IncidentRecord.from_dict(meta)
            search_results.append(SearchResult(
                id=rid,
                content=doc.page_content,
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
        """纯 metadata 精确过滤（不走 embedding 路径）"""
        if not filters:
            return []

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
            record = IncidentRecord.from_dict(meta)
            search_results.append(SearchResult(
                id=rid,
                content=doc,
                score=1.0,
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
        # 复用已有的 client 删除 collection，避免 SharedSystemClient 冲突
        self._store._client.delete_collection(self.collection_name)

        from langchain_chroma import Chroma
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embedding,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info("已清空 ChromaDB 集合")

    def persist(self) -> None:
        """持久化（langchain_chroma 自动持久化，此处为接口兼容）"""
        pass


class _ChromaDefaultEmbeddings(Embeddings):
    """将 chromadb 默认 EmbeddingFunction 包装成 LangChain Embeddings 接口

    无 API key 时的回退方案，使用本地 all-MiniLM-L6-v2。
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]