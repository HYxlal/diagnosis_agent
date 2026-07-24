"""[DEPRECATED] 语义检索工具

已废弃：请使用 langchain_retrievers.py 中的 ChromaVectorRetriever
保留此文件仅用于兼容旧版本代码（hybrid.py → react_agent.py）。

原说明：基于向量相似度检索相似工况。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..storage.vector_store import SearchResult, VectorStoreAdapter

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """语义检索器

    基于向量相似度检索相似工况。
    """

    def __init__(
        self,
        store: VectorStoreAdapter,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        """语义检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据过滤条件
            min_score: 最低相似度阈值

        Returns:
            SearchResult 列表，按相似度降序
        """
        k = top_k or self.top_k
        threshold = min_score if min_score is not None else self.score_threshold

        results = self.store.search(query=query, top_k=k, filters=filters)

        # 过滤低分结果
        filtered = [r for r in results if r.score >= threshold]

        logger.info(
            f"语义检索: query='{query[:50]}...', "
            f"返回 {len(filtered)}/{len(results)} 条 (阈值={threshold})"
        )

        return filtered

    def search_by_description(
        self,
        description: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """按故障描述检索

        Args:
            description: 故障描述
            vehicle_type: 车型（可选过滤）
            top_k: 返回数量

        Returns:
            SearchResult 列表
        """
        query_parts = []
        if vehicle_type:
            query_parts.append(f"车型: {vehicle_type}")
        if description:
            query_parts.append(description)

        query = " | ".join(query_parts) if query_parts else ""

        if not query:
            return []

        filters = {}
        if vehicle_type:
            filters["vehicle_type"] = vehicle_type

        return self.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None,
        )
