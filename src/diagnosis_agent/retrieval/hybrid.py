"""混合检索工具

结合语义检索和精确过滤，综合排序。

得分策略：加权 sum（非 max）
- 同一记录被两路检索命中时，得分 = semantic_score * semantic_weight + filter_score * filter_weight
- filter_score 固定为 1.0，语义 score 为余弦相似度（0~1），量纲一致
- 双路命中获得交叉验证加成，比取 max 更有利于高相关性记录
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..models.input import ParsedInput
from ..storage.vector_store import SearchResult, VectorStoreAdapter
from .filter import FilterRetriever
from .semantic import SemanticRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器

    策略：
    1. 同时执行语义检索和精确过滤
    2. 根据权重合并结果（加权 sum，非 max）
    3. 去重并排序
    """

    def __init__(
        self,
        store: VectorStoreAdapter,
        semantic_top_k: int = 5,
        filter_top_k: int = 10,
        semantic_weight: float = 0.7,
        filter_weight: float = 0.3,
        score_threshold: float = 0.3,
        filter_expansion_ratio: int = 2,
        filter_fields: Optional[list[str]] = None,
    ):
        self.store = store
        self.semantic_retriever = SemanticRetriever(
            store, top_k=semantic_top_k, score_threshold=score_threshold
        )
        self.filter_retriever = FilterRetriever(
            store,
            default_top_k=filter_top_k,
            filter_fields=filter_fields,
        )
        self.semantic_weight = semantic_weight
        self.filter_weight = filter_weight
        self.score_threshold = score_threshold
        self.filter_expansion_ratio = filter_expansion_ratio

    def retrieve(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        drive_code: Optional[str] = None,
        dtc_code: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """混合检索

        Args:
            query: 查询文本
            vehicle_type: 车型（可选过滤）
            drive_code: 驱动代码（可选过滤）
            dtc_code: DTC 故障码（可选过滤）
            top_k: 返回数量

        Returns:
            SearchResult 列表
        """
        # 语义检索
        semantic_results = self.semantic_retriever.search(
            query=query,
            top_k=top_k,
        )

        # 精确过滤（如果有过滤条件）
        filter_results: list[SearchResult] = []
        if vehicle_type or drive_code or dtc_code:
            filter_results = self.filter_retriever.filter_by(
                vehicle_type=vehicle_type,
                drive_code=drive_code,
                dtc_code=dtc_code,
                top_k=top_k * self.filter_expansion_ratio,
            )

        # 合并去重 — 加权 sum 策略
        # 同一记录被两路命中时，得分 = semantic_score*w_s + filter_score*w_f
        # 比取 max 更好：双路命中的交叉验证信号被保留
        merged: dict[str, SearchResult] = {}

        for r in semantic_results:
            score = r.score * self.semantic_weight
            if r.id in merged:
                # 加权 sum：累加而非取 max
                merged[r.id].score += score
            else:
                merged[r.id] = SearchResult(
                    id=r.id,
                    content=r.content,
                    score=score,
                    metadata=r.metadata,
                    record=r.record,
                )

        for r in filter_results:
            score = r.score * self.filter_weight  # filter score 固定 1.0
            if r.id in merged:
                # 加权 sum：累加而非取 max
                merged[r.id].score += score
            else:
                merged[r.id] = SearchResult(
                    id=r.id,
                    content=r.content,
                    score=score,
                    metadata=r.metadata,
                    record=r.record,
                )

        # 排序并截断
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        logger.info(
            f"混合检索: query='{query[:50]}...', "
            f"semantic={len(semantic_results)}, filter={len(filter_results)}, "
            f"merged={len(results)}"
        )

        return results

    def retrieve_from_parsed(
        self, parsed: ParsedInput, top_k: int = 5
    ) -> list[SearchResult]:
        """从 ParsedInput 构建查询并检索

        如果 parsed 有 search_query（由 InputRouter 提取），优先使用；
        否则回退到 description。

        Args:
            parsed: 解析后的输入
            top_k: 返回数量

        Returns:
            SearchResult 列表
        """
        # 优先使用 InputRouter 提取的 search_query
        query = parsed.search_query or parsed.description or ""

        # 从 bulk_records 提取过滤条件
        vehicle_type = None
        drive_code = None
        dtc_code = None
        if parsed.bulk_records:
            first = parsed.bulk_records[0]
            vehicle_type = first.get("vehicle_type") or None
            drive_code = first.get("drive_code") or None
            dtc_code = first.get("dtc_code") or None

        return self.retrieve(
            query=query,
            vehicle_type=vehicle_type,
            drive_code=drive_code,
            dtc_code=dtc_code,
            top_k=top_k,
        )
