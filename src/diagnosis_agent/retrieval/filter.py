"""[DEPRECATED] 精确过滤检索工具

已废弃：请使用 langchain_retrievers.py 中的 ChromaVectorRetriever.search_with_filters()
保留此文件仅用于兼容旧版本代码（hybrid.py → react_agent.py）。

原说明：使用 VectorStoreAdapter.filter() 做纯 metadata 精确过滤，
不走 embedding 路径，彻底消除 query 占位符问题。

支持的可过滤字段由 config.yaml 中 retrieval.filter.filter_fields 配置，
filter_by() 接受任意 key-value 组合，便于后续扩展。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..storage.vector_store import SearchResult, VectorStoreAdapter

logger = logging.getLogger(__name__)


class FilterRetriever:
    """精确过滤检索器

    基于 metadata 精确过滤（车型、驱动代码、DTC 码等），
    不依赖向量相似度。

    可过滤的字段由 config.yaml 的 retrieval.filter.filter_fields 配置，
    filter_by() 接受任意 key-value，只需 key 在 filter_fields 列表中。
    """

    def __init__(
        self,
        store: VectorStoreAdapter,
        default_top_k: int = 10,
        filter_fields: Optional[list[str]] = None,
    ):
        self.store = store
        self.default_top_k = default_top_k
        # 可配置的过滤字段列表，默认为全部 8 列中的可过滤字段
        self.filter_fields = filter_fields or [
            "vehicle_type",
            "drive_code",
            "dtc_code",
            "dashboard_indicator",
            "fault_scenario",
        ]

    def filter_by(
        self,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """按任意 metadata 字段精确过滤

        使用 VectorStoreAdapter.filter() 做纯 metadata 过滤，
        不走 embedding 路径，无 query 占位符。

        Args:
            top_k: 返回数量，默认使用 default_top_k
            **kwargs: 任意 key-value 过滤条件（key 需在 filter_fields 中）
                      常用: vehicle_type, drive_code, dtc_code,
                            dashboard_indicator, fault_scenario

        Returns:
            SearchResult 列表，score 固定为 1.0

        Examples:
            >>> retriever.filter_by(vehicle_type="SUV-A", dtc_code="P0301")
            >>> retriever.filter_by(drive_code="EDU-3", top_k=20)
        """
        k = top_k or self.default_top_k

        # 构建过滤条件，只保留 filter_fields 中配置的字段且值非 None
        filters: dict[str, Any] = {}
        for key, value in kwargs.items():
            if value is not None and key in self.filter_fields:
                filters[key] = value
            elif value is not None and key not in self.filter_fields:
                logger.warning(
                    f"过滤字段 '{key}' 不在配置的 filter_fields 中，已忽略。"
                    f"如需启用，请在 config.yaml 的 retrieval.filter.filter_fields 中添加。"
                )

        if not filters:
            logger.warning("filter_by: 无有效过滤条件")
            return []

        # 调用 store.filter()，不走 embedding 路径
        results = self.store.filter(filters=filters, top_k=k)

        logger.info(
            f"精确过滤: filters={filters}, 返回 {len(results)} 条"
        )

        return results

    def get_by_vehicle_type(
        self, vehicle_type: str, top_k: int = 10
    ) -> list[SearchResult]:
        """按车型检索"""
        return self.filter_by(vehicle_type=vehicle_type, top_k=top_k)

    def get_by_dtc_code(
        self, dtc_code: str, top_k: int = 10
    ) -> list[SearchResult]:
        """按 DTC 故障码检索"""
        return self.filter_by(dtc_code=dtc_code, top_k=top_k)

    def get_by_drive_code(
        self, drive_code: str, top_k: int = 10
    ) -> list[SearchResult]:
        """按驱动代码检索"""
        return self.filter_by(drive_code=drive_code, top_k=top_k)
