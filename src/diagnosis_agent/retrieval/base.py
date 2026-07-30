"""检索器抽象接口

统一 ChromaVectorRetriever / Neo4jFaultRetriever / HybridRetriever 的对外接口，
让 HybridRetriever 编排时只依赖抽象，不依赖具体实现。

对外接口契约：
- retrieve(query, fields, top_k) → list[Document]
  统一以 LangChain Document 返回，metadata 带 source 标签（"chroma"/"neo4j"）
- available → bool
  检索器是否可用（Neo4j 连不上时返回 False）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.documents import Document


class FaultRetriever(ABC):
    """故障检索器抽象基类"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """检索器是否可用"""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        fields: Optional[dict[str, Any]] = None,
        top_k: int = 5,
    ) -> list[Document]:
        """执行检索

        Args:
            query: 查询文本（自然语言故障描述）
            fields: 结构化字段，key 可选：
                mcuid, dtc_codes, scenarios, indicators, vehicle_types, component
            top_k: 返回数量

        Returns:
            Document 列表，metadata 必须包含 source 标签
        """
        ...
