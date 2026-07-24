"""VectorStoreAdapter 抽象层

定义向量存储的标准接口，便于后续扩展其他向量数据库。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..models.incident import IncidentRecord


@dataclass
class SearchResult:
    """向量检索结果"""
    id: str                          # 记录ID
    content: str                     # 文档内容
    score: float                     # 相似度分数
    metadata: dict[str, Any]         # 元数据
    record: Optional[IncidentRecord] = None  # 关联的 IncidentRecord


class VectorStoreAdapter(ABC):
    """向量存储适配器抽象基类

    所有具体的向量存储实现（ChromaDB / FAISS / Pinecone 等）
    都需要实现此接口。

    接口分离原则：
    - search() 做语义检索（需要 embedding）
    - filter() 做纯 metadata 精确过滤（不需要 embedding）
    """

    @abstractmethod
    def add_records(self, records: list[IncidentRecord]) -> int:
        """批量添加故障记录"""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """语义检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            filters: 元数据过滤条件

        Returns:
            SearchResult 列表，按相似度降序
        """
        ...

    @abstractmethod
    def filter(
        self,
        filters: dict[str, Any],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """纯 metadata 精确过滤（不走 embedding 路径）

        Args:
            filters: 元数据过滤条件（key=value 的 AND 组合）
            top_k: 返回结果数

        Returns:
            SearchResult 列表，score 固定为 1.0（表示 100% 匹配过滤条件）
        """
        ...

    @abstractmethod
    def get_by_id(self, record_id: str) -> Optional[IncidentRecord]:
        """根据ID获取记录"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回存储中的记录总数"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有记录"""
        ...

    @abstractmethod
    def persist(self) -> None:
        """持久化到磁盘"""
        ...
