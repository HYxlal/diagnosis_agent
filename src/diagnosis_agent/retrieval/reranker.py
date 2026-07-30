"""Embedding 精排器

对召回层（Neo4j 或 Chroma）返回的候选列表，用 raw_query 的 embedding
与每个候选 description 的 embedding 比余弦相似度，重排取 top-K。

设计要点：
- embedding_fn 复用 ChromaVectorStore 的 embedding function，保证语义空间一致
- 带 LRU 缓存（description hash → 向量），避免重复算
- embedding 不可用时降级到只按 structural_match_ratio 排序，不阻断流程
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Callable, Optional

from ..models.neo4j_result import FaultCandidate

logger = logging.getLogger(__name__)


class SemanticReranker:
    """Embedding 精排器

    final_score = semantic_score * (1 - structural_weight)
                + structural_match_ratio * structural_weight
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        cache_size: int = 1000,
        structural_weight: float = 0.2,
    ):
        """
        Args:
            embedding_fn: 文本 → 向量的函数（复用 Chroma 的 embedding_fn）
            cache_size: description 向量缓存上限（条数）
            structural_weight: 结构化匹配度权重，0~1，默认 0.2
        """
        self._embedding_fn = embedding_fn
        self._cache_size = cache_size
        self._structural_weight = structural_weight
        self._cache: dict[str, list[float]] = {}

        if self._embedding_fn is None:
            logger.info(
                "未传入 embedding_fn，尝试用 chromadb 默认 embedding function"
            )
            self._embedding_fn = self._init_default_embedding_fn()

    @staticmethod
    def _init_default_embedding_fn():
        """chromadb 默认 embedding（all-MiniLM-L6-v2）"""
        try:
            from chromadb.utils.embedding_functions import (
                DefaultEmbeddingFunction,
            )
            return DefaultEmbeddingFunction()
        except Exception as e:
            logger.warning(f"无法加载 chromadb 默认 embedding: {e}")
            return None

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _embed(self, text: str) -> Optional[list[float]]:
        """算单条文本 embedding，带缓存"""
        if self._embedding_fn is None:
            return None

        key = self._hash_text(text)
        if key in self._cache:
            return self._cache[key]

        try:
            vectors = self._embedding_fn([text])
            if not vectors:
                return None
            vec = list(vectors[0])
        except Exception as e:
            logger.warning(f"embedding 计算失败: {e}")
            return None

        # LRU 式淘汰：超容量时丢掉最早插入的
        if len(self._cache) >= self._cache_size and self._cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = vec
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def rerank(
        self,
        query: str,
        candidates: list[FaultCandidate],
        top_k: int = 5,
        structural_fields: Optional[dict[str, Any]] = None,
    ) -> list[FaultCandidate]:
        """精排

        Args:
            query: 用户原始查询文本（raw_query）
            candidates: 召回层返回的候选
            top_k: 返回数量
            structural_fields: 输入中结构化字段，用于算 structural_match_ratio。
                key 可选: mcuid, dtc_code, project, component, working_condition, software_version
                命中的字段越多，ratio 越高

        Returns:
            排序后的 top-K 候选（原地修改 final_score 字段）
        """
        if not candidates:
            return []

        query_vec = self._embed(query) if query else None

        for c in candidates:
            # 语义相似度
            if query_vec is not None:
                cand_vec = self._embed(c.description)
                c.semantic_score = self._cosine(query_vec, cand_vec) if cand_vec else 0.0
            else:
                # embedding 不可用 → 语义分置零，只靠结构化匹配度排序
                c.semantic_score = 0.0

            # 结构化匹配度
            c.structural_match_ratio = self._calc_structural_ratio(
                c, structural_fields or {}
            )

            # 综合分
            c.final_score = (
                c.semantic_score * (1 - self._structural_weight)
                + c.structural_match_ratio * self._structural_weight
            )

        ranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _calc_structural_ratio(
        candidate: FaultCandidate,
        fields: dict[str, Any],
    ) -> float:
        """计算候选对输入结构化字段的命中率

        遍历输入提供的每个结构化字段，判断候选是否命中，命中数 / 总字段数。
        """
        if not fields:
            return 0.0

        total = 0
        hit = 0

        mcuid = fields.get("mcuid")
        if mcuid:
            total += 1
            if mcuid and (mcuid in candidate.motor_code or mcuid in candidate.description):
                hit += 1

        dtc_code = fields.get("dtc_code")
        if dtc_code:
            total += 1
            if isinstance(dtc_code, list):
                if any(d in candidate.dtc_codes for d in dtc_code):
                    hit += 1
            elif isinstance(dtc_code, str) and dtc_code in candidate.dtc_codes:
                hit += 1

        project = fields.get("project")
        if project:
            total += 1
            if project in candidate.vehicle_type or project in candidate.description:
                hit += 1

        component = fields.get("component")
        if component:
            total += 1
            if component in candidate.description:
                hit += 1

        working_condition = fields.get("working_condition")
        if working_condition:
            total += 1
            if working_condition in candidate.scenario or working_condition in candidate.description:
                hit += 1

        software_version = fields.get("software_version")
        if software_version:
            total += 1
            if software_version in candidate.description:
                hit += 1

        return hit / total if total > 0 else 0.0
