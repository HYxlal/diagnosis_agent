"""混合检索编排

按 strategy 配置切换检索路径：
- chroma_only : 只走 Chroma（Neo4j 不可用时用这个）
- neo4j_first: Neo4j 召回 + Embedding 精排，不足时走 Chroma 兜底（默认）
- hybrid     : 两段式全开

对外只暴露 retrieve()，上层不感知走了哪条路。
返回统一 list[Document]，metadata 带 source 标签（"neo4j" / "chroma"）。

字段分通道匹配设计：
每个字段独立走自己的匹配路径，不强制拼进 keyword：
  - vehicleModel          → Neo4j motor_codes 独立匹配 / Chroma metadata vehicle_type 独立过滤
  - dtcCode               → Neo4j dtc_inputs 独立匹配 / Chroma metadata dtc_code 独立过滤
  - faultWorkConditionList → Neo4j scenarios 独立匹配 / Chroma metadata fault_scenario 独立过滤
  - instrumentIndicatorList → Neo4j indicators 独立匹配 / Chroma metadata dashboard_indicator 独立过滤
  - softwareVersion/motorPosition/VIN → keyword 补充通道
空字段自动跳过，不做强过滤。
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import Settings, get_settings
from ..models.input import ParsedInput
from .base import FaultRetriever
from .field_mapper import FieldMapper
from .langchain_retrievers import ChromaVectorRetriever, create_chroma_retriever
from .neo4j_retriever import Neo4jFaultRetriever
from .reranker import SemanticReranker
from .search_condition import SearchCondition

logger = logging.getLogger(__name__)


class HybridRetriever(FaultRetriever, BaseRetriever):
    """两段式混合检索器 — 字段分通道独立匹配

    同时继承 FaultRetriever（内部接口）和 BaseRetriever（LangChain 标准接口）。
    Neo4j 召回 → Embedding 精排 → Chroma 兜底（按 strategy 配置）。
    """

    def __init__(
        self,
        neo4j_retriever: Neo4jFaultRetriever,
        chroma_retriever: ChromaVectorRetriever,
        reranker: SemanticReranker,
        min_candidates: int = 3,
        top_k: int = 5,
        fallback_to_chroma: bool = True,
        strategy: str = "neo4j_first",
    ):
        super().__init__()
        self._neo4j = neo4j_retriever
        self._chroma = chroma_retriever
        self._reranker = reranker
        self._min_candidates = min_candidates
        self._top_k = top_k
        self._fallback_to_chroma = fallback_to_chroma
        self._strategy = strategy

    @property
    def available(self) -> bool:
        """始终可用：至少 Chroma 在工作"""
        return True

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def neo4j(self) -> Neo4jFaultRetriever:
        """暴露 Neo4j 召回器，供 query_fault_graph 工具直接调用"""
        return self._neo4j

    @property
    def chroma(self) -> ChromaVectorRetriever:
        """暴露 Chroma 检索器，供 search_similar_incidents 工具调用"""
        return self._chroma

    def semantic_search(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        drive_code: Optional[str] = None,
        top_k: int = 5,
    ) -> list[Document]:
        """纯语义检索（供 Agent 工具在循环中主动调用）

        直接走 Chroma 语义检索，不走两段式编排（预检索已走过了）。
        """
        try:
            if hasattr(self._chroma, "search_with_filters"):
                docs = self._chroma.search_with_filters(
                    query=query,
                    vehicle_type=vehicle_type,
                    drive_code=drive_code,
                    top_k=top_k,
                )
            else:
                docs = self._chroma.invoke(query)[:top_k]
        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            return []

        for doc in docs:
            doc.metadata.setdefault("source", "chroma")
        return docs

    def retrieve(self, query: str, fields: Optional[dict] = None, top_k: int = 5) -> list[Document]:
        """FaultRetriever 接口：直接走 semantic_search

        供外部按统一接口调用时用，不参与两段式编排。
        """
        return self.semantic_search(query=query, top_k=top_k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None, **kwargs: Any,
    ) -> list[Document]:
        """BaseRetriever 接口：纯语义检索

        让 HybridRetriever 可被 create_retriever_tool 包装。
        由于只接受 query 字符串，无法传递结构化字段，因此走纯 Chroma 语义检索。
        两段式检索仍通过 retrieve_parsed() 使用。
        """
        return self._chroma.invoke(query)[:self._top_k]

    def retrieve_parsed(self, parsed_input: ParsedInput) -> list[Document]:
        """按 ParsedInput 执行两段式检索（编排入口）

        Agent 内部走这个方法，会用到 strategy 配置和字段分通道独立匹配。
        """
        query = parsed_input.search_query or parsed_input.description or ""
        if not query:
            return []

        condition = FieldMapper.extract_search_condition(parsed_input)
        top_k = self._top_k

        # 按策略分流
        if self._strategy == "chroma_only":
            logger.info("strategy=chroma_only，直接走 Chroma 语义检索")
            return self._chroma_fallback(query, top_k, condition)

        # neo4j_first / hybrid 都走 Neo4j 召回
        candidates = self._neo4j_recall(condition)

        neo4j_available = self._neo4j.available
        if not candidates:
            if not neo4j_available:
                if self._fallback_to_chroma:
                    logger.warning("Neo4j 不可用，已降级到 Chroma 语义检索。")
                else:
                    logger.warning("Neo4j 不可用且未启用 Chroma 降级。")
            else:
                logger.info("Neo4j 召回为空，走 Chroma 兜底。")
            return self._chroma_fallback(query, top_k, condition)

        # Stage 2: 精排（返回 FaultCandidate 列表）
        ranked_candidates = self._reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            structural_fields=condition.to_rerank_fields(),
        )

        ranked = [c.to_document() for c in ranked_candidates]

        # Stage 3: 召回不足 → 补 Chroma 兜底
        if len(ranked) < self._min_candidates and self._fallback_to_chroma:
            logger.info(
                f"精排后仅 {len(ranked)} 条 (< {self._min_candidates})，"
                "用 Chroma 语义检索补充。"
            )
            chroma_docs = self._chroma_fallback(query, top_k, condition)
            ranked = self._merge_no_duplicate(ranked, chroma_docs, top_k)

        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Stage 1: Neo4j 召回 — 各字段独立走自己的匹配通道
    # ------------------------------------------------------------------

    def _neo4j_recall(self, condition: SearchCondition) -> list:
        """调用 Neo4j 结构化召回 — 字段分通道独立匹配，不强拼keyword"""
        if not self._neo4j.available:
            return []

        # 工况独立通道（转成字符串列表，多个工况场景独立匹配）
        scenarios = []
        if condition.faultWorkConditionList and condition.faultWorkConditionList != "无法确认故障工况":
            scenarios = [condition.faultWorkConditionList]

        # 走各字段独立通道
        return self._neo4j.structured_recall(
            vehicleModel=condition.vehicleModel if condition.vehicleModel else None,
            dtc_codes=condition.dtcCode if condition.dtcCode else None,
            indicators=condition.instrumentIndicatorList if condition.instrumentIndicatorList else None,
            scenarios=scenarios if scenarios else None,
            keyword=condition.to_keyword() or None,
        )

    # ------------------------------------------------------------------
    # Stage 3: Chroma 兜底 — metadata 独立字段过滤
    # ------------------------------------------------------------------

    def _chroma_fallback(
        self,
        query: str,
        top_k: int,
        condition: SearchCondition,
    ) -> list[Document]:
        """Chroma 语义检索兜底 — 各字段走独立 metadata 过滤通道，不强拼到文本"""
        # 构建独立 metadata 过滤条件
        filters = {}
        if condition.vehicleModel:
            filters["vehicle_type"] = condition.vehicleModel
        if condition.dtcCode and len(condition.dtcCode) > 0:
            # Chroma metadata 里 dtc_code 是字符串，有DTC码的直接精确匹配
            filters["dtc_code"] = condition.dtcCode[0]

        # 查询增强文本：仅补充没有独立metadata通道的字段
        enriched_query_parts = [query]
        kw = condition.to_keyword()
        if kw:
            enriched_query_parts.append(kw)
        enriched_query = " ".join(enriched_query_parts)

        try:
            if hasattr(self._chroma, "search_with_filters"):
                docs = self._chroma.search_with_filters(
                    query=enriched_query,
                    vehicle_type=filters.get("vehicle_type"),
                    top_k=top_k,
                )
            else:
                docs = self._chroma.invoke(enriched_query)[:top_k]
        except Exception as e:
            logger.error(f"Chroma 兜底检索失败: {e}")
            return []

        for doc in docs:
            doc.metadata["source"] = "chroma"
        return docs

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_no_duplicate(
        primary: list[Document],
        secondary: list[Document],
        top_k: int,
    ) -> list[Document]:
        """合并两份结果，按 id 去重，primary 优先"""
        seen_ids: set[str] = set()
        merged: list[Document] = []

        for doc in primary + secondary:
            doc_id = doc.metadata.get("id", "")
            if doc_id and doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            merged.append(doc)
            if len(merged) >= top_k:
                break

        return merged


def create_hybrid_retriever(settings: Optional[Settings] = None) -> HybridRetriever:
    """工厂函数：组装 neo4j + chroma + reranker

    embedding_fn 复用 ChromaVectorStore 的 embedding function，
    保证精排和 Chroma 语义空间一致。
    """
    settings = settings or get_settings()
    cfg = settings.neo4j

    chroma = create_chroma_retriever()
    neo4j = Neo4jFaultRetriever(settings=settings)

    # 复用 chroma 的 embedding_fn，避免重复初始化模型
    embedding_fn = None
    if hasattr(chroma, "store") and chroma.store is not None:
        embedding_fn = chroma.store.embedding_function

    reranker = SemanticReranker(embedding_fn=embedding_fn)

    return HybridRetriever(
        neo4j_retriever=neo4j,
        chroma_retriever=chroma,
        reranker=reranker,
        min_candidates=cfg.min_candidates,
        top_k=settings.retrieval.semantic.top_k,
        fallback_to_chroma=cfg.fallback_to_chroma,
        strategy=settings.retrieval.strategy,
    )
