"""Neo4j 故障知识图谱召回层

用本项目自实现的 cypher_builder（retrieval/cypher_builder.py）生成 Cypher，
自己建 Neo4j 驱动连接，不再依赖 fault_knowledge_graph 项目。

职责：
- 输入结构化字段（mcuid/DTC/keyword/...）
- 调用 cypher_builder.build_query 生成 Cypher
- 执行查询，把原始记录展平为 FaultCandidate
- 连接失败时 catch 住，返回空列表（上层走 Chroma 兜底）

当前阶段：
- 外部字段已对齐 StandardInput.entities：dtc_code, project, component, working_condition, software_version。
- Neo4j schema 已重构为中文语义关系（出现于/关联DTC/亮起/发生于/配备 等）。
- 当前映射：mcuid → motor_codes，dtc_code → dtc_inputs，其余字段统一作为 keyword。
- 待优化：project/component/working_condition/software_version 映射到对应节点属性。

不在这里做的事：
- 不做 embedding 精排（在 reranker.py）
- 不做 Chroma 兜底（在 hybrid_retriever.py）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import GraphDatabase

from ..config import Settings, get_settings
from ..models.neo4j_result import FaultCandidate
from .base import FaultRetriever
from .cypher_builder import QueryCondition, build_query

logger = logging.getLogger(__name__)


class Neo4jFaultRetriever(FaultRetriever):
    """Neo4j 结构化召回器

    用 cypher_builder.build_query(QueryCondition) 生成 Cypher，自己建驱动执行。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        default_depth: int = 1,
        default_limit: int = 50,
    ):
        self._settings = settings or get_settings()
        cfg = self._settings.neo4j
        self._default_depth = default_depth or cfg.default_depth
        self._default_limit = default_limit
        self._driver = None

        if not cfg.url:
            logger.warning(
                "Neo4j URL 未配置 (neo4j.url 为空)，Neo4j 召回将不可用，"
                "检索会降级到 Chroma 语义兜底。"
            )
            return

        try:
            self._driver = GraphDatabase.driver(
                cfg.url,
                auth=(cfg.user, cfg.password),
            )
        except Exception as e:
            logger.warning(
                f"Neo4j 驱动创建失败: {e}。检索将降级到 Chroma 语义兜底。"
            )
            self._driver = None
            return

        logger.info(
            f"Neo4jFaultRetriever 初始化: url={cfg.url}, "
            f"default_depth={self._default_depth}"
        )

    @property
    def available(self) -> bool:
        """是否可用（驱动已建立）"""
        return self._driver is not None

    def structured_recall(
        self,
        mcuid: Optional[str] = None,
        dtc_codes: Optional[list[str]] = None,
        keyword: Optional[str] = None,
        depth: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[FaultCandidate]:
        """结构化召回

        Args:
            mcuid: MCU 标识 → 透传给 MotorType.code 模糊匹配
            dtc_codes: DTC 列表 → DTC.code 精确 / DTC.description 模糊
            keyword: 关键词 → 匹配 Fault.description / full_text
            depth: 关系扩展深度，None 用默认
            limit: 返回上限，None 用默认

        Returns:
            FaultCandidate 列表；Neo4j 不可用时返回空列表（不抛异常）。
        """
        if not self.available:
            logger.info("Neo4j 召回不可用，返回空列表（上层走 Chroma 兜底）")
            return []

        motor_list = [mcuid] if mcuid else None

        condition = QueryCondition(
            motor_codes=motor_list,
            dtc_inputs=dtc_codes,
            keyword=keyword,
            limit=limit or self._default_limit,
            depth=depth or self._default_depth,
        )

        try:
            query, params = build_query(condition)
        except Exception as e:
            logger.error(f"构建 Cypher 失败: {e}")
            return []

        try:
            with self._driver.session() as session:
                records = session.run(query, params).data()
        except Exception as e:
            logger.warning(
                f"Neo4j 查询执行失败: {e}。上层将降级到 Chroma 兜底。"
            )
            return []

        candidates: list[FaultCandidate] = []
        for record in records:
            try:
                candidates.append(FaultCandidate.from_neo4j_record(record))
            except Exception as e:
                logger.warning(f"展平 Neo4j 记录失败: {e}, record={record}")
                continue
        logger.info(
            f"Neo4j 召回: mcuid={mcuid}, dtc={dtc_codes}, "
            f"keyword={keyword} → {len(candidates)} 条候选"
        )
        return candidates

    def retrieve(
        self,
        query: str,
        fields: Optional[dict[str, Any]] = None,
        top_k: int = 5,
    ) -> list:
        """FaultRetriever 接口实现：把结构化字段转成 structured_recall 调用

        本类不做 embedding 精排，返回 FaultCandidate 原始对象（由 HybridRetriever
        在编排层负责转 Document + 精排）。
        """
        fields = fields or {}
        candidates = self.structured_recall(
            mcuid=fields.get("mcuid") or None,
            dtc_codes=fields.get("dtc_code") or None,
            keyword=fields.get("keyword") or None,
            limit=top_k or self._default_limit,
        )
        return candidates

    def close(self) -> None:
        """关闭驱动连接"""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
