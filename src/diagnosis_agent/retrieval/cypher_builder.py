"""Neo4j 故障知识图谱 Cypher 构建器

本项目自实现，脱离 fault_knowledge_graph 项目的 query_builder.py 依赖。

支持的查询字段（AND 组合）：
- motor_codes   → MotorType.code CONTAINS（模糊）
- vehicle_types → VehicleType.type CONTAINS（模糊）
- indicators    → Indicator.name CONTAINS（模糊）
- dtc_inputs    → DTC.code 精确 / DTC.description CONTAINS（按首字符判断）
- scenarios     → Scenario 三级结构（完整三级精确 / 部分层级模糊）
- keyword       → f.description / f.full_text CONTAINS

关系扩展深度 depth：1-5，LIMIT $limit 作用于 Fault。
返回 f + collect(DISTINCT path) AS paths，path 结构与同事项目对齐：
[[source_data, rel_type, target_data], ...]

调用方拿到 records 后，可用 models.neo4j_result.FaultCandidate.from_neo4j_record()
展平。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QueryCondition:
    """故障知识图谱查询条件

    所有字段都可选，None 表示不参与过滤。多个字段之间是 AND 关系。
    """
    motor_codes: Optional[List[str]] = None
    vehicle_types: Optional[List[str]] = None
    indicators: Optional[List[str]] = None
    dtc_inputs: Optional[List[str]] = None
    scenarios: Optional[List[str]] = None
    keyword: Optional[str] = None
    limit: int = 100
    depth: int = 1


@dataclass
class _QueryBuilder:
    """内部状态：收集 where 子句和参数

    每次 build() 重置。
    """
    _params: Dict[str, Any] = field(default_factory=dict)
    _where_clauses: List[str] = field(default_factory=list)
    _param_counter: int = 0

    def _add_param(self, value: Any) -> str:
        """注册一个参数，返回占位符名（p1, p2, ...）"""
        self._param_counter += 1
        name = f"p{self._param_counter}"
        self._params[name] = value
        return name

    def _build_motor(self, values: List[str]) -> None:
        # f 必须有一条 OCCURS_ON → MotorType 的关系，且 motor code 模糊匹配
        parts = []
        for v in values:
            name = self._add_param(v)
            parts.append(f"m.code CONTAINS ${name}")
        self._where_clauses.append(
            f"EXISTS {{ MATCH (f)-[:OCCURS_ON]->(m:MotorType) WHERE {' OR '.join(parts)} }}"
        )

    def _build_vehicle(self, values: List[str]) -> None:
        parts = []
        for v in values:
            name = self._add_param(v)
            parts.append(f"v.type CONTAINS ${name}")
        self._where_clauses.append(
            f"EXISTS {{ MATCH (f)-[:OCCURS_ON_VEHICLE]->(v:VehicleType) WHERE {' OR '.join(parts)} }}"
        )

    def _build_indicator(self, values: List[str]) -> None:
        parts = []
        for v in values:
            name = self._add_param(v)
            parts.append(f"i.name CONTAINS ${name}")
        self._where_clauses.append(
            f"EXISTS {{ MATCH (f)-[:SHOWS_INDICATOR]->(i:Indicator) WHERE {' OR '.join(parts)} }}"
        )

    def _build_dtc(self, values: List[str]) -> None:
        # DTC 码首字符为 P/B/U 且长度 ≥6 时按精确匹配 d.code
        # 否则按 d.description CONTAINS 模糊匹配
        parts = []
        for v in values:
            v_clean = v.strip()
            if not v_clean:
                continue
            first = v_clean[0].upper()
            if first in ('P', 'B', 'U') and len(v_clean) >= 6:
                name = self._add_param(v_clean.upper())
                parts.append(f"d.code = ${name}")
            else:
                name = self._add_param(v_clean)
                parts.append(f"d.description CONTAINS ${name}")
        if parts:
            self._where_clauses.append(
                f"EXISTS {{ MATCH (f)-[:HAS_DTC]->(d:DTC) WHERE {' OR '.join(parts)} }}"
            )

    def _build_scenario(self, values: List[str]) -> None:
        # 完整三级结构（含两个 -）→ 精确匹配 category/subcategory/detail
        # 否则 → 模糊匹配任意层级
        parts = []
        for v in values:
            v_clean = v.strip()
            if not v_clean:
                continue
            if v_clean.count('-') == 2:
                cat, sub, det = [p.strip() for p in v_clean.split('-')]
                conds = []
                if cat:
                    name = self._add_param(cat)
                    conds.append(f"s.category = ${name}")
                if sub:
                    name = self._add_param(sub)
                    conds.append(f"s.subcategory = ${name}")
                if det:
                    name = self._add_param(det)
                    conds.append(f"s.detail = ${name}")
                if conds:
                    parts.append(
                        f"EXISTS {{ MATCH (f)-[:OCCURS_IN_SCENARIO]->(s:Scenario) WHERE {' AND '.join(conds)} }}"
                    )
            else:
                name = self._add_param(v_clean)
                parts.append(
                    f"EXISTS {{ MATCH (f)-[:OCCURS_IN_SCENARIO]->(s:Scenario) "
                    f"WHERE s.category CONTAINS ${name} "
                    f"OR s.subcategory CONTAINS ${name} "
                    f"OR s.detail CONTAINS ${name} }}"
                )
        if parts:
            self._where_clauses.append(f"({' OR '.join(parts)})")

    def _build_keyword(self, keyword: str) -> None:
        kw = keyword.strip()
        if not kw:
            return
        name = self._add_param(kw)
        self._where_clauses.append(
            f"(f.description CONTAINS ${name} OR f.full_text CONTAINS ${name})"
        )

    def build(self, condition: QueryCondition) -> Tuple[str, Dict[str, Any]]:
        """构建 Cypher 查询

        Returns:
            (query, params)，可直接传 session.run(query, params)
        """
        self._params = {}
        self._where_clauses = []
        self._param_counter = 0

        if condition.motor_codes:
            self._build_motor(condition.motor_codes)
        if condition.vehicle_types:
            self._build_vehicle(condition.vehicle_types)
        if condition.indicators:
            self._build_indicator(condition.indicators)
        if condition.dtc_inputs:
            self._build_dtc(condition.dtc_inputs)
        if condition.scenarios:
            self._build_scenario(condition.scenarios)
        if condition.keyword:
            self._build_keyword(condition.keyword)

        where_clause = (
            " AND ".join(self._where_clauses)
            if self._where_clauses
            else "1=1"
        )

        depth = condition.depth
        if not isinstance(depth, int) or depth < 1:
            depth = 1
        if depth > 5:
            depth = 5

        self._params["limit"] = condition.limit

        query = f"""
        MATCH (f:Fault)
        WHERE {where_clause}
        WITH f
        LIMIT $limit
        OPTIONAL MATCH path = (f)-[*1..{depth}]-(neighbor)
        WHERE NOT (neighbor:Fault)
        RETURN f, collect(DISTINCT path) AS paths
        """
        return query, self._params


def build_query(condition: QueryCondition) -> Tuple[str, Dict[str, Any]]:
    """构建查询 Cypher（对应原 fault.query_builder.build_query）"""
    return _QueryBuilder().build(condition)


def build_count_query(condition: QueryCondition) -> Tuple[str, Dict[str, Any]]:
    """构建计数 Cypher（只统计 Fault 数量，不扩展关系）"""
    qb = _QueryBuilder()
    # 复用 build 的 where 构建逻辑，但只取 where 子句
    if condition.motor_codes:
        qb._build_motor(condition.motor_codes)
    if condition.vehicle_types:
        qb._build_vehicle(condition.vehicle_types)
    if condition.indicators:
        qb._build_indicator(condition.indicators)
    if condition.dtc_inputs:
        qb._build_dtc(condition.dtc_inputs)
    if condition.scenarios:
        qb._build_scenario(condition.scenarios)
    if condition.keyword:
        qb._build_keyword(condition.keyword)

    where_clause = " AND ".join(qb._where_clauses) if qb._where_clauses else "1=1"
    query = f"""
    MATCH (f:Fault)
    WHERE {where_clause}
    RETURN count(f) AS total
    """
    return query, qb._params
