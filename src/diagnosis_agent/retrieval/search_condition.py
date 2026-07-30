"""搜索条件模型

按 StandardInput / ParsedInput 的结构定义搜索条件，
不映射到 IncidentRecord 的 8 列表头，也不映射到 Neo4j 节点属性。

目前作为过渡：把输入 JSON 里的字段原样透传到检索层，
Neo4j / Chroma 侧仍按现有 schema 消费这些字段；
等 Neo4j schema 重构后再对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchCondition:
    """标准化搜索条件

    字段直接来自 StandardInput/ParsedInput，检索层消费时不再做字段翻译。
    """

    raw_query: str = ""  # 用户原始问题文本（用于语义检索和精排）
    mcuid: Optional[str] = None
    dtc_code: list[str] = field(default_factory=list)
    project: Optional[str] = None
    component: Optional[str] = None
    working_condition: Optional[str] = None
    software_version: Optional[str] = None

    # 保留原始 entities 字典，方便调试和后续扩展
    extras: dict = field(default_factory=dict)

    def to_keyword(self) -> str:
        """把所有结构化字段拼成 keyword 文本（用于 Neo4j keyword 模糊 / Chroma 兜底）"""
        parts = []
        if self.mcuid:
            parts.append(self.mcuid)
        if self.dtc_code:
            parts.extend(self.dtc_code)
        if self.project:
            parts.append(self.project)
        if self.component:
            parts.append(self.component)
        if self.working_condition:
            parts.append(self.working_condition)
        if self.software_version:
            parts.append(self.software_version)
        return " ".join(parts)

    def to_rerank_fields(self) -> dict:
        """用于精排的结构化字段字典"""
        return {
            "mcuid": self.mcuid,
            "dtc_code": self.dtc_code,
            "project": self.project,
            "component": self.component,
            "working_condition": self.working_condition,
            "software_version": self.software_version,
        }
