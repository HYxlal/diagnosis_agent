"""字段映射层

把 ParsedInput 中的 entities 字段原样透传到 SearchCondition，
不再映射到 IncidentRecord 的 8 列表头，也不映射到 Neo4j 节点属性。

当前目标：搜索条件与输入 JSON 的字段结构保持一致，
让检索层直接消费 dtc_code / project / component / working_condition / software_version。
"""

from __future__ import annotations

from ..models.input import ParsedInput
from .search_condition import SearchCondition


class FieldMapper:
    """ParsedInput → SearchCondition

    只做字段透传，不做字段翻译。
    等后续 Neo4j / Chroma schema 重构完成后再精确映射。
    """

    @classmethod
    def extract_search_condition(cls, parsed_input: ParsedInput) -> SearchCondition:
        """从 ParsedInput 提取搜索条件

        优先用 entities 里的字段；entities 为空时只保留 mcuid / raw_query。
        """
        raw_query = parsed_input.search_query or parsed_input.description or ""

        cond = SearchCondition(
            raw_query=raw_query,
            mcuid=parsed_input.mcuid or None,
        )

        entities = parsed_input.entities
        if entities is None:
            return cond

        cond.dtc_code = list(entities.dtc_code or [])
        cond.project = entities.project or None
        cond.component = entities.component or None
        cond.working_condition = entities.working_condition or None
        cond.software_version = entities.software_version or None

        return cond
