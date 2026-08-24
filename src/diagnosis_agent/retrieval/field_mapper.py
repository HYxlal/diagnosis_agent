"""字段映射层

ParsedInput → SearchCondition 100% 透传，零字段翻译逻辑。
直接从 ParsedInput 扁平化字段取值，不经过任何 entities 中间对象。
"""

from __future__ import annotations

from ..models.input import ParsedInput
from .search_condition import SearchCondition


class FieldMapper:
    """字段提取器 — 纯透传，零映射

    没有任何字段名翻译，所有字段直接从 ParsedInput 赋值到 SearchCondition，
    字段名完全对齐，路径一目了然。
    """

    @classmethod
    def extract_search_condition(cls, parsed_input: ParsedInput) -> SearchCondition:
        """从 ParsedInput 直接提取搜索条件，纯透传零逻辑

        所有字段直接赋值，不做转换、不做翻译、不做过滤。
        空值由 SearchCondition 侧按需跳过。
        """
        raw_query = parsed_input.search_query or parsed_input.description or ""

        return SearchCondition(
            raw_query=raw_query,
            vehicleModel=parsed_input.vehicleModel or None,
            dtcCode=list(parsed_input.dtcCode or []),
            softwareVersion=parsed_input.softwareVersion or None,
            motorPosition=parsed_input.motorPosition.value,
            faultWorkConditionList=parsed_input.faultWorkConditionList.value,
            instrumentIndicatorList=[item.value for item in parsed_input.instrumentIndicatorList],
            VIN=parsed_input.VIN or None,
            mileage=parsed_input.mileage,
        )
