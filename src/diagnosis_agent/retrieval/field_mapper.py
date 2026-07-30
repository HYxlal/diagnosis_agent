"""字段映射层

把 ParsedInput.entities 里的字段映射到 Neo4j 召回用的结构化字段。

当前映射关系（业务现状）：
- mcuid → motor_codes（mcuid 是平台编码，MotorType.code 是电驱代号如 L200A-2，
  当前直接透传，后续可在此处加映射表）
- entities.dtc_code → dtc_codes
- entities.project → vehicle_types（项目编码 → 车辆类型）
- entities.working_condition → scenarios
- entities.component → component（keyword 模糊匹配 description）

重构字段映射时，只动这个文件。比如 mcuid → MotorType.code 映射表确定后，
在 _map_mcuid() 里加 lookup 即可，其他调用方无感知。
"""

from __future__ import annotations

from typing import Any

from ..models.input import ParsedInput


class FieldMapper:
    """ParsedInput → Neo4j 结构化字段映射"""

    # mcuid → MotorType.code 映射表（待业务确认后填充）
    # 当前空字典表示直接透传
    MCU_TO_MOTOR: dict[str, str] = {}

    # working_condition → Scenario 三级结构映射表（待业务确认后填充）
    # key 是自然语言描述，value是"category-subcategory-detail" 格式
    WORKING_CONDITION_TO_SCENARIO: dict[str, str] = {}

    @classmethod
    def extract_structural_fields(cls, parsed_input: ParsedInput) -> dict[str, Any]:
        """从 ParsedInput 提取结构化字段，供召回和精排使用

        Returns:
            dict，key 可能包含：
            mcuid, dtc_codes, scenarios, indicators, vehicle_types, component
            entities 为 None 时只返回 mcuid。
        """
        fields: dict[str, Any] = {}

        if parsed_input.mcuid:
            fields["mcuid"] = cls._map_mcuid(parsed_input.mcuid)

        entities = parsed_input.entities
        if entities is None:
            return fields

        if entities.dtc_code:
            fields["dtc_codes"] = list(entities.dtc_code)

        if entities.project:
            # project 语义上对应车辆/车型代号，暂当 vehicle_types
            # 数据字段重构后再精确映射
            fields["vehicle_types"] = [entities.project]

        if entities.working_condition:
            mapped = cls._map_working_condition(entities.working_condition)
            fields["scenarios"] = [mapped] if mapped else []

        if entities.component:
            fields["component"] = entities.component

        return fields

    @classmethod
    def _map_mcuid(cls, mcuid: str) -> str:
        """mcuid → MotorType.code

        当前映射表为空，直接透传。
        确定映射关系后填 MCU_TO_MOTOR 即可。
        """
        return cls.MCU_TO_MOTOR.get(mcuid, mcuid)

    @classmethod
    def _map_working_condition(cls, working_condition: str) -> str:
        """working_condition → Scenario 三级结构

        当前映射表为空，直接透传（走 Scenario 模糊匹配）。
        确定映射关系后填 WORKING_CONDITION_TO_SCENARIO，
        比如 "爬坡满载、高温" → "工况-动态-激烈驾驶"
        """
        return cls.WORKING_CONDITION_TO_SCENARIO.get(working_condition, working_condition)
