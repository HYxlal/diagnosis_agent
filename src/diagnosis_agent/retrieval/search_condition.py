"""搜索条件模型

字段与 StandardInput / ParsedInput 完全对齐，零字段翻译。
所有字段默认都有独立匹配通道，不拼入 keyword。
空字段自动跳过，查不到就跳过，不做强过滤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchCondition:
    """标准化搜索条件 — 与输入层字段 1:1 完全对齐，所有字段独立走各自匹配通道

    不做任何字段头转换，不做任何字段拼接，字段为空时自动跳过。
    """

    raw_query: str = ""  # 用户原始问题文本（用于语义检索和精排）

    # ========== 所有字段默认各自独立匹配通道，互不干扰，不拼keyword ==========
    vehicleModel: Optional[str] = None          # → Neo4j MotorType.code CONTAINS / Chroma vehicle_type
    dtcCode: list[str] = field(default_factory=list)  # → Neo4j DTC.code / Chroma dtc_code
    softwareVersion: Optional[str] = None      # → Neo4j Fault.software_version CONTAINS / Chroma software_version
    motorPosition: Optional[str] = None        # → Neo4j Fault.motor_position CONTAINS / Chroma motor_position
    faultWorkConditionList: Optional[str] = None  # → Neo4j Scenario / Chroma fault_scenario
    instrumentIndicatorList: list[str] = field(default_factory=list)  # → Neo4j Indicator.name / Chroma dashboard_indicator
    VIN: Optional[str] = None                  # → Neo4j Fault.vin CONTAINS / Chroma vin
    mileage: float = 0.0                       # → Neo4j Fault.mileage / Chroma mileage

    extras: dict = field(default_factory=dict)

    def to_rerank_fields(self) -> dict:
        """精排结构化字段字典，所有字段独立参与，空值自动跳过"""
        result = {}
        if self.vehicleModel:
            result["vehicleModel"] = self.vehicleModel
        if self.dtcCode:
            result["dtcCode"] = self.dtcCode
        if self.faultWorkConditionList and self.faultWorkConditionList != "无法确认故障工况":
            result["faultWorkConditionList"] = self.faultWorkConditionList
        if self.instrumentIndicatorList:
            valid_indicators = [i for i in self.instrumentIndicatorList if i != "无"]
            if valid_indicators:
                result["instrumentIndicatorList"] = valid_indicators
        if self.softwareVersion:
            result["softwareVersion"] = self.softwareVersion
        if self.motorPosition and self.motorPosition != "无法确认具体电机":
            result["motorPosition"] = self.motorPosition
        if self.VIN:
            result["VIN"] = self.VIN
        if self.mileage > 0:
            result["mileage"] = self.mileage
        return result