"""搜索条件模型

字段与 StandardInput / ParsedInput 完全对齐，零字段翻译。
每个字段有自己独立的匹配通道，不强制拼入 keyword，空字段自动跳过不参与过滤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchCondition:
    """标准化搜索条件 — 与输入层字段 1:1 完全对齐，每个字段独立走自己的匹配通道

    不做任何字段头转换，字段为空时自动跳过该条件，不影响其他字段召回。
    """

    raw_query: str = ""  # 用户原始问题文本（用于语义检索和精排）

    # ========== 独立匹配通道字段 ==========
    # 下面所有字段各走各的匹配通道，互不干扰，空值直接跳过
    vehicleModel: Optional[str] = None          # 车型代号 → Neo4j MotorType.code CONTAINS / Chroma vehicle_type 匹配
    dtcCode: list[str] = field(default_factory=list)  # DTC故障码列表 → Neo4j DTC.code 精确匹配 / Chroma dtc_code 匹配
    softwareVersion: Optional[str] = None      # MCU软件版本 → 补充keyword通道
    motorPosition: Optional[str] = None        # 电机位置枚举值 → 补充keyword通道
    faultWorkConditionList: Optional[str] = None  # 故障工况枚举值 → Neo4j Scenario层级匹配 / Chroma fault_scenario 匹配
    instrumentIndicatorList: list[str] = field(default_factory=list)  # 仪表指示灯列表 → Neo4j Indicator.name 匹配 / Chroma dashboard_indicator 匹配
    VIN: Optional[str] = None                  # VIN码 → 补充keyword通道
    mileage: float = 0.0                       # 行驶里程 → 补充keyword通道

    extras: dict = field(default_factory=dict)

    def to_keyword(self) -> str:
        """生成补充keyword文本 — 只进入keyword通道的字段，独立匹配通道的字段不在这里拼入

        严格过滤掉"无法确认"类无意义枚举值，避免垃圾文本进入检索。
        """
        parts = []

        # 只把没有独立数据库匹配通道的字段拼进去
        if self.softwareVersion:
            parts.append(self.softwareVersion)
        if self.motorPosition and self.motorPosition != "无法确认具体电机":
            parts.append(self.motorPosition)
        if self.VIN:
            parts.append(self.VIN)

        return " ".join(parts)

    def to_rerank_fields(self) -> dict:
        """精排结构化字段字典，字段名与输入层完全一致"""
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
        return result
