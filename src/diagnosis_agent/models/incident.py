"""故障工单数据模型

对齐新 8 列表头：
1. problem_description - 问题描述
2. root_cause          - 根本原因
3. countermeasure      - 对策/解决措施
4. drive_code          - 驱动代码
5. vehicle_type        - 车型
6. dashboard_indicator - 仪表盘指示
7. dtc_code            - DTC 故障码
8. fault_scenario      - 故障场景
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 标准 8 列定义
# ---------------------------------------------------------------------------

INCIDENT_COLUMNS: list[str] = [
    "problem_description",
    "root_cause",
    "countermeasure",
    "drive_code",
    "vehicle_type",
    "dashboard_indicator",
    "dtc_code",
    "fault_scenario",
]

# 中文表头 → 标准英文列名映射（用于精确匹配，LLM 映射的补充）
COLUMN_CN_MAP: dict[str, str] = {
    # problem_description
    "问题描述": "problem_description",
    "问题": "problem_description",
    "故障描述": "problem_description",
    "故障现象": "problem_description",
    "现象": "problem_description",
    "描述": "problem_description",
    # root_cause
    "根本原因": "root_cause",
    "原因": "root_cause",
    "故障原因": "root_cause",
    "根因": "root_cause",
    # countermeasure
    "对策": "countermeasure",
    "解决措施": "countermeasure",
    "措施": "countermeasure",
    "解决方案": "countermeasure",
    "处理方案": "countermeasure",
    "方案": "countermeasure",
    # drive_code
    "电驱代号": "drive_code",
    "驱动代码": "drive_code",
    "驱动码": "drive_code",
    "驱动": "drive_code",
    # vehicle_type
    "车型": "vehicle_type",
    "车辆类型": "vehicle_type",
    "车辆型号": "vehicle_type",
    # dashboard_indicator
    "仪表盘指示": "dashboard_indicator",
    "仪表指示灯": "dashboard_indicator",
    "仪表盘": "dashboard_indicator",
    "仪表指示": "dashboard_indicator",
    "指示灯": "dashboard_indicator",
    # dtc_code
    "DTC码": "dtc_code",
    "故障DTC": "dtc_code",
    "DTC故障码": "dtc_code",
    "故障码": "dtc_code",
    "DTC": "dtc_code",
    # fault_scenario
    "故障场景": "fault_scenario",
    "场景": "fault_scenario",
    "故障情景": "fault_scenario",
}

COLUMN_EN_TO_CN: dict[str, str] = {
    "problem_description": "问题描述",
    "root_cause": "根因",
    "countermeasure": "对策",
    "drive_code": "电驱代号",
    "vehicle_type": "车辆类型",
    "dashboard_indicator": "仪表指示灯",
    "dtc_code": "故障DTC",
    "fault_scenario": "故障场景",
}


class IncidentRecord(BaseModel):
    """故障工单记录 — 对齐新 8 列表头

    """

    problem_description: str = Field("", description="问题描述")
    root_cause: str = Field("", description="根本原因")
    countermeasure: str = Field("", description="对策")
    drive_code: str = Field("", description="电驱代号")
    vehicle_type: str = Field("", description="车型")
    dashboard_indicator: str = Field("", description="仪表指示灯")
    dtc_code: str = Field("", description="故障DTC")
    fault_scenario: str = Field("", description="故障场景")

    def to_searchable_text(self) -> str:
        """生成用于向量索引的可搜索文本"""
        parts = [
            f"问题描述: {self.problem_description}",
            f"根本原因: {self.root_cause}",
            f"对策: {self.countermeasure}",
            f"驱动代码: {self.drive_code}",
            f"车型: {self.vehicle_type}",
            f"仪表盘指示: {self.dashboard_indicator}",
            f"DTC码: {self.dtc_code}",
            f"故障场景: {self.fault_scenario}",
        ]
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """转换为字典（用于 DataFrame / 内部存储，英文键名）"""
        return {
            "problem_description": self.problem_description,
            "root_cause": self.root_cause,
            "countermeasure": self.countermeasure,
            "drive_code": self.drive_code,
            "vehicle_type": self.vehicle_type,
            "dashboard_indicator": self.dashboard_indicator,
            "dtc_code": self.dtc_code,
            "fault_scenario": self.fault_scenario,
        }

    def to_dict_cn(self) -> dict:
        """转换为字典（中文表头，用于数据库写入 / CSV / JSON 输出）"""
        return {
            "问题描述": self.problem_description,
            "根因": self.root_cause,
            "对策": self.countermeasure,
            "电驱代号": self.drive_code,
            "车辆类型": self.vehicle_type,
            "仪表指示灯": self.dashboard_indicator,
            "故障DTC": self.dtc_code,
            "故障场景": self.fault_scenario,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IncidentRecord:
        """从字典创建记录，只提取标准 8 列（支持中文和英文键）"""
        return cls(
            problem_description=str(data.get("problem_description", "") or data.get("问题描述", "")),
            root_cause=str(data.get("root_cause", "") or data.get("根因", "")),
            countermeasure=str(data.get("countermeasure", "") or data.get("对策", "")),
            drive_code=str(data.get("drive_code", "") or data.get("电驱代号", "")),
            vehicle_type=str(data.get("vehicle_type", "") or data.get("车辆类型", "")),
            dashboard_indicator=str(data.get("dashboard_indicator", "") or data.get("仪表指示灯", "")),
            dtc_code=str(data.get("dtc_code", "") or data.get("故障DTC", "")),
            fault_scenario=str(data.get("fault_scenario", "") or data.get("故障场景", "")),
        )
