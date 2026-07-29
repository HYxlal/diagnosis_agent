"""诊断输出数据模型 — 双层输出

第一层：Markdown 诊断报告（给人看）
第二层：可录入数据库的结构化条目（给系统用）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .incident import IncidentRecord


# ---------------------------------------------------------------------------
# ReAct 推理过程结构化记录
# ---------------------------------------------------------------------------

class ReActStep(BaseModel):
    """ReAct 单步推理记录

    对应 ReAct 循环的一次 Thought → Action → Observation 迭代。
    """
    step: int = Field(..., description="步骤序号（从1开始）")
    thought: str = Field("", description="Agent 的思考内容")
    action: str = Field("", description="选择的工具名称")
    action_input: dict[str, Any] = Field(default_factory=dict, description="工具输入参数")
    observation: str = Field("", description="工具返回结果")
    timestamp: datetime = Field(default_factory=datetime.now, description="本步骤时间")


class ToolCallRecord(BaseModel):
    """工具调用详细记录

    记录每次工具调用的完整信息：参数、返回值、耗时。
    """
    tool_name: str = Field(..., description="工具名称")
    parameters: dict[str, Any] = Field(..., description="调用参数")
    result: Any = Field(None, description="返回结果")
    duration_ms: float = Field(0.0, description="耗时（毫秒）")
    timestamp: datetime = Field(default_factory=datetime.now, description="调用时间")


# ---------------------------------------------------------------------------
# 相似工况与诊断发现
# ---------------------------------------------------------------------------

class SimilarCase(BaseModel):
    """检索到的相似工况"""
    record_id: str = Field("", description="向量库记录ID")
    problem_description: str = Field("", description="问题描述")
    root_cause: str = Field("", description="根本原因")
    countermeasure: str = Field("", description="对策/解决措施")
    drive_code: str = Field("", description="驱动代码")
    vehicle_type: str = Field("", description="车型")
    dashboard_indicator: str = Field("", description="仪表盘指示")
    dtc_code: str = Field("", description="DTC 故障码")
    fault_scenario: str = Field("", description="故障场景")
    similarity: float = Field(..., description="相似度分数 0-1")


class DiagnosticFinding(BaseModel):
    """诊断结论中的单条发现"""
    title: str = Field(..., description="发现标题")
    description: str = Field(..., description="详细描述")
    confidence: float = Field(..., description="置信度 0-1")
    evidence: list[str] = Field(default_factory=list, description="支持证据")


# ---------------------------------------------------------------------------
# 诊断报告与数据库条目
# ---------------------------------------------------------------------------

class DiagnosticReport(BaseModel):
    """诊断报告 — 第一层输出"""
    diagnosis_id: str = Field(..., description="诊断会话ID")
    diagnosis_time: datetime = Field(default_factory=datetime.now, description="诊断时间")
    input_summary: str = Field("", description="输入摘要")
    has_similar_cases: bool = Field(False, description="是否找到相似工况")
    similar_cases: list[SimilarCase] = Field(default_factory=list, description="相似工况列表")
    findings: list[DiagnosticFinding] = Field(default_factory=list, description="诊断发现")
    recommended_countermeasure: str = Field("", description="推荐对策")

    # 推理过程（结构化 ReAct 步骤 + 叙述性推理过程）
    react_steps: list[ReActStep] = Field(default_factory=list, description="ReAct 每步推理记录")
    reasoning_narrative: str = Field("", description="完整的推断过程叙述")
    reasoning_chain: list[str] = Field(default_factory=list, description="推理链（兼容旧字段，从 react_steps 提取）")

    # 工具调用记录
    tool_calls: list[ToolCallRecord] = Field(default_factory=list, description="工具调用详细记录")
    tools_used: list[str] = Field(default_factory=list, description="使用的工具名称列表")

    agent_version: str = Field("0.3.0", description="Agent 版本")


class DatabaseEntry(BaseModel):
    """可录入数据库的结构化条目 — 第二层输出

    基于标准 8 列表头，填充诊断结果。
    8 列业务字段：problem_description / root_cause / countermeasure /
    drive_code / vehicle_type / dashboard_indicator / dtc_code / fault_scenario。
    """
    diagnosis_id: str = Field(..., description="诊断会话ID")
    diagnosis_time: datetime = Field(default_factory=datetime.now, description="诊断时间")

    # 新 8 列（诊断结果填充）
    problem_description: str = Field("", description="问题描述")
    root_cause: str = Field("", description="根本原因")
    countermeasure: str = Field("", description="对策/解决措施")
    drive_code: str = Field("", description="驱动代码")
    vehicle_type: str = Field("", description="车型")
    dashboard_indicator: str = Field("", description="仪表盘指示")
    dtc_code: str = Field("", description="DTC 故障码")
    fault_scenario: str = Field("", description="故障场景")

    # 诊断元数据
    diagnostic_confidence: float = Field(0.0, description="诊断置信度")
    based_on_similar: bool = Field(False, description="是否基于相似工况")
    similar_record_ids: list[str] = Field(default_factory=list, description="相似记录ID列表")

    def to_dict(self) -> dict:
        """转换为字典（用于数据库写入）

        8 列业务字段使用中文表头，与数据库列名对齐；
        元数据字段（diagnosis_id/confidence 等）保持英文。
        """
        return {
            "diagnosis_id": self.diagnosis_id,
            "diagnosis_time": self.diagnosis_time.isoformat(),
            "问题描述": self.problem_description,
            "根因": self.root_cause,
            "对策": self.countermeasure,
            "电驱代号": self.drive_code,
            "车辆类型": self.vehicle_type,
            "仪表指示灯": self.dashboard_indicator,
            "故障DTC": self.dtc_code,
            "故障场景": self.fault_scenario,
            "diagnostic_confidence": self.diagnostic_confidence,
            "based_on_similar": self.based_on_similar,
            "similar_record_ids": "|".join(self.similar_record_ids),
        }


class DiagnosticOutput(BaseModel):
    """双层输出的聚合模型

    - report：人类可读的诊断报告（含 ReAct 步骤、工具调用、相似工况等）
    - database_entry：机器可录入的结构化条目（标准 8 列 + 元数据）
    - reasoning_result：LLM 原始推理结果（dict，含 fault_root_cause /
      classification / solution 等新结构字段，供 converter 转标准输出时使用）
    """
    report: DiagnosticReport = Field(..., description="第一层：诊断报告")
    database_entry: DatabaseEntry = Field(..., description="第二层：可录入条目")
    reasoning_result: dict = Field(
        default_factory=dict,
        description="LLM 推理结果（包含新结构字段: fault_root_cause, classification, solution 等）",
    )
