"""诊断输出 Schema

定义 Agent 最终输出的结构化数据模型，使用 Pydantic 约束输出格式。
包含内部使用的 DiagnosticResult 和对外的 StandardOutput。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .diagnosis import DiagnosticFinding


class DiagnosticResult(BaseModel):
    """LLM 诊断结果（内部使用）

    用于约束 LLM 输出格式，确保返回结构化的诊断结果。
    字段对应对外的 StandardDiagnosisResult。
    """
    fault_root_cause: list[str] = Field(
        default_factory=list,
        description="故障根本原因列表",
    )
    fault_trigger_condition: str = Field(
        "",
        description="故障触发条件",
    )
    classification: str = Field(
        "",
        description="故障分类（从指定列表中选择）",
    )
    solution: list[str] = Field(
        default_factory=list,
        description="结构化解决方案列表",
    )
    risk_warning: str = Field(
        "",
        description="风险预警等级（V1/V2/V3）",
    )
    maintenance_suggestions: str = Field(
        "",
        description="维护建议",
    )
    confidence: float = Field(description="置信度 0.0-1.0")
    reasoning_narrative: str = Field(description="完整的推断过程叙述")
    findings: list[DiagnosticFinding] = Field(default_factory=list, description="诊断发现列表")


# ---------------------------------------------------------------------------
# 标准输出接口（对外输出给平台 Agent）
# ---------------------------------------------------------------------------

class OutputCode(IntEnum):
    """标准输出状态码

    0: 正常返回
    -1: 入参缺失关键信息
    -2: 内部推理异常
    -3: 不属于电驱系统范畴
    """
    SUCCESS = 0
    MISSING_INPUT = -1
    INTERNAL_ERROR = -2
    OUT_OF_SCOPE = -3


# 故障分类枚举（固定选项）
FaultClassification = Literal[
    "驱动异常故障",
    "控制异常故障",
    "超速故障",
    "高压异常故障",
    "低压异常故障",
    "过温故障",
    "通信故障",
    "旋变故障",
    "状态机故障",
    "油泵故障",
]


class StandardDiagnosisResult(BaseModel):
    """核心诊断结果（对外输出）

    严格按照需求文档定义的结构。
    """
    fault_root_cause: list[str] = Field(
        default_factory=list,
        description="故障根本原因列表",
    )
    fault_trigger_condition: str = Field(
        "",
        description="故障触发条件，如'满载爬坡、环境温度＞35℃'",
    )
    classification: str = Field(
        "",
        description="故障分类（从固定列表中选择）",
    )
    solution: list[str] = Field(
        default_factory=list,
        description="结构化解决方案列表",
    )
    risk_warning: str = Field(
        "",
        description="风险预警等级，如 V1/V2/V3",
    )
    maintenance_suggestions: str = Field(
        "",
        description="维护建议",
    )
    similar_cases: str = Field(
        "",
        description="相似工况描述，如'历史案例XXX'",
    )


class StandardOutput(BaseModel):
    """电驱 Agent 标准输出（对外接口）

    严格按照需求文档定义的结构。
    """
    code: int = Field(
        ...,
        description="状态码: 0=正常, -1=入参缺失, -2=内部异常, -3=超出范围",
    )
    msg: str = Field(..., description="状态描述")
    input_ref: dict = Field(
        ...,
        description="输入引用信息（回显关键输入字段）",
    )
    diagnosis_result: Optional[StandardDiagnosisResult] = Field(
        None,
        description="核心诊断结果（成功时填充）",
    )
    reference_material: list[str] = Field(
        default_factory=list,
        description="参考材料列表",
    )
    need_multi_round: bool = Field(
        False,
        description="是否需要追问用户补充信息",
    )
    follow_up_question: str = Field(
        "",
        description="追问问题（need_multi_round=true 时填充）",
    )
    diagnosis_confidence: float = Field(
        0.0,
        ge=0,
        le=1,
        description="本次诊断结论置信度",
    )