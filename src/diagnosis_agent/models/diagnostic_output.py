"""诊断输出 Schema

定义 Agent 最终输出的结构化数据模型，使用 Pydantic 约束输出格式。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiagnosticFinding(BaseModel):
    """诊断发现"""
    title: str = Field(description="发现标题")
    description: str = Field(description="详细描述")
    confidence: float = Field(description="置信度 0.0-1.0")
    evidence: list[str] = Field(description="证据列表")


class DiagnosticResult(BaseModel):
    """诊断结果（Agent 最终输出）

    用于约束 LLM 输出格式，确保返回结构化的诊断结果。
    """
    root_cause: str = Field(description="根本原因分析")
    countermeasure: str = Field(description="推荐对策/解决措施")
    confidence: float = Field(description="置信度 0.0-1.0")
    reasoning_narrative: str = Field(description="完整的推断过程叙述，描述从故障描述到最终结论的完整推理链路")
    findings: list[DiagnosticFinding] = Field(description="诊断发现列表")