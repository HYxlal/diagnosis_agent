"""输入输出转换层

负责在标准接口模型与内部模型之间进行双向转换：
1. StandardInput → ParsedInput （外部输入转内部解析，扁平直传）
2. DiagnosticOutput → StandardOutput （内部诊断结果转标准输出）
"""
from __future__ import annotations

import logging
from typing import Optional

from .diagnosis import DiagnosticOutput
from .diagnostic_output import (
    OutputCode,
    StandardDiagnosisResult,
    StandardOutput,
)
from .input import (
    InputIntent,
    InputType,
    ParsedInput,
    StandardInput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 输入转换：StandardInput → ParsedInput
# ---------------------------------------------------------------------------

def standard_input_to_parsed(
    standard_input: StandardInput,
) -> ParsedInput:
    """将标准输入转换为内部解析输入 — 扁平直传，不经过中间对象"""
    # 构建检索用 query：vehicleModel + raw_query
    prefix = f"[{standard_input.vehicleModel}] " if standard_input.vehicleModel else ""
    search_query = f"{prefix}{standard_input.raw_query}"

    # 确定初始意图
    initial_intent = InputIntent.DIAGNOSTIC_QUERY
    if standard_input.intentId:
        intent_map = {
            "MCU_001": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_002": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_003": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_004": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_005": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_006": InputIntent.DIAGNOSTIC_QUERY,
        }
        initial_intent = intent_map.get(
            standard_input.intentId,
            InputIntent.DIAGNOSTIC_QUERY,
        )

    return ParsedInput(
        input_type=InputType.NATURAL_LANGUAGE,
        description=standard_input.raw_query,
        raw_input=standard_input.raw_query,
        intent=initial_intent,
        search_query=search_query,
        vehicleModel=standard_input.vehicleModel,
        VIN=standard_input.VIN,
        faultOccurTime=standard_input.faultOccurTime,
        mileage=standard_input.mileage,
        faultWorkConditionList=standard_input.faultWorkConditionList,
        instrumentIndicatorList=standard_input.instrumentIndicatorList,
        dtcCode=standard_input.dtcCode,
        softwareVersion=standard_input.softwareVersion,
        motorPosition=standard_input.motorPosition,
        faultDataUrl=standard_input.faultDataUrl,
        intentId=standard_input.intentId,
        query_confidence=standard_input.query_confidence,
        conversationId=standard_input.conversationId,
        context_history=standard_input.context_history,
    )


# ---------------------------------------------------------------------------
# 输出转换：DiagnosticOutput → StandardOutput
# ---------------------------------------------------------------------------

def diagnostic_output_to_standard(
    diagnostic_output: DiagnosticOutput,
    standard_input: StandardInput,
) -> StandardOutput:
    """将内部诊断结果转换为标准输出"""
    report = diagnostic_output.report
    database_entry = diagnostic_output.database_entry
    reasoning = diagnostic_output.reasoning_result

    # 构建输入回显
    input_ref = {
        "raw_query": standard_input.raw_query,
        "intent_id": standard_input.intentId or standard_input.vehicleModel,
        "VIN": standard_input.VIN,
        "vehicleModel": standard_input.vehicleModel,
        "dtcCode": standard_input.dtcCode,
        "faultOccurTime": standard_input.faultOccurTime,
        "mileage": standard_input.mileage,
    }

    # 从推理结果中提取字段
    fault_root_cause = reasoning.get("fault_root_cause", [])
    if not fault_root_cause and report.findings:
        fault_root_cause = [report.findings[0].description]

    fault_trigger_condition = reasoning.get("fault_trigger_condition", "")
    classification = reasoning.get("classification", "")

    solution = reasoning.get("solution", [])
    if not solution:
        if database_entry.countermeasure:
            solution = [database_entry.countermeasure]
        elif report.recommended_countermeasure:
            solution = [report.recommended_countermeasure]

    risk_warning = reasoning.get("risk_warning", "")
    maintenance_suggestions = reasoning.get("maintenance_suggestions", "")

    # 构建相似工况描述字符串
    similar_cases_str = ""
    if report.similar_cases:
        case_descriptions = []
        for sc in report.similar_cases[:3]:
            desc = f"历史案例{sc.record_id}" if sc.record_id else "历史案例"
            if sc.problem_description:
                desc += f"：{sc.problem_description[:50]}"
            case_descriptions.append(desc)
        similar_cases_str = "、".join(case_descriptions)

    # 构建参考材料
    reference_material = []
    for sc in report.similar_cases:
        if sc.problem_description:
            reference_material.append(
                f"[相似工况] {sc.problem_description[:80]}... (根因: {sc.root_cause[:50]})"
            )

    diagnosis_result = StandardDiagnosisResult(
        fault_root_cause=fault_root_cause,
        fault_trigger_condition=fault_trigger_condition,
        classification=classification,
        solution=solution,
        risk_warning=risk_warning,
        maintenance_suggestions=maintenance_suggestions,
        similar_cases=similar_cases_str,
    )

    confidence = database_entry.diagnostic_confidence

    return StandardOutput(
        code=OutputCode.SUCCESS.value,
        msg="诊断推理完成",
        input_ref=input_ref,
        diagnosis_result=diagnosis_result,
        reference_material=reference_material,
        need_multi_round=False,
        follow_up_question="",
        diagnosis_confidence=confidence,
    )


def build_error_output(
    code: OutputCode,
    msg: str,
    standard_input: Optional[StandardInput] = None,
) -> StandardOutput:
    """构建错误状态的标准输出"""
    input_ref = {}
    if standard_input:
        input_ref = {
            "raw_query": standard_input.raw_query,
            "intent_id": standard_input.intentId or standard_input.vehicleModel,
            "VIN": standard_input.VIN,
            "vehicleModel": standard_input.vehicleModel,
            "dtcCode": standard_input.dtcCode,
        }

    return StandardOutput(
        code=code.value,
        msg=msg,
        input_ref=input_ref,
        diagnosis_result=None,
        reference_material=[],
        need_multi_round=False,
        follow_up_question="",
        diagnosis_confidence=0.0,
    )


def validate_standard_input(standard_input: StandardInput) -> Optional[str]:
    """验证标准输入的关键字段"""
    if not standard_input.raw_query or not standard_input.raw_query.strip():
        return "缺少必填字段: raw_query（用户原始提问文本）不能为空"

    if not standard_input.VIN or not standard_input.VIN.strip():
        return "缺少必填字段: VIN（VIN码）不能为空"

    if not standard_input.vehicleModel or not standard_input.vehicleModel.strip():
        return "缺少必填字段: vehicleModel（车型代号）不能为空"

    if not standard_input.faultOccurTime or not standard_input.faultOccurTime.strip():
        return "缺少必填字段: faultOccurTime（故障发生时间）不能为空"

    if standard_input.mileage < 0:
        return "mileage（里程）不能为负数"

    return None