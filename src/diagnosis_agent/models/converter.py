"""输入输出转换层

负责在标准接口模型与内部模型之间进行双向转换：
1. StandardInput → ParsedInput （外部输入转内部解析，方案B：直接传递 entities）
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
    """将标准输入转换为内部解析输入（方案B：entities 直接传递）

    不再构建 IncidentRecord 中间对象，直接将 StandardEntities 赋值给 ParsedInput.entities。
    """
    # 构建检索用 query：mcuid + raw_query（mcuid 作为前缀便于检索时定位）
    mcuid_prefix = f"[{standard_input.mcuid}] " if standard_input.mcuid else ""
    search_query = f"{mcuid_prefix}{standard_input.raw_query}"

    # 初始意图：平台传入的 intent_name 全部先映射为 DIAGNOSTIC_QUERY，
    # 后续由 InputRouter 做二次分类（识别 instruction/supplement/out_of_scope）。
    # intent_map 当前所有 MCU_* 都映射到同一个值，保留映射结构便于后续差异化处理。
    initial_intent = InputIntent.DIAGNOSTIC_QUERY
    if standard_input.intent_name:
        intent_map = {
            "MCU_001": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_002": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_003": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_004": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_005": InputIntent.DIAGNOSTIC_QUERY,
            "MCU_006": InputIntent.DIAGNOSTIC_QUERY,
        }
        initial_intent = intent_map.get(
            standard_input.intent_name,
            InputIntent.DIAGNOSTIC_QUERY,
        )

    return ParsedInput(
        input_type=InputType.NATURAL_LANGUAGE,
        description=standard_input.raw_query,
        raw_input=standard_input.raw_query,
        intent=initial_intent,
        search_query=search_query,
        mcuid=standard_input.mcuid,
        entities=standard_input.entities,
        # field_extraction 不再从 entities 转换，保持 None
    )


# ---------------------------------------------------------------------------
# 输出转换：DiagnosticOutput → StandardOutput
# ---------------------------------------------------------------------------

def diagnostic_output_to_standard(
    diagnostic_output: DiagnosticOutput,
    standard_input: StandardInput,
) -> StandardOutput:
    """将内部诊断结果转换为标准输出

    字段来源优先级：LLM 推理结果（reasoning_result）> 内部报告字段（report/database_entry）。
    LLM 输出为空时，回退到 DiagnosticReport 的 findings / recommended_countermeasure。
    """
    report = diagnostic_output.report
    database_entry = diagnostic_output.database_entry
    reasoning = diagnostic_output.reasoning_result

    # input_ref 回显输入的关键字段，供平台 Agent 核对
    input_ref = {
        "raw_query": standard_input.raw_query,
        "intent_id": standard_input.intent_name or standard_input.mcuid,
        "entities": {
            "dtc_code": standard_input.entities.dtc_code,
            "project": standard_input.entities.project,
        },
    }

    # 根因：优先用 LLM 输出的 fault_root_cause，空则回退到 findings 首条
    fault_root_cause = reasoning.get("fault_root_cause", [])
    if not fault_root_cause and report.findings:
        fault_root_cause = [report.findings[0].description]

    fault_trigger_condition = reasoning.get("fault_trigger_condition", "")

    classification = reasoning.get("classification", "")

    # 解决方案：LLM solution 空 → database_entry.countermeasure → report.recommended_countermeasure
    solution = reasoning.get("solution", [])
    if not solution:
        if database_entry.countermeasure:
            solution = [database_entry.countermeasure]
        elif report.recommended_countermeasure:
            solution = [report.recommended_countermeasure]

    risk_warning = reasoning.get("risk_warning", "")
    maintenance_suggestions = reasoning.get("maintenance_suggestions", "")

    # 相似工况：拼成人类可读的字符串（取前 3 条）
    similar_cases_str = ""
    if report.similar_cases:
        case_descriptions = []
        for sc in report.similar_cases[:3]:
            desc = f"历史案例{sc.record_id}" if sc.record_id else "历史案例"
            if sc.problem_description:
                desc += f"：{sc.problem_description[:50]}"
            case_descriptions.append(desc)
        similar_cases_str = "、".join(case_descriptions)

    # 参考材料：所有相似工况的摘要列表
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
            "intent_id": standard_input.intent_name or standard_input.mcuid,
            "entities": {
                "dtc_code": standard_input.entities.dtc_code,
                "project": standard_input.entities.project,
            },
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

    if not standard_input.mcuid or not standard_input.mcuid.strip():
        return "缺少必填字段: mcuid（MCU标识）不能为空"

    return None
