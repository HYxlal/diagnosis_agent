"""LangChain Agent 实现

基于 LangChain 最新架构的诊断 Agent，使用多源检索。
支持两种输入模式：
1. 传统模式：ParsedInput（内部解析输入）
2. 标准接口模式：StandardInput（平台 Agent 传入的标准 JSON）
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever

from ..config import Settings
from ..models.converter import (
    build_error_output,
    diagnostic_output_to_standard,
    standard_input_to_parsed,
    validate_standard_input,
)
from ..models.diagnostic_output import (
    DiagnosticResult,
    OutputCode,
    StandardOutput,
)
from ..models.diagnosis import (
    DatabaseEntry,
    DiagnosticFinding,
    DiagnosticOutput,
    DiagnosticReport,
    SimilarCase,
    ToolCallRecord,
    ReActStep,
)
from ..models.input import (
    InputIntent,
    ParsedInput,
    StandardInput,
)
from ..models.incident import IncidentRecord
from ..retrieval.langchain_retrievers import (
    ChromaVectorRetriever,
    document_to_record,
    create_chroma_retriever,
)
from ..utils.llm_factory import create_llm
from .prompts import (
    build_similar_case_prompt,
    build_no_similar_case_prompt,
    format_similar_cases_for_prompt,
)
from .tools import DiagnosticTools

logger = logging.getLogger(__name__)


class LangChainDiagnosticAgent:
    """基于 LangChain 最新架构的诊断 Agent

    使用多源检索：
    - ChromaVectorRetriever: 向量语义检索
    - BM25KeywordRetriever: 关键词检索（可选）
    - Neo4jGraphRetriever: 知识图谱检索（可选）

    使用 langchain.agents.create_agent 构建 Agent。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = self._init_llm()
        self._retriever = self._init_retriever()
        self._tools = DiagnosticTools(
            retriever=self._retriever,
            settings=settings,
        )
        self._agent = self._build_agent()
        self._last_diagnostic_output: Optional[DiagnosticOutput] = None

    def _build_agent(self):
        """构建并缓存 Agent"""
        system_prompt = self._build_system_prompt()
        tools = self._tools.get_tool_list()
        return create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def _init_llm(self):
        """初始化 LLM"""
        return create_llm()

    def _init_retriever(self) -> BaseRetriever:
        """初始化检索器"""
        return create_chroma_retriever()

    def register_working_condition_converter(self, converter) -> None:
        """注册工况文件转换工具"""
        self._tools.register_working_condition_converter(converter)

    def diagnose_with_standard_input(
        self,
        standard_input: StandardInput,
    ) -> StandardOutput:
        """使用标准输入接口执行诊断

        这是对外暴露的主入口方法，接收平台 Agent 传入的标准 JSON。

        流程：
        1. 验证输入
        2. 转换为内部 ParsedInput
        3. 调用内部 diagnose 方法
        4. 转换为标准输出
        """
        # 1. 验证输入
        validation_error = validate_standard_input(standard_input)
        if validation_error:
            logger.error(f"标准输入验证失败: {validation_error}")
            return build_error_output(
                code=OutputCode.MISSING_INPUT,
                msg=validation_error,
                standard_input=standard_input,
            )

        # 2. 转换为内部 ParsedInput
        try:
            parsed_input = standard_input_to_parsed(standard_input)
        except Exception as e:
            logger.error(f"输入转换失败: {e}")
            return build_error_output(
                code=OutputCode.INTERNAL_ERROR,
                msg=f"输入转换失败: {str(e)}",
                standard_input=standard_input,
            )

        # 3. 执行意图路由和诊断
        try:
            from ..agent.input_router import InputRouter
            router = InputRouter(self.settings)
            parsed_input = router.route(parsed_input)

            # 检查是否为非电驱问题
            if parsed_input.intent == InputIntent.OUT_OF_SCOPE:
                logger.info("识别为非电驱系统问题，返回 -3 状态码")
                return build_error_output(
                    code=OutputCode.OUT_OF_SCOPE,
                    msg="识别为非电驱系统问题，不执行诊断",
                    standard_input=standard_input,
                )

            diagnostic_output = self.diagnose(parsed_input)
        except Exception as e:
            logger.error(f"诊断执行失败: {e}")
            return build_error_output(
                code=OutputCode.INTERNAL_ERROR,
                msg=f"诊断执行失败: {str(e)}",
                standard_input=standard_input,
            )

        # 4. 转换为标准输出
        try:
            standard_output = diagnostic_output_to_standard(
                diagnostic_output,
                standard_input,
            )
            return standard_output
        except Exception as e:
            logger.error(f"输出转换失败: {e}")
            return build_error_output(
                code=OutputCode.INTERNAL_ERROR,
                msg=f"输出转换失败: {str(e)}",
                standard_input=standard_input,
            )

    def diagnose(self, parsed_input: ParsedInput) -> DiagnosticOutput:
        """执行完整诊断流程（内部方法）"""
        description = parsed_input.description or ""

        diagnosis_id = f"DIAG-{uuid.uuid4().hex[:8]}"
        diagnosis_time = datetime.now()

        logger.info(f"开始诊断 [{diagnosis_id}]: intent={parsed_input.intent.value}, {description[:100]}...")

        if parsed_input.intent == InputIntent.WORKING_CONDITION_FILE:
            logger.info("检测到工况文件意图，先调用转换工具")
            convert_result = self._tools.convert_working_condition_file(parsed_input.source_file or "")
            if convert_result.get("description"):
                description = convert_result["description"]
                parsed_input.description = description
            if convert_result.get("records"):
                parsed_input.bulk_records = convert_result["records"]
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY

        if parsed_input.intent == InputIntent.DIAGNOSTIC_QUERY:
            similar_cases = self._retrieve_similar_cases(parsed_input)
        else:
            similar_cases = []

        has_similar = len(similar_cases) > 0
        logger.info(f"初始检索到 {len(similar_cases)} 条相似工况")

        reasoning_result, tool_calls, react_steps = self._run_agent(
            description=description,
            similar_cases=similar_cases,
            has_similar=has_similar,
        )

        findings = self._build_findings(reasoning_result, similar_cases)

        similar_case_models = [
            SimilarCase(
                record_id=c.get("record_id", ""),
                problem_description=c.get("problem_description", ""),
                root_cause=c.get("root_cause", ""),
                countermeasure=c.get("countermeasure", ""),
                drive_code=c.get("drive_code", ""),
                vehicle_type=c.get("vehicle_type", ""),
                dashboard_indicator=c.get("dashboard_indicator", ""),
                dtc_code=c.get("dtc_code", ""),
                fault_scenario=c.get("fault_scenario", ""),
                similarity=c.get("similarity", 0.0),
            )
            for c in similar_cases
        ]

        tools_used = list(set(tc.tool_name for tc in tool_calls))

        # 从新格式的推理结果中提取字段
        root_causes = reasoning_result.get("fault_root_cause", [])
        root_cause_text = root_causes[0] if root_causes else ""
        solutions = reasoning_result.get("solution", [])
        countermeasure_text = solutions[0] if solutions else ""

        report = DiagnosticReport(
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            input_summary=description[:500],
            has_similar_cases=has_similar,
            similar_cases=similar_case_models,
            findings=findings,
            recommended_countermeasure=countermeasure_text,
            react_steps=react_steps,
            reasoning_narrative=reasoning_result.get("reasoning_narrative", ""),
            reasoning_chain=[s.thought for s in react_steps if s.thought],
            tool_calls=tool_calls,
            tools_used=tools_used,
        )

        confidence = reasoning_result.get("confidence", 0.3)

        # 从 entities / field_extraction / bulk_records 中获取字段
        entities = parsed_input.entities
        field_extraction = parsed_input.field_extraction

        problem_desc = description[:500]
        if field_extraction and field_extraction.problem_description:
            problem_desc = field_extraction.problem_description

        dtc_code = ""
        if entities and entities.dtc_code:
            dtc_code = ", ".join(entities.dtc_code)
        elif field_extraction and field_extraction.dtc_code:
            dtc_code = field_extraction.dtc_code
        else:
            dtc_code = self._extract_from_records(parsed_input, "dtc_code")

        vehicle_type = ""
        if entities and entities.project:
            vehicle_type = entities.project
        elif field_extraction and field_extraction.vehicle_type:
            vehicle_type = field_extraction.vehicle_type
        else:
            vehicle_type = self._extract_from_records(parsed_input, "vehicle_type")

        component = ""
        if entities and entities.component:
            component = entities.component

        fault_scenario = ""
        if entities and entities.working_condition:
            fault_scenario = entities.working_condition
        elif field_extraction and field_extraction.fault_scenario:
            fault_scenario = field_extraction.fault_scenario
        else:
            fault_scenario = self._extract_from_records(parsed_input, "fault_scenario")

        drive_code = ""
        if field_extraction and field_extraction.drive_code:
            drive_code = field_extraction.drive_code
        else:
            drive_code = self._extract_from_records(parsed_input, "drive_code")

        dashboard_indicator = ""
        if field_extraction and field_extraction.dashboard_indicator:
            dashboard_indicator = field_extraction.dashboard_indicator
        else:
            dashboard_indicator = self._extract_from_records(parsed_input, "dashboard_indicator")

        database_entry = DatabaseEntry(
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            problem_description=problem_desc,
            root_cause=root_cause_text,
            countermeasure=countermeasure_text,
            drive_code=drive_code,
            vehicle_type=vehicle_type,
            dashboard_indicator=dashboard_indicator,
            dtc_code=dtc_code,
            fault_scenario=fault_scenario,
            diagnostic_confidence=confidence,
            based_on_similar=has_similar,
            similar_record_ids=[c.get("record_id", "") for c in similar_cases],
        )

        diagnostic_output = DiagnosticOutput(
            report=report,
            database_entry=database_entry,
            reasoning_result=reasoning_result,
        )
        self._last_diagnostic_output = diagnostic_output
        return diagnostic_output

    def _run_agent(
        self,
        description: str,
        similar_cases: list[dict],
        has_similar: bool,
    ) -> tuple[dict, list[ToolCallRecord], list[ReActStep]]:
        """使用缓存的 Agent 执行循环"""
        if has_similar:
            cases_text = format_similar_cases_for_prompt(similar_cases)
            user_prompt = build_similar_case_prompt(description=description, similar_cases_text=cases_text)
        else:
            user_prompt = build_no_similar_case_prompt(description=description)

        logger.info("开始 Agent 循环")
        try:
            result = self._agent.invoke({
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            })
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            return {}, [], []

        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None

        tool_calls = self._extract_tool_calls(messages)
        react_steps = self._extract_react_steps(messages)
        reasoning_result = self._parse_final_result(last_message)

        return reasoning_result, tool_calls, react_steps

    def _build_system_prompt(self) -> str:
        """构建系统 prompt"""
        return f"""你是一个专业的车辆故障诊断专家 Agent，专注于电驱系统（MCU/电机/逆变器）故障诊断。

你的任务：分析故障描述，调用工具检索历史工单，输出结构化的诊断结论。

## 故障分类选项（必须从以下列表中选择一个）

- 驱动异常故障
- 控制异常故障
- 超速故障
- 高压异常故障
- 低压异常故障
- 过温故障
- 通信故障
- 旋变故障
- 状态机故障
- 油泵故障

## 诊断推理原则

1. **理解问题**：仔细分析故障描述，识别关键信息（车型、DTC码、仪表盘指示、驱动代码等）
2. **检索相似工况**：使用 search_similar_incidents 工具搜索历史工单
3. **对比分析**：将当前故障与历史工单对比，分析异同
4. **推理根因**：基于证据推理可能的根本原因，排除不合理的假设
5. **结构化输出**：
   - fault_root_cause：列出2-3个最可能的具体原因
   - solution：列出可执行的解决步骤
   - classification：从故障分类选项中选择最匹配的分类
   - risk_warning：评估故障的风险等级

## 最终答案格式

当你得到足够信息后，必须输出以下 JSON 格式的最终结果：

```json
{{
  "fault_root_cause": ["具体原因1", "具体原因2", "具体原因3"],
  "fault_trigger_condition": "故障触发条件描述",
  "classification": "从故障分类选项中选择一个",
  "solution": ["解决方案步骤1", "解决方案步骤2", "解决方案步骤3"],
  "risk_warning": "风险预警等级（V1=高风险/V2=中风险/V3=低风险）",
  "maintenance_suggestions": "长期维护建议",
  "confidence": 0.85,
  "reasoning_narrative": "完整的推断过程叙述（200-500字）",
  "findings": [
    {{
      "title": "发现标题",
      "description": "详细描述",
      "confidence": 0.9,
      "evidence": ["证据1", "证据2"]
    }}
  ]
}}
```

注意：所有字段都必须填充，不可为空数组或空字符串。
classification 必须从给定的故障分类选项中选择。
confidence 反映你对诊断结论的把握程度，0.9 以上为非常确定，0.5 以下为推测。"""

    def _extract_tool_calls(self, messages: list[BaseMessage]) -> list[ToolCallRecord]:
        """从消息列表中提取工具调用记录，关联 ToolMessage 结果"""
        tool_calls = []
        tool_call_map = {}

        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get('id', '')
                    tc_record = ToolCallRecord(
                        tool_name=tc.get('name', ''),
                        parameters=tc.get('args', {}),
                        result={},
                        duration_ms=0,
                    )
                    if tc_id:
                        tool_call_map[tc_id] = tc_record
                    tool_calls.append(tc_record)

        for msg in messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id'):
                tc_id = msg.tool_call_id
                if tc_id in tool_call_map:
                    result_content = msg.content
                    if isinstance(result_content, str):
                        try:
                            result_content = json.loads(result_content)
                        except json.JSONDecodeError:
                            pass
                    tool_call_map[tc_id].result = result_content if result_content else {}

        return tool_calls

    def _extract_react_steps(self, messages: list[BaseMessage]) -> list[ReActStep]:
        """从消息列表中提取推理步骤，区分消息类型

        处理逻辑：
        - AIMessage + tool_calls → Thought + Action，等待 ToolMessage
        - ToolMessage → 填充上一步的 Observation
        - AIMessage (无 tool_calls) → 纯 Thought 步骤
        - HumanMessage → 跳过
        """
        steps = []
        step_num = 1
        pending_step = None

        for msg in messages:
            if isinstance(msg, HumanMessage):
                continue

            elif isinstance(msg, AIMessage):
                content = getattr(msg, 'content', '')
                if not content or isinstance(content, dict) or content == 'null':
                    continue

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    action_name = msg.tool_calls[0].get('name', '') if msg.tool_calls else ''
                    action_input = msg.tool_calls[0].get('args', {}) if msg.tool_calls else {}
                    pending_step = ReActStep(
                        step=step_num,
                        thought=content,
                        action=action_name,
                        action_input=action_input,
                        observation='',
                    )
                    steps.append(pending_step)
                    step_num += 1
                else:
                    steps.append(ReActStep(
                        step=step_num,
                        thought=content,
                        action='',
                        action_input={},
                        observation='',
                    ))
                    step_num += 1

            elif isinstance(msg, ToolMessage):
                if pending_step is not None and pending_step.action:
                    result_content = msg.content
                    if isinstance(result_content, str):
                        try:
                            result_content = json.loads(result_content)
                        except json.JSONDecodeError:
                            pass
                    observation = str(result_content) if result_content else ''
                    pending_step.observation = observation
                    pending_step = None

        return steps

    def _parse_final_result(self, last_message) -> dict:
        """解析最终结果，使用 Pydantic 模型验证"""
        if not last_message:
            return self._get_default_result()

        content = getattr(last_message, 'content', '')

        try:
            if isinstance(content, dict):
                data = content
            else:
                import re
                json_match = re.search(r'\{[\s\S]*\}', str(content))
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return self._get_default_result()

            result = DiagnosticResult(**data)
            return result.dict()

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"解析最终结果失败: {e}")
            return self._get_default_result()

    def _get_default_result(self) -> dict:
        """返回默认结果"""
        return {
            "fault_root_cause": ["诊断过程中未能获取完整根因"],
            "fault_trigger_condition": "",
            "classification": "",
            "solution": ["请查看推理叙述获取详细方案"],
            "risk_warning": "",
            "maintenance_suggestions": "",
            "confidence": 0.3,
            "reasoning_narrative": "诊断过程中未能获取完整的结构化输出",
            "findings": [],
        }

    def _retrieve_similar_cases(self, parsed_input: ParsedInput) -> list[dict]:
        """使用检索器检索相似工况

        优先使用 search_query（由 InputRouter 提取），
        否则回退到 description。mcuid 用于精确过滤。
        """
        try:
            query = parsed_input.search_query or parsed_input.description or ""
            if not query:
                return []

            docs = self._retriever.invoke(query)

            cases: list[dict] = []
            for doc in docs:
                record = document_to_record(doc)
                record_dict = record.to_dict()
                record_dict["record_id"] = doc.metadata.get("id", "")
                record_dict["similarity"] = doc.metadata.get("score", 0.0)
                cases.append(record_dict)

            return cases
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []

    def _build_findings(
        self,
        reasoning_result: dict,
        similar_cases: list[dict],
    ) -> list[DiagnosticFinding]:
        """从推理结果构建诊断发现"""
        findings: list[DiagnosticFinding] = []

        for f in reasoning_result.get("findings", []):
            findings.append(DiagnosticFinding(
                title=f.get("title", "发现"),
                description=f.get("description", ""),
                confidence=f.get("confidence", 0.5),
                evidence=f.get("evidence", []),
            ))

        if not findings:
            root_causes = reasoning_result.get("fault_root_cause", [])
            description = root_causes[0] if root_causes else "无明确根因"
            findings.append(DiagnosticFinding(
                title="诊断结论",
                description=description,
                confidence=reasoning_result.get("confidence", 0.3),
                evidence=[],
            ))

        return findings

    def _extract_from_records(self, parsed_input: ParsedInput, field: str) -> str:
        """从 ParsedInput 的 bulk_records 中提取指定字段值"""
        if parsed_input.bulk_records:
            first = parsed_input.bulk_records[0]
            val = first.get(field, "")
            return str(val) if val else ""
        return ""