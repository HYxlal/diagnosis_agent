"""LangChain Agent 实现

基于 LangChain 最新架构的诊断 Agent，使用多源检索。
支持两种输入模式：
1. 传统模式：ParsedInput（内部解析输入）
2. 标准接口模式：StandardInput（平台 Agent 传入的标准 JSON）
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

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
from ..retrieval.hybrid_retriever import HybridRetriever, create_hybrid_retriever
from ..utils.llm_factory import create_llm
from .prompts import (
    build_similar_case_prompt,
    build_no_similar_case_prompt,
    format_similar_cases_for_prompt,
)
from .context_manager import SimpleContextManager
from .tools import DiagnosticTools

logger = logging.getLogger(__name__)


class LangChainDiagnosticAgent:
    """基于 LangChain 最新架构的诊断 Agent

    使用 ChromaVectorRetriever 做向量语义检索，LangChain create_agent 构建 Agent。
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
        self._context_manager = SimpleContextManager(
            window_size=settings.context.window_size,
            max_tokens=settings.context.max_tokens,
            emergency_min_turns=settings.context.emergency_min_turns,
        )
        self._last_diagnostic_output: Optional[DiagnosticOutput] = None
        self._stream_callback = None  # 实时回调，打印 ReAct 步骤

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
        return create_llm(settings=self.settings)

    def _init_retriever(self) -> HybridRetriever:
        """初始化检索器

        使用 HybridRetriever：Neo4j 召回 + Embedding 精排 + Chroma 兜底。
        Neo4j 不可用或未配置时，自动降级到纯 Chroma 语义检索。
        """
        return create_hybrid_retriever(settings=self.settings)

    def register_working_condition_converter(self, converter) -> None:
        """注册工况文件转换工具"""
        self._tools.register_working_condition_converter(converter)

    def diagnose_with_standard_input(
        self,
        standard_input: StandardInput,
        history_messages: list[dict] | None = None,
        prepared_messages: list | None = None,
    ) -> StandardOutput:
        """使用标准输入接口执行诊断

        Args:
            standard_input: 平台 Agent 传入的标准输入
            history_messages: 历史消息列表（来自 SessionManager.get_context()），
                             None 或空列表表示首轮对话。
                             dict 格式与 LangChain messages_to_dict 一致。

        为什么需要 history_messages？
        - LangChain Agent 是无状态的，每次 invoke() 只根据当前 messages 推理。
        - 多轮时，需要把上一轮的完整消息列表（HumanMessage + AIMessage + ToolMessage）
          直接拼入本次 invoke 的 messages 列表，Agent 才能看到之前做了什么。
        - 不使用文本摘要，因为文本摘要会丢失工具调用结果等关键信息。
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

            # 把 history_messages / prepared_messages 透传给内部 diagnose
            diagnostic_output = self.diagnose(
                parsed_input,
                history_messages=history_messages,
                prepared_messages=prepared_messages,
            )
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

    def diagnose(
        self,
        parsed_input: ParsedInput,
        history_messages: list[dict] | None = None,
        prepared_messages: list | None = None,
    ) -> DiagnosticOutput:
        """执行完整诊断流程（内部方法）

        主流程编排：
        1. 工况文件意图：先转换文件为结构化数据，再把意图改回 DIAGNOSTIC_QUERY
        2. 诊断查询意图：预检索相似工况（供 Agent 循环使用）
        3. 其他意图（指令/补充）：跳过预检索，直接进 Agent
        4. 运行 LangChain Agent → 提取工具调用 / ReAct 步骤 / 推理结果
        5. 组装双层输出：DiagnosticReport（人读）+ DatabaseEntry（机读）

        Args:
            parsed_input: 解析后的输入
            history_messages: 历史消息列表（dict 格式），None 或空列表表示首轮对话
            prepared_messages: 预构建的消息列表（BaseMessage 格式），优先级高于 history_messages。
                              当使用 prepare_from_context() 时传入，跳过 SessionManager 的扁平化历史。
        """
        description = parsed_input.description or ""

        diagnosis_id = f"DIAG-{uuid.uuid4().hex[:8]}"
        diagnosis_time = datetime.now()

        logger.info(f"开始诊断 [{diagnosis_id}]: intent={parsed_input.intent.value}, {description[:100]}...")

        # 工况文件：先转换，然后把意图改回 DIAGNOSTIC_QUERY 进入正常诊断路径
        if parsed_input.intent == InputIntent.WORKING_CONDITION_FILE:
            logger.info("检测到工况文件意图，先调用转换工具")
            convert_result = self._tools.convert_working_condition_file(parsed_input.source_file or "")
            if convert_result.get("description"):
                description = convert_result["description"]
                parsed_input.description = description
            if convert_result.get("records"):
                parsed_input.bulk_records = convert_result["records"]
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY

        # 检索全部交给 Agent 工具执行，不在主流程做预检索。
        # 原因：1) 预检索与话题检测的上下文裁剪冲突；
        #       2) Agent 能根据当前上下文决定是否需要补充检索。
        similar_cases: list = []
        has_similar = False

        reasoning_result, tool_calls, react_steps = self._run_agent(
            description=description,
            similar_cases=similar_cases,
            has_similar=has_similar,
            history_messages=history_messages,
            prepared_messages=prepared_messages,
        )

        findings = self._build_findings(reasoning_result, similar_cases)

        # 将工具调用中检索到的 Document 转为 SimilarCase 模型
        # 优先从 tool_calls 中提取 search/query_fault_graph 的结果
        similar_case_models = self._extract_similar_cases_from_tools(tool_calls)

        tools_used = list(set(tc.tool_name for tc in tool_calls))

        # 从 LLM 推理结果中取根因/对策（取列表第一项作为主结论）
        root_causes = reasoning_result.get("fault_root_cause", [])
        root_cause_text = root_causes[0] if root_causes else ""
        solutions = reasoning_result.get("solution", [])
        countermeasure_text = solutions[0] if solutions else ""

        report = DiagnosticReport(
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            input_summary=description[:500],
            has_similar_cases=len(similar_case_models) > 0,
            similar_cases=similar_case_models,
            findings=findings,
            recommended_countermeasure=countermeasure_text,
            react_steps=react_steps,
            reasoning_narrative=reasoning_result.get("reasoning_narrative", ""),
            reasoning_chain=[s.thought for s in react_steps if s.thought],
            tool_calls=tool_calls,
            tools_used=tools_used,
        )


        database_entry = self._build_database_entry(
            parsed_input=parsed_input,
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            root_cause_text=root_cause_text,
            countermeasure_text=countermeasure_text,
            confidence=reasoning_result.get("confidence", 0.3),
            based_on_similar=has_similar,
            similar_record_ids=[doc.metadata.get("id", "") for doc in similar_cases],
        )

        diagnostic_output = DiagnosticOutput(
            report=report,
            database_entry=database_entry,
            reasoning_result=reasoning_result,
        )
        self._last_diagnostic_output = diagnostic_output
        return diagnostic_output

    def _build_database_entry(
        self,
        parsed_input: ParsedInput,
        diagnosis_id: str,
        diagnosis_time: datetime,
        root_cause_text: str,
        countermeasure_text: str,
        confidence: float,
        based_on_similar: bool,
        similar_record_ids: list[str],
        max_summary_length: int = 500,
    ) -> DatabaseEntry:
        """构建可录入数据库的结构化条目

        字段来源优先级：entities（平台 NLP 实体）> bulk_records（文件解析记录）。
        根因、对策、置信度来自 LLM 推理结果（reasoning_result）。
        """
        entities = parsed_input.entities
        description = parsed_input.description or ""

        # 字段回退顺序：entities → bulk_records
        dtc_code = ""
        if entities and entities.dtc_code:
            dtc_code = ", ".join(entities.dtc_code)
        else:
            dtc_code = self._extract_from_records(parsed_input, "dtc_code")

        vehicle_type = ""
        if entities and entities.project:
            vehicle_type = entities.project
        else:
            vehicle_type = self._extract_from_records(parsed_input, "vehicle_type")

        fault_scenario = ""
        if entities and entities.working_condition:
            fault_scenario = entities.working_condition
        else:
            fault_scenario = self._extract_from_records(parsed_input, "fault_scenario")

        return DatabaseEntry(
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            problem_description=description[:max_summary_length],
            root_cause=root_cause_text,
            countermeasure=countermeasure_text,
            drive_code=self._extract_from_records(parsed_input, "drive_code"),
            vehicle_type=vehicle_type,
            dashboard_indicator=self._extract_from_records(parsed_input, "dashboard_indicator"),
            dtc_code=dtc_code,
            fault_scenario=fault_scenario,
            diagnostic_confidence=confidence,
            based_on_similar=based_on_similar,
            similar_record_ids=similar_record_ids,
        )

    def _run_agent(
        self,
        description: str,
        similar_cases: list[Document],
        has_similar: bool,
        history_messages: list[dict] | None = None,
        prepared_messages: list | None = None,
    ) -> tuple[dict, list[ToolCallRecord], list[ReActStep]]:
        """使用缓存的 Agent 执行循环

        Args:
            history_messages: 历史消息列表（来自 SessionManager.get_context()）。
                             dict 格式与 LangChain messages_to_dict 一致。
            prepared_messages: 预构建的消息列表（BaseMessage 格式），
                             优先级高于 history_messages。
                             由 ContextManager.prepare_from_context() 生成，
                             包含 SystemMessage 摘要 + 热层消息。
        """
        if has_similar:
            cases_text = format_similar_cases_for_prompt(similar_cases)
            user_prompt = build_similar_case_prompt(description=description, similar_cases_text=cases_text)
        else:
            user_prompt = build_no_similar_case_prompt(description=description)

        # 多轮历史消息注入：
        # 两种路径：
        # 1. prepared_messages：ContextManager.prepare_from_context() 已构建好
        #    包括 SystemMessage(摘要) + 热层消息，只需追加当前 user_prompt
        # 2. history_messages：SessionManager.get_context() 扁平化 dict 列表
        invoke_messages: list = []
        if prepared_messages:
            # prepared_messages 已包含 SystemMessage(摘要) + 热层消息
            invoke_messages.extend(prepared_messages)
            # 追加当前用户 prompt（含相似工况）
            invoke_messages.append({"role": "user", "content": user_prompt})
        elif history_messages:
            invoke_messages.extend(messages_from_dict(history_messages))
            invoke_messages.append({"role": "user", "content": user_prompt})
        else:
            invoke_messages.append({"role": "user", "content": user_prompt})

        # Token 预算裁剪：在 LLM 调用前按 window_size 和 max_tokens 裁剪消息
        prepare_result = self._context_manager.prepare(invoke_messages)
        invoke_messages = prepare_result.messages

        logger.info("开始 Agent 循环")
        all_messages = list(invoke_messages)
        try:
            # 使用 stream 模式，每个 chunk 包含新增的消息
            last_processed = len(all_messages)
            for chunk in self._agent.stream({"messages": invoke_messages}, stream_mode="values"):
                chunk_messages = chunk.get("messages", [])
                # 打印新增的消息（ReAct 步骤）
                if self._stream_callback:
                    for msg in chunk_messages[last_processed:]:
                        self._stream_callback(msg)
                last_processed = len(chunk_messages)
                all_messages = chunk_messages
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            return {}, [], []

        messages = all_messages
        last_message = messages[-1] if messages else None

        input_len = len(invoke_messages)
        self._last_messages = messages[input_len - 1:] if len(messages) >= input_len else messages

        tool_calls = self._extract_tool_calls(messages)
        react_steps = self._extract_react_steps(messages)
        reasoning_result = self._parse_final_result(last_message)

        return reasoning_result, tool_calls, react_steps

    def _build_system_prompt(self) -> str:
        """构建系统 prompt"""
        from ..prompts.agent import AGENT_SYSTEM_PROMPT
        return AGENT_SYSTEM_PROMPT

    def _extract_tool_calls(self, messages: list[BaseMessage]) -> list[ToolCallRecord]:
        """从消息列表中提取工具调用记录，关联 ToolMessage 结果

        需要两遍循环：第一遍从 AIMessage.tool_calls 建立按 id 索引的记录，
        第二遍从 ToolMessage.tool_call_id 回填结果（工具返回的消息与发起调用的
        AIMessage 是分离的，靠 tool_call_id 关联）。
        """
        tool_calls = []
        tool_call_map = {}

        # 第一遍：建立工具调用记录索引（key=tool_call_id）
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
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

        # 第二遍：根据 tool_call_id 关联 ToolMessage 的返回结果
        for msg in messages:
            if isinstance(msg, ToolMessage):
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
        """解析最终结果，使用 PydanticOutputParser 验证"""
        if not last_message:
            return self._get_default_result()

        content = getattr(last_message, "content", "")

        try:
            if isinstance(content, dict):
                data = content
            else:
                data = self._extract_json_from_text(str(content))
                if data is None:
                    return self._get_default_result()

            from langchain_core.output_parsers import PydanticOutputParser
            parser = PydanticOutputParser(pydantic_object=DiagnosticResult)
            result = parser.parse(json.dumps(data, ensure_ascii=False))
            return result.dict() if isinstance(result, DiagnosticResult) else result

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"解析最终结果失败: {e}")
            return self._get_default_result()

    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON 对象

        分层策略（从严到宽）：
        1. 直接 json.loads 整段文本（LLM 直接返回纯 JSON 时）
        2. 匹配 ```json ... ``` 代码块（LLM 把 JSON 包在代码块里）
        3. 非贪婪匹配第一个 { ... } 块（兜底，处理混杂文本）

        三层都失败则返回 None，由调用方走默认结果。
        """
        # 1. 直接解析整段文本
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 2. 匹配 ```json ... ``` 代码块（非贪婪）
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 3. 兜底：匹配文本中第一个 { ... } 块（非贪婪）
        brace_match = re.search(r"\{[\s\S]*?\}", text)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

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

    def _extract_similar_cases_from_tools(self, tool_calls: list) -> list[SimilarCase]:
        """从工具调用结果中提取相似工况"""
        cases: list[SimilarCase] = []
        seen_ids = set()

        for tc in tool_calls:
            if tc.tool_name not in ("search_similar_incidents", "query_fault_graph"):
                continue
            result = tc.result or {}
            if isinstance(result, str):
                try:
                    import json
                    result = json.loads(result)
                except json.JSONDecodeError:
                    continue
            # 工具返回的格式：`list[dict]`（直接）或 `{"results": [...]}`（包装）
            items = result if isinstance(result, list) else result.get("results", []) if isinstance(result, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                rid = item.get("record_id", item.get("id", ""))
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                cases.append(SimilarCase(
                    record_id=rid,
                    problem_description=item.get("problem_description", ""),
                    root_cause=item.get("root_cause", ""),
                    countermeasure=item.get("countermeasure", ""),
                    drive_code=item.get("drive_code", ""),
                    vehicle_type=item.get("vehicle_type", ""),
                    dashboard_indicator=item.get("dashboard_indicator", ""),
                    dtc_code=item.get("dtc_code", ""),
                    fault_scenario=item.get("fault_scenario", ""),
                    similarity=item.get("similarity", 0.0),
                ))
        return cases

    def _build_findings(
        self,
        reasoning_result: dict,
        similar_cases: list[Document],
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