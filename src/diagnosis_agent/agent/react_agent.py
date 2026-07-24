"""ReAct 推理 Agent

使用 LangChain + LangGraph 进行故障诊断推理。
使用 create_react_agent 构建 ReAct 循环，支持结构化工具调用和输出。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ..config import Settings
from ..models.diagnostic_output import DiagnosticResult
from ..models.diagnosis import (
    DatabaseEntry,
    DiagnosticFinding,
    DiagnosticOutput,
    DiagnosticReport,
    SimilarCase,
    ToolCallRecord,
)
from ..models.input import InputIntent, ParsedInput
from ..retrieval.hybrid import HybridRetriever
from .prompts import (
    build_similar_case_prompt,
    build_no_similar_case_prompt,
    format_similar_cases_for_prompt,
)
from .tools import DiagnosticTools

logger = logging.getLogger(__name__)


class ReActDiagnosticAgent:
    """ReAct 诊断 Agent

    使用 LangGraph create_react_agent 构建，支持结构化工具调用。

    流程：
    1. 接收 ParsedInput
    2. 预检索相似工况（在 Agent 循环之前）
    3. 构建系统 prompt，注入相似工况信息
    4. 调用 create_react_agent 执行 ReAct 循环
    5. 解析结构化输出（Pydantic 模型验证）
    6. 构建双层诊断输出（报告 + 数据库条目）
    """

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
    ):
        self.settings = settings
        self.retriever = retriever
        self.tools = DiagnosticTools(retriever, settings=settings)

        self._llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM — 必须可用，无 fallback"""
        llm_config = self.settings.llm

        if not llm_config.api_key:
            raise RuntimeError(
                "LLM API key 未配置。请设置 DASHSCOPE_API_KEY 环境变量。"
            )

        try:
            return ChatOpenAI(
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base,
            )
        except Exception as e:
            raise RuntimeError(f"LLM 初始化失败: {e}") from e

    def register_working_condition_converter(self, converter) -> None:
        """注册工况文件转换工具"""
        self.tools.register_working_condition_converter(converter)

    # ------------------------------------------------------------------
    # 主诊断入口
    # ------------------------------------------------------------------

    def diagnose(self, parsed_input: ParsedInput) -> DiagnosticOutput:
        """执行完整诊断流程

        根据 InputRouter 分类结果决定是否预检索：
        - DIAGNOSTIC_QUERY: 正常预检索后进 ReAct
        - INSTRUCTION / SUPPLEMENT: 跳过检索直接进 ReAct
        - WORKING_CONDITION_FILE: 工况文件（转换已在 CLI 层处理）

        Args:
            parsed_input: 解析后的输入（已经过 InputRouter 分类）

        Returns:
            DiagnosticOutput 双层诊断结果
        """
        description = parsed_input.description or ""

        diagnosis_id = f"DIAG-{uuid.uuid4().hex[:8]}"
        diagnosis_time = datetime.now()

        logger.info(f"开始诊断 [{diagnosis_id}]: intent={parsed_input.intent.value}, {description[:100]}...")

        # Step 1: 工况文件处理
        if parsed_input.intent == InputIntent.WORKING_CONDITION_FILE:
            logger.info("检测到工况文件意图，先调用转换工具")
            convert_result = self.tools.convert_working_condition_file(parsed_input.source_file or "")
            if convert_result.get("description"):
                description = convert_result["description"]
                parsed_input.description = description
            if convert_result.get("records"):
                parsed_input.bulk_records = convert_result["records"]
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY

        # Step 2: 预检索相似工况（保留现有逻辑）
        if parsed_input.intent == InputIntent.DIAGNOSTIC_QUERY:
            similar_cases = self._retrieve_similar_cases(parsed_input)
        else:
            similar_cases = []

        has_similar = len(similar_cases) > 0
        logger.info(f"初始检索到 {len(similar_cases)} 条相似工况")

        # Step 3: 构建 Agent 并执行 ReAct 循环
        reasoning_result, tool_calls, react_steps = self._run_react_agent(
            description=description,
            similar_cases=similar_cases,
            has_similar=has_similar,
        )

        # Step 4: 构建诊断发现
        findings = self._build_findings(reasoning_result, similar_cases)

        # Step 5: 构建相似工况模型
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

        # Step 6: 构建报告
        tools_used = list(set(tc.tool_name for tc in tool_calls))

        report = DiagnosticReport(
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            input_summary=description[:500],
            has_similar_cases=has_similar,
            similar_cases=similar_case_models,
            findings=findings,
            recommended_countermeasure=reasoning_result.get("countermeasure", ""),
            react_steps=react_steps,
            reasoning_narrative=reasoning_result.get("reasoning_narrative", ""),
            reasoning_chain=[s.thought for s in react_steps if s.thought],
            tool_calls=tool_calls,
            tools_used=tools_used,
        )

        # Step 7: 构建数据库条目
        confidence = reasoning_result.get("confidence", 0.3)
        database_entry = DatabaseEntry(
            diagnosis_id=diagnosis_id,
            diagnosis_time=diagnosis_time,
            problem_description=description[:500],
            root_cause=reasoning_result.get("root_cause", ""),
            countermeasure=reasoning_result.get("countermeasure", ""),
            drive_code=self._extract_from_records(parsed_input, "drive_code"),
            vehicle_type=self._extract_from_records(parsed_input, "vehicle_type"),
            dashboard_indicator=self._extract_from_records(parsed_input, "dashboard_indicator"),
            dtc_code=self._extract_from_records(parsed_input, "dtc_code"),
            fault_scenario=self._extract_from_records(parsed_input, "fault_scenario"),
            diagnostic_confidence=confidence,
            based_on_similar=has_similar,
            similar_record_ids=[c.get("record_id", "") for c in similar_cases],
        )

        return DiagnosticOutput(report=report, database_entry=database_entry)

    # ------------------------------------------------------------------
    # ReAct Agent 执行
    # ------------------------------------------------------------------

    def _run_react_agent(
        self,
        description: str,
        similar_cases: list[dict],
        has_similar: bool,
    ) -> tuple[dict, list[ToolCallRecord], list]:
        """使用 create_react_agent 执行 ReAct 循环

        保留预检索逻辑，将相似工况注入 prompt，使用结构化输出。

        Returns:
            (reasoning_result, tool_calls, react_steps)
        """
        # 构建 system prompt
        system_prompt = self._build_system_prompt()

        # 构建 user prompt（包含预检索的相似工况）
        if has_similar:
            cases_text = format_similar_cases_for_prompt(similar_cases)
            user_prompt = build_similar_case_prompt(description=description, similar_cases_text=cases_text)
        else:
            user_prompt = build_no_similar_case_prompt(description=description)

        # 获取工具列表
        tools = self.tools.get_tool_list()

        # 创建 ReAct Agent（使用 prompt 参数传入系统提示词）
        agent = create_react_agent(
            model=self._llm,
            tools=tools,
            prompt=system_prompt,
        )

        # 执行 Agent
        logger.info("开始 ReAct 循环")
        try:
            result = agent.invoke({
                "messages": [
                    HumanMessage(content=user_prompt)
                ]
            })
        except Exception as e:
            logger.error(f"ReAct Agent 执行失败: {e}")
            return {}, [], []

        # 解析结果
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None

        # 提取工具调用记录
        tool_calls = self._extract_tool_calls(messages)

        # 提取推理步骤
        react_steps = self._extract_react_steps(messages)

        # 解析最终答案
        reasoning_result = self._parse_final_result(last_message)

        return reasoning_result, tool_calls, react_steps

    def _build_system_prompt(self) -> str:
        """构建系统 prompt，包含结构化输出要求"""
        return f"""你是一个专业的车辆故障诊断专家 Agent。

你的任务：分析故障描述，调用工具检索历史工单，输出诊断结论和推荐对策。

## 诊断推理原则

1. **理解问题**：仔细分析故障描述，识别关键信息（车型、DTC码、仪表盘指示、驱动代码等）
2. **检索相似工况**：使用 search_similar_incidents 工具搜索历史工单
3. **对比分析**：将当前故障与历史工单对比，分析异同
4. **推理根因**：基于证据推理可能的根本原因，排除不合理的假设
5. **制定对策**：给出针对性的解决措施

## 最终答案格式

当你得到足够信息后，必须输出以下 JSON 格式的最终结果：

```json
{{
  "root_cause": "根本原因分析",
  "countermeasure": "推荐对策/解决措施",
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

confidence 反映你对诊断结论的把握程度，0.9 以上为非常确定，0.5 以下为推测。"""

    def _extract_tool_calls(self, messages) -> list[ToolCallRecord]:
        """从消息列表中提取工具调用记录"""
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_record = ToolCallRecord(
                        tool_name=tc.get('name', ''),
                        parameters=tc.get('args', {}),
                        result={},
                        duration_ms=0,
                    )
                    tool_calls.append(tc_record)
        return tool_calls

    def _extract_react_steps(self, messages) -> list:
        """从消息列表中提取推理步骤"""
        from ..models.diagnosis import ReActStep

        steps = []
        step_num = 1
        for msg in messages:
            content = getattr(msg, 'content', '')
            if content and not isinstance(content, dict):
                step = ReActStep(
                    step=step_num,
                    thought=content[:500],
                    action='',
                    action_input={},
                    observation='',
                )
                steps.append(step)
                step_num += 1
        return steps

    def _parse_final_result(self, last_message) -> dict:
        """解析最终结果，使用 Pydantic 模型验证"""
        if not last_message:
            return self._get_default_result()

        content = getattr(last_message, 'content', '')

        # 尝试解析 JSON
        try:
            if isinstance(content, dict):
                data = content
            else:
                # 提取 JSON 部分
                import re
                json_match = re.search(r'\{[\s\S]*\}', str(content))
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return self._get_default_result()

            # 使用 Pydantic 验证
            result = DiagnosticResult(**data)
            return result.dict()

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"解析最终结果失败: {e}")
            return self._get_default_result()

    def _get_default_result(self) -> dict:
        """返回默认结果（解析失败时）"""
        return {
            "root_cause": "详见推理叙述",
            "countermeasure": "详见推理叙述",
            "confidence": 0.3,
            "reasoning_narrative": "诊断过程中未能获取完整的结构化输出",
            "findings": [],
        }

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _retrieve_similar_cases(self, parsed_input: ParsedInput) -> list[dict]:
        """检索相似工况"""
        try:
            results = self.retriever.retrieve_from_parsed(parsed_input, top_k=5)
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []

        cases: list[dict] = []
        for r in results:
            record_dict = r.record.to_dict() if r.record else r.metadata
            record_dict["record_id"] = r.id
            record_dict["similarity"] = r.score
            cases.append(record_dict)

        return cases

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
            findings.append(DiagnosticFinding(
                title="诊断结论",
                description=reasoning_result.get("root_cause", "无明确根因"),
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