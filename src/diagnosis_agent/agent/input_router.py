"""InputRouter — 输入意图路由层

在 parse_input 与 ReAct 之间插入一个轻量 LLM 调用，
完成意图分类和检索 query 提取。

4 类意图：
- diagnostic_query:  故障描述 → 正常预检索后进 ReAct
- instruction:       对 LLM 的指令 → 跳过检索直接进 ReAct
- supplement:        信息补充 → 跳过检索直接进 ReAct（作为上下文）
- working_condition_file: 工况文件 → 先调转换工具再重新路由

设计理由：
- 当前架构在 ReAct 循环前就做预检索，结果进 system prompt
- 如果输入是指令/补充，预检索结果会污染 prompt 上下文
- InputRouter 独立前置：职责分离，ReAct 专注诊断推理
- 开销：一次轻量 LLM 调用（qwen-turbo，短输入短输出，成本极低）
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..models.input import InputIntent, InputType, ParsedInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 意图分类输出模型（使用 Pydantic 约束 LLM 输出格式）
# ---------------------------------------------------------------------------

class IntentClassificationResult(BaseModel):
    """意图分类结果模型"""
    intent: str = Field(
        description="意图类别：diagnostic_query | instruction | supplement | working_condition_file",
        examples=["diagnostic_query", "instruction", "supplement", "working_condition_file"]
    )
    search_query: str = Field(
        description="如果是 diagnostic_query，提取适合检索的核心故障描述（去除指令性语言）；否则为空字符串",
        examples=["发动机故障灯亮 怠速不稳", "电池温升过快"]
    )
    reason: str = Field(
        description="简要说明分类理由",
        examples=["用户描述了具体的故障现象，属于诊断查询"]
    )


# ---------------------------------------------------------------------------
# 意图分类 prompt
# ---------------------------------------------------------------------------

INTENT_CLASSIFICATION_PROMPT = """你是一个输入意图分类器。请判断用户输入属于以下哪一类：

1. **diagnostic_query**：故障描述/诊断查询。用户在描述车辆故障现象，希望得到诊断。
   示例："发动机故障灯亮，怠速不稳"、"电池温升过快"、"行驶中熄火"

2. **instruction**：操作指令。用户在给你下达指令，告诉你要关注什么或调整诊断方向。
   示例："请重点关注电池系统"、"换一种思路分析"、"只看最近半年的工单"

3. **supplement**：信息补充。用户在补充之前诊断的背景信息。
   示例："补充信息：该车已行驶8万公里"、"之前换过电池包"、"这是新车，刚交付3天"

4. **working_condition_file**：工况文件。用户上传的是工况数据文件（非标准 xlsx/csv 工单），需要先转换。
   示例：文件扩展名是 .asc/.blf/.mdf 等非标准格式，或文件内容是时间序列工况数据

请根据输入内容进行分类，并提取检索关键词。"""


# 非 LLM 回退的关键词规则（LLM 不可用时的降级方案）
_INSTRUCTION_KEYWORDS = ["请", "重点", "换", "不要", "只看", "调整", "重新"]
_SUPPLEMENT_KEYWORDS = ["补充", "之前", "背景", "额外", "已知", "前提"]
_WORKING_CONDITION_EXTENSIONS = {".asc", ".blf", ".mdf"}


class InputRouter:
    """输入意图路由器

    用一次轻量 LLM 调用完成：
    1. 意图分类（4 选 1）
    2. 如果是 diagnostic_query：提取适合检索的 query
    3. 如果是 working_condition_file：标记需调用转换工具
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._structured_chain = self._init_structured_chain() if settings.input_router.enabled else None

    def _init_structured_chain(self):
        """初始化意图分类用的结构化输出链"""
        llm_config = self.settings.llm

        if not llm_config.api_key:
            logger.warning(
                "LLM API key 未配置，InputRouter 将使用规则回退模式。"
                "建议配置 DASHSCOPE_API_KEY 以启用 LLM 意图分类。"
            )
            return None

        try:
            from langchain_openai import ChatOpenAI
            from langchain.chains import create_structured_output_chain

            llm = ChatOpenAI(
                model=self.settings.input_router.model,
                temperature=self.settings.input_router.temperature,
                max_tokens=512,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base,
            )

            chain = create_structured_output_chain(
                IntentClassificationResult,
                llm,
                INTENT_CLASSIFICATION_PROMPT,
                verbose=False,
            )
            return chain

        except Exception as e:
            logger.warning(f"InputRouter LLM 初始化失败，使用规则回退: {e}")
            return None

    def route(
        self,
        parsed_input: ParsedInput,
    ) -> ParsedInput:
        """对 ParsedInput 进行意图分类和路由

        Args:
            parsed_input: 原始解析结果

        Returns:
            带有 intent 和 search_query 的 ParsedInput（原地修改并返回）
        """
        # 文件输入（xlsx/csv）默认为 diagnostic_query（批量工单数据）
        if parsed_input.input_type in (InputType.XLSX, InputType.CSV):
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY
            parsed_input.search_query = parsed_input.description
            logger.info("文件输入，意图=diagnostic_query（默认）")
            return parsed_input

        # 工况文件检测：非标准扩展名
        if parsed_input.source_file:
            from pathlib import Path
            ext = Path(parsed_input.source_file).suffix.lower()
            if ext in _WORKING_CONDITION_EXTENSIONS:
                parsed_input.intent = InputIntent.WORKING_CONDITION_FILE
                logger.info(f"检测到工况文件扩展名 {ext}，意图=working_condition_file")
                return parsed_input

        # 纯文本输入：用 LLM 分类
        text = parsed_input.description or parsed_input.raw_input
        if not text:
            logger.warning("InputRouter: 无文本内容，默认 diagnostic_query")
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY
            return parsed_input

        if self._structured_chain is not None:
            return self._route_with_llm(parsed_input, text)
        else:
            return self._route_with_rules(parsed_input, text)

    def _route_with_llm(
        self, parsed_input: ParsedInput, text: str
    ) -> ParsedInput:
        """用 LLM 进行意图分类（使用结构化输出链）"""
        try:
            response = self._structured_chain.invoke({"input": text[:2000]})

            if isinstance(response, dict):
                result = response.get("structured_output", response)
            elif hasattr(response, "structured_output"):
                result = response.structured_output
            else:
                result = {"intent": "diagnostic_query", "search_query": "", "reason": ""}

            intent_str = result.get("intent", "diagnostic_query")
            try:
                parsed_input.intent = InputIntent(intent_str)
            except ValueError:
                logger.warning(f"InputRouter: 未知意图 '{intent_str}'，回退 diagnostic_query")
                parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY

            parsed_input.search_query = result.get("search_query") or None

            logger.info(
                f"InputRouter (LLM): intent={parsed_input.intent.value}, "
                f"search_query={parsed_input.search_query[:50] if parsed_input.search_query else 'None'}"
            )

        except Exception as e:
            logger.warning(f"InputRouter LLM 调用失败，使用规则回退: {e}")
            return self._route_with_rules(parsed_input, text)

        return parsed_input

    def _route_with_rules(
        self, parsed_input: ParsedInput, text: str
    ) -> ParsedInput:
        """规则回退模式（LLM 不可用时使用）"""
        text_lower = text.lower()

        # 检查指令关键词
        if any(kw in text for kw in _INSTRUCTION_KEYWORDS):
            parsed_input.intent = InputIntent.INSTRUCTION
            logger.info("InputRouter (规则): intent=instruction")
            return parsed_input

        # 检查补充信息关键词
        if any(kw in text for kw in _SUPPLEMENT_KEYWORDS):
            parsed_input.intent = InputIntent.SUPPLEMENT
            logger.info("InputRouter (规则): intent=supplement")
            return parsed_input

        # 默认：诊断查询
        parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY
        parsed_input.search_query = text
        logger.info("InputRouter (规则): intent=diagnostic_query")
        return parsed_input