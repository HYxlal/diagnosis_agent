"""InputRouter — 输入意图路由层

轻量静态路由，只负责：
1. 文件类型识别（xlsx/csv → diagnostic_query）
2. 工况文件识别（.asc/.blf/.mdf → working_condition_file）
3. 文本输入的简单规则分类（instruction / supplement / diagnostic_query）

设计变化：
- 移除了 LLM 调用和 out_of_scope 判断
- 领域范围判断（is_in_scope）下沉到 TopicDetector 阶段2 精判中统一处理
- 这样避免两次 LLM 调用，也避免多轮对话中 supplement/instruction 被误判为 out_of_scope
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models.input import InputIntent, InputType, ParsedInput, StandardEntities
from ..parsers.field_extractor import FieldExtractor

logger = logging.getLogger(__name__)


# 非 LLM 回退的关键词规则
_INSTRUCTION_KEYWORDS = ["请", "重点", "换", "不要", "只看", "调整", "重新", "关注"]
_SUPPLEMENT_KEYWORDS = ["补充", "之前", "背景", "额外", "已知", "前提"]
_WORKING_CONDITION_EXTENSIONS = {".asc", ".blf", ".mdf"}


class InputRouter:
    """轻量输入意图路由器

    只做静态规则路由，不调用 LLM。
    """

    def __init__(self, settings=None):
        self.settings = settings

    def route(self, parsed_input: ParsedInput) -> ParsedInput:
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
            ext = Path(parsed_input.source_file).suffix.lower()
            if ext in _WORKING_CONDITION_EXTENSIONS:
                parsed_input.intent = InputIntent.WORKING_CONDITION_FILE
                logger.info(f"检测到工况文件扩展名 {ext}，意图=working_condition_file")
                return parsed_input

        # 纯文本输入：简单规则分类
        text = parsed_input.description or parsed_input.raw_input or ""
        if not text:
            logger.warning("InputRouter: 无文本内容，默认 diagnostic_query")
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY
            return parsed_input

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

        # 字段提取（仅当 entities 为空时）
        if parsed_input.entities is None:
            try:
                extractor = FieldExtractor()
                record = extractor.extract(text)
                parsed_input.entities = StandardEntities(
                    dtc_code=[record.dtc_code] if record.dtc_code else [],
                    project=record.vehicle_type or "",
                    component="",
                    working_condition=record.fault_scenario or "",
                    software_version="",
                )
            except Exception as e:
                logger.warning(f"字段提取失败: {e}")

        logger.info(f"InputRouter (规则): intent=diagnostic_query, query={text[:50]}")
        return parsed_input