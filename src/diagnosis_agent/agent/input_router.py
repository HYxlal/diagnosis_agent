"""InputRouter — 输入意图路由层

极简路由，只负责：
1. 文件类型识别（xlsx/csv → diagnostic_query）
2. 工况文件识别（.asc/.blf/.mdf → working_condition_file）
3. 文本输入统一归为 diagnostic_query

简化原因：
- instruction/supplement 的关键词规则太粗糙，频繁误判（如"换一个问题"被误判为 instruction）
- 领域判断（is_in_scope）下沉到 TopicDetector 统一处理
- LLM 在 ReAct 中自己判断是否需要工具调用，无需在路由层区分
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models.input import InputIntent, InputType, ParsedInput, StandardEntities
from ..parsers.field_extractor import FieldExtractor

logger = logging.getLogger(__name__)


_WORKING_CONDITION_EXTENSIONS = {".asc", ".blf", ".mdf"}


class InputRouter:
    """极简输入意图路由器

    只做静态路由，不调用 LLM，不区分 instruction/supplement。
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

        # 纯文本输入：统一归为 diagnostic_query
        text = parsed_input.description or parsed_input.raw_input or ""
        if not text:
            logger.warning("InputRouter: 无文本内容，默认 diagnostic_query")
            parsed_input.intent = InputIntent.DIAGNOSTIC_QUERY
            return parsed_input

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

        logger.info(f"InputRouter: intent=diagnostic_query, query={text[:50]}")
        return parsed_input