"""自然语言解析器

。
此模块仅负责将文本包装为 ParsedInput，实际的语义解析
（提取车型、DTC 码等信息）由 ReAct Agent 在推理阶段完成。
"""

from __future__ import annotations

from ..models.input import InputType, ParsedInput


def parse_natural_language(text: str) -> ParsedInput:
    """解析自然语言输入
    将原始文本直接作为 description，
    后续由 LLM Agent 进行语义理解和推理。

    Args:
        text: 自然语言故障描述

    Returns:
        ParsedInput 实例
    """
    text = text.strip()

    return ParsedInput(
        input_type=InputType.NATURAL_LANGUAGE,
        description=text,
        raw_input=text,
    )


def parse_mixed(
    text: str,
) -> ParsedInput:
    """解析混合输入：自然语言 + 文件内容

    统一走自然语言路径，不再提取结构化字段。

    Args:
        text: 自然语言描述

    Returns:
        ParsedInput 实例
    """
    text = text.strip()

    return ParsedInput(
        input_type=InputType.MIXED,
        description=text,
        raw_input=text,
    )
