"""输入数据模型"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .incident import IncidentRecord


class InputType(str, Enum):
    """输入类型枚举"""
    XLSX = "xlsx"
    CSV = "csv"
    NATURAL_LANGUAGE = "natural_language"
    MIXED = "mixed"   # 混合输入：自然语言 + 文件


class InputIntent(str, Enum):
    """输入意图枚举（由 InputRouter 分类）

    用于决定是否执行预检索：
    - DIAGNOSTIC_QUERY: 故障描述，需要预检索相似工况
    - INSTRUCTION: 对 LLM 的操作指令，跳过预检索
    - SUPPLEMENT: 信息补充，跳过预检索，作为上下文注入
    - WORKING_CONDITION_FILE: 工况文件，需先调用转换工具再重新路由
    """
    DIAGNOSTIC_QUERY = "diagnostic_query"
    INSTRUCTION = "instruction"
    SUPPLEMENT = "supplement"
    WORKING_CONDITION_FILE = "working_condition_file"


class ParsedInput(BaseModel):
    """解析后的输入数据

    无论原始输入是 xlsx / csv / 自然语言 / 混合，
    最终都统一为这个模型。
    """
    input_type: InputType = Field(..., description="输入类型")

    # 主要文本描述（对应 problem_description 列）
    description: str = Field("", description="故障的自然语言描述")

    # 来自文件的多条记录（批量输入时，每条记录包含新 8 列）
    bulk_records: list[dict] = Field(
        default_factory=list, description="批量解析的记录（来自 xlsx/csv）"
    )

    # 原始输入信息
    source_file: Optional[str] = Field(None, description="源文件路径")
    raw_input: str = Field("", description="原始输入文本")

    # InputRouter 相关字段
    intent: InputIntent = Field(
        InputIntent.DIAGNOSTIC_QUERY,
        description="输入意图（由 InputRouter 分类）",
    )
    search_query: Optional[str] = Field(
        None,
        description="LLM 提取的检索用 query（仅 diagnostic_query 有值，去除指令性语言）",
    )

    # 字段提取结果（由 FieldExtractor 生成）
    field_extraction: Optional[IncidentRecord] = Field(
        None,
        description="自然语言输入的字段提取结果（分类到 8 个标准字段）",
    )

    def is_bulk(self) -> bool:
        """是否为批量输入"""
        return len(self.bulk_records) > 0
