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
    """输入意图枚举（由 InputRouter 静态路由分类）"""
    DIAGNOSTIC_QUERY = "diagnostic_query"
    WORKING_CONDITION_FILE = "working_condition_file"


# ---------------------------------------------------------------------------
# 标准输入接口（平台 Agent 传入）
# ---------------------------------------------------------------------------

class StandardEntities(BaseModel):
    """平台 NLP 实体识别结果（结构化）

    字段对应需求文档中的实体抽取结果，每个字段明确类型。
    """
    dtc_code: list[str] = Field(
        default_factory=list,
        description="故障DTC代码列表",
        examples=[["P1A3E98", "U162483"]],
    )
    project: str = Field(
        "",
        description="项目型号/车型代号",
        examples=["H37A"],
    )
    component: str = Field(
        "",
        description="涉及的部件",
        examples=["IGBT/MCU"],
    )
    working_condition: str = Field(
        "",
        description="工作条件描述",
        examples=["爬坡满载、低温冷启动"],
    )
    software_version: str = Field(
        "",
        description="软件版本信息",
        examples=["H37A3621830AW"],
    )


class HistoryContext(BaseModel):
    """多轮对话历史上下文单条记录"""
    query: str = Field("", description="历史用户问题")
    entities: dict = Field(default_factory=dict, description="历史实体抽取结果")
    agent_answer: str = Field("", description="上一轮 Agent 回复")


class StandardInput(BaseModel):
    """电驱诊断 Agent 标准输入

    由平台 Agent 传入的结构化 JSON 数据，作为诊断系统的统一入口。
    """
    # 必填字段
    raw_query: str = Field(..., description="用户原始提问文本")
    mcuid: str = Field(..., description="MCU标识，用于精确搜索相关信息")

    # 平台 Agent 可选输出
    intent_name: Optional[str] = Field(
        None,
        description="意图名称（平台Agent识别结果），如 MCU_001",
    )
    query_confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="意图识别置信度（0-1）",
    )

    # 平台 NLP 实体识别结果
    entities: StandardEntities = Field(
        default_factory=StandardEntities,
        description="结构化实体: dtc_code[], project, component, working_condition, software_version",
    )

    # 多轮对话（仅定义结构，逻辑待实现）
    session_id: Optional[str] = Field(None, description="会话唯一标识")
    context_history: list[HistoryContext] = Field(
        default_factory=list,
        description="历史对话上下文列表",
    )

    # 辅助字段
    user_id: Optional[str] = Field(None, description="用户唯一标识")


# ---------------------------------------------------------------------------
# 内部解析输入（系统内部使用）
# ---------------------------------------------------------------------------

class ParsedInput(BaseModel):
    """解析后的输入数据

    无论原始输入是 xlsx / csv / 自然语言 / 混合，
    最终都统一为这个模型。

    重构后：直接持有结构化实体（entities），不再依赖 field_extraction 中间转换。
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

    # -----------------------------------------------------------------------
    # 直接结构化实体（方案B：StandardInput.entities 直接传递）
    # -----------------------------------------------------------------------
    mcuid: str = Field("", description="MCU标识，用于精确搜索")
    entities: Optional[StandardEntities] = Field(
        None,
        description="结构化实体（来自平台NLP或本地提取）",
    )

    # 兼容字段（保留用于文件解析路径，标记为 deprecated）
    field_extraction: Optional[IncidentRecord] = Field(
        None,
        description="(deprecated) 自然语言输入的字段提取结果，新代码请使用 entities",
    )

    def is_bulk(self) -> bool:
        """是否为批量输入"""
        return len(self.bulk_records) > 0

    def get_entity(self, field: str, default: str = "") -> str:
        """从 entities 中安全获取实体字段值"""
        if self.entities:
            return getattr(self.entities, field, default) or default
        return default
