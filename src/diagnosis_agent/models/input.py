"""输入数据模型 — 电驱诊断系统统一入口定义

所有来自平台侧的输入都通过 StandardInput 定义，完全扁平化，无嵌套实体对象，
字段语义与平台最新 Schema 严格保持一一对应。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
# 表单枚举 — 严格对齐平台表单选项，避免自由文本不一致
# ---------------------------------------------------------------------------

class FaultWorkConditionEnum(str, Enum):
    """故障工况多选枚举"""
    SMOOTH_DRIVE = "平稳驾驶"
    AGGRESSIVE_DRIVE = "激烈驾驶"
    VEHICLE_START = "车辆启动"
    CHARGING = "充电"
    UNKNOWN = "无法确认故障工况"


class InstrumentIndicatorEnum(str, Enum):
    """仪表指示灯多选枚举"""
    MOTOR_RED = "电机故障红灯"
    MOTOR_YELLOW = "电机故障黄灯"
    POWER_SYSTEM_RED = "动力系统故障红灯"
    POWER_SYSTEM_YELLOW = "动力系统故障黄灯"
    MOTOR_OVERTEMP = "电机过温故障灯"
    NONE = "无"


class MotorPositionEnum(str, Enum):
    """电机位置枚举（前驱/后驱/四驱 场景下定位到具体电机）"""
    FRONT_MOTOR = "前电机"
    REAR_MOTOR = "后电机"
    UNKNOWN = "无法确认具体电机"


class ContextTurn(BaseModel):
    """单轮对话历史记录"""
    query: str = Field("", description="上一轮用户提问文本")
    agent_answer: str = Field("", description="Agent 上一轮完整诊断回答文本")


# ---------------------------------------------------------------------------
# 标准输入接口（平台 Agent 传入）
# 完全扁平化，无任何嵌套实体对象，字段全部平铺开
# ---------------------------------------------------------------------------

class StandardInput(BaseModel):
    """电驱诊断 Agent 标准输入

    与平台最新提交的 JSON Schema 严格对齐，所有字段直接平铺，不留中间嵌套层。
    """

    # ========== 平台交互字段（链路追踪与多轮）==========
    intentId: str = Field(
        "",
        description="意图识别名称，Agent 平台输出，如 MCU_001",
        examples=["MCU_001", "MCU_002"],
    )
    query_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="意图识别置信度，0~1 浮点数，平台 NLP 引擎输出",
    )
    conversationId: str = Field(
        "",
        description="多轮对话会话唯一 ID，用于关联 SessionManager 中的上下文历史",
    )
    context_history: list[ContextTurn] = Field(
        default_factory=list,
        description="多轮对话历史降级路径，按时间顺序最新一轮放末尾。"
                    "注意：本字段为纯文本降级备用，不优先使用。"
                    "内部优先通过 conversationId 关联 SessionManager，后者保留完整 LangChain 消息含工具调用历史，"
                    "信息完整度远高于纯文本历史。",
    )
    clientRequestId: str = Field(
        "",
        description="由 AI 诊断平台生成，用于提交幂等控制和请求链路关联",
    )
    traceId: str = Field(
        "",
        description="用于 Agent 之间调用链路、日志串联排查，接入 OpenTelemetry 用",
    )

    # ========== 表单必填业务字段 ==========
    VIN: str = Field(
        ...,
        description="整车 VIN 码，17 位字符，用于查询该车历史维修档案",
        examples=["LSVAF09E2HN1234567"],
    )
    raw_query: str = Field(
        ...,
        description="用户所填写故障问题描述，自然语言自由文本",
        examples=["行驶中加速无力，故障灯点亮"],
    )
    faultOccurTime: str = Field(
        ...,
        description="故障工况发生时间，ISO 8601 date-time 格式，如 2026-08-21T14:30:00+08:00",
        examples=["2026-08-21T14:30:00+08:00"],
    )
    mileage: float = Field(
        ...,
        description="车辆行驶总里程，单位 km",
        ge=0.0,
        examples=[35280.5],
    )
    instrumentIndicatorList: list[InstrumentIndicatorEnum] = Field(
        default_factory=list,
        description="仪表指示灯多选，支持选多个或不选",
    )
    vehicleModel: str = Field(
        ...,
        description="车型代号，用于车型级精确过滤检索",
        examples=["H37A"],
    )

    # ========== 表单可选业务字段 ==========
    faultWorkConditionList: FaultWorkConditionEnum = Field(
        FaultWorkConditionEnum.UNKNOWN,
        description="表单单选故障工况",
    )
    dtcCode: list[str] = Field(
        default_factory=list,
        description="数据湖获取 DTC 故障码；无故障码传空数组 []",
        examples=[["P0A2B"], ["P0A2B", "U162483"], []],
    )
    softwareVersion: str = Field(
        "",
        description="MCU 软件版本号，用于匹配相同版本的历史故障案例",
        examples=["H37A3621830AW"],
    )
    motorPosition: MotorPositionEnum = Field(
        MotorPositionEnum.UNKNOWN,
        description="电机位置，定位前电机 / 后电机 / 无法确认",
    )
    faultDataUrl: str = Field(
        "",
        description="故障数据链接，指向 CAN 总线日志 / DBC 文件下载地址，多个文件逗号分隔",
        examples=["s3://bucket/fault-data/can_001.blf,s3://bucket/fault-data/can.dbc"],
    )


# ---------------------------------------------------------------------------
# 内部解析输入（系统内部使用）
# ---------------------------------------------------------------------------

class ParsedInput(BaseModel):
    """解析后的输入数据

    无论原始输入是 xlsx / csv / 自然语言 / 混合，最终都统一为这个模型。
    从 StandardInput 直接透传所有业务字段，不再经过任何中间对象转换。
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

    # ========================================
    # 业务字段直接透传，与 StandardInput 结构 1:1
    # ========================================
    VIN: str = Field("", description="整车 VIN 码")
    vehicleModel: str = Field("", description="车型代号，用于精确过滤检索")
    faultOccurTime: str = Field("", description="故障发生时间")
    mileage: float = Field(0.0, description="行驶里程")
    faultWorkConditionList: FaultWorkConditionEnum = Field(
        FaultWorkConditionEnum.UNKNOWN,
        description="单选故障工况",
    )
    instrumentIndicatorList: list[InstrumentIndicatorEnum] = Field(
        default_factory=list,
        description="仪表指示灯枚举",
    )
    dtcCode: list[str] = Field(default_factory=list, description="数据湖获取 DTC 故障码，无故障码传空数组")
    softwareVersion: str = Field("", description="MCU 软件版本")
    motorPosition: MotorPositionEnum = Field(
        MotorPositionEnum.UNKNOWN,
        description="电机位置枚举",
    )
    faultDataUrl: str = Field("", description="故障数据文件链接")
    intentId: str = Field("", description="平台意图识别名称")
    query_confidence: float = Field(0.0, description="意图置信度 0-1")
    conversationId: str = Field("", description="会话唯一 ID")
    context_history: list[ContextTurn] = Field(
        default_factory=list,
        description="多轮对话历史",
    )

    def is_bulk(self) -> bool:
        """是否为批量输入"""
        return len(self.bulk_records) > 0
