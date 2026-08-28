"""适配层数据模型 — 平台协议 ↔ 内部模型映射

输入请求完全扁平化，与内部 StandardInput 1:1 对齐，删除所有冗余嵌套对象。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ========================================================================
# 平台请求
# ========================================================================

class PlatformDiagnosisRequest(BaseModel):
    """平台诊断请求 — 统一接入协议标准格式，完全扁平化，无嵌套 data/entities 子对象

    与内部 StandardInput 字段 1:1 对齐，没有冗余字段。
    """
    # ---- 平台交互字段 ----
    intentId: str = Field("", description="意图识别名称，Agent 平台输出，如 MCU_001")
    query_confidence: float = Field(0.0, ge=0.0, le=1.0, description="意图识别置信度 0~1 浮点数")
    conversationId: str = Field("", description="多轮对话会话唯一 ID")
    context_history: list[dict] = Field(
        default_factory=list,
        description="多轮对话历史降级备用路径，每个元素为 {query: str, agent_answer: str}。"
                    "内部优先通过 conversationId 关联 SessionManager，保留完整工具调用历史，"
                    "仅在本地会话不可用时使用此列表重建纯文本消息。"
    )
    clientRequestId: str = Field(..., description="由 AI 诊断平台生成，用于提交幂等和请求关联")
    traceId: str = Field("", description="调用链路、日志串联排查 trace ID")
    callbackUrl: str = Field(..., description="诊断完成后平台回调 URL")

    # ---- 表单必填业务字段 ----
    VIN: str = Field(..., description="VIN 码，整车唯一标识，17 位字符")
    raw_query: str = Field(..., description="用户所填写故障问题描述，自然语言")
    faultOccurTime: str = Field(..., description="故障工况发生时间，ISO 8601 date-time 格式")
    mileage: float = Field(..., ge=0.0, description="车辆行驶里程，单位 km")
    instrumentIndicatorList: list[str] = Field(
        default_factory=list,
        description="仪表指示灯多选，枚举值列表: 电机故障红灯|电机故障黄灯|动力系统故障红灯|动力系统故障黄灯|电机过温故障灯|无",
    )
    vehicleModel: str = Field(..., description="车型代号 / 项目型号")

    # ---- 表单可选业务字段 ----
    faultWorkConditionList: str = Field(
        "无法确认故障工况",
        description="表单单选故障工况: 平稳驾驶|激烈驾驶|车辆启动|充电|无法确认故障工况",
    )
    dtcCode: list[str] = Field(default_factory=list, description="数据湖获取 DTC 故障码，无故障码传空数组")
    softwareVersion: str = Field("", description="MCU 软件版本信息")
    motorPosition: str = Field(
        "无法确认具体电机",
        description="电机位置: 前电机|后电机|无法确认具体电机",
    )
    faultDataUrl: str = Field("", description="故障数据链接，CAN 总线日志/DBC 文件下载地址，多文件逗号分隔")


# ========================================================================
# 平台受理响应
# ========================================================================

class PlatformSubmitResponse(BaseModel):
    """平台提交诊断的受理响应（立即返回，不等待诊断完成）"""
    code: int = 0
    message: str = "accepted"
    success: bool = True
    requestId: str
    timestamp: str = ""
    data: dict = Field(default_factory=lambda: {"status": "PENDING"})


# ========================================================================
# 回调结果
# ========================================================================

class ExecutionStatus(str, Enum):
    """异步任务执行状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DiagnosisStatus(str, Enum):
    """诊断结果状态（区别于执行状态，描述诊断结论）"""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    NEED_MORE_INFO = "NEED_MORE_INFO"
    DOMAIN_REJECTED = "DOMAIN_REJECTED"
    NOT_EVALUATED = "NOT_EVALUATED"
    ERROR = "ERROR"


class PlatformResult(BaseModel):
    """诊断结果核心字段 — 平台回调协议的标准格式"""
    faultRootCause: list[str] = Field(default_factory=list)
    faultTriggerCondition: str = ""
    classification: str = ""
    solution: list[str] = Field(default_factory=list)
    riskWarning: str = ""
    needMultiRound: bool = False
    followUpQuestion: str = ""
    diagnosisConfidence: float = 0.0
    missingFields: list[str] = Field(default_factory=list)


class PlatformCallbackData(BaseModel):
    """回调数据体 — 包含 agent 标识、执行状态、诊断状态和结果"""
    clientRequestId: str
    agentCode: str = "ev_drive_agent"
    domainCode: str = "EV_DRIVE"
    executionStatus: str = ExecutionStatus.SUCCESS.value
    diagnosisStatus: str = DiagnosisStatus.COMPLETED.value
    result: PlatformResult = Field(default_factory=PlatformResult)


class PlatformCallbackBody(BaseModel):
    """回调平台的完整请求体"""
    code: int = 0
    message: str = "success"
    success: bool = True
    requestId: str
    timestamp: str = ""
    data: PlatformCallbackData = Field(default_factory=PlatformCallbackData)


# ========================================================================
# 内部任务状态
# ========================================================================

class PlatformStatusResponse(BaseModel):
    """状态查询响应"""
    clientRequestId: str
    executionStatus: str
    diagnosisStatus: str
    updatedAt: str = ""
    result: dict = Field(default_factory=dict)


class DiagnosisTask:
    def __init__(self, req: PlatformDiagnosisRequest):
        self.requestId = f"EV-REQ-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:8]}"
        self.clientRequestId = req.clientRequestId
        self.conversationId = req.conversationId
        self.callbackUrl = req.callbackUrl
        self.status = ExecutionStatus.PENDING.value
        self.result: Optional[PlatformCallbackBody] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self._req = req
        # CAN报文兜底字段：暂存文件引用、诊断轮次
        self.stored_fault_data_url: str | None = req.faultDataUrl if req.faultDataUrl else None
        self.diagnose_round: int = 0

    def touch(self):
        self.updated_at = datetime.now()


# ========================================================================
# 平台 → 内部模型映射
# ========================================================================

def to_standard_input(req: PlatformDiagnosisRequest):
    """将平台请求转为内部 StandardInput — 扁平化，字段 1:1 对应"""
    from ..models.input import StandardInput

    return StandardInput(
        intentId=req.intentId,
        query_confidence=req.query_confidence,
        conversationId=req.conversationId,
        context_history=req.context_history,
        clientRequestId=req.clientRequestId,
        traceId=req.traceId,
        VIN=req.VIN,
        raw_query=req.raw_query,
        faultOccurTime=req.faultOccurTime,
        mileage=req.mileage,
        instrumentIndicatorList=req.instrumentIndicatorList,
        vehicleModel=req.vehicleModel,
        faultWorkConditionList=req.faultWorkConditionList,
        dtcCode=req.dtcCode,
        softwareVersion=req.softwareVersion,
        motorPosition=req.motorPosition,
        faultDataUrl=req.faultDataUrl,
    )


def to_callback_result(
    task: DiagnosisTask,
    standard_output,
) -> PlatformCallbackBody:
    """将内部 StandardOutput 转为平台回调格式"""
    result = standard_output.diagnosis_result
    confidence = standard_output.diagnosis_confidence

    return PlatformCallbackBody(
        requestId=task.requestId,
        timestamp=datetime.now().isoformat(),
        data=PlatformCallbackData(
            clientRequestId=task.clientRequestId,
            executionStatus=ExecutionStatus.SUCCESS.value,
            diagnosisStatus=DiagnosisStatus.COMPLETED.value,
            result=PlatformResult(
                faultRootCause=result.fault_root_cause if result else [],
                faultTriggerCondition=result.fault_trigger_condition if result else "",
                classification=result.classification if result else "",
                solution=result.solution if result else [],
                riskWarning=result.risk_warning if result else "",
                needMultiRound=standard_output.need_multi_round,
                followUpQuestion=standard_output.follow_up_question,
                diagnosisConfidence=confidence,
            ),
        ),
    )


def to_error_callback(
    task: DiagnosisTask,
    message: str,
    diagnosis_status: str = DiagnosisStatus.ERROR.value,
) -> PlatformCallbackBody:
    """构建错误回调体 — code=1, success=False"""
    return PlatformCallbackBody(
        code=1,
        message=message,
        success=False,
        requestId=task.requestId,
        timestamp=datetime.now().isoformat(),
        data=PlatformCallbackData(
            clientRequestId=task.clientRequestId,
            executionStatus=ExecutionStatus.FAILED.value,
            diagnosisStatus=diagnosis_status,
        ),
    )
