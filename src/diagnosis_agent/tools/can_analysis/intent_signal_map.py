"""意图 ID 到 CAN 信号/消息 ID 的映射配置

平台诊断接口会根据用户问题输出 intentId（如 MCU_001 ~ MCU_006），
本模块把每个意图 ID 映射到诊断时需要重点关注的信号名称和消息 ID。

工具在解码报文前会根据传入的 intent_id 查找本映射，仅解析并导出
相关信号的数据，避免每次全量处理。

注意：
- 以下信号名为示例占位符，需根据项目实际 DBC 进行填充。
- 信号名必须与 DBC 中定义完全一致；不存在的名称会在运行时产生警告并被忽略。
- 消息 ID 使用十进制整数，与 DBC frame_id 对齐。
"""

from typing import Dict, List, Optional


class IntentSignalMapping:
    """单个意图 ID 对应的筛选配置"""

    def __init__(
        self,
        signals: Optional[List[str]] = None,
        message_ids: Optional[List[int]] = None,
        description: str = "",
    ):
        self.signals = signals or []
        self.message_ids = message_ids or []
        self.description = description


# 意图 ID -> 信号/消息映射表
# key 必须与平台 manifest 中 capabilities[].intentIds 保持一致。
INTENT_SIGNAL_MAP: Dict[str, IntentSignalMapping] = {
    "MCU_001": IntentSignalMapping(
        description="电驱 DTC 相关诊断",
        signals=[
            # TODO: 按实际 DBC 填充 DTC 相关信号
            # "DTC_P0A2B",
            # "DTC_U1624",
        ],
        message_ids=[
            # TODO: 按实际 DBC 填充 DTC 消息 ID（十进制）
            # 0x123,
        ],
    ),
    "MCU_002": IntentSignalMapping(
        description="电机功能异常",
        signals=[
            # TODO: 按实际 DBC 填充电机功能相关信号
            # "MotorSpeed",
            # "MotorTorque",
            # "MotorCurrent",
            # "MotorVoltage",
        ],
        message_ids=[
            # TODO: 按实际 DBC 填充电机功能消息 ID（十进制）
        ],
    ),
    "MCU_003": IntentSignalMapping(
        description="电机 NVH",
        signals=[
            # TODO: 按实际 DBC 填充 NVH 相关信号
            # "MotorSpeed",
            # "MotorTorqueRipple",
            # "VibrationIndex",
        ],
        message_ids=[
            # TODO: 按实际 DBC 填充 NVH 消息 ID（十进制）
        ],
    ),
    "MCU_004": IntentSignalMapping(
        description="热管理",
        signals=[
            # 示例：温度信号、水泵信号等
            "MotorTemp",
            "InverterTemp",
            "CoolantTemp",
            "WaterPumpSpeed",
            "WaterPumpDuty",
            "RadiatorFanSpeed",
            "IGBTTemp",
        ],
        message_ids=[
            # TODO: 按实际 DBC 填充热管理消息 ID（十进制）
            # 0x201,
        ],
    ),
    "MCU_005": IntentSignalMapping(
        description="通信链路",
        signals=[
            # 示例：超时信号、通信状态信号等
            "ECU_CommTimeout",
            "MCU_CommTimeout",
            "BMS_CommTimeout",
            "VCU_CommTimeout",
            "CAN_BusOffCounter",
            "MessageTimeoutFlag",
        ],
        message_ids=[
            # TODO: 按实际 DBC 填充通信诊断消息 ID（十进制）
            # 0x301,
        ],
    ),
    "MCU_006": IntentSignalMapping(
        description="其他电驱问题",
        signals=[
            # TODO: 按实际 DBC 填充其他常用电驱信号
            # "DCBusVoltage",
            # "DCBusCurrent",
        ],
        message_ids=[
            # TODO: 按实际 DBC 填充其他消息 ID（十进制）
        ],
    ),
}


def get_mapping(intent_id: str) -> Optional[IntentSignalMapping]:
    """根据意图 ID 获取信号/消息映射配置"""
    return INTENT_SIGNAL_MAP.get(intent_id)


def list_supported_intents() -> List[str]:
    """返回当前支持的所有意图 ID"""
    return list(INTENT_SIGNAL_MAP.keys())