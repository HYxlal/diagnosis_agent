"""Agent Manifest — 元信息配置"""

MANIFEST = {
    "manifestVersion": "1.0",
    "agentCode": "ev_drive_agent",
    "agentName": "新能源电驱诊断Agent",
    "version": "0.5.0",
    "domains": ["EV_DRIVE"],
    "capabilities": [
        {
            "capabilityCode": "ev_drive_diagnosis",
            "intentIds": [
                "MCU_001",  # 电驱DTC
                "MCU_002",  # 电机功能异常
                "MCU_003",  # 电机NVH
                "MCU_004",  # 热管理
                "MCU_005",  # 通信链路
                "MCU_006",  # 其他电驱问题
            ],
            "requestSchemaVersion": "1.0",
            "callbackSchemaVersion": "1.0",
        }
    ],
    "endpoints": {
        "asyncSubmit": "/api/v1/diagnoses/async",
        "statusQuery": "/api/v1/diagnoses/{requestId}/status",
        "liveness": "/health/live",
        "readiness": "/health/ready",
    },
    "formSchema": {
        "fields": [
            {
                "fieldCode": "dtcCode",
                "fieldName": "故障DTC码",
                "dataType": "string",
                "required": False,
                "source": "USER_FORM",
                "missingMessage": "请补充故障DTC码",
            },
            {
                "fieldCode": "component",
                "fieldName": "涉及部件/根因",
                "dataType": "string",
                "required": False,
                "source": "USER_FORM",
            },
            {
                "fieldCode": "workingCondition",
                "fieldName": "故障发生工况",
                "dataType": "string",
                "required": False,
                "source": "USER_FORM",
                "missingMessage": "请补充故障发生时的车辆工况",
            },
            {
                "fieldCode": "project",
                "fieldName": "项目/车型代号",
                "dataType": "string",
                "required": False,
                "source": "PLATFORM_CONTEXT",
            },
            {
                "fieldCode": "softwareVersion",
                "fieldName": "软件版本",
                "dataType": "string",
                "required": False,
                "source": "USER_FORM",
            },
        ]
    },
}