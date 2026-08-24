"""FastAPI 适配层服务 — 平台接入路由"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings, reset_settings
from .handler import task_manager, execute_diagnosis
from .manifest import MANIFEST
from .models import (
    DiagnosisStatus,
    ExecutionStatus,
    PlatformDiagnosisRequest,
    PlatformStatusResponse,
    PlatformSubmitResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EV Drive Diagnosis Agent",
    version="0.5.0",
    description="新能源电驱诊断 Agent — 适配 AI 诊断平台统一接入协议",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================================================
# 核心接口
# ========================================================================

@app.post("/api/v1/diagnoses/async")
async def submit_diagnosis(req: PlatformDiagnosisRequest):
    """平台提交诊断请求"""
    reset_settings()

    # 幂等检查
    existing = task_manager.get_by_client_id(req.clientRequestId)
    if existing:
        return PlatformSubmitResponse(
            requestId=existing.requestId,
            timestamp=datetime.now().isoformat(),
            data={"clientRequestId": req.clientRequestId, "status": existing.status},
        )

    # 创建任务
    task = task_manager.create_task(req)
    logger.info(f"[{task.requestId}] 受理诊断: {req.raw_query[:60]}...")

    # 后台异步执行
    asyncio.create_task(execute_diagnosis(task, req))

    return PlatformSubmitResponse(
        requestId=task.requestId,
        timestamp=datetime.now().isoformat(),
        data={"clientRequestId": req.clientRequestId, "status": ExecutionStatus.PENDING.value},
    )


@app.get("/api/v1/diagnoses/{request_id}/status")
async def query_status(request_id: str):
    """状态查询（回调补偿）"""
    task = task_manager.get(request_id)
    if not task:
        raise HTTPException(status_code=404, detail="requestId not found")

    # 状态映射：PENDING/RUNNING → RUNNING，SUCCESS → COMPLETED
    result = {}
    diagnosis_status = DiagnosisStatus.RUNNING.value
    if task.status == ExecutionStatus.SUCCESS.value and task.result:
        diagnosis_status = DiagnosisStatus.COMPLETED.value
        # 提取诊断结果中的核心字段，避免返回完整回调体
        result = task.result.data.result.model_dump() if hasattr(task.result.data.result, 'model_dump') else {}

    return PlatformStatusResponse(
        clientRequestId=task.clientRequestId,
        executionStatus=task.status,
        diagnosisStatus=diagnosis_status,
        updatedAt=task.updated_at.isoformat(),
        result=result,
    )


# ========================================================================
# 健康检查
# ========================================================================

@app.get("/health/live")
async def health_live():
    return {"status": "UP"}


@app.get("/health/ready")
async def health_ready():
    settings = get_settings()
    issues = []
    if settings.neo4j.url:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.neo4j.url,
                auth=(settings.neo4j.user, settings.neo4j.password),
            )
            driver.verify_connectivity()
            driver.close()
        except Exception as e:
            issues.append(f"neo4j: {e}")
    return {
        "status": "UP" if not issues else "DOWN",
        "issues": issues,
    }


# ========================================================================
# Agent Manifest
# ========================================================================

@app.get("/.well-known/diagnostic-agent-manifest")
async def get_manifest():
    return MANIFEST