"""Adapter 模块测试 — 平台协议模型 + FastAPI 端点"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from diagnosis_agent.adapter.models import (
    DiagnosisStatus,
    ExecutionStatus,
    PlatformCallbackBody,
    PlatformCallbackData,
    PlatformContextTurn,
    PlatformData,
    PlatformDiagnosisRequest,
    PlatformEntities,
    PlatformResult,
    PlatformStatusResponse,
    PlatformSubmitResponse,
    DiagnosisTask,
    to_callback_result,
    to_error_callback,
    to_standard_input,
)
from diagnosis_agent.models.diagnostic_output import (
    StandardDiagnosisResult,
    StandardOutput,
    OutputCode,
)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def sample_request() -> PlatformDiagnosisRequest:
    return PlatformDiagnosisRequest(
        clientRequestId="client-001",
        taskId="task-001",
        traceId="trace-001",
        conversationId="conv-001",
        faultDescription="车辆行驶中加速无力，DTC P0087",
        callbackUrl="http://platform/callback",
        vehicleModel="SUV-X1",
        data=PlatformData(
            rawQuery="加速无力",
            entities=PlatformEntities(
                dtcCode=["P0087"],
                project="SUV-X1",
                component="燃油系统",
            ),
        ),
    )


@pytest.fixture
def sample_standard_output() -> StandardOutput:
    return StandardOutput(
        code=0,
        msg="success",
        input_ref={"raw_query": "加速无力", "mcuid": "SUV-X1"},
        diagnosis_confidence=0.85,
        need_multi_round=False,
        diagnosis_result=StandardDiagnosisResult(
            fault_root_cause=["燃油泵控制模块故障"],
            fault_trigger_condition="急加速时",
            classification="驱动异常故障",
            solution=["更换燃油泵控制模块"],
            risk_warning="继续行驶可能导致熄火",
            confidence=0.85,
        ),
    )


# ========================================================================
# Models
# ========================================================================


class TestPlatformDiagnosisRequest:
    def test_minimal(self):
        """仅必填字段"""
        req = PlatformDiagnosisRequest(
            clientRequestId="c1",
            faultDescription="故障",
            callbackUrl="http://cb",
        )
        assert req.clientRequestId == "c1"
        assert req.protocolVersion == "1.0"
        assert req.domainCode == "EV_DRIVE"

    def test_full(self, sample_request):
        assert sample_request.vin == ""
        assert sample_request.data.entities.dtcCode == ["P0087"]
        assert sample_request.data.entities.component == "燃油系统"

    def test_context_history(self):
        req = PlatformDiagnosisRequest(
            clientRequestId="c2",
            faultDescription="异响",
            callbackUrl="http://cb",
            data=PlatformData(
                contextHistory=[
                    PlatformContextTurn(userQuery="之前有故障码", agentResponse="请提供DTC"),
                ],
            ),
        )
        assert len(req.data.contextHistory) == 1
        assert req.data.contextHistory[0].userQuery == "之前有故障码"


class TestPlatformResult:
    def test_defaults(self):
        r = PlatformResult()
        assert r.faultRootCause == []
        assert r.diagnosisConfidence == 0.0
        assert r.needMultiRound is False

    def test_full(self):
        r = PlatformResult(
            faultRootCause=["原因A"],
            classification="控制异常",
            solution=["方案B"],
            diagnosisConfidence=0.9,
        )
        assert r.faultRootCause == ["原因A"]
        assert r.classification == "控制异常"
        assert r.solution == ["方案B"]


class TestPlatformStatusResponse:
    def test_fields(self):
        resp = PlatformStatusResponse(
            clientRequestId="c1",
            executionStatus="SUCCESS",
            diagnosisStatus="COMPLETED",
            updatedAt="2026-01-01T00:00:00",
            result={"key": "val"},
        )
        assert resp.clientRequestId == "c1"
        assert resp.executionStatus == "SUCCESS"
        assert resp.result == {"key": "val"}


class TestPlatformSubmitResponse:
    def test_defaults(self):
        resp = PlatformSubmitResponse(
            requestId="req-1",
            timestamp="ts",
        )
        assert resp.code == 0
        assert resp.success is True
        assert resp.data["status"] == "PENDING"


class TestDiagnosisTask:
    def test_create(self, sample_request):
        task = DiagnosisTask(sample_request)
        assert task.requestId.startswith("EV-REQ-")
        assert task.clientRequestId == "client-001"
        assert task.status == ExecutionStatus.PENDING.value
        assert task.result is None

    def test_touch(self, sample_request):
        task = DiagnosisTask(sample_request)
        old = task.updated_at
        task.touch()
        assert task.updated_at >= old

    def test_idempotent(self, sample_request):
        """相同 clientRequestId 应返回同一 task"""
        task1 = DiagnosisTask(sample_request)
        task2 = DiagnosisTask(sample_request)
        assert task1.requestId != task2.requestId  # 每次新建 ID 不同


# ========================================================================
# Mapping
# ========================================================================


class TestToStandardInput:
    def test_basic(self, sample_request):
        std = to_standard_input(sample_request)
        assert std.raw_query == "车辆行驶中加速无力，DTC P0087"
        assert std.mcuid == "SUV-X1"
        assert std.entities.dtc_code == ["P0087"]
        assert std.entities.project == "SUV-X1"

    def test_entities_fallback(self):
        """data 层 entities 为空时用顶层字段"""
        req = PlatformDiagnosisRequest(
            clientRequestId="c1",
            faultDescription="故障",
            callbackUrl="http://cb",
            vehicleModel="SEDAN-A2",
            data=PlatformData(
                rawQuery="故障描述",
                entities=PlatformEntities(),
            ),
        )
        std = to_standard_input(req)
        assert std.mcuid == "SEDAN-A2"

    def test_conversation_id(self, sample_request):
        std = to_standard_input(sample_request)
        assert std.conversation_id == "conv-001"


class TestToCallbackResult:
    def test_success(self, sample_request, sample_standard_output):
        task = DiagnosisTask(sample_request)
        cb = to_callback_result(task, sample_standard_output)
        assert cb.code == 0
        assert cb.data.executionStatus == ExecutionStatus.SUCCESS.value
        assert cb.data.diagnosisStatus == DiagnosisStatus.COMPLETED.value
        assert cb.data.result.faultRootCause == ["燃油泵控制模块故障"]
        assert cb.data.result.classification == "驱动异常故障"
        assert cb.data.result.diagnosisConfidence == 0.85

    def test_request_id(self, sample_request, sample_standard_output):
        task = DiagnosisTask(sample_request)
        cb = to_callback_result(task, sample_standard_output)
        assert cb.requestId == task.requestId


class TestToErrorCallback:
    def test_basic(self, sample_request):
        task = DiagnosisTask(sample_request)
        cb = to_error_callback(task, "模型调用失败")
        assert cb.code == 1
        assert cb.success is False
        assert cb.data.executionStatus == ExecutionStatus.FAILED.value
        assert cb.data.diagnosisStatus == DiagnosisStatus.ERROR.value
        assert cb.message == "模型调用失败"

    def test_custom_status(self, sample_request):
        task = DiagnosisTask(sample_request)
        cb = to_error_callback(task, "超出范围", DiagnosisStatus.DOMAIN_REJECTED.value)
        assert cb.data.diagnosisStatus == DiagnosisStatus.DOMAIN_REJECTED.value


# ========================================================================
# FastAPI 端点
# ========================================================================


class TestServer:
    """FastAPI 端点测试（使用 TestClient）"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from diagnosis_agent.adapter.server import app
        self.client = TestClient(app)

    def test_health_live(self):
        resp = self.client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "UP"}

    def test_health_ready(self):
        resp = self.client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_get_manifest(self):
        resp = self.client.get("/.well-known/diagnostic-agent-manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["manifestVersion"] == "1.0"
        assert data["agentCode"] == "ev_drive_agent"
        assert "capabilities" in data
        assert "endpoints" in data

    def test_submit_diagnosis_accepts_request(self):
        payload = {
            "clientRequestId": "test-client-001",
            "faultDescription": "加速无力",
            "callbackUrl": "http://localhost/callback",
            "vehicleModel": "SUV-X1",
            "data": {
                "rawQuery": "加速无力",
                "entities": {"dtcCode": ["P0087"]},
            },
        }
        resp = self.client.post("/api/v1/diagnoses/async", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["status"] == "PENDING"
        assert "requestId" in data

    def test_submit_idempotent(self):
        payload = {
            "clientRequestId": "test-client-idempotent",
            "faultDescription": "怠速不稳",
            "callbackUrl": "http://localhost/callback",
        }
        # 第一次提交
        r1 = self.client.post("/api/v1/diagnoses/async", json=payload)
        assert r1.status_code == 200
        rid1 = r1.json()["requestId"]

        # 相同 clientRequestId 再次提交
        r2 = self.client.post("/api/v1/diagnoses/async", json=payload)
        assert r2.status_code == 200
        rid2 = r2.json()["requestId"]
        assert rid2 == rid1  # 幂等，返回相同 requestId

    def test_status_query_not_found(self):
        resp = self.client.get("/api/v1/diagnoses/nonexistent/status")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "requestId not found"

    def test_status_query_found(self):
        # 先提交一个请求
        payload = {
            "clientRequestId": "test-client-status",
            "faultDescription": "异响",
            "callbackUrl": "http://localhost/callback",
        }
        submit = self.client.post("/api/v1/diagnoses/async", json=payload)
        request_id = submit.json()["requestId"]

        # 查询状态
        resp = self.client.get(f"/api/v1/diagnoses/{request_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["clientRequestId"] == "test-client-status"
        assert data["executionStatus"] in ("PENDING", "RUNNING", "SUCCESS", "FAILED")
        assert "updatedAt" in data