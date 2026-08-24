"""Adapter 模块测试 — 平台协议模型 + 映射函数 + FastAPI 端点"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from diagnosis_agent.adapter.models import (
    DiagnosisStatus,
    ExecutionStatus,
    PlatformCallbackBody,
    PlatformCallbackData,
    PlatformDiagnosisRequest,
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
        VIN="LSVAF09E2HN1234567",
        raw_query="车辆行驶中加速无力，DTC P0087",
        faultOccurTime="2026-08-21T14:30:00+08:00",
        mileage=35280.5,
        vehicleModel="SUV-X1",
        clientRequestId="client-001",
        callbackUrl="http://platform/callback",
        traceId="trace-001",
        conversationId="conv-001",
        dtcCode=["P0087"],
        motorPosition="前电机",
    )


@pytest.fixture
def sample_standard_output() -> StandardOutput:
    return StandardOutput(
        code=0,
        msg="success",
        input_ref={"raw_query": "加速无力", "VIN": "LSVAF09E2HN1234567"},
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
            VIN="V123",
            raw_query="故障",
            faultOccurTime="2026-01-01T00:00:00",
            mileage=1000,
            vehicleModel="H37A",
            clientRequestId="c1",
            callbackUrl="http://cb",
        )
        assert req.VIN == "V123"
        assert req.raw_query == "故障"
        assert req.vehicleModel == "H37A"

    def test_full(self, sample_request):
        assert sample_request.VIN == "LSVAF09E2HN1234567"
        assert sample_request.raw_query == "车辆行驶中加速无力，DTC P0087"
        assert sample_request.dtcCode == ["P0087"]
        assert sample_request.motorPosition == "前电机"


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


class TestPlatformSubmitResponse:
    def test_defaults(self):
        resp = PlatformSubmitResponse(requestId="req-1", timestamp="ts")
        assert resp.code == 0
        assert resp.success is True
        assert resp.data["status"] == "PENDING"


class TestDiagnosisTask:
    def test_create(self, sample_request):
        task = DiagnosisTask(sample_request)
        assert task.requestId.startswith("EV-REQ-")
        assert task.clientRequestId == "client-001"
        assert task.status == ExecutionStatus.PENDING.value

    def test_touch(self, sample_request):
        task = DiagnosisTask(sample_request)
        old = task.updated_at
        task.touch()
        assert task.updated_at >= old


# ========================================================================
# Mapping
# ========================================================================


class TestToStandardInput:
    def test_basic(self, sample_request):
        std = to_standard_input(sample_request)
        assert std.VIN == "LSVAF09E2HN1234567"
        assert std.raw_query == "车辆行驶中加速无力，DTC P0087"
        assert std.vehicleModel == "SUV-X1"
        assert std.dtcCode == ["P0087"]
        assert std.mileage == 35280.5
        assert std.motorPosition.value == "前电机"

    def test_conversation_id(self, sample_request):
        std = to_standard_input(sample_request)
        assert std.conversationId == "conv-001"

    def test_defaults(self):
        req = PlatformDiagnosisRequest(
            VIN="V123", raw_query="test", faultOccurTime="2026-01-01T00:00:00",
            mileage=100, vehicleModel="H37A", clientRequestId="c1", callbackUrl="http://cb",
        )
        std = to_standard_input(req)
        assert std.dtcCode == []
        assert std.faultWorkConditionList.value == "无法确认故障工况"


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

    def test_submit_diagnosis_accepts_request(self):
        payload = {
            "clientRequestId": "test-client-001",
            "VIN": "LSVAF09E2HN1234567",
            "raw_query": "加速无力",
            "faultOccurTime": "2026-01-01T00:00:00",
            "mileage": 1000,
            "vehicleModel": "SUV-X1",
            "callbackUrl": "http://localhost/callback",
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
            "VIN": "V123",
            "raw_query": "怠速不稳",
            "faultOccurTime": "2026-01-01T00:00:00",
            "mileage": 1000,
            "vehicleModel": "H37A",
            "callbackUrl": "http://localhost/callback",
        }
        r1 = self.client.post("/api/v1/diagnoses/async", json=payload)
        assert r1.status_code == 200
        rid1 = r1.json()["requestId"]

        r2 = self.client.post("/api/v1/diagnoses/async", json=payload)
        assert r2.status_code == 200
        rid2 = r2.json()["requestId"]
        assert rid2 == rid1

    def test_status_query_not_found(self):
        resp = self.client.get("/api/v1/diagnoses/nonexistent/status")
        assert resp.status_code == 404

    def test_status_query_found(self):
        payload = {
            "clientRequestId": "test-client-status",
            "VIN": "V123",
            "raw_query": "异响",
            "faultOccurTime": "2026-01-01T00:00:00",
            "mileage": 1000,
            "vehicleModel": "H37A",
            "callbackUrl": "http://localhost/callback",
        }
        submit = self.client.post("/api/v1/diagnoses/async", json=payload)
        request_id = submit.json()["requestId"]

        resp = self.client.get(f"/api/v1/diagnoses/{request_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["clientRequestId"] == "test-client-status"
        assert "updatedAt" in data