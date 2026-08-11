"""测试推理 Agent 和报告生成"""

import pytest
from datetime import datetime

from diagnosis_agent.models.diagnosis import (
    DatabaseEntry,
    DiagnosticFinding,
    DiagnosticOutput,
    DiagnosticReport,
    ReActStep,
    SimilarCase,
    ToolCallRecord,
)
from diagnosis_agent.reporting.markdown import generate_markdown_report
from diagnosis_agent.reporting.entries import (
    generate_database_entry_csv,
    generate_database_entry_json,
)
from diagnosis_agent.agent.prompts import (
    build_no_similar_case_prompt,
    build_similar_case_prompt,
    build_system_prompt,
    format_similar_cases_for_prompt,
)


class TestPrompts:
    """测试 Prompt 模板"""

    def test_system_prompt(self):
        prompt = build_system_prompt("tool1: description1")
        assert "tool1: description1" in prompt
        assert "故障诊断" in prompt
        assert "ReAct" in prompt

    def test_similar_case_prompt(self):
        from langchain_core.documents import Document
        cases = [
            Document(
                page_content="发动机故障灯亮",
                metadata={
                    "id": "REC-001",
                    "problem_description": "发动机故障灯亮",
                    "root_cause": "传感器故障",
                    "countermeasure": "更换传感器",
                    "drive_code": "DRV-001",
                    "vehicle_type": "SUV-X1",
                    "dashboard_indicator": "发动机故障灯",
                    "dtc_code": "P0107",
                    "fault_scenario": "冷启动",
                    "score": 0.85,
                },
            )
        ]
        text = format_similar_cases_for_prompt(cases)
        assert "REC-001" in text
        assert "发动机故障灯亮" in text
        assert "0.85" in text

    def test_no_similar_case_prompt(self):
        prompt = build_no_similar_case_prompt("发动机故障灯亮了")
        assert "发动机故障灯亮了" in prompt
        assert "Final Answer" in prompt


class TestReActStep:
    """测试 ReAct 步骤模型"""

    def test_create_step(self):
        step = ReActStep(
            step=1,
            thought="需要检索相似工况",
            action="search_similar_incidents",
            action_input={"query": "发动机故障灯亮"},
            observation="找到2条相似记录",
        )
        assert step.step == 1
        assert step.action == "search_similar_incidents"


class TestToolCallRecord:
    """测试工具调用记录模型"""

    def test_create_record(self):
        record = ToolCallRecord(
            tool_name="search_similar_incidents",
            parameters={"query": "发动机故障灯亮"},
            result=[{"record_id": "REC-001"}],
            duration_ms=52.3,
        )
        assert record.tool_name == "search_similar_incidents"
        assert record.duration_ms == 52.3


class TestReportGeneration:
    """测试报告生成"""

    @pytest.fixture
    def sample_output(self):
        """构建样例诊断输出（含 ReAct 步骤和工具调用记录）"""
        similar_case = SimilarCase(
            record_id="REC-001",
            problem_description="发动机故障灯亮，怠速不稳",
            root_cause="进气压力传感器故障",
            countermeasure="更换进气压力传感器",
            drive_code="DRV-001",
            vehicle_type="SUV-X1",
            dashboard_indicator="发动机故障灯",
            dtc_code="P0107",
            fault_scenario="冷启动后低速行驶",
            similarity=0.85,
        )

        finding = DiagnosticFinding(
            title="进气压力传感器故障",
            description="根据DTC码P0107和相似工况，判断为进气压力传感器线路接触不良",
            confidence=0.85,
            evidence=["DTC P0107", "相似工况 REC-001"],
        )

        react_steps = [
            ReActStep(
                step=1,
                thought="需要检索是否有相似工况",
                action="search_similar_incidents",
                action_input={"query": "发动机故障灯亮 怠速不稳"},
                observation="找到2条相似记录",
            ),
            ReActStep(
                step=2,
                thought="已有足够信息进行诊断",
                action="Final Answer",
                action_input={},
                observation='{"root_cause": "传感器故障", "confidence": 0.85}',
            ),
        ]

        tool_calls = [
            ToolCallRecord(
                tool_name="search_similar_incidents",
                parameters={"query": "发动机故障灯亮 怠速不稳"},
                result=[{"record_id": "REC-001"}],
                duration_ms=52.3,
            ),
        ]

        report = DiagnosticReport(
            diagnosis_id="DIAG-TEST-001",
            diagnosis_time=datetime.now(),
            input_summary="发动机故障灯亮，怠速不稳",
            has_similar_cases=True,
            similar_cases=[similar_case],
            findings=[finding],
            recommended_countermeasure="更换进气压力传感器",
            react_steps=react_steps,
            reasoning_narrative="根据DTC码P0107和检索到的相似工况REC-001，推断为进气压力传感器故障。",
            reasoning_chain=["需要检索是否有相似工况", "已有足够信息进行诊断"],
            tool_calls=tool_calls,
            tools_used=["search_similar_incidents"],
        )

        entry = DatabaseEntry(
            diagnosis_id="DIAG-TEST-001",
            diagnosis_time=datetime.now(),
            problem_description="发动机故障灯亮",
            root_cause="进气压力传感器故障",
            countermeasure="更换传感器",
            drive_code="DRV-001",
            vehicle_type="SUV-X1",
            dashboard_indicator="发动机故障灯",
            dtc_code="P0107",
            fault_scenario="冷启动",
            diagnostic_confidence=0.85,
            based_on_similar=True,
            similar_record_ids=["REC-001"],
        )

        return DiagnosticOutput(report=report, database_entry=entry)

    def test_markdown_report(self, sample_output, tmp_path):
        """测试 Markdown 报告生成"""
        md_path = generate_markdown_report(sample_output, output_dir=tmp_path)
        assert md_path.exists()

        content = md_path.read_text(encoding="utf-8")
        assert "车辆故障诊断报告" in content
        assert "DIAG-TEST-001" in content
        # 检查 ReAct 步骤
        assert "思维链" in content
        assert "search_similar_incidents" in content
        # 检查工具调用记录
        assert "工具调用记录" in content
        # 检查推断过程叙述
        assert "推断过程叙述" in content
        assert "P0107" in content
        # 检查相似工况索引
        assert "相似工况索引" in content
        assert "REC-001" in content

    def test_csv_entry(self, sample_output, tmp_path):
        csv_path = generate_database_entry_csv(sample_output, output_dir=tmp_path)
        assert csv_path.exists()

        content = csv_path.read_text(encoding="utf-8-sig")
        assert "DIAG-TEST-001" in content
        assert "发动机故障灯亮" in content
        # CSV 表头应为中文
        assert "问题描述" in content
        assert "根因" in content

    def test_json_entry(self, sample_output, tmp_path):
        import json

        json_path = generate_database_entry_json(sample_output, output_dir=tmp_path)
        assert json_path.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["diagnosis_id"] == "DIAG-TEST-001"
        # 8 列业务字段使用中文表头
        assert data["问题描述"] == "发动机故障灯亮"
        assert data["根因"] == "进气压力传感器故障"
        # 元数据保持英文
        assert data["diagnostic_confidence"] == 0.85

    def test_double_layer_output(self, sample_output):
        """测试双层输出完整性"""
        report = sample_output.report
        assert report.diagnosis_id == sample_output.database_entry.diagnosis_id
        assert len(report.similar_cases) > 0
        assert len(report.findings) > 0
        assert report.recommended_countermeasure != ""
        assert len(report.react_steps) > 0
        assert len(report.tool_calls) > 0
        assert report.reasoning_narrative != ""

        entry = sample_output.database_entry
        assert entry.diagnosis_id == report.diagnosis_id
        assert entry.diagnostic_confidence > 0
        assert entry.based_on_similar == report.has_similar_cases
