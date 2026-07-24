"""测试数据模型"""

import pytest
from datetime import datetime

from diagnosis_agent.models.incident import (
    COLUMN_CN_MAP,
    COLUMN_EN_TO_CN,
    INCIDENT_COLUMNS,
    IncidentRecord,
)
from diagnosis_agent.models.diagnosis import (
    DatabaseEntry,
    DiagnosticFinding,
    DiagnosticOutput,
    DiagnosticReport,
    SimilarCase,
)
from diagnosis_agent.models.input import InputType, ParsedInput


class TestIncidentRecord:
    """测试故障工单模型"""

    def test_create_record(self):
        """测试创建记录"""
        record = IncidentRecord(
            problem_description="发动机故障灯常亮",
            root_cause="传感器故障",
            countermeasure="更换传感器",
            drive_code="DRV-001",
            vehicle_type="SUV-X1",
            dashboard_indicator="发动机故障灯",
            dtc_code="P0107",
            fault_scenario="冷启动",
        )
        assert record.problem_description == "发动机故障灯常亮"
        assert record.root_cause == "传感器故障"
        assert record.vehicle_type == "SUV-X1"

    def test_to_searchable_text(self):
        """测试生成可搜索文本"""
        record = IncidentRecord(
            problem_description="发动机故障灯常亮",
            root_cause="传感器故障",
            countermeasure="更换传感器",
            drive_code="DRV-001",
            vehicle_type="SUV-X1",
            dashboard_indicator="发动机故障灯",
            dtc_code="P0107",
            fault_scenario="冷启动",
        )
        text = record.to_searchable_text()
        assert "发动机故障灯常亮" in text
        assert "传感器故障" in text
        assert "SUV-X1" in text

    def test_to_dict(self):
        """测试转字典（英文键名，内部存储用）"""
        record = IncidentRecord(
            problem_description="测试问题",
            root_cause="测试原因",
            countermeasure="测试对策",
            drive_code="DRV-002",
            vehicle_type="SEDAN-A2",
            dashboard_indicator="ABS灯",
            dtc_code="P0108",
            fault_scenario="高速行驶",
        )
        d = record.to_dict()
        assert d["problem_description"] == "测试问题"
        assert d["vehicle_type"] == "SEDAN-A2"
        assert len(d) == 8  # 标准 8 列

    def test_to_dict_cn(self):
        """测试转中文字典（用于数据库写入）"""
        record = IncidentRecord(
            problem_description="测试问题",
            root_cause="测试原因",
            countermeasure="测试对策",
            drive_code="DRV-002",
            vehicle_type="SEDAN-A2",
            dashboard_indicator="ABS灯",
            dtc_code="P0108",
            fault_scenario="高速行驶",
        )
        d = record.to_dict_cn()
        assert d["问题描述"] == "测试问题"
        assert d["根因"] == "测试原因"
        assert d["对策"] == "测试对策"
        assert d["电驱代号"] == "DRV-002"
        assert d["车辆类型"] == "SEDAN-A2"
        assert d["仪表指示灯"] == "ABS灯"
        assert d["故障DTC"] == "P0108"
        assert d["故障场景"] == "高速行驶"
        assert len(d) == 8

    def test_from_dict_english(self):
        """测试从英文字典创建"""
        data = {
            "problem_description": "测试",
            "root_cause": "原因",
            "countermeasure": "对策",
            "drive_code": "DRV-003",
            "vehicle_type": "SUV-X1",
            "dashboard_indicator": "灯",
            "dtc_code": "P0109",
            "fault_scenario": "场景",
            "extra_field": "忽略",
        }
        record = IncidentRecord.from_dict(data)
        assert record.problem_description == "测试"
        assert not hasattr(record, "extra_field")

    def test_from_dict_chinese(self):
        """测试从中文字典创建"""
        data = {
            "问题描述": "中文测试",
            "根因": "中文原因",
            "对策": "中文对策",
            "电驱代号": "DRV-004",
            "车辆类型": "SUV-X2",
            "仪表指示灯": "发动机灯",
            "故障DTC": "P0110",
            "故障场景": "中文场景",
        }
        record = IncidentRecord.from_dict(data)
        assert record.problem_description == "中文测试"
        assert record.root_cause == "中文原因"
        assert record.vehicle_type == "SUV-X2"
        assert record.dtc_code == "P0110"

    def test_column_count(self):
        """测试标准列数量为 8"""
        assert len(INCIDENT_COLUMNS) == 8

    def test_cn_map_has_entries(self):
        """测试中文映射表非空"""
        assert len(COLUMN_CN_MAP) > 0
        assert "问题描述" in COLUMN_CN_MAP
        assert COLUMN_CN_MAP["问题描述"] == "problem_description"

    def test_en_to_cn_map(self):
        """测试英文→中文映射"""
        assert len(COLUMN_EN_TO_CN) == 8
        assert COLUMN_EN_TO_CN["problem_description"] == "问题描述"
        assert COLUMN_EN_TO_CN["root_cause"] == "根因"
        assert COLUMN_EN_TO_CN["dtc_code"] == "故障DTC"


class TestParsedInput:
    """测试输入模型"""

    def test_natural_language_input(self):
        parsed = ParsedInput(
            input_type=InputType.NATURAL_LANGUAGE,
            description="发动机故障灯亮了",
        )
        assert not parsed.is_bulk()

    def test_bulk_input(self):
        parsed = ParsedInput(
            input_type=InputType.CSV,
            bulk_records=[{"id": 1}, {"id": 2}],
        )
        assert parsed.is_bulk()


class TestDiagnosticModels:
    """测试诊断输出模型"""

    def test_similar_case(self):
        case = SimilarCase(
            record_id="REC-001",
            problem_description="故障描述",
            root_cause="原因",
            countermeasure="对策",
            drive_code="DRV-001",
            vehicle_type="SUV-X1",
            dashboard_indicator="灯",
            dtc_code="P0107",
            fault_scenario="场景",
            similarity=0.85,
        )
        assert case.similarity == 0.85
        assert case.problem_description == "故障描述"

    def test_database_entry(self):
        entry = DatabaseEntry(
            diagnosis_id="DIAG-001",
            diagnosis_time=datetime.now(),
            problem_description="故障",
            root_cause="原因",
            countermeasure="对策",
            drive_code="DRV-001",
            vehicle_type="SUV-X1",
            dashboard_indicator="灯",
            dtc_code="P0107",
            fault_scenario="场景",
        )
        d = entry.to_dict()
        assert d["diagnosis_id"] == "DIAG-001"
        # 8 列业务字段使用中文表头
        assert d["问题描述"] == "故障"
        assert d["根因"] == "原因"
        assert d["对策"] == "对策"
        assert d["电驱代号"] == "DRV-001"
        # 元数据保持英文
        assert "diagnostic_confidence" in d
        assert "based_on_similar" in d
