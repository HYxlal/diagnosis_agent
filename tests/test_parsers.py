"""测试输入解析器"""

import pytest
from pathlib import Path

from diagnosis_agent.parsers.csv_parser import parse_csv
from diagnosis_agent.parsers.nl_parser import parse_mixed, parse_natural_language
from diagnosis_agent.parsers.unified import parse_input
from diagnosis_agent.parsers.xlsx_parser import parse_xlsx
from diagnosis_agent.models.input import InputType


class TestNaturalLanguageParser:
    """测试自然语言解析器（纯文本透传，不使用正则）"""

    def test_basic_parsing(self):
        text = "车辆启动后发动机故障灯常亮，怠速不稳"
        result = parse_natural_language(text)
        assert result.input_type == InputType.NATURAL_LANGUAGE
        assert result.description == text

    def test_no_regex_extraction(self):
        """确认不再使用正则提取 — 原文本直接作为 description"""
        text = "系统: web-server\nnginx 响应缓慢"
        result = parse_natural_language(text)
        # 不再提取 system 字段
        assert not hasattr(result, "system") or result.system is None
        assert result.description == text

    def test_parse_mixed(self):
        """测试混合输入"""
        result = parse_mixed("发动机故障灯亮了")
        assert result.input_type == InputType.MIXED
        assert result.description == "发动机故障灯亮了"


class TestUnifiedParser:
    """测试统一解析器"""

    def test_parse_text(self):
        result = parse_input(text="发动机故障灯亮了")
        assert result.input_type == InputType.NATURAL_LANGUAGE
        assert result.description == "发动机故障灯亮了"

    def test_parse_file_csv(self, sample_csv_path):
        result = parse_input(file_path=str(sample_csv_path))
        assert result.input_type == InputType.CSV
        assert len(result.bulk_records) == 10
        # 检查新 8 列字段存在
        first = result.bulk_records[0]
        assert "problem_description" in first
        assert "vehicle_type" in first
        assert "dtc_code" in first

    def test_parse_no_input_raises(self):
        with pytest.raises(ValueError):
            parse_input()

    def test_parse_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_input(file_path="/nonexistent/file.csv")
