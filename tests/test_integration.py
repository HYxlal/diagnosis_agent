"""集成测试 — 端到端流程

测试从输入到诊断输出的完整链路。
使用 ChromaVectorStore（chromadb 为必需依赖）。
LLM 推理路径需要 API key，无 key 时跳过诊断步骤。
"""

import os
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnosis_agent.models.incident import IncidentRecord
from diagnosis_agent.storage.chroma_store import ChromaVectorStore
from diagnosis_agent.retrieval.langchain_retrievers import ChromaVectorRetriever
from diagnosis_agent.parsers.unified import parse_input
from diagnosis_agent.config import Settings


class TestEndToEnd:
    """端到端测试"""

    def test_data_pipeline(self, tmp_path):
        """测试数据加载 + 检索流程（不依赖 LLM）"""

        # 1. 初始化向量库
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_e2e",
        )

        # 2. 加载记录
        records = [
            IncidentRecord(
                problem_description="发动机故障灯亮，怠速不稳",
                root_cause="进气压力传感器故障",
                countermeasure="更换进气压力传感器",
                drive_code="DRV-001",
                vehicle_type="SUV-X1",
                dashboard_indicator="发动机故障灯",
                dtc_code="P0107",
                fault_scenario="冷启动后低速行驶",
            ),
            IncidentRecord(
                problem_description="ABS 故障灯亮，制动距离变长",
                root_cause="轮速传感器损坏",
                countermeasure="更换轮速传感器",
                drive_code="DRV-002",
                vehicle_type="SEDAN-A2",
                dashboard_indicator="ABS故障灯",
                dtc_code="C0040",
                fault_scenario="高速行驶中",
            ),
        ]
        store.add_records(records)
        assert store.count() == 2

        # 3. 检索（使用 ChromaVectorRetriever 替代废弃的 HybridRetriever）
        retriever = ChromaVectorRetriever(
            store=store, top_k=2, score_threshold=0.0
        )
        docs = retriever.invoke("发动机故障灯 怠速")
        assert len(docs) > 0

        # 4. 验证结果
        top = docs[0]
        assert top.metadata.get("problem_description") is not None
        assert top.metadata.get("vehicle_type") is not None

    def test_parsing_pipeline(self, sample_csv_path):
        """测试解析流程"""
        parsed = parse_input(file_path=str(sample_csv_path))
        assert parsed.is_bulk()
        assert len(parsed.bulk_records) == 10

        first = parsed.bulk_records[0]
        assert "problem_description" in first
        assert "vehicle_type" in first
        assert "dtc_code" in first

    @pytest.mark.skipif(
        not os.getenv("DASHSCOPE_API_KEY"),
        reason="需要 DASHSCOPE_API_KEY 才能测试 LLM 推理路径",
    )
    def test_full_diagnosis(self, tmp_path):
        """测试完整诊断流程（需要 LLM）"""
        from diagnosis_agent.agent.langchain_agent import LangChainDiagnosticAgent
        from diagnosis_agent.reporting.markdown import generate_markdown_report

        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_full",
        )

        records = [
            IncidentRecord(
                problem_description="发动机故障灯亮，怠速不稳",
                root_cause="进气压力传感器故障",
                countermeasure="更换进气压力传感器",
                drive_code="DRV-001",
                vehicle_type="SUV-X1",
                dashboard_indicator="发动机故障灯",
                dtc_code="P0107",
                fault_scenario="冷启动后低速行驶",
            ),
        ]
        store.add_records(records)

        settings = Settings()
        agent = LangChainDiagnosticAgent(settings=settings)

        parsed = parse_input(text="发动机故障灯亮了，怠速不稳")
        output = agent.diagnose(parsed)

        # 生成报告
        md_path = generate_markdown_report(output, output_dir=tmp_path)
        assert md_path.exists()
        assert "故障诊断报告" in md_path.read_text(encoding="utf-8")
