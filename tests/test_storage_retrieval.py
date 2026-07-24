"""测试存储和检索

chromadb 为必需依赖，不再跳过。
"""

import pytest

from diagnosis_agent.storage.chroma_store import ChromaVectorStore
from diagnosis_agent.retrieval.hybrid import HybridRetriever
from diagnosis_agent.retrieval.semantic import SemanticRetriever
from diagnosis_agent.retrieval.filter import FilterRetriever
from diagnosis_agent.models.incident import IncidentRecord


@pytest.fixture
def sample_records():
    """样例记录列表 — 新 8 列"""
    return [
        IncidentRecord(
            problem_description=f"发动机故障灯亮，怠速不稳 {i}",
            root_cause="进气压力传感器故障" if i < 3 else "点火线圈老化",
            countermeasure="更换传感器" if i < 3 else "更换点火线圈",
            drive_code=f"DRV-{i:03d}",
            vehicle_type="SUV-X1" if i < 3 else "SEDAN-A2",
            dashboard_indicator="发动机故障灯",
            dtc_code=f"P010{i % 10}",
            fault_scenario="冷启动" if i < 3 else "高速行驶",
        )
        for i in range(5)
    ]


class TestChromaVectorStore:
    """测试 ChromaDB 向量存储"""

    def test_add_and_count(self, tmp_path, sample_records):
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_add",
        )
        assert store.count() == 0
        count = store.add_records(sample_records)
        assert count == 5
        assert store.count() == 5

    def test_search(self, tmp_path, sample_records):
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_search",
        )
        store.add_records(sample_records)
        results = store.search(query="发动机故障灯 怠速", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_clear(self, tmp_path, sample_records):
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_clear",
        )
        store.add_records(sample_records)
        assert store.count() == 5
        store.clear()
        assert store.count() == 0


class TestRetrievers:
    """测试检索器"""

    def test_semantic_retriever(self, tmp_path, sample_records):
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_semantic",
        )
        store.add_records(sample_records)

        retriever = SemanticRetriever(store, score_threshold=0.0)
        results = retriever.search(query="发动机故障灯", top_k=3)
        assert len(results) > 0

    def test_filter_retriever(self, tmp_path, sample_records):
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_filter",
        )
        store.add_records(sample_records)

        retriever = FilterRetriever(store)
        results = retriever.get_by_vehicle_type("SUV-X1")
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("vehicle_type") == "SUV-X1"

    def test_hybrid_retriever(self, tmp_path, sample_records):
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_hybrid",
        )
        store.add_records(sample_records)

        retriever = HybridRetriever(store, score_threshold=0.0)
        results = retriever.retrieve(
            query="发动机故障灯 怠速",
            vehicle_type="SUV-X1",
            top_k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3
