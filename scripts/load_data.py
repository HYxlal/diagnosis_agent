"""数据加载脚本

将数据文件加载到 ChromaDB 向量库。
可独立运行: python -m scripts.load_data data/samples/your_data.csv
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnosis_agent.config import get_settings, reset_settings
from diagnosis_agent.models.incident import IncidentRecord
from diagnosis_agent.parsers.unified import parse_input
from diagnosis_agent.storage.chroma_store import ChromaVectorStore


def load_sample_data(file_path: str = "data/samples/your_data.csv"):
    """加载数据文件到 ChromaDB 向量库

    Args:
        file_path: CSV 或 XLSX 文件路径
    """
    path = Path(file_path)
    if not path.exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    print(f"📁 加载文件: {file_path}")

    # 解析文件
    parsed = parse_input(file_path=str(path))

    if not parsed.is_bulk():
        print("❌ 文件中无有效记录")
        return

    print(f"📊 解析到 {len(parsed.bulk_records)} 条记录")

    # 配置
    reset_settings()
    settings = get_settings()

    # 初始化向量库
    store = ChromaVectorStore(
        persist_dir=settings.vector_store.persist_dir,
        collection_name=settings.vector_store.collection_name,
        embedding_model=settings.embedding.model,
        api_key=settings.embedding.api_key or None,
        api_base=settings.embedding.api_base or None,
    )
    print(f"📦 向量库: {store.persist_dir} (当前 {store.count()} 条)")

    # 构建 IncidentRecord 列表
    records = []
    for rec_dict in parsed.bulk_records:
        try:
            record = IncidentRecord.from_dict(rec_dict)
            records.append(record)
        except Exception as e:
            print(f"  ⚠️ 跳过记录: {e}")

    # 加载
    count = store.add_records(records)
    print(f"\n✅ 成功加载 {count} 条记录")
    print(f"📦 当前向量库总数: {store.count()}")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/your_data.csv"
    load_sample_data(file_path)
