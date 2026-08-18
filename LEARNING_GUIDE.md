# Diagnosis Agent 学习指南

> 本指南带你从零理解 Diagnosis Agent 项目涉及的每个核心技术。每章包含概念讲解、通用代码示例，以及**项目中的实际实现**（文件路径、类名、函数名一一对应）。

---

## 目录

- [学习路径总览](#学习路径总览)
- [第一章 Pydantic 数据模型](#第一章-pydantic-数据模型)
- [第二章 向量数据库与 ChromaDB](#第二章-向量数据库与-chromadb)
- [第三章 语义检索与混合检索](#第三章-语义检索与混合检索)
- [第四章 LLM 表头容错映射](#第四章-llm-表头容错映射)
- [第五章 ReAct Agent — 纯 LLM 推理](#第五章-react-agent--纯-llm-推理)
- [第六章 Chain-of-Thought 推理](#第六章-chain-of-thought-推理)
- [第七章 双层输出设计](#第七章-双层输出设计)
- [第八章 模块串联与主流程](#第八章-模块串联与主流程)
- [第九章 会话生命周期与状态机](#第九章-会话生命周期与状态机)
- [第十章 Redis 会话存储](#第十章-redis-会话存储)
- [第十一章 异步摘要生成](#第十一章-异步摘要生成)
- [附录 推荐学习资源](#附录-推荐学习资源)
- [附录：新增 CLI 命令](#附录新增-cli-命令)

---

## 学习路径总览

```
入门 ────────────────────────────────────────────── 进阶
  │                                                   │
  ├─ 1. Pydantic ──────→ 数据校验与模型定义           │
  ├─ 2. ChromaDB ──────→ 向量存储与持久化             │
  ├─ 3. 混合检索 ──────→ 语义检索 + 精确过滤          │
  ├─ 4. 表头容错 ──────→ LLM 智能映射                 │
  ├─ 5. ReAct Agent ──→ 纯 LLM 推理框架               │
  ├─ 6. CoT 推理 ─────→ 思维链 Prompt 设计            │
  ├─ 7. 双层输出 ─────→ 报告+条目的设计模式            │
  ├─ 8. 主流程 ───────→ CLI 编排与配置管理            │
  ├─ 9. 会话生命周期 ──→ 状态机 + Redis 存储          │
  ├─ 10. Redis 存储 ───→ 热层/温层持久化              │
  └─ 11. 异步摘要 ────→ 后台线程不阻塞诊断            │
                                                       |
                                             实战项目 ←
```

**建议学习顺序**：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

每章约需 30-60 分钟。学完 1-3 章即可理解"数据 + 检索"部分，4 章理解"输入容错"设计，5-6 章是"推理"核心，7-8 章是"输出与整合"。

---

## 第一章 Pydantic 数据模型

### 1.1 什么是 Pydantic？

Pydantic 是 Python 最流行的数据校验库。定义一个类继承 `BaseModel`，声明字段类型，Pydantic 自动帮你做：

- **类型校验**：传错类型自动报错或转换
- **数据序列化**：轻松转 dict / JSON
- **文档生成**：字段说明自动变成 schema

### 1.2 基础示例

```python
from pydantic import BaseModel, Field
from typing import Optional

class CaseInput(BaseModel):
    """故障诊断的输入模型"""
    device_name: str = Field(..., description="故障设备名称")
    symptom: str = Field(..., description="故障现象描述")
    operating_condition: Optional[str] = Field(None, description="工况条件")
    severity: Optional[str] = Field("info", description="严重等级")

# 创建实例 — 自动校验类型
case = CaseInput(
    device_name="电机 M-301",
    symptom="启动时有异响",
    operating_condition="空载启动",
    severity="warning"
)

# 类型校验：传错类型会自动转换或报错
case2 = CaseInput(
    device_name="电机 M-302",
    symptom="温度过高",
    severity=123  # Pydantic 会尝试把 int 转成 str
)
print(case2.severity)  # "123"

# 序列化
print(case.model_dump())       # → dict
print(case.model_dump_json())  # → JSON 字符串
```

### 1.3 为什么用 Pydantic 而不是 dataclass？

| 特性 | dataclass | Pydantic |
|------|-----------|----------|
| 类型校验 | ❌ 仅注解 | ✅ 运行时校验 |
| 嵌套模型 | 手动处理 | 自动递归 |
| JSON 序列化 | 需要额外库 | 内置 |
| 字段默认值 | 支持 | 支持（更强大） |
| 自定义校验器 | 不支持 | ✅ `@validator` |
| Schema 导出 | ❌ | ✅ JSON Schema |

### 1.4 项目中的应用

Diagnosis Agent 的数据模型分布在 `src/diagnosis_agent/models/` 下的三个文件中，全部使用 Pydantic `BaseModel`。

#### 1.4.1 `IncidentRecord` — 故障工单记录

**文件**: `src/diagnosis_agent/models/incident.py`

这是项目的核心数据模型，对齐标准 8 列表头。不再包含 `incident_id` / `timestamp` / `severity` / `status` 等旧字段。

```python
# models/incident.py（节选）

INCIDENT_COLUMNS: list[str] = [
    "problem_description",
    "root_cause",
    "countermeasure",
    "drive_code",
    "vehicle_type",
    "dashboard_indicator",
    "dtc_code",
    "fault_scenario",
]

class IncidentRecord(BaseModel):
    """故障工单记录 — 对齐新 8 列表头"""

    problem_description: str = Field("", description="问题描述")
    root_cause: str = Field("", description="根本原因")
    countermeasure: str = Field("", description="对策/解决措施")
    drive_code: str = Field("", description="驱动代码")
    vehicle_type: str = Field("", description="车型")
    dashboard_indicator: str = Field("", description="仪表盘指示")
    dtc_code: str = Field("", description="DTC 故障码")
    fault_scenario: str = Field("", description="故障场景")

    def to_searchable_text(self) -> str:
        """生成用于向量索引的可搜索文本"""
        parts = [
            f"问题描述: {self.problem_description}",
            f"根本原因: {self.root_cause}",
            # ... 8 列全部拼接
        ]
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """转换为字典（用于 DataFrame / 数据库写入）"""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> IncidentRecord:
        """从字典创建记录，只提取标准 8 列"""
        ...
```

同文件中还定义了 `COLUMN_CN_MAP`：中文表头 → 标准英文列名的映射字典，供 `header_mapper.py` 的精确匹配层使用（详见第四章）。

#### 1.4.2 `ParsedInput` — 统一输入模型

**文件**: `src/diagnosis_agent/models/input.py`

无论原始输入是 xlsx / csv / 自然语言 / 混合，最终都统一为这个模型：

```python
# models/input.py（节选）

class InputType(str, Enum):
    XLSX = "xlsx"
    CSV = "csv"
    NATURAL_LANGUAGE = "natural_language"
    MIXED = "mixed"

class ParsedInput(BaseModel):
    input_type: InputType = Field(..., description="输入类型")
    description: str = Field("", description="故障的自然语言描述")
    bulk_records: list[dict] = Field(
        default_factory=list, description="批量解析的记录（来自 xlsx/csv）"
    )
    source_file: Optional[str] = Field(None, description="源文件路径")
    raw_input: str = Field("", description="原始输入文本")

    def is_bulk(self) -> bool:
        """是否为批量输入"""
        return len(self.bulk_records) > 0
```

**设计要点**：自然语言输入只有 `description` 字段，`bulk_records` 为空；文件输入则将每行解析为 `bulk_records` 中的字典，同时提取第一条记录的 `problem_description` 作为 `description`。

#### 1.4.3 双层输出模型

**文件**: `src/diagnosis_agent/models/diagnosis.py`

诊断结果分为两层，由 `DiagnosticOutput` 聚合：

```python
# models/diagnosis.py（节选）

class SimilarCase(BaseModel):
    """检索到的相似工况"""
    record_id: str
    problem_description: str
    root_cause: str
    # ... 8 列 + similarity
    similarity: float

class DiagnosticFinding(BaseModel):
    """诊断结论中的单条发现"""
    title: str
    description: str
    confidence: float          # 0-1
    evidence: list[str]

class DiagnosticReport(BaseModel):
    """诊断报告 — 第一层输出（给人看）"""
    diagnosis_id: str
    diagnosis_time: datetime
    input_summary: str
    has_similar_cases: bool
    similar_cases: list[SimilarCase]
    findings: list[DiagnosticFinding]
    recommended_countermeasure: str
    reasoning_chain: list[str]   # 推理链
    tools_used: list[str]
    agent_version: str = "0.2.0"

class DatabaseEntry(BaseModel):
    """可录入数据库的结构化条目 — 第二层输出（给机器用）"""
    diagnosis_id: str
    diagnosis_time: datetime
    # 新 8 列（诊断结果填充）
    problem_description: str
    root_cause: str
    # ... 8 列
    # 诊断元数据
    diagnostic_confidence: float
    based_on_similar: bool
    similar_record_ids: list[str]

class DiagnosticOutput(BaseModel):
    """双层输出的聚合模型"""
    report: DiagnosticReport
    database_entry: DatabaseEntry
```

**设计要点**：
- `IncidentRecord` 是输入端的标准结构（8 列）
- `DatabaseEntry` 是输出端的标准结构（8 列 + 诊断元数据）
- `DiagnosticOutput` 聚合两层输出，保证一致性（详见第七章）

> **学习练习**：尝试定义一个 `Book` 模型，包含 title、author、price、isbn 字段，price 用 float 且必须 > 0。提示：用 `Field(gt=0)`。

### 1.5 进阶：自定义校验器

```python
from pydantic import field_validator

class CaseInput(BaseModel):
    symptom: str
    severity: str = "info"

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        allowed = {"info", "warning", "critical"}
        if v not in allowed:
            raise ValueError(f"severity 必须是 {allowed} 之一")
        return v
```

---

## 第二章 向量数据库与 ChromaDB

### 2.1 为什么需要向量数据库？

传统数据库（SQL/NoSQL）擅长**精确匹配**：`WHERE vehicle_type = 'SUV'`。

但故障诊断需要**相似匹配**：
- "发动机怠速不稳" ≈ "发动机转速波动" ≈ "怠速忽高忽低"

这些描述措辞不同但语义相同。要实现语义匹配，需要：
1. 把文字转成向量（Embedding）
2. 用向量距离衡量语义相似度
3. 存储和检索向量 → **向量数据库**

### 2.2 Embedding 是什么？

Embedding 是把文字映射成高维向量（如 1536 维浮点数数组），使得**语义相近的文字向量距离也近**。

```
"发动机怠速不稳" → [0.12, -0.34, 0.56, ..., 0.78]  (1536维)
"发动机转速波动" → [0.11, -0.32, 0.55, ..., 0.77]  (1536维)
                                         ↑ 两个向量很接近

"刹车失灵"       → [-0.45, 0.67, -0.12, ..., 0.03]  (1536维)
                                         ↑ 和上面两个差很远
```

### 2.3 相似度计算

| 方法 | 公式（简化） | 特点 |
|------|-------------|------|
| 余弦相似度 | cos(θ) = A·B / (\|A\|·\|B\|) | 最常用，关注方向 |
| 欧氏距离 | √(Σ(aᵢ-bᵢ)²) | 关注绝对距离 |
| 点积 | A·B | 简单快速 |

ChromaDB 使用 **余弦距离**（`metadata={"hnsw:space": "cosine"}`），距离越小越相似。

### 2.4 ChromaDB 快速入门

```python
import chromadb

# 1. 创建客户端（持久化模式）
client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. 创建集合（相当于数据库的表）
collection = client.get_or_create_collection(
    name="incidents",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 3. 添加数据
collection.add(
    ids=["case_001", "case_002"],
    documents=[
        "发动机怠速不稳，转速波动",
        "刹车踏板偏软，制动距离变长"
    ],
    metadatas=[
        {"vehicle_type": "SUV", "dtc_code": "P0506"},
        {"vehicle_type": "Sedan", "dtc_code": "C0267"}
    ]
    # embeddings 可省略 — 配置了 embedding_function 时自动生成
)

# 4. 语义检索
results = collection.query(
    query_texts=["发动机转速忽高忽低"],
    n_results=3
)
```

### 2.5 ChromaDB 核心概念

| 概念 | 类比传统数据库 | 说明 |
|------|---------------|------|
| Client | 数据库连接 | 管理持久化和连接 |
| Collection | 表 | 存储同类向量 |
| Document | 行的文本字段 | 原始文本，用于生成 Embedding |
| Embedding | 行的向量字段 | 由 embedding_function 自动生成 |
| Metadata | 行的其他列 | 结构化字段，用于过滤 |
| ID | 主键 | 唯一标识 |

### 2.6 项目中的应用：`ChromaVectorStore`

**文件**: `src/diagnosis_agent/storage/chroma_store.py`
**抽象接口**: `src/diagnosis_agent/storage/vector_store.py`

项目通过 **适配器模式** 封装 ChromaDB。`VectorStoreAdapter` 是抽象基类，`ChromaVectorStore` 是具体实现：

```python
# storage/vector_store.py — 抽象接口

@dataclass
class SearchResult:
    """向量检索结果"""
    id: str
    content: str
    score: float                  # 相似度分数
    metadata: dict[str, Any]
    record: Optional[IncidentRecord] = None

class VectorStoreAdapter(ABC):
    @abstractmethod
    def add_records(self, records: list[IncidentRecord]) -> int: ...

    @abstractmethod
    def search(
        self, query: str, top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def get_by_id(self, record_id: str) -> Optional[IncidentRecord]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def persist(self) -> None: ...
```

```python
# storage/chroma_store.py — ChromaDB 实现（节选）

class ChromaVectorStore(VectorStoreAdapter):
    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "incidents",
        embedding_model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = self._create_embedding_fn(api_key, api_base)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

    def add_records(self, records: list[IncidentRecord]) -> int:
        """批量添加故障记录"""
        ids, documents, metadatas = [], [], []
        for record in records:
            record_id = f"REC-{uuid.uuid4().hex[:12]}"
            doc_text = record.to_searchable_text()  # 8 列拼接为可搜索文本
            meta = record.to_dict()
            ids.append(record_id)
            documents.append(doc_text)
            metadatas.append(meta)

        # 分批添加（batch_size=100）
        ...

    def search(
        self, query: str, top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """语义检索 — ChromaDB cosine distance → similarity"""
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            where=filters if filters else None,
        )
        # ChromaDB distance → similarity: score = max(0.0, 1.0 - distance)
        ...
```

**关键设计**：
- `to_searchable_text()` 将 `IncidentRecord` 的 8 列拼接为一段文本，作为 ChromaDB 的 `document`（Embedding 的输入）
- `metadata` 存储完整的 8 列字典，用于精确过滤和结果重建
- ChromaDB 返回的 `distance`（余弦距离）被转换为 `similarity`（`1 - distance`）
- `SearchResult.record` 字段从 `metadata` 重建 `IncidentRecord`，保持类型安全

### 2.7 为什么用 Adapter 抽象层？

```python
# storage/vector_store.py
from abc import ABC, abstractmethod

class VectorStoreAdapter(ABC):
    @abstractmethod
    def add_records(self, records: list[IncidentRecord]) -> int: ...
    @abstractmethod
    def search(self, query: str, top_k: int = 5, filters=None) -> list[SearchResult]: ...
```

**好处**：今天用 ChromaDB，明天想换 FAISS 或 Milvus，只需写新 Adapter，业务代码零改动。这就是**依赖倒置** — 业务逻辑依赖抽象接口，不依赖具体实现。

> **学习练习**：用 ChromaDB 创建一个集合，添加 5 条故障描述，然后写 3 个不同的查询文本，观察检索结果的距离差异。

---

## 第三章 语义检索与混合检索

### 3.1 关键词匹配 vs 语义检索

| 方式 | 示例 | 问题 |
|------|------|------|
| 关键词匹配 | `WHERE problem_description LIKE '%发动机%'` | "引擎抖动" 搜不到（没有"发动机"三字） |
| 语义检索 | Embedding 向量相似度 | "引擎抖动" ≈ "发动机异响" → 能搜到 |

### 3.2 Embedding 模型选择

| 模型 | 来源 | 维度 | 特点 |
|------|------|------|------|
| text-embedding-ada-002 | OpenAI | 1536 | 通用，OpenAI API 调用 |
| text-embedding-3-small | OpenAI | 1536 | 性价比高 |
| text-embedding-3-large | OpenAI | 3072 | 精度更高 |
| bge-large-zh | 智源 | 1024 | 中文效果好，可本地部署 |

项目默认使用 `text-embedding-ada-002`，通过 `config.yaml` 的 `embedding.model` 配置。

### 3.3 语义检索 + 阈值过滤

不是所有检索结果都有参考价值。设定阈值，低于阈值的结果视为"不相似"：

```python
# 项目中的阈值配置（config.yaml）
retrieval:
  semantic:
    top_k: 5
    score_threshold: 0.3   # 相似度低于 0.3 的结果被过滤
```

### 3.4 项目中的应用：三层检索架构

项目的检索模块位于 `src/diagnosis_agent/retrieval/`，包含三个检索器：

```
HybridRetriever（混合检索器，对外接口）
  ├── SemanticRetriever（语义检索器）
  │     └── VectorStoreAdapter.search() → 余弦相似度排序
  └── FilterRetriever（精确过滤检索器）
        └── VectorStoreAdapter.search(where={...}) → metadata 精确匹配
```

#### 3.4.1 `SemanticRetriever` — 语义检索

**文件**: `src/diagnosis_agent/retrieval/semantic.py`

```python
class SemanticRetriever:
    def __init__(
        self,
        store: VectorStoreAdapter,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(
        self, query: str, top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[SearchResult]:
        k = top_k or self.top_k
        threshold = min_score if min_score is not None else self.score_threshold
        results = self.store.search(query=query, top_k=k, filters=filters)
        # 过滤低分结果
        return [r for r in results if r.score >= threshold]
```

#### 3.4.2 `FilterRetriever` — 精确过滤

**文件**: `src/diagnosis_agent/retrieval/filter.py`

基于 metadata 精确过滤（车型、驱动代码、DTC 码等），不依赖向量相似度：

```python
class FilterRetriever:
    def filter_by(
        self,
        vehicle_type: Optional[str] = None,
        drive_code: Optional[str] = None,
        dtc_code: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        filters: dict[str, Any] = {}
        if vehicle_type:
            filters["vehicle_type"] = vehicle_type
        if drive_code:
            filters["drive_code"] = drive_code
        if dtc_code:
            filters["dtc_code"] = dtc_code

        return self.store.search(
            query="故障 问题 原因 对策 车型 代码",  # 占位查询
            top_k=k,
            filters=filters if filters else None,
        )

    # 便捷方法
    def get_by_vehicle_type(self, vehicle_type: str, top_k: int = 10): ...
    def get_by_dtc_code(self, dtc_code: str, top_k: int = 10): ...
    def get_by_drive_code(self, drive_code: str, top_k: int = 10): ...
```

> **注意**：`FilterRetriever` 仍然调用 `store.search()`，但通过 `where` 参数让 ChromaDB 做 metadata 精确匹配，向量相似度在此场景下不是重点（query 为占位文本）。

#### 3.4.3 `HybridRetriever` — 混合检索

**文件**: `src/diagnosis_agent/retrieval/hybrid.py`

同时执行语义检索和精确过滤，按权重合并结果：

```python
class HybridRetriever:
    def __init__(
        self,
        store: VectorStoreAdapter,
        semantic_top_k: int = 5,
        filter_top_k: int = 10,
        semantic_weight: float = 0.7,   # 语义权重
        filter_weight: float = 0.3,     # 过滤权重
        score_threshold: float = 0.3,
    ):
        self.store = store
        self.semantic_retriever = SemanticRetriever(store, ...)
        self.filter_retriever = FilterRetriever(store, ...)
        self.semantic_weight = semantic_weight
        self.filter_weight = filter_weight

    def retrieve(
        self, query: str,
        vehicle_type: Optional[str] = None,
        drive_code: Optional[str] = None,
        dtc_code: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        # 1. 语义检索
        semantic_results = self.semantic_retriever.search(query=query, top_k=top_k)

        # 2. 精确过滤（有过滤条件时）
        filter_results = []
        if vehicle_type or drive_code or dtc_code:
            filter_results = self.filter_retriever.filter_by(...)

        # 3. 加权合并 + 去重
        merged: dict[str, SearchResult] = {}
        for r in semantic_results:
            score = r.score * self.semantic_weight   # 乘语义权重
            ...
        for r in filter_results:
            score = r.score * self.filter_weight     # 乘过滤权重
            ...

        # 4. 排序并截断
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return results[:top_k]
```

**混合检索的策略**：
1. 同时执行语义检索和精确过滤
2. 各自分数乘以权重（语义 0.7，过滤 0.3）
3. 同一条记录在两个结果中都出现时，取**最大值**
4. 合并后按总分排序，截取 Top-K

> **学习练习**：收集 10 条故障描述，分别用纯语义检索和混合检索（加车型过滤）搜索，对比结果差异。

---

## 第四章 LLM 表头容错映射

### 4.1 为什么需要表头容错？

用户提供的 CSV/XLSX 文件表头可能千差万别：

| 用户表头 | 期望映射到的标准列 |
|----------|-------------------|
| "问题描述" | `problem_description` |
| "故障现象" | `problem_description` |
| "原因分析" | `root_cause` |
| "解决方案" | `countermeasure` |
| "DTC码" | `dtc_code` |
| "车的型号" | `vehicle_type`（需要 LLM 推理） |

如果硬编码所有可能的表头变体，维护成本极高。项目采用**两层映射**策略：精确匹配优先，匹配不完全时调用 LLM 智能映射。

### 4.2 两层映射逻辑

**文件**: `src/diagnosis_agent/parsers/header_mapper.py`
**核心函数**: `map_headers_to_standard(headers, settings)`

```
输入表头 → 第一层：精确匹配 → 覆盖全部 8 列？
                              │
                   ├─ 是 → 直接返回（不调用 LLM）
                   │
                   └─ 否 → 第二层：LLM 智能映射
                            │
                            ├─ 构建 prompt（标准 8 列 + 含义 + 已匹配项）
                            ├─ LLM 返回 JSON 映射
                            └─ 合并（精确匹配优先，LLM 补充）
```

#### 4.2.1 第一层：精确匹配

**函数**: `_try_exact_match(headers)`

通过内置的 `COLUMN_CN_MAP`（中文→英文映射表）和英文大小写归一化进行匹配：

```python
# models/incident.py 中的 COLUMN_CN_MAP（节选）

COLUMN_CN_MAP: dict[str, str] = {
    # problem_description
    "问题描述": "problem_description",
    "故障描述": "problem_description",
    "故障现象": "problem_description",
    # root_cause
    "根本原因": "root_cause",
    "故障原因": "root_cause",
    "根因": "root_cause",
    # countermeasure
    "对策": "countermeasure",
    "解决措施": "countermeasure",
    "处理方案": "countermeasure",
    # dtc_code
    "DTC码": "dtc_code",
    "故障码": "dtc_code",
    # ... 更多映射
}
```

```python
# header_mapper.py — 精确匹配

def _try_exact_match(headers: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in headers:
        col_str = str(col).strip()
        if col_str in COLUMN_CN_MAP:            # 中文精确匹配
            mapping[col_str] = COLUMN_CN_MAP[col_str]
        elif col_str.lower() in INCIDENT_COLUMNS:  # 英文大小写归一化
            mapping[col_str] = col_str.lower()
    return mapping
```

如果精确匹配已覆盖全部 8 列标准列名，直接返回，不调用 LLM。

#### 4.2.2 第二层：LLM 智能映射

**函数**: `_llm_map_headers(headers, exact_mapping, settings)`

精确匹配不完全时，将未映射的表头连同标准 8 列定义发送给 LLM：

```python
# header_mapper.py — LLM 映射（节选）

def _llm_map_headers(headers, exact_mapping, settings=None) -> dict[str, str]:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=0.0,          # 确定性输出
        max_tokens=1024,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base,
    )

    prompt = f"""你是表头映射专家。请将以下输入文件的表头映射到标准列名。

    ## 标准 8 列
    {json.dumps(INCIDENT_COLUMNS, ensure_ascii=False, indent=2)}

    ## 各列含义
    - problem_description: 问题描述（故障的主要描述文本）
    - root_cause: 根本原因（故障的根因分析）
    ...

    ## 输入表头
    {json.dumps(headers, ensure_ascii=False)}

    ## 已通过精确匹配的映射
    {json.dumps(exact_mapping, ensure_ascii=False)}

    ## 要求
    请输出一个 JSON 对象，key 为原始表头，value 为对应的标准列名。
    只输出 JSON，不要任何其他文字。"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content

    # 解析 JSON 响应
    json_match = re.search(r'\{[\s\S]*\}', content)
    llm_mapping = json.loads(json_match.group())

    # 合并：精确匹配优先，LLM 补充
    merged = dict(llm_mapping)
    merged.update(exact_mapping)  # 精确匹配覆盖 LLM 结果
    return merged
```

**合并策略**：`merged = dict(llm_mapping); merged.update(exact_mapping)` — LLM 结果先放入，精确匹配结果后放入并覆盖相同 key，确保精确匹配优先。

### 4.3 应用映射到 DataFrame

**函数**: `apply_header_mapping(df, mapping)`

```python
def apply_header_mapping(df, mapping: dict[str, str]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in mapping:
            rename_map[col] = mapping[col_str]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
```

### 4.4 调用链

`parse_csv()` / `parse_xlsx()` 中的完整流程：

```python
# parsers/csv_parser.py（节选）

def parse_csv(file_path, encoding="utf-8") -> ParsedInput:
    df = pd.read_csv(file_path, encoding=encoding)

    # 1. 表头映射
    headers = [str(c).strip() for c in df.columns.tolist()]
    mapping = map_headers_to_standard(headers)  # 两层映射
    df = apply_header_mapping(df, mapping)       # 应用到 DataFrame

    # 2. 只保留标准 8 列
    available_cols = [c for c in INCIDENT_COLUMNS if c in df.columns]
    if not available_cols:
        raise ValueError(f"表头映射后无任何标准列匹配。原始表头: {headers}")
    df = df[available_cols]

    # 3. 转为记录列表
    clean_records = []
    for _, row in df.iterrows():
        rec = {col: str(row.get(col, "")).strip() for col in INCIDENT_COLUMNS}
        clean_records.append(rec)

    # 4. 包装为 ParsedInput
    return ParsedInput(
        input_type=InputType.CSV,
        description=clean_records[0].get("problem_description", ""),
        bulk_records=clean_records,
        source_file=str(file_path),
    )
```

`parse_xlsx()` 的逻辑完全相同，只是用 `pd.read_excel()` 替代 `pd.read_csv()`。

### 4.5 自然语言输入的处理

**文件**: `src/diagnosis_agent/parsers/nl_parser.py`

自然语言输入**不做表头映射**，直接将文本包装为 `ParsedInput`：

```python
def parse_natural_language(text: str) -> ParsedInput:
    return ParsedInput(
        input_type=InputType.NATURAL_LANGUAGE,
        description=text.strip(),
        raw_input=text.strip(),
    )
```

语义理解（提取车型、DTC 码等信息）由 ReAct Agent 在推理阶段通过 LLM 完成，而非在解析阶段用正则匹配。这是"纯 LLM 推理"设计的一部分 — 不预判输入的结构，交给 LLM 理解。

> **学习练习**：构造一个 CSV 文件，表头用"故障描述/原因/措施/车型/DTC码/驱动版本/仪表灯/场景"，验证 `map_headers_to_standard()` 能否正确映射。其中"驱动版本"和"仪表灯"不在 `COLUMN_CN_MAP` 中，需要 LLM 映射。

---

## 第五章 ReAct Agent — 纯 LLM 推理

### 5.1 什么是 Agent？

**普通 Chain**：固定流程，输入 → 处理 → 输出
**Agent**：自主决策，根据情况选择不同工具，多步推理

| 类型 | 类比 | 适用场景 |
|------|------|----------|
| Chain | 食谱（按步骤做） | 流程固定 |
| Agent | 厨师（看食材决定怎么做） | 流程不确定，需动态决策 |

故障诊断是典型的不确定流程：有时需要检索，有时不需要；有时查一次就够，有时要查多次。所以用 Agent。

### 5.2 ReAct 模式

ReAct = **Reasoning + Acting**，核心思想：先想后做，边想边做。

```
循环：
  Thought: 我需要先搜索相似工况
  Action: search_similar_incidents("发动机怠速不稳")
  Observation: 找到3个相似工况：...

  Thought: 信息足够了，可以下诊断结论
  Final Answer: { "root_cause": "...", "countermeasure": "...", ... }
```

### 5.3 项目中的应用：`ReActDiagnosticAgent`

**文件**: `src/diagnosis_agent/agent/react_agent.py`

项目没有使用 LangChain 的 `create_react_agent` + `AgentExecutor`，而是实现了**自定义的 ReAct 风格 Agent**。核心设计是**纯 LLM 推理**，无规则降级路径：

```python
# agent/react_agent.py（节选）

class ReActDiagnosticAgent:
    """ReAct 诊断 Agent — 纯 LLM 推理路径

    1. 接收 ParsedInput
    2. 使用检索器搜索相似工况
    3. 构建 prompt，调用 LLM 进行 ReAct 推理
    4. 解析 LLM 输出，生成双层诊断结果
    """

    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.settings = settings
        self.retriever = retriever
        self.tools = DiagnosticTools(retriever)
        self._llm = self._init_llm()   # ChatOpenAI，必须可用

    def _init_llm(self):
        """初始化 LLM — 必须可用，无 fallback"""
        llm_config = self.settings.llm
        if not llm_config.api_key:
            raise RuntimeError(
                "LLM API key 未配置。纯 LLM 推理模式下 API key 为必需依赖。"
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_key=llm_config.api_key,
            base_url=llm_config.api_base,
        )

    def diagnose(self, parsed_input: ParsedInput) -> DiagnosticOutput:
        """执行诊断 — 主流程"""
        # Step 1: 检索相似工况
        similar_cases = self._retrieve_similar_cases(parsed_input)
        has_similar = len(similar_cases) > 0

        # Step 2: LLM 推理（根据有无相似工况选择不同 prompt）
        reasoning_result = self._llm_reason(
            description=parsed_input.description,
            parsed_input=parsed_input,
            similar_cases=similar_cases,
            has_similar=has_similar,
        )

        # Step 3-6: 构建诊断发现、相似工况列表、报告、数据库条目
        ...
        return DiagnosticOutput(report=report, database_entry=database_entry)
```

**关键设计点**：
- `_init_llm()` 中明确标注"无 fallback" — API key 缺失直接 `raise RuntimeError`
- `diagnose()` 方法是主流程入口，按步骤执行：检索 → 推理 → 构建输出
- LLM 推理失败也直接 `raise RuntimeError`，不降级到规则匹配

### 5.4 诊断工具集：`DiagnosticTools`

**文件**: `src/diagnosis_agent/agent/tools.py`

工具集封装了检索能力，供 Agent 调用：

```python
class DiagnosticTools:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def search_similar_incidents(
        self, query: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """工具1：检索相似工单"""
        results = self.retriever.retrieve(query=query, vehicle_type=vehicle_type, top_k=top_k)
        return [
            {
                "record_id": r.id,
                "problem_description": r.metadata.get("problem_description", ""),
                "root_cause": r.metadata.get("root_cause", ""),
                # ... 8 列 + similarity
                "similarity": r.score,
            }
            for r in results
        ]

    def filter_by_vehicle_type(self, vehicle_type: str, top_k: int = 10) -> list[dict]:
        """工具2：按车型精确过滤"""
        ...

    def get_incident_detail(self, record_id: str) -> Optional[dict]:
        """工具3：根据记录ID获取详细信息"""
        ...

    def get_tool_descriptions(self) -> str:
        """返回工具描述文本（用于构建 prompt）"""
        return """1. search_similar_incidents(query, vehicle_type=None, top_k=5)
   - 检索与当前故障相似的历史工单。
   ...
2. filter_by_vehicle_type(vehicle_type, top_k=10)
   ...
3. get_incident_detail(record_id)
   ..."""
```

> **注意**：项目没有使用 LangChain 的 `@tool` 装饰器，而是将工具方法封装在 `DiagnosticTools` 类中。工具描述通过 `get_tool_descriptions()` 方法返回纯文本，嵌入到 system prompt 中供 LLM 参考。

### 5.5 LLM 推理流程

```python
# agent/react_agent.py — LLM 推理（节选）

def _llm_reason(self, description, parsed_input, similar_cases, has_similar) -> dict:
    # 提取上下文信息
    vehicle_type = self._extract_from_records(parsed_input, "vehicle_type")
    dtc_code = self._extract_from_records(parsed_input, "dtc_code")

    # 构建 prompt — 有/无相似工况走不同路径
    system_prompt = build_system_prompt(self.tools.get_tool_descriptions())

    if has_similar:
        similar_text = format_similar_cases_for_prompt(similar_cases)
        user_prompt = build_similar_case_prompt(
            description=description,
            vehicle_type=vehicle_type,
            dtc_code=dtc_code,
            similar_cases_text=similar_text,
        )
    else:
        user_prompt = build_no_similar_case_prompt(
            description=description,
            vehicle_type=vehicle_type,
            dtc_code=dtc_code,
        )

    # 调用 LLM
    response = self._llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    content = response.content

    # 解析 LLM 响应为结构化字典
    return self._parse_llm_response(content)
```

### 5.6 LLM 响应解析

```python
def _parse_llm_response(self, content: str) -> dict:
    """尝试从 LLM 输出中提取 JSON 格式的诊断结果"""
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            result = json.loads(json_match.group())
            # 确保必要字段存在
            result.setdefault("root_cause", "")
            result.setdefault("countermeasure", "")
            result.setdefault("confidence", 0.5)
            result.setdefault("findings", [])
            result.setdefault("reasoning_chain", ["LLM 推理"])
            return result
        except json.JSONDecodeError:
            logger.warning("LLM 响应 JSON 解析失败，使用原始文本")

    # JSON 解析失败时的兜底（非规则降级，仅保留原始文本）
    return {
        "root_cause": content[:500],
        "countermeasure": "",
        "confidence": 0.3,
        "findings": [...],
        "reasoning_chain": ["LLM 推理（JSON 解析失败）"],
    }
```

> **注意**：JSON 解析失败时的兜底不是"规则降级"，而是保留 LLM 原始文本作为 `root_cause`，置信度降至 0.3。这是容错处理，不是替代推理路径。

> **学习练习**：用 `langchain_openai.ChatOpenAI` 创建一个 LLM 实例，给它一个系统 prompt 和用户 prompt，问它"发动机怠速不稳可能的原因有哪些"，观察 LLM 的输出格式。

---

## 第六章 Chain-of-Thought 推理

### 6.1 什么是 Chain-of-Thought（CoT）？

CoT 是一种 Prompt 技术，让 LLM **把推理过程写出来**，而不是直接给答案。

**不用 CoT**：
```
Q: 发动机抖动 + DTC P0301 + 怠速偏低，什么原因？
A: 喷油嘴堵塞。
```

**用 CoT**：
```
Q: 发动机抖动 + DTC P0301 + 怠速偏低，什么原因？
A: 让我逐步分析：
1. DTC P0301 → 1号缸失火
2. 发动机抖动 → 失火导致做功不平衡
3. 怠速偏低 → 失火缸不输出动力，转速下降
综合：1号缸失火，可能原因：火花塞/喷油嘴/点火线圈。
结论：1号缸点火系统故障。
```

### 6.2 为什么 CoT 有效？

| 原因 | 说明 |
|------|------|
| 分解问题 | 复杂问题拆成小步，每步更容易推理 |
| 显式推理 | 避免跳步导致的逻辑断裂 |
| 可追溯 | 能看到推理过程，便于审查和纠错 |
| 自我纠偏 | 推理过程中能发现矛盾并修正 |

### 6.3 项目中的 CoT 设计

**文件**: `src/diagnosis_agent/agent/prompts.py`

项目的 prompt 设计融合了 ReAct 格式和 CoT 推理要求：

#### 6.3.1 System Prompt

```python
# agent/prompts.py（节选）

SYSTEM_PROMPT = """你是一个专业的车辆故障诊断专家 Agent。

你的任务：分析故障描述，检索历史工单，输出诊断结论和推荐对策。

## 可用工具

{tool_descriptions}

## 工作流程（ReAct）

每一步使用以下格式：
Thought: 思考下一步该做什么
Action: 工具名称
Action Input: 工具参数（JSON 格式）
Observation: 工具返回的结果

当你得到足够信息后：
Thought: 我已经有了足够的信息来进行诊断
Final Answer: 最终诊断结果

## 诊断推理原则（Chain-of-Thought）

1. **理解问题**：仔细分析故障描述，识别关键信息（车型、DTC码、仪表盘指示等）
2. **检索相似工况**：使用工具搜索历史工单
3. **对比分析**：将当前故障与历史工单对比，分析异同
4. **推理根因**：基于证据推理可能的根本原因
5. **制定对策**：给出针对性的解决措施

## 输出要求

你的最终诊断结果必须包含以下 JSON 字段：
{{
  "root_cause": "根本原因分析",
  "countermeasure": "推荐对策/解决措施",
  "confidence": 0.0-1.0 的置信度,
  "findings": [
    {{"title": "发现标题", "description": "详细描述", "confidence": 0.0-1.0, "evidence": ["证据1", "证据2"]}}
  ],
  "reasoning_chain": ["推理步骤1", "推理步骤2", ...]
}}
"""
```

**关键点**：
- `{tool_descriptions}` 占位符：运行时由 `DiagnosticTools.get_tool_descriptions()` 填充
- ReAct 格式定义了 Thought / Action / Observation 循环
- CoT 原则要求"基于证据推理"和"说明排除其他可能性"
- 输出格式要求 JSON，包含 `reasoning_chain`（推理链）

#### 6.3.2 有/无相似工况的双路径 Prompt

```python
def build_similar_case_prompt(
    description, vehicle_type, dtc_code, similar_cases_text
) -> str:
    return f"""## 当前故障

**问题描述**: {description}
**车型**: {vehicle_type or "未指定"}
**DTC码**: {dtc_code or "未指定"}

## 检索到的相似工况

{similar_cases_text}

## 请基于以上信息进行诊断推理，输出最终诊断结果（JSON 格式）。"""


def build_no_similar_case_prompt(
    description, vehicle_type, dtc_code
) -> str:
    return f"""## 当前故障

**问题描述**: {description}
**车型**: {vehicle_type or "未指定"}
**DTC码**: {dtc_code or "未指定"}

## 未找到相似工况

未检索到与当前故障相似的历史工单。请基于你的专业知识进行诊断推理，输出最终诊断结果（JSON 格式）。"""
```

**设计要点**：
- 有相似工况时，将检索结果格式化后注入 prompt，要求 LLM 参考历史案例推理
- 无相似工况时，明确告知 LLM"未找到相似工况"，要求基于专业知识推理
- 两条路径都要求输出 JSON 格式，保持输出结构一致

#### 6.3.3 相似工况格式化

```python
def format_similar_cases_for_prompt(cases: list[dict]) -> str:
    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"**工单 {i}** (ID: {case.get('record_id', 'N/A')}, "
                     f"相似度: {case.get('similarity', 0):.2f})")
        lines.append(f"  - 问题描述: {case.get('problem_description', 'N/A')}")
        lines.append(f"  - 根本原因: {case.get('root_cause', 'N/A')}")
        lines.append(f"  - 对策: {case.get('countermeasure', 'N/A')}")
        # ... 8 列全部输出
    return "\n".join(lines)
```

### 6.4 CoT + ReAct 的配合

```
ReAct 负责外部动作：搜索、查询、过滤
CoT 负责内部推理：分析、推断、下结论

Thought（CoT推理）: DTC P0301 指向1号缸失火，需要搜索是否有类似失火案例。
Action（ReAct动作）: search_similar_incidents("1号缸失火 P0301")
Observation: 找到2个相似工况，均为点火线圈故障...

Thought（CoT推理）: 案例都指向点火线圈。但车型不同，需确认当前车型。
Action（ReAct动作）: filter_by_vehicle_type("Model X")
Observation: 同车型有1条记录，也是点火线圈故障...

Thought（CoT推理）: 结合案例和症状，点火线圈老化是最可能的诊断。
Final Answer: { "root_cause": "1号缸点火线圈老化", ... }
```

### 6.5 CoT 的局限

- **不是万能的**：对需要精确计算的领域，CoT 的分步推理仍可能出错
- **增加 token 消耗**：推理过程占用输出 token
- **可能"过度推理"**：简单问题也长篇大论
- **需要审查**：推理过程不一定正确，但至少能发现问题

> **学习练习**：写一个 CoT prompt 让 LLM 诊断"汽车刹车踏板偏软"，对比加和不加"让我们一步一步思考"的输出质量差异。

---

## 第七章 双层输出设计

### 7.1 什么是双层输出？

同一次诊断，同时产出两种格式：

```
        ┌─── Markdown 报告 ───→ 给人看（工程师做决策）
诊断结果 ┤
        └─── CSV/JSON 条目 ───→ 给机器用（数据库录入）
```

### 7.2 为什么要双层？

| 需求 | 单层报告 | 单层条目 | 双层输出 |
|------|---------|---------|---------|
| 工程师阅读 | ✅ | ❌ | ✅ |
| 数据库录入 | 手动 | ✅ | ✅ |
| 推理可追溯 | ✅ | ❌ | ✅ |
| 批量分析 | ❌ | ✅ | ✅ |

**现实问题**：只输出报告 → 数据库积累不了；只输出条目 → 工程师看不懂。双层并行是最佳解。

### 7.3 项目中的实现

#### 7.3.1 数据模型

**文件**: `src/diagnosis_agent/models/diagnosis.py`

```python
class DiagnosticOutput(BaseModel):
    """双层输出的聚合模型"""
    report: DiagnosticReport       # 第一层：给人看
    database_entry: DatabaseEntry  # 第二层：给机器用
```

两层输出都来自同一个 `DiagnosticOutput`，保证一致性。

#### 7.3.2 Markdown 报告生成

**文件**: `src/diagnosis_agent/reporting/markdown.py`
**函数**: `generate_markdown_report(output, output_dir)`

```python
def generate_markdown_report(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
    filename: str | None = None,
) -> Path:
    report = output.report
    filepath = output_dir / f"diagnosis_{report.diagnosis_id}.md"

    lines: list[str] = []
    lines.append("# 故障诊断报告")
    lines.append(f"**诊断ID**: {report.diagnosis_id}")
    lines.append(f"**诊断时间**: {report.diagnosis_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Agent版本**: {report.agent_version}")

    # 输入摘要
    lines.append("## 输入摘要")
    lines.append(f"```\n{report.input_summary}\n```")

    # 相似工况
    lines.append("## 相似工况检索")
    if report.has_similar_cases:
        for i, case in enumerate(report.similar_cases, 1):
            lines.append(f"### 工单 {i} (相似度: {case.similarity:.1%})")
            lines.append(f"- **问题描述**: {case.problem_description}")
            lines.append(f"- **根本原因**: {case.root_cause}")
            # ...

    # 诊断发现（含置信度和证据）
    lines.append("## 诊断发现")
    for finding in report.findings:
        lines.append(f"### {finding.title}")
        lines.append(f"- **置信度**: {finding.confidence:.1%}")
        lines.append(f"- **描述**: {finding.description}")
        if finding.evidence:
            lines.append("- **证据**:")
            for ev in finding.evidence:
                lines.append(f"  - {ev}")

    # 推荐对策
    lines.append("## 推荐对策")
    lines.append(f"{report.recommended_countermeasure}")

    # 推理链
    lines.append("## 推理链")
    for i, step in enumerate(report.reasoning_chain, 1):
        lines.append(f"{i}. {step}")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath
```

#### 7.3.3 结构化条目生成

**文件**: `src/diagnosis_agent/reporting/entries.py`
**函数**: `generate_database_entry_csv()` / `generate_database_entry_json()` / `generate_both()`

```python
def generate_database_entry_csv(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
) -> Path:
    """生成可录入数据库的 CSV 条目"""
    entry = output.database_entry
    entry_dict = entry.to_dict()

    filepath = output_dir / f"db_entry_{entry.diagnosis_id}.csv"
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(entry_dict.keys()))
        writer.writeheader()
        writer.writerow(entry_dict)
    return filepath


def generate_database_entry_json(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
) -> Path:
    """生成可录入数据库的 JSON 条目"""
    entry = output.database_entry
    entry_dict = entry.to_dict()

    filepath = output_dir / f"db_entry_{entry.diagnosis_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry_dict, f, ensure_ascii=False, indent=2)
    return filepath


def generate_both(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
) -> dict[str, Path]:
    """同时生成 CSV 和 JSON 格式"""
    return {
        "csv": generate_database_entry_csv(output, output_dir),
        "json": generate_database_entry_json(output, output_dir),
    }
```

`DatabaseEntry.to_dict()` 将 8 列数据 + 诊断元数据（`diagnostic_confidence`、`based_on_similar`、`similar_record_ids`）输出为扁平字典，其中 `similar_record_ids` 用 `|` 分隔。

### 7.4 设计原则

1. **同一数据源**：两层输出都来自同一个 `DiagnosticOutput`，保证一致性
2. **格式适配**：Markdown 优化可读性（含推理链、证据），CSV/JSON 优化结构性（扁平字典）
3. **ID 关联**：两层输出共享 `diagnosis_id`，可互相追溯
4. **元数据分离**：`DatabaseEntry` 中的诊断元数据（置信度、相似记录ID）与业务数据（8 列）分离

> **学习练习**：设计一个双层输出的场景（如"餐厅评价分析"），定义两个输出格式：给人看的评价摘要 + 给数据库的结构化评分。

---

## 第八章 模块串联与主流程

### 8.1 CLI 入口

**文件**: `src/diagnosis_agent/cli.py`

项目使用 **Typer** 框架构建 CLI，提供 5 个命令：

```python
import typer
from rich.console import Console

app = typer.Typer(name="diagnosis-agent", no_args_is_help=True)
console = Console()

@app.command()
def diagnose(
    text: Optional[str] = typer.Option(None, "--text", "-t"),
    file: Optional[str] = typer.Option(None, "--file", "-f"),
    output_dir: str = typer.Option("output", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """执行故障诊断"""
    _setup_logging(verbose)
    settings = get_settings()

    # 1. 解析输入
    parsed = parse_input(text=text, file_path=file)

    # 2. 构建存储 + 检索器
    store, retriever, _ = _build_components(settings)

    # 3. 执行诊断
    agent = ReActDiagnosticAgent(settings=settings, retriever=retriever)
    output = agent.diagnose(parsed)

    # 4. 生成双层输出
    md_path = generate_markdown_report(output, output_dir=output_dir)
    db_paths = generate_db_entries(output, output_dir=output_dir)

    console.print(f"  📄 Markdown 报告: {md_path}")
    console.print(f"  📊 CSV 条目: {db_paths['csv']}")
    console.print(f"  📊 JSON 条目: {db_paths['json']}")

@app.command()
def load_data(file: str = typer.Option(..., "--file", "-f")):
    """加载批量数据到向量库"""
    ...

@app.command()
def search(query: str = typer.Option(..., "--query", "-q"), ...):
    """检索相似工单"""
    ...

@app.command()
def stats():
    """查看向量库统计"""
    ...

@app.command()
def clear(confirm: bool = typer.Option(False, "--confirm", "-y")):
    """清空向量库"""
    ...
```

### 8.2 组件构建

```python
# cli.py — 构建存储 + 检索器

def _build_components(settings: Settings):
    store = ChromaVectorStore(
        persist_dir=settings.vector_store.persist_dir,     # data/chroma
        collection_name=settings.vector_store.collection_name,  # incidents
        embedding_model=settings.embedding.model,
        api_key=settings.embedding.api_key or None,
        api_base=settings.embedding.api_base or None,
    )
    retriever = HybridRetriever(
        store,
        semantic_top_k=settings.retrieval.semantic.top_k,
        filter_top_k=settings.retrieval.filter.default_top_k,
        semantic_weight=settings.retrieval.hybrid.semantic_weight,
        filter_weight=settings.retrieval.hybrid.filter_weight,
        score_threshold=settings.retrieval.semantic.score_threshold,
    )
    return store, retriever, settings
```

### 8.3 配置加载

**文件**: `src/diagnosis_agent/config.py`

配置通过 Pydantic `Settings` 模型统一管理，从 `config.yaml` + `.env` 加载：

```python
# config.py（节选）

class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"

class VectorStoreConfig(BaseModel):
    type: str = "chroma"
    persist_dir: str = "data/chroma"
    collection_name: str = "incidents"

class RetrievalConfig(BaseModel):
    semantic: SemanticRetrievalConfig     # top_k, score_threshold
    filter: FilterRetrievalConfig         # default_top_k
    hybrid: HybridRetrievalConfig         # semantic_weight, filter_weight

class Settings(BaseModel):
    """全局配置根模型"""
    app: AppConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    agent: AgentConfig
    report: ReportConfig
    paths: PathsConfig

def load_settings(config_path=None, env_path=None) -> Settings:
    """加载配置 — 先 .env 后 config.yaml"""
    # 1. 加载 .env（API key 等敏感信息）
    load_dotenv(env_path or project_root / ".env")

    # 2. 加载 config.yaml
    with open(config_path or project_root / "config.yaml") as f:
        raw_config = yaml.safe_load(f)

    # 3. 解析 ${ENV_VAR:default} 占位符
    resolved_config = _resolve_env_placeholders(raw_config)

    return Settings(**resolved_config)

# 全局单例
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings

def reset_settings() -> None:
    """重置全局配置（主要用于测试）"""
    global _settings
    _settings = None
```

**环境变量占位符**：`config.yaml` 中的 `${OPENAI_API_KEY}` 会被 `_resolve_env_placeholders()` 替换为环境变量 `OPENAI_API_KEY` 的值，`${LLM_MODEL_NAME:gpt-4o-mini}` 语法则提供默认值。

### 8.4 完整数据流

```
用户输入（--text 或 --file）
    │
    ▼
parse_input()                    # parsers/unified.py
    │ 自动检测输入类型
    ├── 文件输入 → parse_csv() / parse_xlsx()
    │     └── map_headers_to_standard()   # header_mapper.py
    │           ├── 第一层：精确匹配（COLUMN_CN_MAP + 英文归一化）
    │           └── 第二层：LLM 智能映射（ChatOpenAI, temperature=0）
    │     └── apply_header_mapping() → 只保留标准 8 列
    │
    └── 自然语言 → parse_natural_language()  # 纯透传
    │
    ▼
ParsedInput                      # models/input.py
    │
    ▼
_build_components()              # cli.py
    ├── ChromaVectorStore        # storage/chroma_store.py
    └── HybridRetriever          # retrieval/hybrid.py
          ├── SemanticRetriever  # retrieval/semantic.py
          └── FilterRetriever    # retrieval/filter.py
    │
    ▼
ReActDiagnosticAgent.diagnose()  # agent/react_agent.py
    ├── Step 1: _retrieve_similar_cases()
    │     └── DiagnosticTools.search_similar_incidents()
    │           └── HybridRetriever.retrieve()
    │
    ├── Step 2: _llm_reason()
    │     ├── build_system_prompt()           # agent/prompts.py
    │     ├── build_similar_case_prompt()     # 有相似工况
    │     │   或 build_no_similar_case_prompt() # 无相似工况
    │     ├── ChatOpenAI.invoke()             # 纯 LLM 推理
    │     └── _parse_llm_response()           # 提取 JSON
    │
    ├── Step 3: _build_findings()             # 构建 DiagnosticFinding 列表
    ├── Step 4: 构建 SimilarCase 列表
    ├── Step 5: 构建 DiagnosticReport（第一层）
    └── Step 6: 构建 DatabaseEntry（第二层）
    │
    ▼
DiagnosticOutput                 # models/diagnosis.py
    │
    ▼
generate_markdown_report()       # reporting/markdown.py → .md 文件
generate_both()                  # reporting/entries.py → .csv + .json 文件
```

### 8.5 错误处理

项目采用**快速失败**策略，关键组件不可用时直接报错：

```python
# react_agent.py — LLM 初始化失败
def _init_llm(self):
    if not llm_config.api_key:
        raise RuntimeError(
            "LLM API key 未配置。纯 LLM 推理模式下 API key 为必需依赖。"
        )
    # ...

# react_agent.py — LLM 推理失败
def _llm_reason(self, ...):
    try:
        response = self._llm.invoke(...)
        return self._parse_llm_response(content)
    except Exception as e:
        raise RuntimeError(f"LLM 推理失败: {e}") from e
```

```python
# react_agent.py — 检索失败（非致命，降级为空列表）
def _retrieve_similar_cases(self, parsed_input):
    try:
        return self.tools.search_similar_incidents(...)
    except Exception as e:
        logger.warning(f"检索相似工况失败: {e}")
        return []   # 检索失败不阻塞，走纯推理路径
```

**设计原则**：
- LLM 是核心依赖，不可降级 → 快速失败
- 检索是辅助能力，可降级为空 → 走"无相似工况"纯推理路径
- 这不是"规则降级"，而是"有/无参考案例"的两条 LLM 推理路径

> **学习练习**：画出从用户输入到最终输出的完整数据流图（可以用纸笔），标出每个模块的输入和输出类型。

---

## 附录 推荐学习资源

### Pydantic
- 官方文档：https://docs.pydantic.dev/
- 关键章节：Models, Validators, Settings

### ChromaDB
- 官方文档：https://docs.trychroma.com/
- 关键章节：Collections, Query, Filtering, Embedding Functions

### LangChain
- 官方文档：https://python.langchain.com/
- 关键章节：Chat Models, Prompts, Output Parsers
- 本项目使用 `langchain-openai` 的 `ChatOpenAI`，未使用 `create_react_agent` / `AgentExecutor`

### Typer
- 官方文档：https://typer.tiangolo.com/
- 关键章节：Commands, Options, Callbacks

### Embedding & 语义检索
- OpenAI Embedding Guide：https://platform.openai.com/docs/guides/embeddings
- 中文 Embedding 模型：https://huggingface.co/BAAI/bge-large-zh

### Chain-of-Thought
- 原始论文：Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- ReAct 论文：Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"

### 工程实践
- 《Designing Data-Intensive Applications》— 数据系统设计经典
- 适配器模式：本项目 `VectorStoreAdapter` → `ChromaVectorStore` 的实现示例

---

## 总结

| 章节 | 核心概念 | 在项目中的角色 | 关键文件 |
|------|---------|---------------|----------|
| 1. Pydantic | 数据模型与校验 | 定义 IncidentRecord / ParsedInput / DiagnosticOutput | `models/incident.py`, `models/input.py`, `models/diagnosis.py` |
| 2. ChromaDB | 向量数据库 | 存储和检索历史故障工况 | `storage/chroma_store.py`, `storage/vector_store.py` |
| 3. 混合检索 | 语义+精确加权合并 | 找到措辞不同但语义相同的案例 | `retrieval/semantic.py`, `retrieval/filter.py`, `retrieval/hybrid.py` |
| 4. 表头容错 | LLM 智能映射 | 兼容用户各种表头命名 | `parsers/header_mapper.py`, `parsers/csv_parser.py` |
| 5. ReAct Agent | 纯 LLM 推理 | Agent 自主决定何时检索、何时推理 | `agent/react_agent.py`, `agent/tools.py` |
| 6. CoT | 思维链推理 | 让推理过程显式、可追溯 | `agent/prompts.py` |
| 7. 双层输出 | 报告+条目 | 同时服务人和机器 | `reporting/markdown.py`, `reporting/entries.py` |
| 8. 主流程 | CLI 编排与配置 | Pipeline 编排全链路 | `cli.py`, `config.py` |
| 9. 会话生命周期 | 状态机 + Redis 存储 | 多轮对话上下文持久化 | `session_manager.py`, `context/types.py` |
| 10. 异步摘要 | asyncio + ThreadPoolExecutor | 摘要不阻塞诊断主流程 | `context_manager.py` (AsyncContextManager) |

**学完这十章，你就理解了 Diagnosis Agent 的每一个组件。** 接下来建议：
1. 阅读项目源码，对照每章找到对应实现
2. 尝试修改一个模块（如添加新的检索工具、扩展表头映射字典）
3. 用自己的数据跑一遍完整流程：`load-data` → `diagnose`
4. 自己设计一个类似的双层输出 Agent 项目

---
## 第九章 会话生命周期与状态机

### 9.1 为什么需要会话管理？

多轮对话场景中，每次 LLM 调用都是独立的。如果没有会话管理，Agent 会"忘记"上一轮说了什么：

```
# 第一轮
用户：电机抖动是什么原因？
Agent：可能是相电流不平衡...

# 第二轮（没有会话上下文）
用户：换了一个MCU还是报错
Agent：？？？你在说什么MCU？
```

**会话管理器**的职责就是把历史消息保存下来，在下一次调用时拼入 messages 列表，让 Agent 感知上下文。

### 9.2 会话状态机

本项目的会话有 6 个状态：

```
状态机：created → active → idle → closing → archived（+ error）
```

| 状态 | 含义 | 触发条件 |
|------|------|---------|
| `created` | 刚创建，无任何消息 | `_create_new()` |
| `active` | 活跃对话中 | 首次 `update()` |
| `idle` | 空闲超时 | Redis TTL 过期 |
| `closing` | 正在归档 | `archive()` 调用 |
| `archived` | 已归档到冷层 | 归档完成 |
| `error` | 异常状态 | 任意异常 |

### 9.3 代码实现：`ConversationContext`

```python
@dataclass
class ConversationContext:
    session_id: str = ""
    status: str = "created"  # ← 核心：状态字段
    hot_messages: list[dict] = field(default_factory=list)
    warm_summaries: list[TopicSnapshot] = field(default_factory=list)
    total_turns: int = 0
    created_at: str = ""
    last_activity_at: str = ""
```

**关键语法：`@dataclass` 的 `field(default_factory=...)`**

```python
# 错误写法：所有实例共享同一个空列表！
hot_messages: list[dict] = []

# 正确写法：每个实例独立创建新列表
hot_messages: list[dict] = field(default_factory=list)
```

为什么？Python 的默认参数在**函数定义时**求值（def-time evaluation）。对于 `@dataclass`，不加 `field(default_factory=...)` 会导致所有实例共享同一个可变对象。一个实例修改了列表，所有实例都会受影响。

### 9.4 状态转换逻辑

状态转换发生在 `SessionManager` 的三个方法中：

```python
def update(self, session_id, query, messages):
    ctx = self._load_to_memory(session_id)
    # ...
    if ctx.status == "created":
        ctx.status = "active"  # 首次消息 → active

def archive(self, session_id, user_id=""):
    ctx.status = "closing"  # 开始归档
    # ... 写入冷层 ...
    ctx.status = "archived"  # 归档完成
```

### 9.5 生命周期检查

两个检查点，分别在 `_load_to_memory()` 和 `update()` 中：

```python
def _check_expired(self, ctx) -> bool:
    """检查会话是否超过最大存活时间"""
    created = datetime.fromisoformat(ctx.created_at)
    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    return elapsed > self._max_lifetime  # 默认 86400 秒（24 小时）
```

**空闲超时**由 Redis TTL 天然支持——每次 `update()` 刷新 TTL，Redis 过期后自动删除 key，下次读取时判为 idle。

### 9.6 单例模式详解

`SessionManager` 使用 `__new__` 实现单例：

```python
class SessionManager:
    _instance: Optional["SessionManager"] = None

    def __new__(cls, persist_dir: str = "data/sessions") -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**`__new__` vs `__init__` 的区别：**

| 方法 | 调用时机 | 职责 | 参数 |
|------|---------|------|------|
| `__new__` | 对象创建前（类方法） | **分配内存**，返回实例 | `cls` + 构造参数 |
| `__init__` | 对象创建后（实例方法） | **初始化**属性 | `self` + 构造参数 |

单例的常见陷阱：

```python
# 问题：__init__ 每次都会执行，即使实例已经存在
def __init__(self, persist_dir: str = "data/sessions"):
    self.persist_dir = persist_dir  # 第二次调用会覆盖！

# 修复：用标志位防止重复初始化
def __init__(self, persist_dir: str = "data/sessions"):
    if not self._persist_dir:  # 类变量，只初始化一次
        self._persist_dir = persist_dir
```

---
## 第十章 Redis 会话存储

### 10.1 为什么用 Redis？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 纯内存 | 最快 | 进程重启丢失 |
| 磁盘 JSON | 持久化 | 读写慢，需手动管理 |
| **Redis** | 快 + 持久化 + TTL 自动过期 | 需要额外服务 |

本项目的三层存储策略：
- **热层/温层 → Redis**：活跃会话，需要快速读写 + TTL 自动过期
- **冷层 → 磁盘 JSON**：已归档会话，长期保存，不需要频繁访问

### 10.2 SessionRedisStore 设计

```python
class SessionRedisStore:
    def __init__(self, redis_url, key_prefix, default_ttl):
        self._redis = None
        self._local = {}  # 内存兜底
        self._available = False
        self._connect(redis_url)

    def _connect(self, redis_url):
        try:
            import redis as r
            self._redis = r.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._available = True
        except Exception:
            self._available = False  # 降级到内存
```

**核心设计模式：降级（Degradation）**
- Redis 可用 → 读写 Redis
- Redis 不可用 → 自动降级到 `self._local` 内存 dict
- 下次操作时不会自动重连（避免每次调用都尝试连接），但 `save()` 和 `load()` 中如果检测到异常会再次尝试

### 10.3 Redis 数据模型

```
Key:   session:{session_id}     ← 前缀可配置
Value: JSON(ConversationContext) ← 完整的会话状态
TTL:   session_idle_timeout     ← 每次访问刷新，过期自动删除
```

### 10.4 Redis TTL 自动过期

```python
def save(self, ctx, ttl=None):
    ttl = ttl or self._default_ttl  # 默认 3600 秒
    self._redis.setex(self._key(ctx.session_id), ttl, json_data)

def refresh_ttl(self, session_id, ttl):
    self._redis.expire(self._key(session_id), ttl)
```

`setex` = SET + EXPIRE 的原子操作。`EXPIRE` 命令设置 TTL，到期后 Redis 自动删除 key。

**TTL 驱动状态机**：Redis key 过期 → 下次 `load()` 返回 `None` → `SessionManager` 判为 idle → 触发归档。不需要额外的定时任务。

### 10.5 序列化与反序列化

```python
# 保存
data = ctx.to_dict()  # ConversationContext → dict
json_str = json.dumps(data, ensure_ascii=False)
redis.setex(key, ttl, json_str)

# 加载
json_str = redis.get(key)
data = json.loads(json_str)
ctx = ConversationContext.from_dict(data)  # dict → ConversationContext
```

**`ensure_ascii=False` 的作用**：中文等非 ASCII 字符不会被转义成 `\uxxxx`，节省空间且可读性更好。

### 10.6 冷层恢复

```python
def restore_from_archive(self, session_id):
    archive_path = f"data/sessions/archive/{session_id}.json"
    with open(archive_path) as f:
        data = json.load(f)
    # 从 ArchivedSession 格式重建 ConversationContext
    ctx = ConversationContext(
        session_id=session_id,
        status="active",
        hot_messages=[],  # 清空旧消息，只保留摘要
        warm_summaries=[TopicSnapshot(...) for t in data.get("topics", [])],
        total_turns=data.get("total_turns", 0),
    )
    if self._redis_store:
        self._redis_store.save(ctx)  # 写回 Redis
    return ctx
```

---
## 第十一章 异步摘要生成

### 11.1 为什么需要异步？

在同步模式下，摘要生成阻塞诊断流程：

```
用户提问 → 热层溢出 → 调用 LLM 生成摘要（等 2-5 秒）→ 诊断推理 → 返回结果
```

用户需要额外等待摘要完成才能拿到诊断结果。异步模式将摘要移到后台：

```
用户提问 → 热层溢出 → 标记"需要摘要" → 立即返回裁剪后的消息 → 诊断推理 → 返回结果
                                                                  ↓
                                                         后台线程生成摘要 → 更新温层
```

### 11.2 架构设计

```python
class AsyncContextManager(SimpleContextManager):
    """继承 SimpleContextManager，覆写为异步模式"""

    def __init__(self, ..., session_manager=None, max_workers=2):
        super().__init__(..., async_mode=True)  # ← 关键：开启异步模式
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def prepare_messages_async(self, ctx, query):
        # 1. 同步裁剪（毫秒级）
        result = self.prepare_from_context(ctx, query)

        # 2. 如需摘要，异步执行
        if result.metadata.trim_info.needs_summary:
            future = asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._generate_summary_sync,  # 后台线程执行
                overflow, ctx, trim_info,
            )
            # 摘要完成后更新温层，不阻塞当前请求
            asyncio.create_task(self._update_warm_async(future, session_id))

        return result  # 立即返回，不等摘要
```

### 11.3 关键语法详解：`ThreadPoolExecutor` + `asyncio`

这是一个 Python 中**同步代码 + 异步编排**的经典模式。

**`ThreadPoolExecutor`**：线程池，管理一组工作线程。提交任务后返回 `Future` 对象。

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summarizer")

# 同步提交（阻塞调用方线程）
future = executor.submit(my_slow_function, arg1, arg2)
result = future.result()  # 阻塞直到完成

# 异步提交（不阻塞 asyncio 事件循环）
future = asyncio.get_event_loop().run_in_executor(
    executor, my_slow_function, arg1, arg2
)
# 继续执行其他代码...
result = await asyncio.wrap_future(future)  # 异步等待
```

**`asyncio.get_event_loop().run_in_executor()`** 的作用：
1. 把同步函数 `my_slow_function` 提交到线程池
2. 返回一个 `asyncio.Future`（不是 `concurrent.futures.Future`）
3. 当前协程可以 `await` 这个 Future，不阻塞事件循环
4. 其他协程可以在等待期间继续执行

**`asyncio.create_task()`**：把协程包装成 Task，在后台调度执行。

```python
# 立即返回，不等待 task 完成
asyncio.create_task(self._update_warm_async(future, session_id))
```

这三个机制配合产生的效果：
```
主线程（事件循环）：
  prepare_messages_async() → 提交摘要到线程池 → 返回裁剪后的消息
  ↓
  LLM 诊断推理（同步，但事件循环可以处理其他协程）
  ↓
  返回结果给用户

后台线程：
  _generate_summary_sync() → 调用 LLM 生成摘要 → 返回 TopicSnapshot

摘要完成后（通过回调协程）：
  _update_warm_async() → 更新温层 → 持久化到 Redis
```

### 11.4 异步模式下的摘要标记

`SimpleContextManager` 中，`async_mode=True` 时 Step 1 不执行摘要，只标记：

```python
if hot_rounds > self.window_size:
    # 移除溢出消息（同步，毫秒级）
    ctx.hot_messages = ctx.hot_messages[keep_from:]

    if overflow:
        if self.async_mode:
            # 只标记，不阻塞！
            trim_info.needs_summary = True
            trim_info.summarized_turns = hot_rounds - self.window_size
        elif self.summarizer:
            # 同步模式：立即生成摘要
            snapshot = self.summarizer.summarize(overflow, ...)
            ctx.warm_summaries.append(snapshot)
```

### 11.5 摘要丢弃策略

后台摘要完成后，通过回调更新温层。但如果会话已经归档，就丢弃摘要：

```python
async def _update_warm_async(self, future, session_id):
    snapshot = await asyncio.wrap_future(future)
    if snapshot is None:
        return

    ctx = sm.get_conversation_context(session_id)
    if ctx is None or ctx.status in ("archived", "closing"):
        logger.info(f"异步摘要丢弃: 会话 {session_id} 已结束")
        return  # 丢弃，不报错

    ctx.warm_summaries.append(snapshot)
    sm.add_warm_summary(session_id, snapshot)
```

**为什么可以丢弃？** 摘要的目的是优化"下一次"的上下文窗口。如果会话已经结束，不需要优化了，丢弃也无害。

### 11.6 完整异步流程

```
用户输入 "电机抖动"
  │
  ▼
chat 命令（异步）：
  │
  ├─ 1. sm.get_conversation_context(session_id)
  │      → Redis 加载或新建
  │
  ├─ 2. acm.prepare_messages_async(ctx, query)
  │      ├─ 同步裁剪（毫秒级）
  │      ├─ 异步提交摘要（后台线程，不阻塞）
  │      └─ 返回 PrepareResult
  │
  ├─ 3. agent.diagnose_with_standard_input(input, messages)
  │      → LLM 诊断推理（可能 5-10 秒）
  │
  ├─ 4. sm.update(session_id, query, messages)
  │      → 持久化到 Redis
  │
  └─ 5. 返回结果给用户
        ↓
  （后台）摘要完成 → 更新温层 → Redis 持久化
```

---
## 附录：新增 CLI 命令

### 会话管理

```bash
# 列出所有会话
python -m diagnosis_agent.cli session list

# 查看会话详情
python -m diagnosis_agent.cli session show <session_id>

# 手动归档会话
python -m diagnosis_agent.cli session archive <session_id> --confirm

# 从冷层恢复会话
python -m diagnosis_agent.cli session restore <session_id>
```

---

*文档版本: 3.0 | 最后更新: 2026-08-14 | 对应代码版本: diagnosis_agent v0.1.0*
