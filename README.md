# Diagnosis Agent — 车辆故障诊断智能助手

> 基于向量检索 + LLM ReAct 推理的双层输出车辆故障诊断 Agent。支持 xlsx / csv / 自然语言多格式输入，LLM 表头自动映射，同时生成人类可读报告和机器可录入条目。

---

## 项目简介

Diagnosis Agent 是一个车辆故障诊断系统，核心能力：

1. **接收多格式输入**：用户可传入 Excel（xlsx）、CSV 文件，或直接用自然语言描述故障现象
2. **LLM 表头容错**：输入文件表头与标准 8 列不完全匹配时，由 LLM 自动映射到标准列名
3. **输入意图路由**：通过 InputRouter 判断输入意图（诊断、批量导入、检索），自动选择处理路径
4. **检索历史案例**：通过 ChromaDB 向量数据库语义检索相似故障工况
5. **纯 LLM 推理诊断**：LangChain + ChatOpenAI 进行 ReAct 风格推理，无规则降级路径
6. **双层输出**：同时生成人类可读的 Markdown 诊断报告和可直接录入数据库的 CSV/JSON 结构化条目

### 标准 8 列表头

| 列名 | 说明 |
|------|------|
| `problem_description` | 问题描述 |
| `root_cause` | 根本原因 |
| `countermeasure` | 对策/解决措施 |
| `drive_code` | 驱动代码 |
| `vehicle_type` | 车型 |
| `dashboard_indicator` | 仪表盘指示 |
| `dtc_code` | DTC 故障码 |
| `fault_scenario` | 故障场景 |

> 中文表头也可识别（如"问题描述"、"故障码"等），`header_mapper.py` 内置中文→英文映射表，匹配不完全时自动调用 LLM 智能映射。

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- pip

### 安装

```bash
# 1. 克隆项目
git clone <repository-url>
cd diagnosis_agent

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# 3. 安装依赖（chromadb + langchain 均为必需依赖，无降级路径）
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 5. 运行诊断
python -m diagnosis_agent.cli diagnose --text "发动机故障灯亮，怠速不稳"
```

### 配置

项目通过 **`config.yaml` + `.env`** 双文件管理配置：

- **`.env`**：存放 `OPENAI_API_KEY` 等敏感信息（不提交到版本控制）
- **`config.yaml`**：存放非敏感的运行参数（检索阈值、路径、权重等），支持 `${ENV_VAR:default}` 语法引用环境变量

`.env` 文件示例：

```ini
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
```

`config.yaml` 中的关键路径配置（与实际代码一致）：

```yaml
vector_store:
  persist_dir: "data/chroma"     # ChromaDB 持久化目录
  collection_name: "incidents"

paths:
  data_dir: "data"
  samples_dir: "data/samples"    # 用户样例数据目录（空，供用户放入）
  output_dir: "output"           # 诊断报告输出目录

report:
  output_dir: "output"

retrieval:
  semantic:
    top_k: 10
    score_threshold: 0.6
  filter:
    default_top_k: 5
    filter_fields: ["vehicle_type", "dtc_code", "drive_code"]
  hybrid:
    semantic_weight: 0.7
    filter_weight: 0.3
    filter_expansion_ratio: 2.0

tools:
  search_top_k: 5
  filter_top_k: 3

input_router:
  enabled: true
  model: gpt-4o-mini
```

配置加载逻辑见 `src/diagnosis_agent/config.py`，由 `Settings` Pydantic 模型统一管理，`get_settings()` 返回全局单例。

---

## 使用方法

### CLI 命令

```bash
python -m diagnosis_agent.cli <command> [options]
```

| 命令 | 说明 | 关键参数 |
|------|------|----------|
| `diagnose` | 执行故障诊断 | `--text` 或 `--file`，`--output` |
| `load-data` | 批量加载数据到向量库 | `--file` |
| `search` | 仅检索相似案例 | `--query`，`--vehicle-type`，`--top-k` |
| `stats` | 查看向量库统计 | — |
| `clear` | 清空向量库 | `--confirm` |
| `config` | 查看当前模型配置状态 | — |

### 命令示例

```bash
# 自然语言诊断
python -m diagnosis_agent.cli diagnose --text "发动机故障灯亮，怠速不稳"

# 从文件诊断（表头无需完全匹配，LLM 会自动映射）
python -m diagnosis_agent.cli diagnose --file data/samples/your_data.xlsx

# 加载数据到向量库
python -m diagnosis_agent.cli load-data --file data/samples/your_data.csv

# 检索相似案例
python -m diagnosis_agent.cli search --query "DTC P0107 进气压力传感器" --top-k 5

# 查看向量库统计
python -m diagnosis_agent.cli stats

# 查看当前配置
python -m diagnosis_agent.cli config
```

### 输入数据格式

CSV/XLSX 文件需包含标准 8 列表头（中英文均可，LLM 会自动映射）：

```
problem_description,root_cause,countermeasure,drive_code,vehicle_type,dashboard_indicator,dtc_code,fault_scenario
```

> 样例数据由用户自行提供，`data/samples/` 为空目录（含 `.gitkeep`），供用户放入自己的数据文件。

---

## 重构后项目架构

### 整体架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                      用户输入层                                   │
│          (xlsx / csv / 自然语言 / 目录批量)                        │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CLI 入口 (cli.py)                              │
│  命令分发 → 参数校验 → 配置加载 → 模型状态展示                      │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                   输入解析层 (parsers/)                           │
│  csv/xlsx → 表头映射(精确+LLM) → ParsedInput                      │
│  自然语言 → ParsedInput (纯文本透传，由 LLM 推理)                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                意图路由层 (agent/input_router.py)                  │
│  判断输入意图: DIAGNOSE / BATCH_IMPORT / SEARCH / UNKNOWN         │
│  根据意图选择处理路径                                             │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                   向量检索层 (storage/ + retrieval/)               │
│  ChromaVectorStore (ChromaDB, 必需)                              │
│  HybridRetriever: 语义检索 ⊕ 精确过滤                            │
│  LangChain Retrievers 封装                                       │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                  推理 Agent 层 (agent/langchain_agent.py)         │
│  LangChain ReAct Agent（纯 LLM 推理，无规则降级）                   │
│  Thought → Action → Observation → Final Answer                   │
│  支持工具调用：检索相似工况、过滤查询等                             │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    双层输出层 (reporting/)                         │
│  Markdown 报告 (人读) + CSV/JSON 条目 (机读)                       │
│  相似工况仅在 Markdown 报告中展示，CSV/JSON 不含相似工况详情         │
└──────────────────────────────────────────────────────────────────┘
```

### 核心模块说明

#### 1. CLI 入口 (`cli.py`)

- 主流程编排：输入解析 → 意图路由 → 检索 → 推理 → 报告生成
- 模型配置状态展示：启动时统一展示 LLM、Embedding、InputRouter、向量存储的配置状态
- 支持批量导入和批量诊断
- 命令分发：`diagnose`、`load-data`、`search`、`stats`、`clear`、`config`

#### 2. 输入解析层 (`parsers/`)

| 模块 | 功能 |
|------|------|
| `unified.py` | 统一解析入口，自动路由到 csv/xlsx/nl 解析器 |
| `csv_parser.py` | CSV 文件解析，调用表头映射 |
| `xlsx_parser.py` | XLSX 文件解析，调用表头映射 |
| `nl_parser.py` | 自然语言解析（纯透传，由 LLM 推理） |
| `header_mapper.py` | LLM 表头映射器（两层逻辑：精确匹配 + LLM 智能映射） |
| `field_extractor.py` | 字段提取工具 |

**LLM 表头映射流程**：
1. **第一层 — 精确匹配**：通过内置 `COLUMN_CN_MAP`（中文→英文映射表）和英文大小写归一化进行匹配。若精确匹配已覆盖全部 8 列，直接返回，不调用 LLM。
2. **第二层 — LLM 智能映射**：精确匹配不完全时，将未映射的表头连同标准 8 列定义和各列含义发送给 LLM（`ChatOpenAI`，`temperature=0.0`），要求 LLM 输出 JSON 映射。最终合并时**精确匹配优先**，LLM 结果仅补充未匹配部分。

#### 3. 意图路由层 (`agent/input_router.py`)

**InputRouter** 负责判断用户输入意图，支持以下意图类型：

| 意图类型 | 说明 | 处理路径 |
|----------|------|----------|
| `DIAGNOSE` | 故障诊断 | 执行完整诊断流程 |
| `BATCH_IMPORT` | 批量导入数据 | 调用 `load-data` 逻辑 |
| `SEARCH` | 检索相似案例 | 调用 `search` 逻辑 |
| `UNKNOWN` | 未知意图 | 回退到默认诊断流程 |

**路由策略**：
- 默认启用 LLM 意图分类（`input_router.enabled: true`）
- 当 LLM 不可用时，自动回退到规则匹配（基于输入内容判断）

#### 4. 向量存储层 (`storage/`)

| 模块 | 功能 |
|------|------|
| `vector_store.py` | VectorStoreAdapter 抽象基类 + SearchResult 模型 |
| `chroma_store.py` | ChromaVectorStore（ChromaDB 实现），支持记录的增删查 |

**ChromaVectorStore** 特性：
- 持久化存储到 `data/chroma/` 目录
- 支持自定义 Embedding 模型
- 批量添加记录（`add_records`）
- 精确检索（`search`）和过滤检索（`search_with_filters`）

#### 5. 检索层 (`retrieval/`)

| 模块 | 功能 |
|------|------|
| `semantic.py` | SemanticRetriever（语义检索 + 阈值过滤） |
| `filter.py` | FilterRetriever（车型/DTC/驱动代码精确过滤） |
| `hybrid.py` | HybridRetriever（混合检索，加权合并语义检索和精确过滤结果） |
| `langchain_retrievers.py` | LangChain Retrievers 封装，提供 LangChain 兼容的检索接口 |

**混合检索算法**：
1. 语义检索：使用 Embedding 模型计算相似度，返回 top_k 条
2. 精确过滤：根据车型、DTC 码、驱动代码等字段进行精确匹配
3. 结果合并：按配置权重（`semantic_weight`、`filter_weight`）合并去重，支持过滤扩倍率（`filter_expansion_ratio`）

#### 6. 推理 Agent 层 (`agent/`)

| 模块 | 功能 |
|------|------|
| `langchain_agent.py` | LangChainDiagnosticAgent（主推理 Agent） |
| `react_agent.py` | ReActDiagnosticAgent（旧版，保留兼容） |
| `prompts.py` | ReAct + CoT Prompt 模板 |
| `tools.py` | DiagnosticTools（检索工具封装，供 Agent 调用） |
| `input_router.py` | 输入意图路由 |

**LangChainDiagnosticAgent 核心流程**：

```
输入描述 + 相似案例 → 构建 Prompt → Agent.invoke() → 解析结果
         │                   │                 │
         ▼                   ▼                 ▼
    检索相似工况        系统 Prompt +          ReAct 循环
    (HybridRetriever)   用户 Prompt        Thought → Action → Observation
                                                    │
                                                    ▼
                                           解析最终 JSON 结果
                                           (Pydantic 模型验证)
```

**工具列表**：
- `search_similar_cases`: 检索相似工况
- `filter_by_field`: 按字段精确过滤
- `get_similar_case_details`: 获取相似工况详情

#### 7. 模型层 (`models/`)

| 模型 | 用途 |
|------|------|
| `incident.py` | IncidentRecord（标准 8 列数据模型）+ COLUMN_CN_MAP |
| `input.py` | ParsedInput（统一输入模型）、InputIntent、InputType |
| `diagnosis.py` | DiagnosticReport、DatabaseEntry、DiagnosticOutput（双层输出模型） |
| `diagnostic_output.py` | 旧版输出模型（保留兼容） |

**双层输出模型**：

```
DiagnosticOutput
├── report: DiagnosticReport          # 第一层：人类可读报告
│   ├── diagnosis_id
│   ├── diagnosis_time
│   ├── input_summary
│   ├── has_similar_cases
│   ├── similar_cases: list[SimilarCase]
│   ├── findings: list[DiagnosticFinding]
│   ├── recommended_countermeasure
│   ├── react_steps: list[ReActStep]      # 思维链步骤
│   ├── reasoning_narrative               # 推断过程叙述
│   ├── tool_calls: list[ToolCallRecord]
│   └── agent_version
└── database_entry: DatabaseEntry     # 第二层：机器可录入条目
    ├── diagnosis_id
    ├── diagnosis_time
    ├── problem_description
    ├── root_cause
    ├── countermeasure
    ├── drive_code
    ├── vehicle_type
    ├── dashboard_indicator
    ├── dtc_code
    ├── fault_scenario
    ├── diagnostic_confidence
    ├── based_on_similar
    └── similar_record_ids (仅内存中，CSV/JSON 不输出)
```

#### 8. 报告生成层 (`reporting/`)

| 模块 | 功能 |
|------|------|
| `markdown.py` | Markdown 报告生成（包含思维链、工具调用、相似工况索引） |
| `entries.py` | CSV/JSON 结构化条目生成（不含相似工况详情） |

**输出内容差异**：

| 内容 | Markdown 报告 | CSV/JSON 条目 |
|------|---------------|---------------|
| 思维链（ReAct 步骤） | ✅ 完整展示 | ❌ 不包含 |
| 工具调用记录 | ✅ 完整展示 | ❌ 不包含 |
| 推断过程叙述 | ✅ 完整展示 | ❌ 不包含 |
| 诊断发现 | ✅ 完整展示 | ❌ 不包含 |
| 相似工况索引 | ✅ 完整展示 | ❌ 不包含 |
| 推荐对策 | ✅ 完整展示 | ✅ 包含 |
| 标准 8 列数据 | ✅ 展示 | ✅ 完整包含 |
| 诊断置信度 | ✅ 展示 | ✅ 包含 |
| 相似记录 ID | ✅ 展示 | ❌ 不包含 |

#### 9. 工具层 (`utils/`)

| 模块 | 功能 |
|------|------|
| `embedding_wrapper.py` | Embedding 模型封装 |
| `llm_factory.py` | LLM 实例工厂，统一创建 LLM 客户端 |

---

## 目录结构

```
diagnosis_agent/
├── README.md
├── LEARNING_GUIDE.md              # 学习指南
├── pyproject.toml
├── requirements.txt               # 全部依赖（chromadb + langchain 必需）
├── config.yaml                    # 运行参数配置
├── .env.example                   # 环境变量模板
├── src/diagnosis_agent/
│   ├── cli.py                     # CLI 入口（typer）
│   ├── config.py                  # 配置加载（Settings Pydantic 模型）
│   ├── models/
│   │   ├── incident.py            # IncidentRecord（新 8 列）+ COLUMN_CN_MAP
│   │   ├── input.py               # ParsedInput（统一输入模型）+ InputIntent/Type
│   │   ├── diagnosis.py           # DiagnosticReport, DatabaseEntry, DiagnosticOutput
│   │   └── diagnostic_output.py   # 旧版输出模型（兼容）
│   ├── parsers/
│   │   ├── unified.py             # 统一解析入口（自动路由 csv/xlsx/nl）
│   │   ├── csv_parser.py          # CSV 解析
│   │   ├── xlsx_parser.py         # XLSX 解析
│   │   ├── nl_parser.py           # 自然语言解析（纯透传）
│   │   ├── header_mapper.py       # ★ LLM 表头映射器（两层逻辑）
│   │   ├── field_extractor.py     # 字段提取工具
│   │   └── __init__.py
│   ├── storage/
│   │   ├── vector_store.py        # VectorStoreAdapter 抽象 + SearchResult
│   │   ├── chroma_store.py        # ChromaVectorStore（ChromaDB 实现）
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── semantic.py            # SemanticRetriever（语义检索）
│   │   ├── filter.py              # FilterRetriever（精确过滤）
│   │   ├── hybrid.py              # HybridRetriever（混合检索）
│   │   ├── langchain_retrievers.py # LangChain 兼容检索接口
│   │   └── __init__.py
│   ├── agent/
│   │   ├── langchain_agent.py     # ★ LangChainDiagnosticAgent（主推理 Agent）
│   │   ├── react_agent.py         # ReActDiagnosticAgent（旧版兼容）
│   │   ├── input_router.py        # ★ 输入意图路由
│   │   ├── prompts.py             # ReAct + CoT Prompt 模板
│   │   ├── tools.py               # DiagnosticTools（Agent 调用的工具）
│   │   └── __init__.py
│   ├── reporting/
│   │   ├── markdown.py            # Markdown 报告生成
│   │   ├── entries.py             # CSV/JSON 结构化条目生成
│   │   └── __init__.py
│   ├── utils/
│   │   ├── embedding_wrapper.py   # Embedding 模型封装
│   │   └── llm_factory.py         # LLM 实例工厂
│   └── __init__.py
├── scripts/
│   ├── load_data.py               # 独立数据加载脚本
│   └── backup_version.py          # 版本备份脚本
├── data/
│   ├── samples/                   # 用户数据目录（空，含 .gitkeep）
│   │   └── processed/             # 已处理文件目录（自动创建）
│   └── chroma/                    # ChromaDB 持久化（运行时自动创建）
├── output/                        # 诊断报告输出（运行时自动创建）
├── docs/
│   └── review_20260723.md         # 评审文档
└── tests/
    ├── conftest.py
    ├── test_models.py             # 模型测试
    ├── test_parsers.py            # 解析器测试
    ├── test_storage_retrieval.py  # 存储检索测试
    ├── test_agent_report.py       # Agent 报告测试
    └── test_integration.py        # 集成测试
```

---

## 技术栈

| 技术 | 用途 | 是否必需 |
|------|------|:--------:|
| Python ≥ 3.10 | 运行时 | ✅ |
| Pydantic ≥ 2.0 | 数据模型与校验 | ✅ |
| pandas ≥ 2.0 | CSV/XLSX 数据处理 | ✅ |
| openpyxl ≥ 3.1 | Excel 解析 | ✅ |
| typer ≥ 0.9 | CLI 框架 | ✅ |
| rich ≥ 13.0 | CLI 美化输出 | ✅ |
| ChromaDB ≥ 0.4.18 | 向量数据库 | ✅ |
| LangChain ≥ 0.1.0 | LLM 调用框架 | ✅ |
| langchain-openai | ChatOpenAI LLM 集成 | ✅ |
| python-dotenv | .env 环境变量加载 | ✅ |
| PyYAML | config.yaml 解析 | ✅ |

---

## License

MIT License
