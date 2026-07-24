# Diagnosis Agent — 车辆故障诊断智能助手

> 基于向量检索 + LLM ReAct 推理的双层输出车辆故障诊断 Agent。支持 xlsx / csv / 自然语言多格式输入，LLM 表头自动映射，同时生成人类可读报告和机器可录入条目。

---

## 项目简介

Diagnosis Agent 是一个车辆故障诊断系统，核心能力：

1. **接收多格式输入**：用户可传入 Excel（xlsx）、CSV 文件，或直接用自然语言描述故障现象
2. **LLM 表头容错**：输入文件表头与标准 8 列不完全匹配时，由 LLM 自动映射到标准列名
3. **检索历史案例**：通过 ChromaDB 向量数据库语义检索相似故障工况
4. **纯 LLM 推理诊断**：LangChain + ChatOpenAI 进行 ReAct 风格推理，无规则降级路径
5. **双层输出**：同时生成人类可读的 Markdown 诊断报告和可直接录入数据库的 CSV/JSON 结构化条目

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
```

### 输入数据格式

CSV/XLSX 文件需包含标准 8 列表头（中英文均可，LLM 会自动映射）：

```
problem_description,root_cause,countermeasure,drive_code,vehicle_type,dashboard_indicator,dtc_code,fault_scenario
```

> 样例数据由用户自行提供，`data/samples/` 为空目录（含 `.gitkeep`），供用户放入自己的数据文件。

---

## 项目架构

```
┌─────────────────────────────────────────────────────┐
│                    用户输入                           │
│           (xlsx / csv / 自然语言)                     │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│              输入解析器 (parsers/)                    │
│  csv/xlsx → 表头映射(精确+LLM) → ParsedInput         │
│  自然语言 → ParsedInput (纯文本透传，由 LLM 推理)     │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│            向量检索 (storage/ + retrieval/)          │
│  ChromaVectorStore (ChromaDB, 必需)                  │
│  HybridRetriever: 语义检索 ⊕ 精确过滤               │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│            推理 Agent (agent/react_agent.py)         │
│  纯 LLM ReAct 推理（无规则降级）                      │
│  Thought → Action → Observation → Final Answer      │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│               双层输出 (reporting/)                   │
│  Markdown 报告 (人读) + CSV/JSON 条目 (机读)         │
└─────────────────────────────────────────────────────┘
```

### LLM 表头映射器

**文件**: `src/diagnosis_agent/parsers/header_mapper.py`
**核心函数**: `map_headers_to_standard()`

当输入文件表头与标准 8 列不完全匹配时，执行两层映射逻辑：

1. **第一层 — 精确匹配**（`_try_exact_match()`）：通过内置 `COLUMN_CN_MAP`（中文→英文映射表，如"问题描述"→`problem_description`）和英文大小写归一化进行匹配。若精确匹配已覆盖全部 8 列，直接返回，不调用 LLM。
2. **第二层 — LLM 智能映射**（`_llm_map_headers()`）：精确匹配不完全时，将未映射的表头连同标准 8 列定义和各列含义发送给 LLM（`ChatOpenAI`，`temperature=0.0`），要求 LLM 输出 JSON 映射。最终合并时**精确匹配优先**，LLM 结果仅补充未匹配部分。

调用链：`parse_csv()` / `parse_xlsx()` → `map_headers_to_standard()` → `apply_header_mapping()` → 只保留标准 8 列数据。

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
│   │   ├── input.py               # ParsedInput（统一输入模型）
│   │   └── diagnosis.py           # DiagnosticReport, DatabaseEntry, DiagnosticOutput
│   ├── parsers/
│   │   ├── unified.py             # 统一解析入口（自动路由 csv/xlsx/nl）
│   │   ├── csv_parser.py          # CSV 解析
│   │   ├── xlsx_parser.py         # XLSX 解析
│   │   ├── nl_parser.py           # 自然语言解析（纯透传，由 LLM 推理）
│   │   └── header_mapper.py       # ★ LLM 表头映射器（两层逻辑）
│   ├── storage/
│   │   ├── vector_store.py        # VectorStoreAdapter 抽象 + SearchResult
│   │   └── chroma_store.py        # ChromaVectorStore（ChromaDB 实现）
│   ├── retrieval/
│   │   ├── semantic.py            # SemanticRetriever（语义检索 + 阈值过滤）
│   │   ├── filter.py              # FilterRetriever（车型/DTC/驱动代码精确过滤）
│   │   └── hybrid.py              # HybridRetriever（混合检索，加权合并）
│   ├── agent/
│   │   ├── react_agent.py         # ReActDiagnosticAgent（纯 LLM 推理）
│   │   ├── prompts.py             # ReAct + CoT Prompt 模板
│   │   └── tools.py               # DiagnosticTools（检索工具封装）
│   └── reporting/
│       ├── markdown.py            # Markdown 报告生成
│       └── entries.py             # CSV/JSON 结构化条目生成
├── scripts/
│   └── load_data.py               # 独立数据加载脚本
├── data/
│   ├── samples/                   # 用户数据目录（空，含 .gitkeep）
│   └── chroma/                    # ChromaDB 持久化（运行时自动创建）
├── output/                        # 诊断报告输出（运行时自动创建）
└── tests/
    ├── conftest.py
    ├── test_models.py
    ├── test_parsers.py
    ├── test_storage_retrieval.py
    ├── test_agent_report.py
    └── test_integration.py
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
