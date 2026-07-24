# Diagnosis Agent 重构总结

> 版本: v0.3.0 | 日期: 2026-07-24

---

## 概述

本次重构旨在提升项目的**可扩展性**、**LangChain 生态兼容性**和**检索能力**。核心改动包括：

1. **推理引擎升级**：从自定义 ReAct 循环迁移到 LangChain `create_agent()` API
2. **检索系统增强**：新增 BM25 关键词检索，实现语义+关键词互补检索
3. **字段提取能力**：新增 FieldExtractor，从自然语言中自动提取标准 8 列字段
4. **接口标准化**：统一基于 LangChain `BaseRetriever` 和 `Document` 格式
5. **输出优化**：分离 Markdown 报告与 CSV/JSON 条目内容

---

## 一、新增文件

| 文件路径 | 功能说明 | 状态 |
|----------|----------|------|
| `src/diagnosis_agent/agent/langchain_agent.py` | **核心改动** - 基于 LangChain `create_agent` API 的新推理 Agent | ✅ 已实现 |
| `src/diagnosis_agent/parsers/field_extractor.py` | 字段提取器，从自然语言输入中提取标准 8 列字段 | ✅ 已实现 |
| `src/diagnosis_agent/retrieval/langchain_retrievers.py` | LangChain 兼容的检索器封装（向量检索 + BM25 + 知识图谱预留） | ✅ 已实现 |
| `src/diagnosis_agent/utils/embedding_wrapper.py` | Embedding 模型封装 | ✅ 已实现 |
| `src/diagnosis_agent/utils/llm_factory.py` | LLM 实例工厂，统一创建 LLM 客户端 | ✅ 已实现 |
| `scripts/backup_version.py` | 版本备份脚本 | ✅ 已实现 |

---

## 二、修改文件

### 2.1 `src/diagnosis_agent/cli.py`

| 改动点 | 重构前 | 重构后 |
|--------|--------|--------|
| Agent 实例化 | `ReActDiagnosticAgent(settings, retriever)` | `LangChainDiagnosticAgent(settings)` |
| 组件构建 | 需要构建 store + retriever | 由 Agent 内部自行构建 |
| search 命令 | 使用 HybridRetriever | 使用 `create_chroma_retriever()` |
| 模型状态展示 | 仅展示 LLM 状态 | 展示 LLM、Embedding、InputRouter、向量存储完整状态 |

### 2.2 `src/diagnosis_agent/agent/input_router.py`

| 改动点 | 重构前 | 重构后 |
|--------|--------|--------|
| LangChain API | `create_structured_output_chain`（旧 API） | `ChatPromptTemplate \| llm \| parser`（LCEL 链式调用） |
| 意图分类提示词 | 简单分类指令 | 添加任务要求：提取 search_query、排除示例关键词 |
| 字段提取 | 无 | 新增 `FieldExtractor`，诊断查询时自动提取标准字段 |

### 2.3 `src/diagnosis_agent/agent/tools.py`

| 改动点 | 重构前 | 重构后 |
|--------|--------|--------|
| Retriever 类型 | 固定 `HybridRetriever` | 泛型 `retriever`，支持多种检索器 |
| 检索实现 | `retriever.retrieve()` | 兼容 `search_with_filters()` 和 `invoke()` 两种接口 |
| 结果转换 | `_to_dict(result: SearchResult)` | `_doc_to_dict(doc)`，适配 LangChain Document 格式 |
| 详情查询 | `store.get_by_id(record_id)` | 通过 `vectorstore._collection.get()` 直接查询 |

### 2.4 `src/diagnosis_agent/config.py`

| 改动点 | 说明 |
|--------|------|
| 新增 `Neo4jConfig` | 添加知识图谱数据库配置（URL、用户名、密码），为后续集成预留接口 |

### 2.5 `src/diagnosis_agent/models/input.py`

| 改动点 | 说明 |
|--------|------|
| 新增 `field_extraction` 字段 | 存储 FieldExtractor 从自然语言中提取的标准 8 列字段结果 |

### 2.6 `src/diagnosis_agent/reporting/entries.py`

| 改动点 | 重构前 | 重构后 |
|--------|--------|--------|
| 相似工况输出 | CSV/JSON 包含 `similar_record_ids` | **移除**，仅 Markdown 报告展示相似工况 |

### 2.7 `src/diagnosis_agent/agent/langchain_agent.py`（新增文件关键修改）

| 改动点 | 说明 |
|--------|------|
| `_extract_react_steps` | 过滤 null 值、移除内容截断，确保思维链完整输出 |
| `_parse_final_result` | 使用 Pydantic 模型验证最终 JSON 结果 |
| Agent 初始化 | 内部自行构建 ChromaVectorRetriever |

---

## 三、核心架构改动

### 3.1 Agent 推理引擎重构

**重构前**：
```
自定义 ReAct 循环 → 手动解析 Thought/Action/Observation → 手动调用工具
```

**重构后**：
```
LangChain create_agent() → 自动管理 ReAct 循环 → 工具自动注册和调用
```

**优势**：
- 利用 LangChain 成熟的 Agent 框架，减少自定义代码
- 支持更丰富的工具调用模式
- 更容易扩展新工具

### 3.2 字段提取能力增强

**重构前**：自然语言输入直接透传给 LLM，完全依赖 LLM 在推理过程中理解输入

**重构后**：
```
自然语言 → FieldExtractor → 提取标准 8 列字段 → 结构化输入给 Agent
```

**优势**：
- 提前从自然语言中提取结构化数据
- 减少 LLM 推理时的理解负担
- 输出更符合标准 8 列表头格式

### 3.3 检索系统升级

**重构前**：`HybridRetriever`（语义检索 + 精确过滤，加权 sum 合并）

**重构后**：
```
┌──────────────────────────────────────────────────────┐
│   ChromaVectorRetriever (向量语义检索)                │
│   BM25KeywordRetriever (关键词检索)                   │
│   Neo4jGraphRetriever (知识图谱检索，预留)             │
└──────────────────────────────────────────────────────┘
         │
         ▼
    LangChain BaseRetriever (统一接口)
         │
         ▼
    document_to_record() → IncidentRecord
```

**BM25 算法引入原因**：
- 语义检索对精确关键词（如 DTC 码、车型名称）不敏感
- BM25 基于词频统计，擅长精确关键词匹配
- 互补提升召回率和准确率

### 3.4 输入路由优化

- 增强意图分类提示词，明确任务要求
- 添加示例内容检测，防止 search_query 被污染
- 诊断查询时自动触发字段提取

### 3.5 接口标准化

**返回格式统一**：
- 重构前：`SearchResult`（自定义）
- 重构后：`Document`（LangChain 标准格式）

**转换流程**：
```
ChromaDB → SearchResult → Document → IncidentRecord
```

---

## 四、检索系统详解

### 4.1 新增 BM25KeywordRetriever

**工作原理**：

1. **分词**：使用 `jieba` 对所有文档和查询进行中文分词
2. **索引构建**：`BM25Okapi` 预计算文档长度、词频等统计信息
3. **查询计算**：对查询分词后，计算与每个文档的 BM25 得分
4. **结果排序**：按得分降序，取前 top_k 条

**BM25 公式**：
```
score(D, Q) = Σ [ IDF(q_i) × (f(q_i,D) × (k1+1)) / (f(q_i,D) + k1 × (1-b + b×|D|/avgdl)) ]
```

**与语义检索的互补性**：

| 检索方式 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| 语义检索 | 理解语义、上下文 | 对精确关键词不敏感 | "发动机抖动"、"动力不足" |
| BM25 | 精确关键词匹配 | 不理解语义 | "P0107"、"SUV-A"、"EDU-3" |

### 4.2 格式转换机制

**`document_to_record()` 函数**：

```python
def document_to_record(doc: Document) -> IncidentRecord:
    data = {}
    for en_col, cn_col in COLUMN_EN_TO_CN.items():
        # 优先取英文字段名
        value = doc.metadata.get(en_col, "")
        if value:
            data[en_col] = value
        else:
            # 回退到中文字段名
            value = doc.metadata.get(cn_col, "")
            if value:
                data[en_col] = value
    
    # 如果没有 problem_description，从 page_content 截取
    if not data.get("problem_description"):
        data["problem_description"] = doc.page_content[:500]
    
    return IncidentRecord(**data)
```

**转换目的**：
1. LangChain 生态与项目内部数据模型的接口适配
2. 统一处理中英文表头（`COLUMN_EN_TO_CN` 映射）
3. 将通用的 `Document` 转换为项目特定的 `IncidentRecord`

---

## 五、输出内容优化

### 5.1 输出内容差异

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

### 5.2 思维链 null 值修复

**问题**：报告中步骤 2、3 的 Thought 字段值为 `null`

**修复**：在 `_extract_react_steps` 中添加 `content != 'null'` 检查，过滤掉内容为字符串 "null" 的步骤

**代码改动**：
```python
# 重构前
if content and not isinstance(content, dict):
    step = ReActStep(thought=content[:500], ...)

# 重构后
if content and not isinstance(content, dict) and content != 'null':
    step = ReActStep(thought=content, ...)
```

### 5.3 reasoning_narrative 输出不全修复

**问题**：步骤 4 的 reasoning_narrative 被截断

**修复**：移除 `content[:500]` 截断，确保完整输出

---

## 六、目录结构变化

### 6.1 新增目录

| 目录 | 说明 |
|------|------|
| `src/diagnosis_agent/utils/` | 工具函数目录（Embedding 封装、LLM 工厂） |

### 6.2 完整目录结构

```
diagnosis_agent/
├── src/diagnosis_agent/
│   ├── agent/
│   │   ├── langchain_agent.py     # ★ 新增：主推理 Agent
│   │   ├── react_agent.py         # 旧版，保留兼容
│   │   ├── input_router.py        # 修改：增强意图路由
│   │   ├── prompts.py
│   │   ├── tools.py               # 修改：工具接口适配
│   │   └── __init__.py
│   ├── parsers/
│   │   ├── field_extractor.py     # ★ 新增：字段提取器
│   │   ├── unified.py
│   │   ├── csv_parser.py
│   │   ├── xlsx_parser.py
│   │   ├── nl_parser.py
│   │   ├── header_mapper.py
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── langchain_retrievers.py # ★ 新增：LangChain 检索器封装
│   │   ├── semantic.py
│   │   ├── filter.py
│   │   ├── hybrid.py
│   │   └── __init__.py
│   ├── storage/
│   ├── models/
│   ├── reporting/
│   ├── utils/                     # ★ 新增：工具函数
│   ├── cli.py                     # 修改：主流程编排
│   ├── config.py                  # 修改：新增 Neo4j 配置
│   └── __init__.py
├── scripts/
│   ├── load_data.py
│   └── backup_version.py          # ★ 新增：版本备份
├── docs/
│   ├── review_20260723.md
│   └── refactor_summary.md        # ★ 本文件
├── data/
├── output/
└── tests/
```

---

## 七、向后兼容性

| 组件 | 兼容性 | 说明 |
|------|--------|------|
| `ReActDiagnosticAgent` | ✅ 兼容 | 旧版 Agent 保留，可继续使用 |
| `HybridRetriever` | ✅ 兼容 | 旧版检索器保留 |
| CLI 命令 | ✅ 兼容 | `diagnose`、`load-data`、`search` 等命令接口不变 |
| 配置文件 | ✅ 兼容 | `config.yaml` 和 `.env` 格式不变，新增配置项向后兼容 |
| 输出格式 | ⚠️ 部分变化 | CSV/JSON 不再包含 `similar_record_ids` |

---

## 八、后续规划

| 优先级 | 功能 | 状态 |
|--------|------|------|
| 高 | Neo4j 知识图谱检索集成 | 🔄 预留接口 |
| 中 | EnsembleRetriever 集成（向量+BM25+知识图谱加权融合） | 🔄 规划中 |
| 中 | Reranker 重排序（CrossEncoder） | 🔄 规划中 |
| 低 | 多语言支持 | ❌ 未规划 |

---

## 九、总结

本次重构的核心价值：

1. **架构升级**：从自定义实现迁移到 LangChain 生态，提升可扩展性和维护性
2. **检索增强**：引入 BM25 关键词检索，实现语义+关键词互补，提升检索精度
3. **能力扩展**：新增字段提取、意图路由优化、格式统一等能力
4. **问题修复**：解决思维链 null 值、reasoning_narrative 截断等问题
5. **输出优化**：分离人读报告与机读条目，更符合实际使用场景

**代码量变化**：
- 新增文件：6 个（约 800+ 行）
- 修改文件：7 个
- 删除文件：0 个（旧版代码保留兼容）
