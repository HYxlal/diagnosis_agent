# voyah.md

本文件为 Voyah Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

**diagnosis_agent** 是面向电驱系统（MCU/电机/逆变器）的车辆故障诊断系统。结合 Neo4j 知识图谱结构化召回 + ChromaDB 向量语义检索 + LangChain ReAct 风格的 LLM 推理，产出双层输出：人类可读的 Markdown 报告 + 机器可录入的 CSV/JSON 数据库条目。

系统对外提供**标准输入/输出契约**与**FastAPI 适配层**两条对接路径，用于与上游平台 Agent 对接：输入 `StandardInput`（含 `raw_query` + `mcuid` + `entities` 的扁平化 JSON），输出 `StandardOutput`（含状态码 `code` 0/-1/-2/-3）。

## 常用命令

```bash
# 安装（可编辑模式，含开发依赖）
pip install -e ".[dev]"          # 或：pip install -r requirements.txt

# 运行 CLI（推荐用模块形式）
python -m diagnosis_agent.cli diagnose --text "发动机故障灯亮，怠速不稳"
python -m diagnosis_agent.cli diagnose --file data/samples/your_data.xlsx
python -m diagnosis_agent.cli diagnose --json-input input.json --std-output
python -m diagnosis_agent.cli load-data --file data/samples/your_data.csv
python -m diagnosis_agent.cli search --query "DTC P0107" --top-k 5
python -m diagnosis_agent.cli stats        # 向量库记录数
python -m diagnosis_agent.cli config        # 查看模型 + 检索配置
python -m diagnosis_agent.cli chat          # 交互式多轮诊断（进程常驻）
python -m diagnosis_agent.cli adapter       # 启动 FastAPI 平台适配层服务（默认 :8000）
python -m diagnosis_agent.cli session-list  # 会话管理（session-list/show/archive/audit/cleanup）

# 测试
pytest                          # 全部测试
pytest tests/test_models.py     # 单文件
pytest tests/test_parsers.py::test_parse_csv -v   # 单个测试
```

环境配置：复制 `.env.example` 为 `.env` 并设置 `DASHSCOPE_API_KEY`。LLM 后端为阿里云通义千问（DashScope，OpenAI 兼容接口）——不是 OpenAI。配置通过 `config.yaml`（非敏感参数，支持 `${ENV_VAR:default}` 占位符）+ `.env`（密钥）双文件管理。

## 架构

### 端到端流程

```
输入 → parse_input() → InputRouter.route() → 预检索相似工况 → Agent.invoke() → 双层输出
```

两条输入路径最终汇入同一个 `LangChainDiagnosticAgent`：

1. **传统路径**（`--text` / `--file` / `--files`）：`parsers/unified.py` 生成 `ParsedInput`，再由 `InputRouter` 做意图分类。
2. **标准接口路径**（`--json-input` 或 FastAPI 适配层）：`StandardInput` → `converter.standard_input_to_parsed()` → `ParsedInput`。这是与平台 Agent 的对接契约。

`diagnose_with_standard_input()` 是对平台暴露的公开入口；`diagnose()` 是内部方法，两条路径最终都会走到这里。Agent 主流程在推理后、工具调用前会异步触发 CAN 兜底（`_maybe_can_fallback()`），检索结果不足时用 `tools/can_fallback.py` 解码工况文件补充上下文。

### 核心分层（均在 `src/diagnosis_agent/` 下）

- **`cli.py`** —— typer CLI，主流程编排。导入所有模块；若改动流程结构，先改这里。命令包括：`diagnose` / `search` / `load-data` / `stats` / `clear` / `config` / `chat`（交互式多轮，进程常驻）/ `session-list` / `session-show` / `session-archive` / `session-audit` / `session-cleanup`（会话管理）/ `adapter`（启动 FastAPI 服务）。typer 自动把下划线命令名暴露为连字符调用。
- **`adapter/`** —— **平台适配层**（FastAPI）。`server.py` 定义路由（异步提交 `/api/v1/diagnoses/async`、状态查询、健康检查、`.well-known/...-manifest`）；`handler.py` 的 `TaskManager` 管理内存任务表（幂等：相同 `clientRequestId` 复用），`execute_diagnosis()` 执行诊断并回调平台，`_extract_knowledge_async()` 触发知识沉淀；`session.py` 的 `build_history_messages()` 优先读本地 SessionManager，降级时从平台 `context_history` 重建纯文本消息；`manifest.py` 的 `MANIFEST` 声明能力与 intentIds（MCU_001~006）；`models.py` 定义扁平化的 `PlatformDiagnosisRequest`/`PlatformCallbackBody` 等（与 `StandardInput` 1:1 对齐）。
- **`parsers/`** —— `unified.py` 路由到 `csv_parser`/`xlsx_parser`/`nl_parser`。`header_mapper.py` 做两层表头映射：先精确匹配（通过 `models/incident.py` 中的 `COLUMN_CN_MAP`），仅对未匹配列调用 LLM。合并时精确匹配始终优先。
- **`agent/input_router.py`** —— `InputRouter` 发起一次轻量 LLM 调用（qwen-turbo），将意图分为 5 类：`diagnostic_query`、`instruction`、`supplement`、`working_condition_file`、`out_of_scope`。LLM 不可用时回退到关键词规则。`OUT_OF_SCOPE` 直接短路返回状态码 `-3`。
- **`agent/langchain_agent.py`** —— `LangChainDiagnosticAgent`，核心 Agent。使用 `langchain.agents.create_agent`（不是 `create_react_agent`）。`_run_agent()` 发送一条 user 消息，再从返回的消息列表中提取工具调用 / ReAct 步骤 / 最终结果。系统 prompt 在 `_build_system_prompt()` 中构建，按配置在轻量/完整两种模式间切换。`_maybe_can_fallback()` 在推理后异步判定是否需要 CAN 兜底解码。
- **`agent/tools.py`** —— `DiagnosticTools` 动态生成 `StructuredTool` 列表：`search_similar_incidents`（语义检索）+ `query_fault_graph`（Neo4j 可用时条件注册）+ `get_incident_detail` + `convert_working_condition_file`。**注意**：`can_converter` 在 prompt 和注释中提及，但未在 `get_tool_list()` 注册——它由 `can_fallback.py` 在 Agent 主流程中直接调用，不走 LLM 工具调用路径。
- **`agent/prompts.py`** —— **仅存放 user prompt 构建函数**（如 `build_similar_case_prompt`）。系统 prompt 已迁移到 `prompts/agent.py`（见下）。
- **`prompts/`** —— **集中管理所有 LLM Prompt**。`agent.py` 提供 `AGENT_SYSTEM_PROMPT_LIGHTWEIGHT`（默认，自然语言 + 5 字段 JSON）和 `AGENT_SYSTEM_PROMPT_FULL`（报告模式，完整 9 字段 JSON）；`knowledge_extraction.py` 知识抽取 prompt；`summarizer.py` 摘要 prompt；`topic_detector.py` 话题检测。`__init__.py` 统一导出。
- **`retrieval/`** —— **混合检索体系**（字段分通道独立匹配，零拼接）：
  - `hybrid_retriever.py` —— `HybridRetriever` 编排入口，按 `retrieval.strategy` 切换：`chroma_only` / `neo4j_first`（默认，Neo4j 召回 + Embedding 精排，不足走 Chroma 兜底）/ `hybrid`。对外只暴露 `retrieve()`，返回统一 `list[Document]`，metadata 带 `source` 标签（"neo4j"/"chroma"）。
  - `neo4j_retriever.py` —— `Neo4jFaultRetriever`，每个字段独立匹配通道（车型/DTC/仪表/工况/关键词），调用 `cypher_builder` 生成 Cypher，原始记录展平为 `FaultCandidate`，连接失败 catch 返回空列表。
  - `langchain_retrievers.py` —— `ChromaVectorRetriever`（LangChain `BaseRetriever`）封装 `ChromaVectorStore`。模块级 `_store_cache` 按 persist_dir 缓存 store。
  - `field_mapper.py` —— 把 `ParsedInput.entities` 字段映射到 Neo4j 节点属性 / Chroma metadata。
  - `cypher_builder.py` / `search_condition.py` / `reranker.py` / `base.py` —— Cypher 生成、查询条件封装、语义精排、检索器抽象基类。
- **`storage/chroma_store.py`** —— `ChromaVectorStore`（ChromaDB `PersistentClient`，cosine 空间）。有 API key 时用 `OpenAIEmbeddingFunction`，否则用 ChromaDB 默认 embedding（`all-MiniLM-L6-v2`）。批量添加限制 25 条（embedding API 约束）。`filter()` 做纯 metadata `where` 查询（不走 embedding 路径）。
- **`tools/`** —— **CAN 报文处理工具集**：
  - `can_converter_tool.py` —— `can_converter_impl()` + `build_can_converter_tool()`，CAN 报文文件转 CSV/Excel（结合 DBC 解码）。
  - `can_fallback.py` —— `preprocess_can_file()`，Agent 主流程调用的异步兜底解码入口（内部调用 `can_converter_impl`），当检索相似度/诊断置信度不足时触发。
  - `can_analysis/` —— 子模块：`decoder.py`（多格式解析：asc/blf/mf4/csv 等）、`exporters.py`（导出）、`schemas.py`（数据模型，`CanLogFormat`/`ExportFormat` 枚举）、`README.md`。
- **`knowledge/`** —— **知识沉淀模块**。`extractor.py` 的 `ConversationKnowledgeExtractor` 做对话知识提取 + 审核队列 + 持久化恢复；`graph_writer.py` 写入 Neo4j；`edit_manager.py` 的 `ManualEditManager` 做编辑保护；`models.py` 定义 `ConversationKnowledge`/`ExtractedEntity`/`ExtractedRelationship`/`KnowledgeStats`；`web.py`/`cli.py` 提供 Web 和 CLI 管理入口。实体类型固定列表：现象/根因/对策/电驱代号/车辆类型/仪表指示灯/故障DTC/故障场景。
- **`utils/`** —— **基础设施封装**。`llm_factory.py` 的 `create_llm()` 统一创建 `ChatOpenAI` 实例；`embedding_wrapper.py` 提供 `DashScopeEmbeddings`（非 OpenAI 兼容 API 的回退模板，通过 `embedding.provider` 配置切换，默认走 OpenAI 兼容路径）。
- **`agent/retention.py`** —— `RetentionPolicy`，扫描 `data/sessions/archive/`，按时间（`retention_days`）+ 空间（`max_archive_size_mb`）双维度清理过期归档，支持定时自动清理 + 手动触发。
- **`models/`** —— Pydantic 模型。**`incident.py`** 含 `IncidentRecord` + 标准 8 列列表（`INCIDENT_COLUMNS`）+ `COLUMN_CN_MAP`（中文→英文表头映射）。**`input.py`** 含 `StandardInput`/`StandardEntities`/`ParsedInput`/`InputIntent`。**`diagnostic_output.py`** 含对外模型 `StandardOutput`/`OutputCode`/`StandardDiagnosisResult`。**`diagnosis.py`** 含内部模型 `DiagnosticOutput`/`DiagnosticReport`/`DatabaseEntry`。**`converter.py`** 桥接两侧（标准↔内部）。**`neo4j_result.py`** 的 `FaultCandidate` 封装 Neo4j 召回结果，展平嵌套结构，`to_document()` 转 LangChain Document 与 Chroma 路径对齐。
- **`reporting/`** —— `markdown.py`（人类报告）和 `entries.py`（CSV/JSON 数据库条目；`similar_record_ids` 不会写入输出文件）。
- **`config.py`** —— `Settings` Pydantic 根模型，`get_settings()` 全局单例，`reset_settings()` 供测试用。子配置：`LLMConfig`/`EmbeddingConfig`/`VectorStoreConfig`/`RetrievalConfig`（`strategy` 默认 `neo4j_first`）/`ToolsConfig`/`ReportConfig`/`Neo4jConfig`（`min_candidates`/`fallback_to_chroma`）/`ContextConfig`（含 `redis: RedisConfig` 子项）/`KnowledgeConfig`（知识沉淀，`persistence_file` 等）/`ToolCallConfig`/`CanFallbackConfig`（`min_similar_record_score`/`min_diagnosis_confidence`/`max_redo_times`）。**`RedisConfig` 挂在 `ContextConfig.redis` 下，不在 `Settings` 根**。
- **`agent/session_manager.py`** —— `SessionManager`（单例）+ `SessionRedisStore`。三层存储：Redis 热层/温层 + 磁盘冷层。状态机 created→active→idle→closing→archived。Redis 不可用时自动降级内存。
- **`agent/context_manager.py`** —— `SimpleContextManager`（同步裁剪）+ `AsyncContextManager`（异步摘要）。`async_mode` 标志控制是否阻塞摘要生成。`ThreadPoolExecutor` 后台线程执行摘要，`asyncio` 协程编排回调。
- **`agent/context/types.py`** —— `ConversationContext` 增加 `status` 字段（状态机）。`TrimInfo` 增加 `needs_summary` / `summarized_start` 标记。
- **`agent/context/summarizer.py`** —— 温层摘要生成器，LLM 或模板策略。`summarize()` 和 `merge_summaries()` 同步方法，被 `AsyncContextManager` 在后台线程中调用。

### 标准 I/O 契约（对接关键）

`StandardInput` 必填 `raw_query` 和 `mcuid`。`entities`（DTC 码、项目、部件、工况、软件版本）直接流入 `ParsedInput.entities` —— "方案B"设计，跳过 `IncidentRecord` 中间对象。平台适配层的 `PlatformDiagnosisRequest` 与 `StandardInput` 字段 1:1 扁平对齐，无嵌套 `data/entities` 子对象。

`StandardOutput.code`：
- `0` SUCCESS —— 完整诊断结果
- `-1` MISSING_INPUT —— `raw_query` 或 `mcuid` 为空
- `-2` INTERNAL_ERROR —— 转换/推理失败
- `-3` OUT_OF_SCOPE —— 非电驱系统问题（InputRouter 识别）

错误输出**只返回** `code` + `msg`（不含 `diagnosis_result`）。

### 故障分类（固定列表）

系统 prompt 强制 LLM 从以下 10 个选项中选择 `classification`：驱动异常故障、控制异常故障、超速故障、高压异常故障、低压异常故障、过温故障、通信故障、旋变故障、状态机故障、油泵故障。修改此列表需同时编辑 `prompts/agent.py`（`AGENT_SYSTEM_PROMPT_LIGHTWEIGHT` 和 `AGENT_SYSTEM_PROMPT_FULL` 两处都列出了该列表）。

## 约定与陷阱

- **模型分散在两处且有差异**：`models/diagnostic_output.py` 定义*对外*的 `DiagnosticResult`（LLM 输出 schema）和 `StandardOutput`；`models/diagnosis.py` 定义*内部*的 `DiagnosticReport`/`DatabaseEntry`/`DiagnosticOutput`。两边都有结构相似的 `DiagnosticFinding` —— 不要混淆。
- **`ParsedInput.field_extraction` 已废弃**，应使用 `entities`。新代码走 `entities`；旧字段仍由 `InputRouter._route_with_llm()` 写入，用于兼容文件解析路径。
- **Prompt 分两层**：系统 prompt 在 `prompts/agent.py`（轻量/完整双模式），user prompt 构建函数在 `agent/prompts.py`。不要在 `agent/prompts.py` 里找系统 prompt 字符串。
- **`can_converter` 的双重身份**：`prompts/agent.py` 和 `tools.py` 注释把它列为可用工具，但 `get_tool_list()` 实际不注册它——它由 `can_fallback.py` 在 Agent 主流程中直接 `can_converter_impl()` 调用，不走 LLM 工具调用路径。改 prompt 列表时注意这个不一致。
- **检索字段分通道独立匹配**：`vehicleModel`/`dtcCode`/`faultWorkConditionList`/`instrumentIndicatorList`/`softwareVersion`/`motorPosition`/`VIN` 各自独立走 Neo4j 节点匹配或 Chroma metadata 过滤，**不做任何字段拼接**。空字段自动跳过，全字段空时跳过 Neo4j 查询避免 `OR 1=1` 返回无关记录。
- **ChromaDB 分数**：`search()` 用 `max(0.0, 1.0 - dist)` 转换 cosine distance；`filter()` 返回固定 `score=1.0`。`score_threshold`（`config.yaml` 默认 0.3）过滤语义检索结果。
- **测试中重载配置**：修改了 env/config 后，先调用 `reset_settings()` 再调用 `get_settings()`，因为 `get_settings()` 是缓存单例。
- **项目根目录发现**：`config.py` 以 `Path(__file__).resolve().parents[2]`（从 `src/diagnosis_agent/config.py` 上溯两级）计算根目录。`config.yaml` 和 `.env` 须放在仓库根目录。
- **`load-data` 会移动文件**：导入成功后，`data/samples/` 下的源文件会被移动到 `data/samples/processed/`，防止重复导入。
- **LangChain 导入**：使用 `from langchain.agents import create_agent`（较新的 API）。
- **CAN 兜底下沉位置**：CAN 解码逻辑在 `tools/can_fallback.py`（Agent 主流程调用），不在 `adapter/handler.py`（适配层只做请求接入/任务调度/回调）。

## 行为准则

1. **全程中文。** 所有输出、交互、注释、内部推理（thinking/思维链）必须使用中文。技术标识符（变量名、函数名、类型名、框架名等）保持原文，但自然语言部分必须全部为中文。在每次思考前必须先提醒自己思维链用中文
2. **编码前先思考。** 明确陈述假设；若存在多种解读，呈现出来而非默默选择其一。遇到不清晰的地方，停下来询问。
3. **简洁优先。** 用解决问题的最少代码。不做投机性抽象、多余配置、为不可能场景做错误处理。200 行能压到 50 行就重写。
4. **手术刀式修改。** 只动任务要求的部分。匹配既有风格。不重构相邻代码、不删除既有死代码（除非被要求）—— 发现问题改为口头提示。
5. **目标驱动执行。** 将任务转化为可验证的目标（如"修 bug"→"写一个复现 bug 的测试，再让它通过"）。多步骤任务先给出简短计划，每步附验证方式。
