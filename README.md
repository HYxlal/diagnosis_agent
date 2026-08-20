# Diagnosis Agent v0.5.0 — 车辆故障诊断智能助手

多轮对话式车辆故障诊断系统，基于语义检索 + Neo4j 知识图谱 + LLM ReAct 推理。

---

## 核心工作流

```
用户输入 → 预检索 → 话题检测 → Agent 循环 → 诊断输出 → 知识沉淀
                                           ↓
                                    ReAct 推理
                             (Thought → Tool → Observation)
                                           ↓
                                    query_fault_graph  (Neo4j 结构化召回)
                                    search_similar_incidents (Chroma 语义检索)
```

### 预检索
用户输入后，系统同时从两个数据源召回相似案例：
- **ChromaDB**（语义检索）：用 embedding 计算文本相似度
- **Neo4j**（结构化召回）：按关键词匹配故障描述/full_text

结果合并去重后传给 Agent 作为参考上下文。

### 话题检测
每轮对话前判断是否与当前话题相关：
- **same**：继续当前对话
- **different**：归档旧话题，开启新对话轮次

### Agent 循环
LangChain ReAct Agent 执行推理，支持的多轮工具调用：
- `query_fault_graph`：按 mcuid/DTC/部件/场景等结构化字段精确查询 Neo4j 知识图谱
- `search_similar_incidents`：语义检索相似历史工单
- `get_incident_detail`：获取特定工单详情

### 知识沉淀
每轮诊断完成后自动提取对话中的实体和关系，存入待审核队列：
- **实体类型**：现象、根因、对策、电驱代号、车辆类型、仪表指示灯、故障DTC、故障场景
- **关系类型**：由...引起、导致、对应对策、适用于、发生于、关联DTC、亮起、配备、出现于、排除、关联、互斥、并存
- 审核通过的实体和关系写入 Neo4j 知识图谱

---

## 数据库

### Neo4j 故障知识图谱

系统核心数据源，存储结构化的故障案例和知识沉淀结果。

**节点标签**（8 种）：

| 标签 | 含义 | 关键属性 |
|---|---|---|
| Fault | 故障现象 | `description`, `root_cause`, `solution`, `entity_type: "现象"` |
| RootCause | 根本原因 | `name`, `description`, `entity_type: "根因"` |
| Solution | 对策 | `name`, `description`, `entity_type: "对策"` |
| MotorType | 电驱代号 | `code`, `entity_type: "电驱代号"` |
| VehicleType | 车辆类型 | `type`, `entity_type: "车辆类型"` |
| Indicator | 仪表指示灯 | `name`, `entity_type: "仪表指示灯"` |
| DTC | 故障码 | `code`, `description`, `entity_type: "故障DTC"` |
| Scenario | 故障场景 | `category`, `subcategory`, `detail`, `entity_type: "故障场景"` |

**关系类型**（13 种）：

| 关系 | 源→目标 | 含义 |
|---|---|---|
| 由...引起 | Fault → RootCause | 现象由根因引起 |
| 导致 | RootCause → Fault | 根因导致现象 |
| 对应对策 | RootCause → Solution | 根因对应对策 |
| 适用于 | Solution → Fault | 对策适用于现象 |
| 发生于 | Fault → Scenario | 现象发生在场景下 |
| 关联DTC | Fault → DTC | 现象关联故障码 |
| 亮起 | Fault → Indicator | 现象亮起指示灯 |
| 配备 | VehicleType → MotorType | 车辆配备某电驱 |
| 出现于 | Fault → MotorType | 现象出现在某电驱 |
| 排除 | Solution → RootCause | 对策排除根因 |
| 关联 | Scenario → DTC / Indicator → DTC | 场景/指示灯关联故障码 |
| 互斥 | RootCause ↔ RootCause / Solution ↔ Solution | 同类互斥 |
| 并存 | Fault ↔ Fault / RootCause ↔ RootCause | 同类并存 |

**部署**：

```bash
docker run -d --name neo4j-fault \
  -p 7687:7687 -p 7474:7474 \
  -v $(pwd)/neo4j/data:/data \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5.22.0
```

### ChromaDB 向量库

存储历史工单的 embedding 向量，用于语义检索相似案例。

---

## 使用方式

### CLI 交互模式（chat）

```bash
python -m diagnosis-agent.cli chat
```

交互式多轮对话，支持：
- `/tool` — 切换工具调用详情的显示（开/关）
- `exit` / `quit` / `q` — 退出

### Web 审核页面

```bash
python -m diagnosis-agent.cli knowledge
```

知识沉淀的审核界面，地址 `http://localhost:8090`，支持：
- 待审核列表：查看待审核的实体和关系
- 审核操作：通过/拒绝，可填写备注
- 历史审核：查看已审核的记录
- 知识库浏览：查看已写入 Neo4j 的实体和关系

### 会话管理

```bash
python -m diagnosis-agent.cli session list        # 列出所有会话
python -m diagnosis-agent.cli session show <id>   # 查看会话详情
python -m diagnosis-agent.cli session resume <id> # 恢复会话
```

---

## 配置

### config.yaml 核心配置

```yaml
# LLM 配置
llm:
  model: "qwen3.5-plus"          # 推理模型
  temperature: 0.3
  max_tokens: 4096

# 知识图谱（Neo4j）
neo4j:
  url: "bolt://localhost:7687"
  user: "neo4j"
  password: "sDK2aesu"
  min_candidates: 3
  default_depth: 2
  fallback_to_chroma: true

# 上下文管理
context:
  max_tokens: 8000               # Token 预算
  window_size: 5                 # 热层保留轮次
  emergency_min_turns: 2         # 紧急截断保留轮次
  effective_window: 15           # 自适应窗口上限
  max_warm_summaries: 3          # 温层摘要上限

# 知识沉淀
knowledge:
  enabled: true
  extraction_model: "deepseek-v3.1"   # 提取专用模型,和话题判断模型一样不推荐使用thinking模型
  # Web 审核页面 Basic Auth
  web_username: "admin"
  web_password: ""

# 工具调用展示
tool_call:
  show_details: true             # 默认开启工具调用详情

# 向量检索
retrieval:
  semantic:
    top_k: 10
    score_threshold: 0.6
```

### .env 环境变量

```ini
DASHSCOPE_API_KEY=sk-your-key
```

### 工具调用详情开关

在 chat 对话中输入 `/tool` 实时切换：

```
第1轮 > /tool
工具调用详情已关闭
第2轮 > /tool
工具调用详情已开启
```

关闭时只显示工具名和返回结果数量，开启时显示完整参数和返回 JSON。

---

## 目录结构

```
src/diagnosis_agent/
├── cli.py                 # CLI 入口（chat / session / knowledge）
├── config.py              # 配置加载
├── agent/
│   ├── langchain_agent.py # Agent 主逻辑
│   ├── tools.py           # 工具定义
│   ├── context_manager.py # 多轮对话上下文管理
│   ├── context/
│   │   ├── summarizer.py  # 对话摘要
│   │   └── topic_detector.py # 话题检测
│   └── session_manager.py # 会话管理
├── knowledge/
│   ├── extractor.py       # 知识提取
│   ├── graph_writer.py    # Neo4j 写入
│   ├── edit_manager.py    # 编辑保护
│   ├── web.py             # Web 审核页面
│   └── cli.py             # 知识沉淀 CLI
├── retrieval/
│   ├── hybrid_retriever.py # 混合检索
│   ├── neo4j_retriever.py # Neo4j 检索
│   ├── cypher_builder.py  # Cypher 查询构建
│   └── reranker.py        # 精排
├── prompts/
│   ├── knowledge_extraction.py # 提取 prompt
│   ├── agent.py            # Agent prompt
│   └── summarizer.py       # 摘要 prompt
├── models/
│   ├── neo4j_result.py    # 图谱查询结果模型
│   ├── diagnosis.py       # 诊断输出模型
│   └── input.py           # 输入模型
└── scripts/
    ├── load_data.py        # 数据导入
    ├── migrate_graph_schema.py # 图谱迁移
    └── backup_version.py   # 版本备份
```