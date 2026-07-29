# 输入输出接口重构说明

## 概述

本次重构旨在为电驱诊断 Agent 添加标准化的输入输出接口，使其能够与平台 Agent 进行规范化的数据交互。

## 变更范围

### 修改的文件

| 文件路径 | 变更类型 | 说明 |
|---------|---------|------|
| `src/diagnosis_agent/models/input.py` | 新增 | 新增 `StandardInput`、`StandardEntities`、`HistoryContext` 模型 |
| `src/diagnosis_agent/models/diagnostic_output.py` | 新增 | 新增 `StandardOutput`、`StandardDiagnosisResult`、`OutputCode`、`SimilarCaseRef` 模型 |
| `src/diagnosis_agent/agent/prompts.py` | 修改 | 更新 LLM Prompt，增加新的结构化输出字段要求 |
| `src/diagnosis_agent/agent/langchain_agent.py` | 修改 | 新增 `diagnose_with_standard_input` 方法，更新 Prompt 和默认结果 |
| `src/diagnosis_agent/cli.py` | 修改 | 新增 `--json-input`、`--generate-md`、`--std-output` 参数 |

### 新增的文件

| 文件路径 | 说明 |
|---------|------|
| `src/diagnosis_agent/models/converter.py` | 输入输出转换层，实现双向转换逻辑 |

---

## 详细变更说明

### 1. 新增标准输入模型（StandardInput）

**文件**: `models/input.py`

新增三个模型类：

#### StandardEntities（结构化实体）
平台 NLP 实体识别结果，每个字段明确类型：
- `dtc_code`: `list[str]` - 故障 DTC 代码列表
- `project`: `str` - 项目型号/车型代号
- `component`: `str` - 涉及的部件
- `working_condition`: `str` - 工作条件描述
- `software_version`: `str` - 软件版本信息

#### HistoryContext（历史对话）
多轮对话历史上下文单条记录：
- `query`: `str` - 历史用户问题
- `entities`: `dict` - 历史实体抽取结果
- `agent_answer`: `str` - 上一轮 Agent 回复

#### StandardInput（标准输入）
对外接口的统一输入格式：
- 必填字段: `raw_query`、`mcuid`
- 可选字段: `intent_name`、`query_confidence`
- 实体字段: `entities`（StandardEntities）
- 多轮对话: `session_id`、`context_history`
- 辅助字段: `user_id`

### 2. 新增标准输出模型（StandardOutput）

**文件**: `models/diagnostic_output.py`

新增四个模型类：

#### OutputCode（状态码枚举）
- `SUCCESS = 0`: 正常返回
- `MISSING_INPUT = -1`: 入参缺失关键信息
- `INTERNAL_ERROR = -2`: 内部推理异常
- `OUT_OF_SCOPE = -3`: 不属于电驱系统范畴

#### SimilarCaseRef（相似工况引用）
对外输出的精简结构：
- `record_id`、`problem_description`、`root_cause`、`countermeasure`、`similarity`

#### StandardDiagnosisResult（核心诊断结果）
LLM 重新生成的结构化解决方案：
- `fault_root_cause`: `list[str]` - 故障根本原因列表
- `fault_trigger_condition`: `str` - 故障触发条件
- `classification`: `str` - 故障分类
- `solution`: `list[str]` - 结构化解决方案列表
- `risk_warning`: `str` - 风险预警等级
- `maintenance_suggestions`: `str` - 维护建议
- `similar_cases`: `list[SimilarCaseRef]` - 相似工况引用列表

#### StandardOutput（标准输出）
对外接口的统一输出格式：
- `code`: 状态码
- `msg`: 状态描述
- `input_ref`: 输入引用信息
- `diagnosis_result`: 核心诊断结果
- `reference_material`: 参考材料列表
- `need_multi_round`: 是否需要追问
- `follow_up_question`: 追问问题
- `diagnosis_confidence`: 诊断置信度

### 3. 新增转换层（converter.py）

**文件**: `models/converter.py`

实现双向转换逻辑：

#### 输入转换：StandardInput → ParsedInput
- `standard_input_to_parsed()`: 将标准输入转换为内部解析输入
- 处理逻辑：
  - `raw_query` → `description`
  - `entities` → `field_extraction`（方案A：直接赋值，跳过 FieldExtractor）
  - `mcuid` → `search_query` 前缀（用于精确搜索）
  - `intent_name` → `intent` 初始值（方案B：仍走 InputRouter 二次确认）

#### 输出转换：DiagnosticOutput → StandardOutput
- `diagnostic_output_to_standard()`: 将内部诊断结果转换为标准输出
- 智能映射：
  - `root_cause` → `fault_root_cause` 列表
  - `recommended_countermeasure` → `solution` 列表
  - 相似工况保留结构化列表格式
  - 参考材料从相似工况摘要生成

#### 输入验证
- `validate_standard_input()`: 验证必填字段
- `build_error_output()`: 构建错误状态的标准输出

### 4. 更新 LLM Prompt

**文件**: `agent/prompts.py` 和 `agent/langchain_agent.py`

更新 LLM 输出格式要求，新增字段：
- `fault_root_cause`: 列出 2-3 个最可能的具体原因
- `fault_trigger_condition`: 故障触发条件描述
- `classification`: 故障的专业分类
- `solution`: 列出可执行的解决步骤
- `risk_warning`: 故障的风险等级
- `maintenance_suggestions`: 长期维护建议

更新诊断推理原则：
- 增加结构化输出要求
- 强调所有新增字段必须填充

### 5. 新增诊断接口

**文件**: `agent/langchain_agent.py`

新增 `diagnose_with_standard_input()` 方法：
1. 验证输入
2. 转换为内部 ParsedInput
3. 调用 InputRouter 进行意图路由
4. 调用内部 diagnose 方法执行诊断
5. 转换为标准输出
6. 处理异常情况，返回错误状态码

### 6. 更新 CLI 接口

**文件**: `cli.py`

新增三个命令行参数：

| 参数 | 类型 | 说明 |
|-----|------|------|
| `--json-input` | `str` | 标准输入 JSON 文件路径 |
| `--generate-md` | `bool` | 生成 Markdown 报告（默认关闭） |
| `--std-output` | `bool` | 输出标准 JSON 格式到控制台 |

新增两种工作模式：

#### 标准接口模式
```bash
python -m diagnosis_agent diagnose --json-input input.json
```
- 从 JSON 文件读取标准输入
- 执行完整诊断流程
- 输出标准 JSON 格式到控制台和文件

#### 传统模式（向后兼容）
```bash
python -m diagnosis_agent diagnose --text "MCU报P1A3E98爬坡IGBT过温" --generate-md --std-output
```
- 使用原有的文本/文件输入方式
- `--generate-md`: 可选生成 Markdown 报告
- `--std-output`: 可选输出标准 JSON 格式

---

## 数据流图

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 平台 Agent   │────▶│ StandardInput    │────▶│ converter.py    │
│  (JSON)     │     │ (标准输入模型)   │     │ (输入转换)      │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                                                     ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 标准输出 JSON │◀────│ StandardOutput   │◀────│ converter.py    │
│ (对外接口)   │     │ (标准输出模型)   │     │ (输出转换)      │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     │
                                                     ▼
                                          ┌─────────────────┐
                                          │ DiagnosticOutput │
                                          │ (内部诊断结果)   │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ LangChainAgent   │
                                          │ (诊断推理)      │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ InputRouter      │
                                          │ (意图路由)      │
                                          └─────────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ ParsedInput      │
                                          │ (内部解析输入)   │
                                          └─────────────────┘
```

---

## 使用示例

### 示例 1：标准接口模式

```bash
# 创建标准输入 JSON 文件
cat > input.json << 'EOF'
{
    "raw_query": "MCU报P1A3E98爬坡IGBT过温",
    "mcuid": "MCU_001",
    "intent_name": "MCU_001",
    "query_confidence": 0.96,
    "entities": {
        "dtc_code": ["P1A3E98"],
        "project": "H37A",
        "component": "IGBT/MCU",
        "working_condition": "爬坡满载、高温",
        "software_version": "H37A3621830AW"
    },
    "user_id": "test_user"
}
EOF

# 执行诊断
python -m diagnosis_agent diagnose --json-input input.json --output output/
```

### 示例 2：传统模式（带标准输出）

```bash
# 使用文本输入
python -m diagnosis_agent diagnose \
    --text "MCU报P1A3E98爬坡IGBT过温" \
    --generate-md \
    --std-output
```

---

## 后续规划

本次重构仅定义了数据结构和转换逻辑，以下功能待后续实现：

1. **多轮对话逻辑**：`session_id` 和 `context_history` 已定义结构，对话管理逻辑待实现
2. **知识库接口**：`reference_material` 字段已预留，知识图谱检索待对接
3. **追问机制**：`need_multi_round` 和 `follow_up_question` 字段已预留，追问逻辑待实现
4. **意图识别对接**：平台 Agent 的意图识别结果（`intent_name`）已支持，后续可根据实际接口调整映射关系
