# 电驱诊断 Agent 输入处理流程说明

## 概述

本文档详细描述电驱诊断 Agent 从接收输入到输出诊断结果的完整处理流程，涵盖所有关键环节的 debug 级细节。

---

## 1. 输入接收阶段

### 1.1 触发入口

```
CLI 命令: python -m diagnosis_agent.cli diagnose --json-input <file.json>
```

**入口函数**: `cli.py` → `diagnose()` 函数  
**文件位置**: `src/diagnosis_agent/cli.py`

### 1.2 JSON 文件加载

**文件路径**: 由 `--json-input` 参数指定

**处理步骤**:
1. 检查文件是否存在 (`cli.py` → 第 182 行)
2. 读取文件内容 (`cli.py` → 第 183 行)
3. 使用 `json.load()` 解析 JSON (`cli.py` → 第 184 行)

**异常处理**:
- 文件不存在 → 返回 `{"code": -1, "msg": "JSON文件不存在: <path>"}` (`cli.py` → 第 185-188 行)
- JSON 格式错误 → 返回 `{"code": -1, "msg": "JSON解析失败: <error>"}` (`cli.py` → 第 189-192 行)

---

## 2. 输入验证阶段

### 2.1 StandardInput 创建

使用 Pydantic 模型将 JSON 数据转换为 `StandardInput` 对象。

**模型定义**: `models/input.py` → `StandardInput` 类  
**文件位置**: `src/diagnosis_agent/models/input.py`

**必填字段**:
| 字段 | 类型 | 说明 |
|-----|-----|------|
| `raw_query` | str | 用户原始提问文本 |
| `mcuid` | str | MCU 标识 |

**可选字段**:
| 字段 | 类型 | 说明 |
|-----|-----|------|
| `intent_name` | str | 平台意图识别结果 |
| `query_confidence` | float | 意图置信度 0-1 |
| `entities` | StandardEntities | 结构化实体信息 (`models/input.py` → `StandardEntities` 类) |
| `session_id` | str | 会话标识（多轮用） |
| `context_history` | list | 历史对话上下文 |
| `user_id` | str | 用户标识 |

**创建位置**: `cli.py` → 第 196 行  
**异常处理**:
- 必填字段缺失 → 返回 `{"code": -1, "msg": "入参缺失关键信息无法诊断: <error>"}` (`cli.py` → 第 197-200 行)

### 2.2 关键字段验证

验证 `raw_query` 和 `mcuid` 是否为非空字符串。

**验证函数**: `converter.py` → `validate_standard_input()` 函数  
**文件位置**: `src/diagnosis_agent/models/converter.py`

```python
# converter.py → validate_standard_input()
def validate_standard_input(standard_input):
    if not standard_input.raw_query or not standard_input.raw_query.strip():
        return "缺少必填字段: raw_query（用户原始提问文本）不能为空"
    if not standard_input.mcuid or not standard_input.mcuid.strip():
        return "缺少必填字段: mcuid（MCU标识）不能为空"
    return None
```

**调用位置**: `langchain_agent.py` → `diagnose_with_standard_input()` 方法 → 第 122 行  
**异常处理**:
- 验证失败 → 返回 `{"code": -1, "msg": "<具体错误信息>"}` (`langchain_agent.py` → 第 124-129 行)

---

## 3. 输入转换阶段

### 3.1 StandardInput → ParsedInput

将标准输入转换为内部解析输入对象。

**转换函数**: `converter.py` → `standard_input_to_parsed()` 函数  
**文件位置**: `src/diagnosis_agent/models/converter.py`

**目标模型**: `models/input.py` → `ParsedInput` 类  
**文件位置**: `src/diagnosis_agent/models/input.py`

**转换逻辑**:
| StandardInput | ParsedInput | 处理方式 |
|---|---|---|
| `raw_query` | `description` / `raw_input` | 直接传递 |
| `mcuid` | → 检索查询前缀 | 格式: `[MCU_001] 用户提问` |
| `entities` | `field_extraction` | 方案A：直接赋值，跳过FieldExtractor |
| `intent_name` | `intent` 初始值 | 方案B：仍走InputRouter二次确认 |

**调用位置**: `langchain_agent.py` → `diagnose_with_standard_input()` 方法 → 第 133 行  
**异常处理**:
- 转换失败 → 返回 `{"code": -2, "msg": "输入转换失败: <error>"}` (`langchain_agent.py` → 第 134-140 行)

---

## 4. 意图路由阶段

### 4.1 InputRouter 初始化

**组件**: `InputRouter` 类  
**文件位置**: `src/diagnosis_agent/agent/input_router.py`

**配置**: 使用 qwen-turbo 模型进行轻量意图分类  
**初始化位置**: `langchain_agent.py` → `diagnose_with_standard_input()` 方法 → 第 144-145 行

### 4.2 意图分类

使用 LLM 对输入进行 5 类意图分类：

**分类方法**: `input_router.py` → `route()` 方法  
**文件位置**: `src/diagnosis_agent/agent/input_router.py`

| 意图类型 | 说明 | 后续处理 |
|---------|-----|---------|
| `diagnostic_query` | 电驱系统故障描述 | 执行检索 + 诊断 |
| `instruction` | 操作指令 | 跳过检索，直接诊断 |
| `supplement` | 信息补充 | 跳过检索，作为上下文 |
| `working_condition_file` | 工况文件 | 先转换再重新路由 |
| `out_of_scope` | 非电驱系统问题 | 返回 -3 状态码 |

**意图枚举**: `models/input.py` → `InputIntent` 类  
**文件位置**: `src/diagnosis_agent/models/input.py`

### 4.3 非电驱问题检测

**判定逻辑**: LLM 判断输入内容是否与电驱系统（MCU/电机/逆变器）相关  
**检测位置**: `langchain_agent.py` → `diagnose_with_standard_input()` 方法 → 第 149 行

**示例**:
- ✅ 电驱问题: "电机报P1A3E98过温"、"MCU通讯丢失"
- ❌ 非电驱问题: "空调不制冷"、"车门打不开"

**异常处理**:
- 识别为非电驱问题 → 返回 `{"code": -3, "msg": "识别为非电驱系统问题，不执行诊断"}` (`langchain_agent.py` → 第 151-155 行)

---

## 5. 检索阶段

### 5.1 相似工况检索

**检索方法**: `langchain_agent.py` → `_retrieve_similar_cases()` 方法  
**文件位置**: `src/diagnosis_agent/agent/langchain_agent.py`

**检索引擎**: `storage/chroma_store.py` → `ChromaVectorStore` 类  
**文件位置**: `src/diagnosis_agent/storage/chroma_store.py`

**检索策略**: 
- 向量语义检索（Embedding）: `retrieval/langchain_retrievers.py` → `LangChainRetriever` 类  
  **文件位置**: `src/diagnosis_agent/retrieval/langchain_retrievers.py`
- BM25 关键词检索（备用）

**检索参数**:
- `top_k`: 返回最相似的 5 条记录
- `score_threshold`: 最低相似度阈值 0.5

**过滤条件**:
- 如果有车型信息，按 `vehicle_type` 过滤
- 如果有 DTC 码，按 `dtc_code` 过滤

### 5.2 检索结果处理

返回相似工况列表，每条记录使用 `SimilarCase` 模型：  
**模型定义**: `models/diagnosis.py` → `SimilarCase` 类  
**文件位置**: `src/diagnosis_agent/models/diagnosis.py`

| 字段 | 说明 |
|-----|------|
| `record_id` | 向量库记录 ID |
| `problem_description` | 问题描述 |
| `root_cause` | 根本原因 |
| `countermeasure` | 解决措施 |
| `similarity` | 相似度分数 0-1 |

---

## 6. LLM 推理阶段

### 6.1 ReAct Agent 初始化

**Agent 类**: `langchain_agent.py` → `LangChainDiagnosticAgent` 类  
**文件位置**: `src/diagnosis_agent/agent/langchain_agent.py`

**模型**: qwen3.6-flash  
**Prompt 定义**: `agent/prompts.py` → `REACT_PROMPT` 常量  
**文件位置**: `src/diagnosis_agent/agent/prompts.py`

**工具集** (定义在 `agent/tools.py` → `AgentTools` 类):  
**文件位置**: `src/diagnosis_agent/agent/tools.py`
- `search_similar_incidents`: 搜索相似工单
- `get_incident_detail`: 获取工单详情
- `filter_by_vehicle_type`: 按车型过滤

### 6.2 推理循环

执行 ReAct 循环，最多 5 步：

**循环方法**: `langchain_agent.py` → `_run_agent()` 方法  
**文件位置**: `src/diagnosis_agent/agent/langchain_agent.py`

```
Step 1: Thought + Action → 调用 search_similar_incidents
Step 2: Thought + Action → 分析检索结果
Step 3: Thought + Action → 调用 get_incident_detail (可选)
Step 4: Thought + Action → 推理根因
Step 5: Final Answer → 输出结构化诊断结果
```

### 6.3 LLM 输出要求

LLM 必须输出符合以下 JSON 格式的结果，由 `agent/prompts.py` 中的 Prompt 约束。  
**解析方法**: `langchain_agent.py` → `_parse_llm_output()` 方法  
**文件位置**: `src/diagnosis_agent/agent/langchain_agent.py`

```json
{
    "fault_root_cause": ["具体原因1", "具体原因2"],
    "fault_trigger_condition": "故障触发条件描述",
    "classification": "从固定分类列表中选择",
    "solution": ["解决方案步骤1", "解决方案步骤2"],
    "risk_warning": "V1/V2/V3",
    "maintenance_suggestions": "维护建议",
    "confidence": 0.9
}
```

**故障分类选项**（必须从以下列表选择）:
- 驱动异常故障
- 控制异常故障
- 超速故障
- 高压异常故障
- 低压异常故障
- 过温故障
- 通信故障
- 旋变故障
- 状态机故障
- 油泵故障

**异常处理**:
- LLM 调用失败 → 返回 `{"code": -2, "msg": "Agent输出异常: <error>"}` (`langchain_agent.py` → 第 158-164 行)
- LLM 输出解析失败 → 在 `_run_agent()` 中处理

---

## 7. 结果转换阶段

### 7.1 DiagnosticOutput → StandardOutput

将内部诊断结果转换为标准输出格式。

**转换函数**: `converter.py` → `diagnostic_output_to_standard()` 函数  
**文件位置**: `src/diagnosis_agent/models/converter.py`

**内部输出模型**: `models/diagnosis.py` → `DiagnosticOutput` 类  
**文件位置**: `src/diagnosis_agent/models/diagnosis.py`

**标准输出模型**: `models/diagnostic_output.py` → `StandardOutput` 类  
**文件位置**: `src/diagnosis_agent/models/diagnostic_output.py`

**转换映射**:
| LLM 推理结果 | 标准输出 | 说明 |
|-------------|---------|------|
| `fault_root_cause` | `diagnosis_result.fault_root_cause` | 直接映射 |
| `fault_trigger_condition` | `diagnosis_result.fault_trigger_condition` | 直接映射 |
| `classification` | `diagnosis_result.classification` | 直接映射 |
| `solution` | `diagnosis_result.solution` | 直接映射 |
| `risk_warning` | `diagnosis_result.risk_warning` | 直接映射 |
| `maintenance_suggestions` | `diagnosis_result.maintenance_suggestions` | 直接映射 |
| `similar_cases` (内部报告) | `diagnosis_result.similar_cases` | 格式化为字符串 |
| 检索结果摘要 | `reference_material` | 相似工况摘要 |
| LLM 置信度 | `diagnosis_confidence` | 直接映射 |

### 7.2 相似工况格式化

将内部相似工况列表格式化为字符串，在 `converter.py` → `diagnostic_output_to_standard()` 函数中处理（约第 210-220 行）。

```python
# converter.py → diagnostic_output_to_standard() 内部
similar_cases_str = "、".join([
    f"历史案例{record_id}：{description[:50]}"
    for each similar case
])
```

**异常处理**:
- 转换失败 → 返回 `{"code": -2, "msg": "输出转换失败: <error>"}` (`langchain_agent.py` → 第 167-173 行)

---

## 8. 输出阶段

### 8.1 成功输出

**输出位置**: `cli.py` → 第 224-225 行（JSON 打印）、第 237-239 行（文件保存）

```json
{
    "code": 0,
    "msg": "诊断推理完成",
    "input_ref": {
        "raw_query": "用户提问",
        "intent_id": "意图标识",
        "entities": {"dtc_code": ["P1A3E98"]}
    },
    "diagnosis_result": {
        "fault_root_cause": ["原因1", "原因2"],
        "fault_trigger_condition": "触发条件",
        "classification": "故障分类",
        "solution": ["步骤1", "步骤2"],
        "risk_warning": "V1",
        "maintenance_suggestions": "维护建议",
        "similar_cases": "历史案例XXX：描述"
    },
    "reference_material": ["参考材料1", "参考材料2"],
    "need_multi_round": false,
    "follow_up_question": "",
    "diagnosis_confidence": 0.9
}
```

### 8.2 错误输出

错误状态只返回 `code` 和 `msg` 两个字段。  
**输出位置**: `cli.py` → 第 229-231 行

| 状态码 | 说明 | 示例 |
|-------|-----|------|
| `0` | 成功返回 | 完整诊断结果 |
| `-1` | 入参缺失关键信息 | `{"code": -1, "msg": "缺少必填字段: mcuid不能为空"}` |
| `-2` | Agent 输出异常 | `{"code": -2, "msg": "Agent输出异常: LLM调用超时"}` |
| `-3` | 非电驱问题 | `{"code": -3, "msg": "识别为非电驱系统问题，不执行诊断"}` |

---

## 9. 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入处理流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  1.接收输入  │───▶│  2.验证输入  │───▶│  3.转换输入  │         │
│  │  JSON加载    │    │  字段校验    │    │  格式转换    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ -1:文件错误  │    │ -1:入参缺失  │    │ -2:转换失败  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  4.意图路由  │───▶│  5.检索相似  │───▶│  6.LLM推理  │         │
│  │  意图分类    │    │  向量检索    │    │  ReAct循环  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ -3:非电驱   │    │  检索结果    │    │ -2:推理异常  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐                           │
│  │  7.结果转换  │───▶│  8.输出结果  │                           │
│  │  格式映射    │    │  JSON输出    │                           │
│  └─────────────┘    └─────────────┘                           │
│         │                   │                                  │
│         ▼                   ▼                                  │
│  ┌─────────────┐    ┌─────────────┐                           │
│  │ -2:转换失败  │    │  0:成功     │                           │
│  └─────────────┘    └─────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 测试场景

### 场景1: 正常诊断

**输入**: 包含完整故障描述、mcuid、dtc_code

**预期输出**: `code: 0`，包含完整诊断结果

### 场景2: mcuid 为空

**输入**: mcuid 字段为空字符串

**预期输出**: `{"code": -1, "msg": "缺少必填字段: mcuid（MCU标识）不能为空"}`

### 场景3: 非电驱问题

**输入**: "空调不制冷"

**预期输出**: `{"code": -3, "msg": "识别为非电驱系统问题，不执行诊断"}`

### 场景4: LLM 调用失败

**输入**: 正常输入，但 LLM API 不可用

**预期输出**: `{"code": -2, "msg": "Agent输出异常: LLM调用失败"}`

---

## 11. 相关代码文件索引

| 文件路径 | 关键类/函数 | 说明 |
|---------|-----------|------|
| `src/diagnosis_agent/cli.py` | `diagnose()` | CLI 入口函数，处理 JSON 输入/输出 |
| `src/diagnosis_agent/models/input.py` | `StandardInput` | 标准输入模型 |
| `src/diagnosis_agent/models/input.py` | `StandardEntities` | 结构化实体模型 |
| `src/diagnosis_agent/models/input.py` | `ParsedInput` | 内部解析输入模型 |
| `src/diagnosis_agent/models/input.py` | `InputIntent` | 意图枚举（含 OUT_OF_SCOPE） |
| `src/diagnosis_agent/models/diagnostic_output.py` | `StandardOutput` | 标准输出模型 |
| `src/diagnosis_agent/models/diagnostic_output.py` | `StandardDiagnosisResult` | 诊断结果模型 |
| `src/diagnosis_agent/models/diagnostic_output.py` | `OutputCode` | 状态码枚举 |
| `src/diagnosis_agent/models/converter.py` | `standard_input_to_parsed()` | 输入转换函数 |
| `src/diagnosis_agent/models/converter.py` | `diagnostic_output_to_standard()` | 输出转换函数 |
| `src/diagnosis_agent/models/converter.py` | `validate_standard_input()` | 输入验证函数 |
| `src/diagnosis_agent/models/converter.py` | `build_error_output()` | 错误输出构建函数 |
| `src/diagnosis_agent/models/diagnosis.py` | `DiagnosticOutput` | 内部诊断输出模型 |
| `src/diagnosis_agent/models/diagnosis.py` | `DiagnosticReport` | 诊断报告模型 |
| `src/diagnosis_agent/models/diagnosis.py` | `SimilarCase` | 相似工况模型 |
| `src/diagnosis_agent/agent/input_router.py` | `InputRouter` | 意图路由类（含非电驱判断） |
| `src/diagnosis_agent/agent/input_router.py` | `InputRouter.route()` | 意图分类方法 |
| `src/diagnosis_agent/agent/langchain_agent.py` | `LangChainDiagnosticAgent` | 核心 Agent 类 |
| `src/diagnosis_agent/agent/langchain_agent.py` | `diagnose_with_standard_input()` | 标准诊断入口方法 |
| `src/diagnosis_agent/agent/langchain_agent.py` | `_run_agent()` | ReAct 推理循环方法 |
| `src/diagnosis_agent/agent/langchain_agent.py` | `_retrieve_similar_cases()` | 相似工况检索方法 |
| `src/diagnosis_agent/agent/tools.py` | `AgentTools` | Agent 工具集定义 |
| `src/diagnosis_agent/agent/prompts.py` | `REACT_PROMPT` | LLM 推理 Prompt |
| `src/diagnosis_agent/storage/chroma_store.py` | `ChromaVectorStore` | 向量存储类 |
| `src/diagnosis_agent/retrieval/langchain_retrievers.py` | `LangChainRetriever` | 检索器类 |

---

## 12. 状态码汇总

| 状态码 | 常量名 | 说明 | 触发位置 |
|-------|-------|------|---------|
| `0` | `OutputCode.SUCCESS` | 正常返回 | `converter.py` → `diagnostic_output_to_standard()` |
| `-1` | `OutputCode.MISSING_INPUT` | 入参缺失关键信息 | `langchain_agent.py` → 验证阶段 |
| `-2` | `OutputCode.INTERNAL_ERROR` | 内部推理异常 | `langchain_agent.py` → 转换/推理阶段 |
| `-3` | `OutputCode.OUT_OF_SCOPE` | 非电驱系统问题 | `langchain_agent.py` → 意图路由后 |

---

**文档版本**: v1.1  
**更新时间**: 2026-07-28  
**适用版本**: diagnosis_agent v0.3.1+
