"""CoT (Chain-of-Thought) Prompt 模板

包含两种路径的 prompt：
1. 有相似工况路径：基于检索到的相似工单进行推理
2. 无相似工况路径：基于领域知识进行推理

强调真正的 ReAct 迭代循环：Thought → Action → Observation。
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# 系统 Prompt（ReAct 格式）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一个专业的车辆故障诊断专家 Agent。

你的任务：分析故障描述，调用工具检索历史工单，输出诊断结论和推荐对策。

## 可用工具

{tool_descriptions}

## 工作流程（ReAct 迭代循环）

你必须在每一步使用以下格式：

Thought: 思考下一步该做什么，分析当前已知信息和未知信息
Action: 工具名称（从可用工具中选择，或填写 Final Answer）
Action Input: 工具参数（JSON 格式，如 {{"query": "发动机故障灯亮"}}）

系统会执行工具并返回：
Observation: 工具返回的结果

然后你继续下一步 Thought → Action → Action Input → Observation 循环。

当你得到足够信息后：
Thought: 我已经有了足够的信息来进行诊断
Action: Final Answer
Action Input: 最终诊断结果（必须为以下 JSON 格式）

最终诊断结果 JSON 格式：
{{
  "root_cause": "根本原因分析",
  "countermeasure": "推荐对策/解决措施",
  "confidence": 0.0-1.0 的置信度,
  "reasoning_narrative": "完整的推断过程叙述（200-500字），描述你从故障描述到最终结论的完整推理链路，包括你考虑了哪些可能性、排除了哪些、为什么",
  "findings": [
    {{
      "title": "发现标题",
      "description": "详细描述",
      "confidence": 0.0-1.0,
      "evidence": ["证据1", "证据2"]
    }}
  ]
}}

## 诊断推理原则（Chain-of-Thought）

1. **理解问题**：仔细分析故障描述，识别关键信息（车型、DTC码、仪表盘指示、驱动代码等）
2. **检索相似工况**：使用 search_similar_incidents 工具搜索历史工单。如果有车型信息，也可用 filter_by_vehicle_type 精确过滤
3. **对比分析**：将当前故障与历史工单对比，分析异同。如果有 DTC 码，可调用 get_incident_detail 查看详情
4. **推理根因**：基于证据推理可能的根本原因，排除不合理的假设
5. **制定对策**：给出针对性的解决措施

## 重要提示

- 你必须至少调用一次工具（search_similar_incidents）后再给出 Final Answer
- Final Answer 中的 reasoning_narrative 必须是完整的推断过程叙述
- 如果未检索到相似工况，基于你的专业领域知识进行推理
- confidence 反映你对诊断结论的把握程度，0.9 以上为非常确定，0.5 以下为推测"""


def build_system_prompt(tool_descriptions: str) -> str:
    """构建系统 prompt"""
    return SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)


# ---------------------------------------------------------------------------
# User Prompt 构建函数
# ---------------------------------------------------------------------------

def build_similar_case_prompt(
    description: str,
    similar_cases_text: str,
) -> str:
    """构建有相似工况时的 user prompt"""
    return f"""## 当前故障描述

{description}

## 已检索到的相似工况

{similar_cases_text}

请开始 ReAct 推理。先分析当前故障与相似工况的关系，然后调用工具进一步检索或查看详情，最后给出 Final Answer。"""


def build_no_similar_case_prompt(
    description: str,
) -> str:
    """构建无相似工况时的 user prompt"""
    return f"""## 当前故障描述

{description}

## 检索结果

未找到相似工况。请基于你的专业领域知识进行推理，给出 Final Answer。

注意：即使没有相似工况，你仍然需要先调用 search_similar_incidents 确认检索结果，再给出 Final Answer。"""


def format_similar_cases_for_prompt(cases: list[dict]) -> str:
    """将相似工单列表格式化为 prompt 文本"""
    if not cases:
        return "（未检索到相似工单）"

    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"**工单 {i}** (ID: {case.get('record_id', 'N/A')}, 相似度: {case.get('similarity', 0):.2f})")
        lines.append(f"  - 问题描述: {case.get('problem_description', 'N/A')}")
        lines.append(f"  - 根本原因: {case.get('root_cause', 'N/A')}")
        lines.append(f"  - 对策: {case.get('countermeasure', 'N/A')}")
        lines.append(f"  - 驱动代码: {case.get('drive_code', 'N/A')}")
        lines.append(f"  - 车型: {case.get('vehicle_type', 'N/A')}")
        lines.append(f"  - 仪表盘指示: {case.get('dashboard_indicator', 'N/A')}")
        lines.append(f"  - DTC码: {case.get('dtc_code', 'N/A')}")
        lines.append(f"  - 故障场景: {case.get('fault_scenario', 'N/A')}")
        lines.append("")

    return "\n".join(lines)
