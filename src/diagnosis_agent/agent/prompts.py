"""CoT (Chain-of-Thought) Prompt 模板

包含两种路径的 prompt：
1. 有相似工况路径：基于检索到的相似工单进行推理
2. 无相似工况路径：基于领域知识进行推理

强调真正的 ReAct 迭代循环：Thought → Action → Observation。
最终输出包含结构化的诊断字段（故障根因列表、解决方案列表等）。
"""

from __future__ import annotations

from typing import Optional


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

## 预检索结果

{similar_cases_text}

请基于以上预检索结果优先进行诊断推理：
1. 如果预检索结果内容充分且与当前故障高度相关，**直接**给出 Final Answer
2. 如果预检索结果不充分、信息不足，**只允许**调用 `get_incident_detail` 查看某个特定工单的完整详情
3. 只有在预检索结果明显偏离主题、无法支撑诊断时，才允许自行编辑检索词调用 `search_similar_incidents` 或 `query_fault_graph` 做补充检索
4. 不要重复执行和当前预检索关键词相同的搜索

完成以上任意一条路径后，给出 Final Answer。"""


def build_no_similar_case_prompt(
    description: str,
) -> str:
    """构建无预检索时的 user prompt"""
    return f"""## 当前故障描述

{description}

## 预检索结果

未找到相似历史工单。请先基于你的电驱系统专业知识进行诊断推理。
如果认为仅靠领域知识不足以给出可信结论，**可以**自行编辑更精准的检索词调用 `search_similar_incidents` 或 `query_fault_graph` 做补充检索，然后给出 Final Answer。"""


def format_similar_cases_for_prompt(cases: list) -> str:
    """将相似工单 Document 列表格式化为 prompt 文本

    Args:
        cases: LangChain Document 列表，metadata 包含 id/problem_description/similarity 等
    """
    if not cases:
        return "（未检索到相似工单）"

    lines = []
    for i, doc in enumerate(cases, 1):
        meta = doc.metadata
        lines.append(
            f"**工单 {i}** (ID: {meta.get('id', 'N/A')}, "
            f"相似度: {meta.get('score', 0):.2f})"
        )
        lines.append(f"  - 问题描述: {meta.get('problem_description', 'N/A')}")
        lines.append(f"  - 根本原因: {meta.get('root_cause', 'N/A')}")
        lines.append(f"  - 对策: {meta.get('countermeasure', 'N/A')}")
        lines.append(f"  - 驱动代码: {meta.get('drive_code', 'N/A')}")
        lines.append(f"  - 车型: {meta.get('vehicle_type', 'N/A')}")
        lines.append(f"  - 仪表盘指示: {meta.get('dashboard_indicator', 'N/A')}")
        lines.append(f"  - DTC码: {meta.get('dtc_code', 'N/A')}")
        lines.append(f"  - 故障场景: {meta.get('fault_scenario', 'N/A')}")
        lines.append("")

    return "\n".join(lines)
