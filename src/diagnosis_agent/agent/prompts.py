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
    search_query: str = "",
) -> str:
    """构建有相似工况时的 user prompt"""
    query_line = f"\n（检索词: \"{search_query}\"）" if search_query else ""
    return f"""## 当前故障描述

{description}

## 预检索结果{query_line}

{similar_cases_text}

请开始诊断。先分析预检索结果与当前故障的关联。如果预检索结果充分且与当前故障高度相关，足以支撑诊断，直接给出 Final Answer。如果预检索结果不足或内容偏离，可以自行编辑更精准的检索词调用 search_similar_incidents 或 query_fault_graph 进一步检索，或调用 get_incident_detail 查看某个特定工单的完整详情，再给出 Final Answer，但不要重复执行和当前预检索关键词相同的搜索。"""


def build_no_similar_case_prompt(
    description: str,
) -> str:
    """构建无预检索时的 user prompt

    # ====== 兜底推理路径 ======
    # 这是「预检索未找到相似工况」时的专门兜底 prompt。
    # 目前：让主 Agent LLM 基于领域知识直接推理（无历史工单证据）。
    #
    # 计划改造（待做任务，见 memory: 故障兜底查询层）：
    #   1. 这条 prompt 的推理职责将移交给新增的「故障循环查询层」
    #   2. 故障循环查询层接收：故障描述 + CAN 信号摘要（如果有）
    #   3. 主 Agent LLM 改为只做 JSON 结构化提取，不再承担推理
    #   4. 这条 prompt 对应改造为：故障循环查询层的输入/系统 prompt
    # ==========================
    """
    return f"""## 当前故障描述

{description}

## 预检索结果

预检索用户语句未能找到相似历史工单，如果觉得需要，可以自行编辑更精准的检索词调用 search_similar_incidents 或 query_fault_graph 补充检索，无论找到与否，都请基于你的专业领域知识进行诊断推理。

**重要：由于未找到相似历史工单，诊断缺乏历史数据支撑，confidence 不可超过 0.5。**"""


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
            f"相似度: {1 - meta.get('score', 0):.2f})"
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
