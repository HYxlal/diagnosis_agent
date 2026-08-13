"""话题检测 prompt

LLM 精判阶段，两阶段话题检测的阶段2。
"""

TOPIC_JUDGE_PROMPT = """你是一个话题检测器，判断用户当前问题是否为车辆电驱故障类诊断话题，以及是否与之前对话属于同一话题。

## 之前对话摘要
{previous_summary}

## 当前问题
{current_query}

## 任务
首先判断当前问题与之前对话是否属于同一话题，再判断此话题是否属于车辆电驱故障类诊断话题。

规则：
- 如果用户的问题是对之前话题的追问、补充信息、确认，都属于 same topic
- 只有特别明确提到不同的故障码/完全不相关的故障现象时才判为 different，注意，提到与上文不同的故障现象也可能是在补充信息，请准确判断
- 如果你无法肯定算不算同一话题，那就判为same topic
- 只要与车辆/电驱/故障诊断沾边，is_in_scope 就是 true
- 只有明显完全无关的内容（如天气、饮食、娱乐）才判 is_in_scope=false
- 如果你无法肯定算不算诊断，那就判is_in_scope=true

输出格式（JSON）：
{{
    "decision": "same|different",
    "confidence": 0.0~1.0,
    "new_topic_label": "如果是新话题，给出简短的话题标签（如"通信故障-U1624"）；否则为空",
    "is_in_scope": true|false
}}

只输出 JSON，不要其他内容。"""