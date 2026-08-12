"""话题检测 prompt

LLM 精判阶段，两阶段话题检测的阶段2。
"""

TOPIC_JUDGE_PROMPT = """你是一个话题检测器，判断用户当前问题是否与之前对话属于同一话题。

## 之前对话摘要
{previous_summary}

## 当前问题
{current_query}

## 任务
判断当前问题与之前对话是否属于同一话题。

输出格式（JSON）：
{{
    "decision": "same|different",
    "confidence": 0.0~1.0,
    "new_topic_label": "如果是新话题，给出简短的话题标签（如"通信故障-U1624"）；否则为空",
    "is_in_scope": true|false  "当前问题是否在电驱系统（MCU/电机/逆变器）故障诊断范围内"
}}

只输出 JSON，不要其他内容。"""