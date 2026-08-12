"""摘要生成器 — 温层摘要

将热层中超出窗口的旧消息压缩为结构化摘要，存入温层。
支持 LLM 摘要和模板回退两种策略。

摘要格式：
{
    "entities": ["P1A3E98", "MCU", "H37A"],
    "conclusion": "电机过温故障，根因为传感器异常",
    "todos": ["检查温度数据流", "对比环境温度"]
}
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ...config import get_settings
from ...utils.llm_factory import create_llm
from .types import TopicSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 摘要 Prompt
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """你是一个电驱系统故障诊断对话的摘要生成器。
请根据对话内容，生成结构化的摘要，包含以下字段：

1. entities: 关键实体列表（DTC码、车型、部件名、软件版本等）
2. conclusion: 诊断结论总结（1-2句话）
3. todos: 待办事项列表（建议的排查步骤）

对话内容可能包含多轮交流，包括用户描述、诊断推理、工具调用结果等。
请提取最核心的信息。

只输出 JSON，不要其他内容。"""

# 模板回退 — 不依赖 LLM 的简单摘要
_TEMPLATE_STRATEGIES = {
    "classification": "故障分类",
    "fault_root_cause": "根因",
    "solution": "方案",
    "risk_warning": "风险",
}


# ---------------------------------------------------------------------------
# 摘要生成器
# ---------------------------------------------------------------------------


class Summarizer:
    """温层摘要生成器

    将热层消息压缩为 TopicSnapshot。
    LLM 不可用时使用模板回退。
    """

    def __init__(self, strategy: str = "llm", max_tokens: int = 500):
        """
        Args:
            strategy: 摘要策略 — "llm" | "rule" | "template"
            max_tokens: 摘要最大 token 数
        """
        self.strategy = strategy
        self.max_tokens = max_tokens

    def summarize(
        self,
        messages: list[dict],
        topic_label: str = "",
        start_turn: int = 0,
        end_turn: int = 0,
    ) -> TopicSnapshot:
        """生成摘要

        Args:
            messages: 本轮次的消息列表（LangChain messages_to_dict 格式）
            topic_label: 话题标签（可选）
            start_turn: 起始轮次
            end_turn: 结束轮次

        Returns:
            TopicSnapshot 对象
        """
        # 提取对话文本
        conversation_text = self._format_conversation(messages)

        # 根据策略生成摘要
        if self.strategy == "llm":
            summary_data = self._summarize_with_llm(conversation_text)
        else:
            summary_data = self._summarize_with_template(messages)

        entities = summary_data.get("entities", [])
        conclusion = summary_data.get("conclusion", conversation_text[:200])
        todos = summary_data.get("todos", [])
        metadata = {"todos": todos, "strategy": self.strategy}

        if not topic_label and entities:
            topic_label = f"{entities[0]}" if len(entities) == 1 else f"{entities[0]}等"

        return TopicSnapshot(
            topic_id=uuid.uuid4().hex[:12],
            topic_label=topic_label or f"对话-{start_turn + 1}-{end_turn + 1}轮",
            start_turn=start_turn,
            end_turn=end_turn,
            summary=conclusion,
            key_entities=entities,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    def merge_summaries(
        self, summaries: list[TopicSnapshot]
    ) -> TopicSnapshot:
        """合并多个温层摘要为一个

        Args:
            summaries: 待合并的摘要列表

        Returns:
            合并后的 TopicSnapshot
        """
        if not summaries:
            return TopicSnapshot(
                topic_id="",
                topic_label="",
                start_turn=0,
                end_turn=0,
                summary="",
            )

        if len(summaries) == 1:
            return summaries[0]

        # 收集所有实体和待办
        all_entities: list[str] = []
        all_todos: list[str] = []
        all_conclusions: list[str] = []

        for s in summaries:
            all_entities.extend(s.key_entities)
            if s.summary:
                all_conclusions.append(s.summary)
            todos = s.metadata.get("todos", [])
            if isinstance(todos, list):
                all_todos.extend(todos)

        # 去重
        unique_entities = list(dict.fromkeys(all_entities))
        unique_todos = list(dict.fromkeys(all_todos))

        # 如果 LLM 可用，用 LLM 压缩
        if self.strategy == "llm":
            merged_text = "\n".join(
                f"- {s.topic_label}: {s.summary}" for s in summaries if s.summary
            )
            try:
                llm = self._create_llm()
                prompt = f"""请将以下多个对话摘要合并为一个精炼的摘要：

{merged_text}

提取：关键实体、核心结论、待办事项。只输出 JSON。"""
                response = llm.invoke(prompt)
                data = self._parse_response(response.content)
                if data:
                    return TopicSnapshot(
                        topic_id=uuid.uuid4().hex[:12],
                        topic_label=f"合并摘要({len(summaries)}个话题)",
                        start_turn=summaries[0].start_turn,
                        end_turn=summaries[-1].end_turn,
                        summary=data.get("conclusion", "；".join(all_conclusions)),
                        key_entities=data.get("entities", unique_entities),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        metadata={"todos": data.get("todos", unique_todos), "merged": True},
                    )
            except Exception as e:
                logger.warning(f"LLM 合并摘要失败，使用模板合并: {e}")

        # 模板回退：拼接
        merged_conclusion = "；".join(
            f"{s.topic_label}: {s.summary}" for s in summaries if s.summary
        )
        return TopicSnapshot(
            topic_id=uuid.uuid4().hex[:12],
            topic_label=f"合并摘要({len(summaries)}个话题)",
            start_turn=summaries[0].start_turn,
            end_turn=summaries[-1].end_turn,
            summary=merged_conclusion[:self.max_tokens],
            key_entities=unique_entities,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"todos": unique_todos, "merged": True, "strategy": "template"},
        )

    # ------------------------------------------------------------------
    # LLM 摘要
    # ------------------------------------------------------------------

    def _create_llm(self):
        """创建用于摘要的轻量 LLM"""
        settings = get_settings()
        # 优先使用 summary 专用模型，没有则用 qwen-turbo（轻量）
        model = settings.context.summary_strategy or "qwen-turbo"
        return create_llm(
            model=model if model != "llm" else "qwen-turbo",
            temperature=0.1,
            max_tokens=self.max_tokens,
        )

    def _summarize_with_llm(self, text: str) -> dict:
        """用 LLM 生成结构化摘要"""
        try:
            llm = self._create_llm()
            prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n对话内容：\n{text[:3000]}"
            response = llm.invoke(prompt)
            return self._parse_response(response.content)
        except Exception as e:
            logger.warning(f"LLM 摘要失败，回退模板: {e}")
            return {}

    def _parse_response(self, content: str) -> dict:
        """解析 LLM 返回的 JSON"""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # 尝试从代码块中提取
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return {}

    # ------------------------------------------------------------------
    # 模板摘要（不依赖 LLM）
    # ------------------------------------------------------------------

    def _summarize_with_template(self, messages: list[dict]) -> dict:
        """从消息中提取关键信息"""
        all_text = ""
        conclusion_parts = []
        entities: list[str] = []
        todos: list[str] = []
        seen_entities: set[str] = set()

        for msg in messages:
            content = msg.get("data", {}).get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, str):
                all_text += content + "\n"
            elif isinstance(content, dict):
                all_text += json.dumps(content, ensure_ascii=False) + "\n"

        # 提取标准字段
        for field, label in _TEMPLATE_STRATEGIES.items():
            for line in all_text.split("\n"):
                if field in line.lower() or label in line:
                    # 提取 JSON 值
                    import re
                    values = re.findall(rf'"{field}":\s*"([^"]+)"', all_text)
                    if values:
                        conclusion_parts.append(f"{label}: {values[0]}")
                        break

        # 提取 DTC 码
        import re
        dtc_codes = re.findall(r'[PUV]\d{4}(?:\d{2})?', all_text)
        for code in dtc_codes:
            if code not in seen_entities:
                entities.append(code)
                seen_entities.add(code)

        return {
            "entities": entities,
            "conclusion": "；".join(conclusion_parts) if conclusion_parts else all_text[:200],
            "todos": todos,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _format_conversation(self, messages: list[dict]) -> str:
        """将消息列表格式化为对话文本"""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            content = msg.get("data", {}).get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, str):
                lines.append(f"[{role}]: {content[:500]}")
            elif isinstance(content, dict):
                lines.append(f"[{role}]: {json.dumps(content, ensure_ascii=False)[:500]}")
        return "\n".join(lines)