"""摘要生成器 — 温层摘要

将热层中超出窗口的旧消息压缩为结构化摘要，存入温层。

摘要格式（按规范）：
关键实体：
- 实体1
- 实体2
核心结论：
- 结论1
未解决问题：
- 问题1
关键约束：
- 约束1
时间戳：
- 时间
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ...config import get_settings
from ...utils.llm_factory import create_llm
from ...prompts.summarizer import SUMMARY_SYSTEM_PROMPT, SUMMARY_MERGE_PROMPT
from .types import TopicSnapshot

logger = logging.getLogger(__name__)


class Summarizer:
    """温层摘要生成器

    将热层消息压缩为 TopicSnapshot。
    使用结构化 prompt 输出 5 类信息：关键实体/核心结论/未解决问题/关键约束/时间戳。
    """

    def __init__(self, strategy: str = "llm", max_tokens: int = 500):
        self.strategy = strategy
        self.max_tokens = max_tokens

    def summarize(
        self,
        messages: list[dict],
        topic_label: str = "",
        start_turn: int = 0,
        end_turn: int = 0,
    ) -> TopicSnapshot:
        conversation_text = self._format_conversation(messages)

        if self.strategy == "llm":
            summary_data = self._summarize_with_llm(conversation_text)
        else:
            summary_data = self._summarize_with_template(messages)

        entities = summary_data.get("entities", [])
        conclusion = summary_data.get("conclusion", "")
        unsolved = summary_data.get("unsolved", [])
        constraints = summary_data.get("constraints", [])
        timestamps = summary_data.get("timestamps", [])

        if not topic_label and entities:
            topic_label = entities[0] if len(entities) == 1 else f"{entities[0]}等"

        now = datetime.now(timezone.utc).isoformat()
        summary_text = "\n".join(summary_data.get("raw_lines", [conclusion]))

        return TopicSnapshot(
            topic_id=uuid.uuid4().hex[:12],
            topic_label=topic_label or f"对话-{start_turn + 1}-{end_turn + 1}轮",
            start_turn=start_turn,
            end_turn=end_turn,
            summary=summary_text or conversation_text[:200],
            key_entities=entities,
            created_at=now,
            metadata={
                "unsolved": unsolved,
                "constraints": constraints,
                "timestamps": timestamps,
                "strategy": self.strategy,
            },
        )

    def merge_summaries(
        self, summaries: list[TopicSnapshot]
    ) -> TopicSnapshot:
        if not summaries:
            return TopicSnapshot(topic_id="", topic_label="", start_turn=0, end_turn=0, summary="")
        if len(summaries) == 1:
            return summaries[0]

        all_entities: list[str] = []
        for s in summaries:
            all_entities.extend(s.key_entities)
        unique_entities = list(dict.fromkeys(all_entities))

        if self.strategy == "llm":
            merged_text = "\n---\n".join(
                f"## {s.topic_label}\n{s.summary}" for s in summaries if s.summary
            )
            try:
                llm = self._create_llm()
                prompt = SUMMARY_MERGE_PROMPT + "\n\n" + merged_text[:3000]
                response = llm.invoke(prompt)
                parsed = self._parse_structured_response(response.content)
                if parsed:
                    return TopicSnapshot(
                        topic_id=uuid.uuid4().hex[:12],
                        topic_label=f"合并摘要({len(summaries)}个话题)",
                        start_turn=summaries[0].start_turn,
                        end_turn=summaries[-1].end_turn,
                        summary="\n".join(parsed.get("raw_lines", [])),
                        key_entities=parsed.get("entities", unique_entities),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        metadata={
                            "unsolved": parsed.get("unsolved", []),
                            "constraints": parsed.get("constraints", []),
                            "timestamps": parsed.get("timestamps", []),
                            "merged": True,
                            "strategy": "llm",
                        },
                    )
            except Exception as e:
                logger.warning(f"LLM 合并摘要失败，使用模板合并: {e}")

        merged_conclusion = "\n".join(f"- {s.topic_label}: {s.summary}" for s in summaries if s.summary)
        return TopicSnapshot(
            topic_id=uuid.uuid4().hex[:12],
            topic_label=f"合并摘要({len(summaries)}个话题)",
            start_turn=summaries[0].start_turn,
            end_turn=summaries[-1].end_turn,
            summary=merged_conclusion[:self.max_tokens],
            key_entities=unique_entities,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"merged": True, "strategy": "template"},
        )

    # ------------------------------------------------------------------
    # LLM 摘要
    # ------------------------------------------------------------------

    def _create_llm(self):
        settings = get_settings()
        return create_llm(
            model=get_settings().llm.model,
            temperature=0.1,
            max_tokens=self.max_tokens,
        )

    def _summarize_with_llm(self, text: str) -> dict:
        try:
            llm = self._create_llm()
            prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n对话内容：\n{text[:3000]}"
            response = llm.invoke(prompt)
            return self._parse_structured_response(response.content)
        except Exception as e:
            logger.warning(f"LLM 摘要失败，回退模板: {e}")
            return {}

    def _parse_structured_response(self, content: str) -> dict:
        """解析 LLM 返回的结构化摘要

        格式：
        关键实体：
        - xxx
        核心结论：
        - xxx
        ...
        """
        lines = content.split("\n")
        result = {
            "entities": [],
            "conclusion": [],
            "unsolved": [],
            "constraints": [],
            "timestamps": [],
            "raw_lines": [l.strip() for l in lines if l.strip()],
        }
        current_section = None

        section_map = {
            "关键实体": "entities",
            "核心结论": "conclusion",
            "未解决问题": "unsolved",
            "关键约束": "constraints",
            "时间戳": "timestamps",
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "：" in line or ":" in line:
                key = line.split("：")[0].split(":")[0].strip()
                if key in section_map:
                    current_section = section_map[key]
                    continue
            if current_section and line.startswith("-"):
                result[current_section].append(line[1:].strip())

        return result

    # ------------------------------------------------------------------
    # 模板摘要（不依赖 LLM）
    # ------------------------------------------------------------------

    def _summarize_with_template(self, messages: list[dict]) -> dict:
        all_text = ""
        for msg in messages:
            content = msg.get("data", {}).get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, str):
                all_text += content + "\n"

        import re
        dtc_codes = re.findall(r'[PUV]\d{4}(?:\d{2})?', all_text)
        entities = list(dict.fromkeys(dtc_codes))

        return {
            "entities": entities,
            "conclusion": all_text[:200],
            "unsolved": [],
            "constraints": [],
            "timestamps": [datetime.now(timezone.utc).isoformat()],
            "raw_lines": [f"关键实体：", f"- {', '.join(entities) if entities else '无'}", f"核心结论：", f"- {all_text[:200]}", f"未解决问题：", f"- 无", f"关键约束：", f"- 无", f"时间戳：", f"- {datetime.now(timezone.utc).isoformat()}"],
        }

    def _format_conversation(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            content = msg.get("data", {}).get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, str):
                lines.append(f"[{role}]: {content[:500]}")
            elif isinstance(content, dict):
                lines.append(f"[{role}]: {json.dumps(content, ensure_ascii=False)[:500]}")
        return "\n".join(lines)