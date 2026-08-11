"""会话管理器 — 多轮问询状态管理

管理每个 session_id 的对话历史，在 Agent 调用前注入上下文。

核心设计：
- SessionManager 独立于 Agent，Agent 不感知多轮
- 存储：当前用内存 dict，可后续升级到 Redis/文件
- 上下文注入：通过 diagnose_with_standard_input(session_context=...) 传递
- StandardOutput 不含对话控制字段，多轮是会话层职责
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 单条会话记录
_SessionEntry = dict  # {"rounds": [{"query": str, "answer": str}, ...]}


class SessionManager:
    """会话管理器（单例）

    使用方式：
        manager = SessionManager()
        context = manager.get_context("sess-123")
        # context 为空字符串表示无历史
        ...

        manager.update("sess-123", "用户问题", "Agent回答")
    """

    _instance: Optional["SessionManager"] = None
    _sessions: dict[str, list[dict[str, str]]] = {}

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_context(self, session_id: str, max_rounds: int = 5) -> str:
        """获取会话历史上下文 prompt

        Args:
            session_id: 会话唯一标识
            max_rounds: 最多保留几轮历史（默认 5，防 prompt 超长）

        Returns:
            格式化的上下文文本，无历史时返回空字符串
        """
        rounds = self._sessions.get(session_id, [])
        if not rounds:
            return ""

        recent = rounds[-max_rounds:]
        lines = ["[历史对话]"]
        for r in recent:
            q = r.get("query", "")
            a = r.get("answer", "")
            if q:
                lines.append(f"用户: {q}")
            if a:
                # 截取 answer 前 300 字，避免太长
                a_short = a[:300] + ("..." if len(a) > 300 else "")
                lines.append(f"Agent: {a_short}")
        lines.append("[当前问题]")
        return "\n".join(lines)

    def update(self, session_id: str, query: str, answer: str) -> None:
        """存储一轮对话

        Args:
            session_id: 会话唯一标识
            query: 用户本轮提问
            answer: Agent 本轮回答（从 StandardOutput 提取）
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"query": query, "answer": answer})
        logger.info(
            f"Session {session_id}: 第 {len(self._sessions[session_id])} 轮已存储"
        )

    def clear(self, session_id: Optional[str] = None) -> None:
        """清除会话

        Args:
            session_id: 指定清除的会话，None 时清除所有
        """
        if session_id:
            self._sessions.pop(session_id, None)
            logger.info(f"Session {session_id} 已清除")
        else:
            self._sessions.clear()
            logger.info("所有会话已清除")

    def get_round_count(self, session_id: str) -> int:
        """获取会话轮次"""
        return len(self._sessions.get(session_id, []))

    def has_history(self, session_id: str) -> bool:
        """是否有历史"""
        return session_id in self._sessions and bool(self._sessions[session_id])

    def _format_answer_for_context(self, standard_output) -> str:
        """从 StandardOutput 提取可读的回答文本

        用于构建上下文 prompt 中的 Agent 回答。
        """
        parts = []
        result = getattr(standard_output, "diagnosis_result", None)
        if result:
            if result.fault_root_cause:
                parts.append("根因: " + "; ".join(result.fault_root_cause[:3]))
            if result.classification:
                parts.append("分类: " + result.classification)
            if result.solution:
                parts.append("方案: " + "; ".join(result.solution[:3]))
        msg = getattr(standard_output, "msg", "")
        if msg and not parts:
            parts.append(msg)
        return " | ".join(parts) if parts else ""