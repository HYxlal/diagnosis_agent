"""多轮对话上下文处理 — 降级重建历史消息

从平台 context_history 重建 {query, agent_answer} 格式的纯文本消息列表，
作为 SessionManager 本地会话不可用时的降级路径。
"""
from __future__ import annotations

from ..agent.session_manager import SessionManager


def build_history_messages(
    conversation_id: str,
    context_history: list[dict],
) -> list[dict] | None:
    """构建历史消息列表

    优先级：
    1. 本地 SessionManager 完整会话（含工具调用）→ 优先读取
    2. 降级：从平台 context_history 重建纯文本 user/assistant 消息
    """
    if not conversation_id and not context_history:
        return None

    # 1. 优先查本地完整会话
    if conversation_id:
        try:
            sm = SessionManager()
            ctx = sm.get_conversation_context(conversation_id)
            if ctx and ctx.hot_messages:
                return ctx.hot_messages
        except Exception:
            pass

    # 2. 降级：从平台 context_history 重建纯文本
    if not context_history:
        return None
    messages = []
    for turn in context_history:
        if turn.get("query"):
            messages.append({"role": "user", "content": turn["query"]})
        if turn.get("agent_answer"):
            messages.append({"role": "assistant", "content": turn["agent_answer"]})
    return messages if messages else None
