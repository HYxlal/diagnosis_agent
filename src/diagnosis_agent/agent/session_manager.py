"""会话管理器 — 多轮问询状态管理

管理每个 session_id 的对话历史，在 Agent 调用前注入上下文。

存储设计：
  热层 (hot_messages)  → 纯内存，进程重启后丢失
  温层 (warm_summaries) → 纯内存，进程重启后丢失
  冷层 (archive)        → 磁盘文件，持久化保留

兼容旧格式（rounds[] 列表）自动迁移。

核心设计：
- SessionManager 独立于 Agent，Agent 不感知多轮
- 消息格式：LangChain messages_to_dict / messages_from_dict 序列化
- 单例模式，同一进程内多次调用共享同一份内存数据
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .context.types import ConversationContext

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器（热层/温层内存，冷层磁盘）

    热层和温层数据存储在内存 dict 中，不写磁盘。
    冷层归档通过 archive() 写入磁盘。

    Args:
        persist_dir: 冷层归档目录，默认 data/sessions/archive。
    """

    _instance: Optional["SessionManager"] = None
    _sessions: dict[str, ConversationContext] = {}
    _persist_dir: str = ""

    def __new__(cls, persist_dir: str = "data/sessions") -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_dir: str = "data/sessions") -> None:
        if not self._persist_dir:
            self._persist_dir = persist_dir

    def _session_path(self, session_id: str) -> Path:
        safe_name = session_id.replace("/", "_").replace("\\", "_")
        return Path(self._persist_dir) / f"{safe_name}.json"

    def _load_to_memory(self, session_id: str) -> None:
        """懒加载：内存中没有则创建新的"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationContext.create_new(session_id)

    # ------------------------------------------------------------------
    # 公共接口（保持兼容）
    # ------------------------------------------------------------------

    def get_context(self, session_id: str) -> list[dict]:
        """获取会话历史消息列表（扁平化）

        Args:
            session_id: 会话唯一标识

        Returns:
            历史消息的 dict 列表（与 messages_to_dict 格式一致），
            无历史时返回空列表。
            调用方直接拼入 Agent invoke 的 messages 即可。
        """
        if not session_id:
            return []
        self._load_to_memory(session_id)
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return []
        # 返回热层消息的副本（避免外部修改）
        return list(ctx.hot_messages) if ctx.hot_messages else []

    def get_conversation_context(self, session_id: str) -> ConversationContext | None:
        """获取完整会话上下文对象（含温层摘要、话题信息等）"""
        if not session_id:
            return None
        self._load_to_memory(session_id)
        return self._sessions.get(session_id)

    def update(self, session_id: str, query: str, messages: list[dict]) -> None:
        """存储一轮对话的新增消息（纯内存）

        Args:
            session_id: 会话唯一标识
            query: 用户本轮提问（保留用于快速查看）
            messages: 本轮 Agent 调用产生的新增消息列表（序列化后的 dict）
        """
        if not session_id:
            return
        self._load_to_memory(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationContext.create_new(session_id)

        ctx = self._sessions[session_id]
        ctx.hot_messages.extend(messages)
        ctx.total_turns += 1
        ctx.touch()
        logger.info(
            f"Session {session_id}: 第 {ctx.total_turns} 轮已存储 "
            f"(热层 {len(ctx.hot_messages)} 条消息)"
        )

    def clear(self, session_id: Optional[str] = None) -> None:
        """清除会话（仅清除内存）"""
        if session_id:
            self._sessions.pop(session_id, None)
            logger.info(f"Session {session_id} 已清除")
        else:
            self._sessions.clear()
            logger.info("所有会话已清除")

    def get_round_count(self, session_id: str) -> int:
        """获取会话总轮次"""
        if not session_id:
            return 0
        self._load_to_memory(session_id)
        ctx = self._sessions.get(session_id)
        return ctx.total_turns if ctx else 0

    def has_history(self, session_id: str) -> bool:
        """是否有历史消息"""
        if not session_id:
            return False
        self._load_to_memory(session_id)
        ctx = self._sessions.get(session_id)
        return ctx is not None and len(ctx.hot_messages) > 0

    # ------------------------------------------------------------------
    # 分层管理接口（预留，Step 1-2 使用）
    # ------------------------------------------------------------------

    def add_warm_summary(
        self, session_id: str, summary: "TopicSnapshot"
    ) -> None:
        """添加温层摘要（Step 1 实现）"""
        if not session_id:
            return
        self._load_to_memory(session_id)
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return
        ctx.warm_summaries.append(summary)

    def set_current_topic(
        self, session_id: str, topic: Optional["TopicSnapshot"]
    ) -> None:
        """设置当前话题（Step 1 实现）"""
        if not session_id:
            return
        self._load_to_memory(session_id)
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return
        ctx.current_topic = topic

    def archive(self, session_id: str, user_id: str = "") -> Optional["ArchivedSession"]:
        """关闭并归档会话（冷层持久化）"""
        if not session_id:
            return None
        self._load_to_memory(session_id)
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return None
        from .context.types import ArchivedSession
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        duration = 0
        if ctx.created_at:
            try:
                created = datetime.fromisoformat(ctx.created_at)
                duration = int(
                    (datetime.now(timezone.utc) - created).total_seconds()
                )
            except (ValueError, TypeError):
                pass

        archived = ArchivedSession(
            session_id=session_id,
            total_turns=ctx.total_turns,
            topics=list(ctx.warm_summaries),
            created_at=ctx.created_at,
            archived_at=now,
            duration=duration,
            user_id=user_id,
        )

        # 写入冷层归档
        if self._persist_dir:
            archive_dir = Path(self._persist_dir) / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{ctx.session_id}.json"
            try:
                with open(archive_path, "w", encoding="utf-8") as f:
                    json.dump(archived.to_dict(), f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.warning(f"归档会话 {session_id} 失败: {e}")

        # 清除内存
        self.clear(session_id)
        logger.info(f"Session {session_id} 已归档 ({archived.total_turns} 轮)")
        return archived