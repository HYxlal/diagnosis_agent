"""会话管理器 — 多轮问询状态管理

管理每个 session_id 的对话历史，在 Agent 调用前注入上下文。

存储格式：
  ConversationContext（分层结构）
  └── hot_messages: 最近 N 轮完整消息（LangChain messages_to_dict 格式）
  └── warm_summaries: 旧对话摘要（TopicSnapshot 列表）
  └── current_topic: 当前话题
  └── total_turns: 总轮次

兼容旧格式（rounds[] 列表）自动迁移。

核心设计：
- SessionManager 独立于 Agent，Agent 不感知多轮
- 消息格式：LangChain messages_to_dict / messages_from_dict 序列化
- 支持文件持久化，跨进程恢复会话
- 单例模式，同一进程内多次调用只读一次文件
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .context.types import ConversationContext

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器（单例 + 文件持久化）

    每次 get_context / update 都会从文件重载，确保跨进程一致性。
    单例让同一进程内多次调用只读一次文件。

    Args:
        persist_dir: 会话持久化目录，默认 data/sessions。为 None 时纯内存。
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
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"SessionManager 持久化目录: {persist_dir}")

    # ------------------------------------------------------------------
    # 文件持久化
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        safe_name = session_id.replace("/", "_").replace("\\", "_")
        return Path(self._persist_dir) / f"{safe_name}.json"

    def _load_from_disk(self, session_id: str) -> ConversationContext:
        """从磁盘加载会话，兼容旧格式自动迁移"""
        if not self._persist_dir:
            return ConversationContext.create_new(session_id)

        path = self._session_path(session_id)
        if not path.exists():
            return ConversationContext.create_new(session_id)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"加载会话 {session_id} 失败: {e}")
            return ConversationContext.create_new(session_id)

        # 兼容旧格式：list[rounds] → 迁移为 ConversationContext
        if isinstance(data, list):
            return self._migrate_from_old_format(session_id, data)

        # 新格式：ConversationContext dict
        return ConversationContext.from_dict(data)

    def _migrate_from_old_format(
        self, session_id: str, rounds: list[dict]
    ) -> ConversationContext:
        """旧格式迁移：rounds[] → ConversationContext"""
        logger.info(f"迁移旧会话格式: {session_id} ({len(rounds)} 轮)")
        ctx = ConversationContext.create_new(session_id)
        # 扁平化所有轮次的消息到 hot_messages
        for r in rounds:
            ctx.hot_messages.extend(r.get("messages", []))
            ctx.total_turns += 1
        # 保存新格式
        self._save_to_disk_from_ctx(ctx)
        return ctx

    def _save_to_disk(self, session_id: str) -> None:
        """保存会话到磁盘"""
        if not self._persist_dir:
            return
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return
        self._save_to_disk_from_ctx(ctx)

    def _save_to_disk_from_ctx(self, ctx: ConversationContext) -> None:
        """直接保存 ConversationContext"""
        if not self._persist_dir:
            return
        path = self._session_path(ctx.session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"保存会话 {ctx.session_id} 失败: {e}")

    def _load_to_memory(self, session_id: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._load_from_disk(session_id)

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
        """存储一轮对话的新增消息

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
        self._save_to_disk(session_id)
        logger.info(
            f"Session {session_id}: 第 {ctx.total_turns} 轮已存储 "
            f"(热层 {len(ctx.hot_messages)} 条消息)"
        )

    def clear(self, session_id: Optional[str] = None) -> None:
        """清除会话"""
        if session_id:
            self._sessions.pop(session_id, None)
            path = self._session_path(session_id)
            if path.exists():
                path.unlink()
            logger.info(f"Session {session_id} 已清除")
        else:
            self._sessions.clear()
            if self._persist_dir:
                import shutil
                shutil.rmtree(self._persist_dir, ignore_errors=True)
                Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
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
        self._save_to_disk(session_id)

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
        self._save_to_disk(session_id)

    def archive(self, session_id: str, user_id: str = "") -> Optional["ArchivedSession"]:
        """关闭并归档会话（Step 2 实现）"""
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

        # 写入归档目录
        if self._persist_dir:
            archive_dir = Path(self._persist_dir) / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{ctx.session_id}.json"
            try:
                with open(archive_path, "w", encoding="utf-8") as f:
                    json.dump(
                        archived.to_dict(), f, ensure_ascii=False, indent=2
                    )
            except OSError as e:
                logger.warning(f"归档会话 {session_id} 失败: {e}")

        # 清除内存和活跃文件
        self.clear(session_id)
        logger.info(f"Session {session_id} 已归档 ({archived.total_turns} 轮)")
        return archived