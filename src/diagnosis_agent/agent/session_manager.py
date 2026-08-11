"""会话管理器 — 多轮问询状态管理

管理每个 session_id 的对话历史，在 Agent 调用前注入上下文。

核心设计：
- SessionManager 独立于 Agent，Agent 不感知多轮
- 存储完整消息列表（HumanMessage / AIMessage / ToolMessage），
  不存业务摘要，不关心具体字段
- 消息列表由 LangChain 自带的 messages_to_dict / messages_from_dict 序列化
- 支持文件持久化，跨进程恢复会话
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器（单例 + 文件持久化）

    每次 get_context / update 都会从文件重载，确保跨进程一致性。
    单例让同一进程内多次调用只读一次文件。

    Args:
        persist_dir: 会话持久化目录，默认 data/sessions。为 None 时纯内存。
    """

    _instance: Optional["SessionManager"] = None
    _sessions: dict[str, list[dict[str, Any]]] = {}
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

    def _load_from_disk(self, session_id: str) -> list[dict[str, Any]]:
        if not self._persist_dir:
            return []
        path = self._session_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"加载会话 {session_id} 失败: {e}")
        return []

    def _save_to_disk(self, session_id: str) -> None:
        if not self._persist_dir:
            return
        rounds = self._sessions.get(session_id, [])
        path = self._session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rounds, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"保存会话 {session_id} 失败: {e}")

    def _load_to_memory(self, session_id: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._load_from_disk(session_id)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_context(self, session_id: str) -> list[dict]:
        """获取会话历史消息列表

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
        rounds = self._sessions.get(session_id, [])
        if not rounds:
            return []

        # 把所有轮次的消息扁平拉出来
        all_messages: list[dict] = []
        for r in rounds:
            all_messages.extend(r.get("messages", []))
        return all_messages

    def update(self, session_id: str, query: str, messages: list[dict]) -> None:
        """存储一轮对话的完整消息列表

        Args:
            session_id: 会话唯一标识
            query: 用户本轮提问（保留用于快速查看）
            messages: 本轮 Agent 调用产生的完整消息列表（序列化后的 dict）
        """
        if not session_id:
            return
        self._load_to_memory(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "query": query,
            "messages": messages,
        })
        self._save_to_disk(session_id)
        logger.info(
            f"Session {session_id}: 第 {len(self._sessions[session_id])} 轮已存储"
        )

    def clear(self, session_id: Optional[str] = None) -> None:
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
        if not session_id:
            return 0
        self._load_to_memory(session_id)
        return len(self._sessions.get(session_id, []))

    def has_history(self, session_id: str) -> bool:
        if not session_id:
            return False
        self._load_to_memory(session_id)
        return session_id in self._sessions and bool(self._sessions[session_id])