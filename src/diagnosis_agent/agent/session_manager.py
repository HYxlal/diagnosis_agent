"""会话管理器 — 多轮问询状态管理

管理每个 session_id 的对话历史，在 Agent 调用前注入上下文。

核心设计：
- SessionManager 独立于 Agent，Agent 不感知多轮
- 存储：支持文件持久化，跨进程恢复会话
- 上下文注入：通过 diagnose_with_standard_input(session_context=...) 传递
- StandardOutput 不含对话控制字段，多轮是会话层职责

使用方式：
    # 文件持久化（默认）：跨进程可用
    sm = SessionManager(persist_dir="data/sessions")
    ctx = sm.get_context("sess-123")
    sm.update("sess-123", "用户的提问", "Agent的回答")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器（单例 + 文件持久化）

    每次 get_context / update 都会从文件重载，确保跨进程一致性。
    单例让同一进程内多次调用只读一次文件。

    Args:
        persist_dir: 会话持久化目录，默认 data/sessions。为 None 时纯内存。
    """

    _instance: Optional["SessionManager"] = None
    _sessions: dict[str, list[dict[str, str]]] = {}
    _persist_dir: str = ""

    def __new__(cls, persist_dir: str = "data/sessions") -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_dir: str = "data/sessions") -> None:
        # 单例模式：只在首次创建时设置 persist_dir
        if not self._persist_dir:
            self._persist_dir = persist_dir
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"SessionManager 持久化目录: {persist_dir}")

    # ------------------------------------------------------------------
    # 文件持久化
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        """会话文件路径"""
        safe_name = session_id.replace("/", "_").replace("\\", "_")
        return Path(self._persist_dir) / f"{safe_name}.json"

    def _load_from_disk(self, session_id: str) -> list[dict[str, str]]:
        """从文件加载会话历史"""
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
        """将会话历史写入文件"""
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
        """确保会话在内存中"""
        if session_id not in self._sessions:
            self._sessions[session_id] = self._load_from_disk(session_id)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_context(self, session_id: str, max_rounds: int = 5) -> str:
        """获取会话历史上下文 prompt

        Args:
            session_id: 会话唯一标识
            max_rounds: 最多保留几轮历史（默认 5，防 prompt 超长）

        Returns:
            格式化的上下文文本，无历史时返回空字符串
        """
        if not session_id:
            return ""
        self._load_to_memory(session_id)
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
        if not session_id:
            return
        self._load_to_memory(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"query": query, "answer": answer})
        self._save_to_disk(session_id)
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
        """获取会话轮次"""
        if not session_id:
            return 0
        self._load_to_memory(session_id)
        return len(self._sessions.get(session_id, []))

    def has_history(self, session_id: str) -> bool:
        """是否有历史"""
        if not session_id:
            return False
        self._load_to_memory(session_id)
        return session_id in self._sessions and bool(self._sessions[session_id])

    def _format_answer_for_context(self, standard_output) -> str:
        """从 StandardOutput 提取可读的回答文本"""
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