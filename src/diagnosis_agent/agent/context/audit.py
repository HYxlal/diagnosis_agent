"""审计日志 — 完整记录上下文变更事件

记录以下事件类型：
- session_create: 会话创建
- session_end: 会话结束
- trim: 消息裁剪
- summarize: 摘要生成
- archive: 会话归档
- topic_switch: 话题切换

写入格式：结构化 JSON，每行一条记录，文件路径 data/sessions/audit/{conversation_id}.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """审计日志记录器（单例）

    异步写入 JSONL 文件，不阻塞主流程。
    """

    _instance: Optional["AuditLogger"] = None

    def __new__(cls) -> "AuditLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._base_dir = Path("data/sessions/audit")
        self._lock = threading.Lock()

    def _ensure_dir(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _write_event(self, conversation_id: str, event: dict) -> None:
        """写入一条审计事件"""
        try:
            self._ensure_dir()
            path = self._base_dir / f"{conversation_id}.jsonl"
            event["_timestamp"] = datetime.now(timezone.utc).isoformat()
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"审计日志写入失败: {e}")

    # ── 事件记录方法 ──

    def log_session_create(self, conversation_id: str) -> None:
        self._write_event(conversation_id, {
            "event": "session_create",
            "conversation_id": conversation_id,
        })

    def log_session_end(self, conversation_id: str, reason: str) -> None:
        self._write_event(conversation_id, {
            "event": "session_end",
            "conversation_id": conversation_id,
            "reason": reason,
        })

    def log_trim(
        self,
        conversation_id: str,
        step: str,
        before_count: int,
        after_count: int,
        token_usage: int = 0,
        max_tokens: int = 0,
    ) -> None:
        self._write_event(conversation_id, {
            "event": "trim",
            "conversation_id": conversation_id,
            "step": step,
            "before_count": before_count,
            "after_count": after_count,
            "dropped": before_count - after_count,
            "token_usage": token_usage,
            "max_tokens": max_tokens,
        })

    def log_summarize(
        self,
        conversation_id: str,
        topic_label: str,
        message_count: int,
        summary_length: int,
        turns: int = 0,
    ) -> None:
        self._write_event(conversation_id, {
            "event": "summarize",
            "conversation_id": conversation_id,
            "topic_label": topic_label,
            "message_count": message_count,
            "summary_length": summary_length,
            "turns": turns,
        })

    def log_archive(
        self,
        conversation_id: str,
        total_turns: int,
        duration: int,
    ) -> None:
        self._write_event(conversation_id, {
            "event": "archive",
            "conversation_id": conversation_id,
            "total_turns": total_turns,
            "duration": duration,
        })

    def log_topic_switch(
        self,
        conversation_id: str,
        old_topic: str,
        new_topic: str,
    ) -> None:
        self._write_event(conversation_id, {
            "event": "topic_switch",
            "conversation_id": conversation_id,
            "old_topic": old_topic,
            "new_topic": new_topic,
        })

    def read_logs(self, conversation_id: str) -> list[dict]:
        """读取指定会话的审计日志"""
        path = self._base_dir / f"{conversation_id}.jsonl"
        if not path.exists():
            return []
        events = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError as e:
            logger.warning(f"读取审计日志失败: {e}")
        return events

    def list_sessions(self) -> list[str]:
        """列出所有有审计日志的会话 ID"""
        self._ensure_dir()
        sessions = []
        for p in self._base_dir.glob("*.jsonl"):
            sessions.append(p.stem)
        return sorted(sessions)

    def cleanup(self, retention_days: int = 30) -> int:
        """清理过期审计日志"""
        self._ensure_dir()
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - retention_days * 86400
        cleaned = 0
        for p in self._base_dir.glob("*.jsonl"):
            try:
                mtime = os.path.getmtime(p)
                if mtime < cutoff:
                    p.unlink()
                    cleaned += 1
            except OSError:
                pass
        return cleaned


# 全局单例
audit = AuditLogger()