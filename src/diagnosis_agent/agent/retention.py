"""数据保留策略 — 自动清理过期归档

扫描 data/sessions/archive/ 目录，按两个维度清理：
1. 时间维度：删除超过 retention_days 的归档文件
2. 空间维度：总大小超过 max_archive_size_mb 时，从旧到新删除至满足限制

支持定时自动清理 + 手动触发。
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RetentionPolicy:
    """数据保留策略

    自动清理过期归档，释放磁盘空间。

    Args:
        archive_dir: 归档目录路径
        retention_days: 保留天数（默认 30 天）
        max_archive_size_mb: 归档目录最大大小（MB），默认 500MB
        cleanup_interval_hours: 自动清理间隔（小时），默认 24
    """

    def __init__(
        self,
        archive_dir: str = "data/sessions/archive",
        retention_days: int = 30,
        max_archive_size_mb: int = 500,
        cleanup_interval_hours: int = 24,
    ):
        self._archive_dir = Path(archive_dir)
        self._retention_days = retention_days
        self._max_size_mb = max_archive_size_mb
        self._interval_hours = cleanup_interval_hours
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def cleanup(self) -> dict:
        """执行一次清理

        Returns:
            清理统计: {"deleted_files": int, "freed_bytes": int, "remaining_files": int}
        """
        if not self._archive_dir.exists():
            return {"deleted_files": 0, "freed_bytes": 0, "remaining_files": 0}

        # 收集所有归档文件及其信息
        files_info: list[tuple[Path, float, int]] = []  # (path, mtime, size)
        for p in self._archive_dir.glob("*.json"):
            if p.name.endswith(".snapshot.json"):
                continue  # 跳过话题快照文件
            try:
                stat = p.stat()
                files_info.append((p, stat.st_mtime, stat.st_size))
            except OSError:
                continue

        if not files_info:
            return {"deleted_files": 0, "freed_bytes": 0, "remaining_files": 0}

        deleted = 0
        freed = 0
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - self._retention_days * 86400

        to_delete: set[Path] = set()
        total_size = 0

        for path, mtime, size in files_info:
            total_size += size
            # 时间维度：超过保留天数
            if mtime < cutoff:
                to_delete.add(path)

        # 空间维度：总大小超过限制
        size_limit = self._max_size_mb * 1024 * 1024
        if total_size > size_limit:
            # 按修改时间排序（从旧到新），删除最旧的直到满足限制
            files_info.sort(key=lambda x: x[1])  # 按 mtime 升序
            remaining = total_size
            for path, mtime, size in files_info:
                if path in to_delete:
                    continue
                if remaining <= size_limit:
                    break
                to_delete.add(path)
                remaining -= size

        # 执行删除
        for path in to_delete:
            try:
                file_size = path.stat().st_size
                path.unlink()
                deleted += 1
                freed += file_size
                logger.info(f"已删除过期归档: {path.name} ({file_size} bytes)")
            except OSError as e:
                logger.warning(f"删除归档失败 {path.name}: {e}")

        # 同时清理审计日志
        try:
            from .context.audit import audit
            audit_cleaned = audit.cleanup(self._retention_days)
            if audit_cleaned > 0:
                logger.info(f"已清理 {audit_cleaned} 个过期审计日志")
        except Exception:
            pass

        if deleted > 0:
            logger.info(
                f"清理完成: 删除 {deleted} 个文件, "
                f"释放 {freed / 1024 / 1024:.1f} MB"
            )

        return {
            "deleted_files": deleted,
            "freed_bytes": freed,
            "remaining_files": len(files_info) - deleted,
        }

    def start_scheduled(self) -> None:
        """启动定时清理（后台线程）"""
        if self._running:
            return
        self._running = True
        self._schedule_next()

    def _schedule_next(self) -> None:
        """安排下一次清理"""
        if not self._running:
            return
        self._timer = threading.Timer(
            self._interval_hours * 3600,
            self._on_timer,
        )
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        """定时器回调"""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"定时清理失败: {e}")
        self._schedule_next()

    def stop(self) -> None:
        """停止定时清理"""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None