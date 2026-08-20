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
from .context.audit import audit as _audit
from .retention import RetentionPolicy

logger = logging.getLogger(__name__)


class SessionRedisStore:
    """Redis 存储 — 热层/温层持久化

    保存/加载 ConversationContext，Redis TTL 自动处理空闲超时。
    Redis 不可用时降级到内存模式。

    Args:
        redis_url: Redis 连接 URL
        key_prefix: 键前缀
        default_ttl: 默认 TTL（秒），会被 session_idle_timeout 覆盖
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "session:",
        default_ttl: int = 3600,
    ):
        self._prefix = key_prefix
        self._default_ttl = default_ttl
        self._redis = None
        self._local: dict[str, ConversationContext] = {}  # 内存兜底
        self._available = False
        self._connect(redis_url)

    def _connect(self, redis_url: str) -> None:
        """尝试连接 Redis，失败时降级到内存"""
        try:
            import redis as r
            self._redis = r.from_url(redis_url, decode_responses=True, protocol=2)
            self._redis.ping()
            self._available = True
            logger.info(f"Redis 连接成功: {redis_url}")
        except Exception as e:
            self._available = False
            self._redis = None
            logger.warning(f"Redis 不可用，降级到内存模式: {e}")

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def save(self, ctx: ConversationContext, ttl: int | None = None) -> None:
        """保存会话上下文到 Redis（或内存兜底）

        Args:
            ctx: 会话上下文
            ttl: TTL 秒数，默认 self._default_ttl
        """
        ttl = ttl or self._default_ttl
        if self._available and self._redis:
            try:
                data = ctx.to_dict()
                # 添加状态字段（to_dict 可能不包含 status）
                self._redis.set(self._key(ctx.session_id), json.dumps(data, ensure_ascii=False), ex=ttl)
                return
            except Exception as e:
                logger.warning(f"Redis 保存失败，降级到内存: {e}")
                self._available = False
        # 内存兜底
        self._local[ctx.session_id] = ctx

    def load(self, session_id: str) -> ConversationContext | None:
        """从 Redis（或内存兜底）加载会话上下文"""
        key = self._key(session_id)
        if self._available and self._redis:
            try:
                raw = self._redis.get(key)
                if raw is None:
                    # Redis key 已过期（TTL 触发的空闲超时）
                    self._local.pop(session_id, None)
                    return None
                data = json.loads(raw)
                return ConversationContext.from_dict(data)
            except Exception as e:
                logger.warning(f"Redis 加载失败，降级到内存: {e}")
                self._available = False
        # 内存兜底
        return self._local.get(session_id)

    def delete(self, session_id: str) -> None:
        """删除会话"""
        key = self._key(session_id)
        if self._available and self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._local.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        key = self._key(session_id)
        if self._available and self._redis:
            try:
                return bool(self._redis.exists(key))
            except Exception:
                self._available = False
        return session_id in self._local

    def list_active(self) -> list[str]:
        """列出所有活跃会话 ID"""
        if self._available and self._redis:
            try:
                cursor, keys = self._redis.scan(match=f"{self._prefix}*")
                return [k[len(self._prefix):] for k in keys]
            except Exception:
                self._available = False
        return list(self._local.keys())

    def refresh_ttl(self, session_id: str, ttl: int) -> None:
        """刷新会话 TTL（续期）"""
        if self._available and self._redis:
            try:
                self._redis.expire(self._key(session_id), ttl)
            except Exception:
                pass


class SessionManager:
    """会话管理器（Redis 热层/温层 + 磁盘冷层）

    Redis 存储活跃会话，TTL 自动处理空闲超时。
    Redis 不可用时降级到内存模式。
    冷层归档通过 archive() 写入磁盘 JSON。

    Args:
        persist_dir: 冷层归档目录，默认 data/sessions。
    """

    _instance: Optional["SessionManager"] = None
    _sessions: dict[str, ConversationContext] = {}
    _persist_dir: str = ""
    _redis_store: Optional[SessionRedisStore] = None
    _idle_timeout: int = 3600
    _max_lifetime: int = 86400

    def __new__(cls, persist_dir: str = "data/sessions") -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, persist_dir: str = "data/sessions") -> None:
        if not self._persist_dir:
            self._persist_dir = persist_dir
            # 初始化 Redis 存储
            if self._redis_store is None:
                try:
                    from ..config import get_settings
                    settings = get_settings()
                    rc = settings.context.redis
                    if rc.enabled:
                        self._redis_store = SessionRedisStore(
                            redis_url=rc.url,
                            key_prefix=rc.key_prefix,
                            default_ttl=settings.context.session_idle_timeout,
                        )
                    self._idle_timeout = settings.context.session_idle_timeout
                    self._max_lifetime = settings.context.session_max_lifetime
                    # 启动数据保留策略定时清理
                    try:
                        from ..config import RetentionConfig
                        rt = settings.context.retention
                        self._retention = RetentionPolicy(
                            archive_dir=f"{persist_dir}/archive",
                            retention_days=rt.retention_days,
                            max_archive_size_mb=rt.max_archive_size_mb,
                            cleanup_interval_hours=rt.cleanup_interval_hours,
                        )
                        self._retention.start_scheduled()
                        logger.info(
                            f"数据保留策略已启动: {rt.retention_days} 天, "
                            f"{rt.max_archive_size_mb} MB, "
                            f"每 {rt.cleanup_interval_hours} 小时清理"
                        )
                    except Exception as e:
                        logger.warning(f"数据保留策略启动失败: {e}")
                except Exception as e:
                    logger.warning(f"SessionManager 初始化 Redis 失败: {e}")

    def _session_path(self, session_id: str) -> Path:
        safe_name = session_id.replace("/", "_").replace("\\", "_")
        return Path(self._persist_dir) / f"{safe_name}.json"

    def _archive_dir(self) -> Path:
        return Path(self._persist_dir) / "archive"

    def _active_dir(self) -> Path:
        return Path(self._persist_dir) / "active"

    def _save_active(self, ctx: ConversationContext) -> None:
        """保存活跃会话到磁盘（Redis 不可用时的兜底）"""
        try:
            active_dir = self._active_dir()
            active_dir.mkdir(parents=True, exist_ok=True)
            path = active_dir / f"{ctx.session_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"保存活跃会话 {ctx.session_id} 失败: {e}")

    def _load_to_memory(self, session_id: str) -> ConversationContext:
        """加载会话到内存 — 优先 Redis → 内存 → 活跃目录 → 冷层归档，最后新建

        状态机：找到 → active（刷新 TTL）
               都不存在 → created（新建）
        """
        if session_id in self._sessions:
            ctx = self._sessions[session_id]
            if self._check_expired(ctx) or self._check_idle(ctx):
                self._archive_session(ctx)
                self._sessions.pop(session_id, None)
                return self._create_new(session_id)
            if self._redis_store:
                self._redis_store.refresh_ttl(session_id, self._idle_timeout)
            return ctx

        # 1. Redis
        if self._redis_store:
            ctx = self._redis_store.load(session_id)
            if ctx is not None:
                if self._check_expired(ctx) or self._check_idle(ctx):
                    self._archive_session(ctx)
                    return self._create_new(session_id)
                ctx.status = "active"
                self._redis_store.refresh_ttl(session_id, self._idle_timeout)
                self._sessions[session_id] = ctx
                logger.info(f"从 Redis 恢复会话 {session_id} ({ctx.total_turns} 轮)")
                return ctx

        # 2. 活跃目录
        active_path = self._active_dir() / f"{session_id}.json"
        if active_path.exists():
            try:
                with open(active_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ctx = ConversationContext.from_dict(data)
                ctx.status = "active"
                self._sessions[session_id] = ctx
                logger.info(f"从活跃目录恢复会话 {session_id} ({ctx.total_turns} 轮)")
                return ctx
            except Exception as e:
                logger.warning(f"从活跃目录加载会话 {session_id} 失败: {e}")

        # 3. 冷层归档（兜底）
        restored = self.restore_from_archive(session_id)
        if restored is not None:
            return restored

        # 新建
        ctx = self._create_new(session_id)
        self._sessions[session_id] = ctx
        return ctx

    def _create_new(self, session_id: str) -> ConversationContext:
        """创建新会话（状态: created）"""
        ctx = ConversationContext.create_new(session_id)
        ctx.status = "created"
        _audit.log_session_create(session_id)
        return ctx

    def _check_expired(self, ctx: ConversationContext) -> bool:
        """检查会话是否已过期（最大存活时间）"""
        if not ctx.created_at:
            return False
        try:
            from datetime import datetime, timezone
            created = datetime.fromisoformat(ctx.created_at)
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()
            return elapsed > self._max_lifetime
        except (ValueError, TypeError):
            return False

    def _check_idle(self, ctx: ConversationContext) -> bool:
        """检查会话是否空闲超时

        基于 last_activity_at 判断，Redis TTL 之外的第二道防线。
        内存模式 / Redis 不可用时，这是唯一的空闲超时检测。
        """
        if not ctx.last_activity_at:
            return False
        try:
            from datetime import datetime, timezone
            last = datetime.fromisoformat(ctx.last_activity_at)
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            return elapsed > self._idle_timeout
        except (ValueError, TypeError):
            return False

    def _archive_session(self, ctx: ConversationContext) -> None:
        """归档过期会话到冷层（使用 ArchivedSession 格式，与 archive() 一致）"""
        try:
            from datetime import datetime, timezone
            from .context.types import ArchivedSession

            ctx.status = "archived"
            now = datetime.now(timezone.utc).isoformat()
            duration = 0
            if ctx.created_at:
                try:
                    created = datetime.fromisoformat(ctx.created_at)
                    duration = int((datetime.now(timezone.utc) - created).total_seconds())
                except (ValueError, TypeError):
                    pass

            archived = ArchivedSession(
                session_id=ctx.session_id,
                total_turns=ctx.total_turns,
                topics=list(ctx.warm_summaries),
                hot_messages=list(ctx.hot_messages),  # 保存完整热层消息
                created_at=ctx.created_at,
                archived_at=now,
                duration=duration,
            )

            archive_dir = self._archive_dir()
            archive_dir.mkdir(parents=True, exist_ok=True)
            path = archive_dir / f"{ctx.session_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(archived.to_dict(), f, ensure_ascii=False, indent=2)
            if self._redis_store:
                self._redis_store.delete(ctx.session_id)
            _audit.log_archive(ctx.session_id, ctx.total_turns, duration)
            logger.info(f"会话 {ctx.session_id} 已自动归档（{ctx.total_turns} 轮）")
        except OSError as e:
            logger.warning(f"归档会话 {ctx.session_id} 失败: {e}")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_context(self, session_id: str) -> list[dict]:
        """获取会话历史消息列表（扁平化）

        Args:
            session_id: 会话唯一标识

        Returns:
            历史消息的 dict 列表（与 messages_to_dict 格式一致），
            无历史时返回空列表。
        """
        if not session_id:
            return []
        ctx = self._load_to_memory(session_id)
        if ctx is None:
            return []
        # 状态检查：idle/archived 的会话返回空（触发重建）
        if ctx.status in ("idle", "archived", "closing"):
            return []
        return list(ctx.hot_messages) if ctx.hot_messages else []

    def get_conversation_context(self, session_id: str) -> ConversationContext | None:
        """获取完整会话上下文对象"""
        if not session_id:
            return None
        return self._load_to_memory(session_id)

    def update(self, session_id: str, query: str, messages: list[dict]) -> None:
        """存储一轮对话的新增消息

        Args:
            session_id: 会话唯一标识
            query: 用户本轮提问
            messages: 本轮新增消息列表（序列化后的 dict）
        """
        if not session_id:
            return
        ctx = self._load_to_memory(session_id)

        ctx.hot_messages.extend(messages)
        ctx.total_turns += 1
        ctx.touch()

        # 状态机：created → active
        if ctx.status == "created":
            ctx.status = "active"

        # 生命周期检查：最大存活时间
        if self._check_expired(ctx):
            self._archive_session(ctx)
            self._sessions.pop(session_id, None)
            logger.info(f"会话 {session_id} 已超最大存活时间，自动归档")
            return

        # 持久化到 Redis
        if self._redis_store:
            self._redis_store.save(ctx, ttl=self._idle_timeout)
        else:
            self._save_active(ctx)

        logger.info(
            f"Session {session_id}: 第 {ctx.total_turns} 轮已存储 "
            f"(状态: {ctx.status}, 热层 {len(ctx.hot_messages)} 条消息)"
        )

    def clear(self, session_id: Optional[str] = None) -> None:
        """清除会话（内存 + Redis）"""
        if session_id:
            self._sessions.pop(session_id, None)
            if self._redis_store:
                self._redis_store.delete(session_id)
            logger.info(f"Session {session_id} 已清除")
        else:
            self._sessions.clear()
            logger.info("所有会话已清除")

    def get_round_count(self, session_id: str) -> int:
        """获取会话总轮次"""
        if not session_id:
            return 0
        ctx = self._sessions.get(session_id)
        return ctx.total_turns if ctx else 0

    def has_history(self, session_id: str) -> bool:
        """是否有历史消息"""
        if not session_id:
            return False
        ctx = self._sessions.get(session_id)
        if ctx is None and self._redis_store:
            ctx = self._redis_store.load(session_id)
        return ctx is not None and len(ctx.hot_messages) > 0

    # ------------------------------------------------------------------
    # 温层管理
    # ------------------------------------------------------------------

    def add_warm_summary(
        self, session_id: str, summary: "TopicSnapshot"
    ) -> None:
        """添加温层摘要"""
        if not session_id:
            return
        ctx = self._load_to_memory(session_id)
        if ctx is None:
            return
        ctx.warm_summaries.append(summary)
        if self._redis_store:
            self._redis_store.save(ctx, ttl=self._idle_timeout)

    def set_current_topic(
        self, session_id: str, topic: Optional["TopicSnapshot"]
    ) -> None:
        """设置当前话题"""
        if not session_id:
            return
        ctx = self._load_to_memory(session_id)
        if ctx is None:
            return
        ctx.current_topic = topic

    # ------------------------------------------------------------------
    # 冷层归档与恢复
    # ------------------------------------------------------------------

    def archive(self, session_id: str, user_id: str = "") -> Optional["ArchivedSession"]:
        """关闭并归档会话（冷层持久化）"""
        if not session_id:
            return None
        ctx = self._load_to_memory(session_id)
        if ctx is None:
            return None

        ctx.status = "closing"
        from .context.types import ArchivedSession
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        duration = 0
        if ctx.created_at:
            try:
                created = datetime.fromisoformat(ctx.created_at)
                duration = int((datetime.now(timezone.utc) - created).total_seconds())
            except (ValueError, TypeError):
                pass

        archived = ArchivedSession(
            session_id=session_id,
            total_turns=ctx.total_turns,
            topics=list(ctx.warm_summaries),
            hot_messages=list(ctx.hot_messages),  # 保存完整热层消息
            created_at=ctx.created_at,
            archived_at=now,
            duration=duration,
            user_id=user_id,
        )

        # 写入冷层归档
        archive_dir = self._archive_dir()
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{ctx.session_id}.json"
        try:
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(archived.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"归档会话 {session_id} 失败: {e}")

        # 清除 Redis 和内存
        if self._redis_store:
            self._redis_store.delete(session_id)
        self._sessions.pop(session_id, None)

        ctx.status = "archived"
        logger.info(f"Session {session_id} 已归档 ({archived.total_turns} 轮)")
        return archived

    def restore_from_archive(self, session_id: str) -> ConversationContext | None:
        """恢复会话（优先冷层 archive/，其次活跃目录 active/）

        恢复后状态为 active。
        - 冷层归档：ArchivedSession 格式，仅保留 topics 摘要
        - 活跃目录：ConversationContext 格式，保留完整 hot_messages + summaries
        """
        archive_path = self._archive_dir() / f"{session_id}.json"
        active_path = self._active_dir() / f"{session_id}.json"

        # 1. 优先尝试冷层归档
        if archive_path.exists():
            try:
                with open(archive_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                from .context.types import TopicSnapshot
                ctx = ConversationContext(
                    session_id=session_id,
                    status="active",
                    hot_messages=data.get("hot_messages", []),  # 恢复完整热层消息
                    warm_summaries=[
                        TopicSnapshot(
                            topic_id=t.get("topic_id", ""),
                            topic_label=t.get("topic_label", ""),
                            start_turn=t.get("start_turn", 0),
                            end_turn=t.get("end_turn", 0),
                            summary=t.get("summary", ""),
                            key_entities=t.get("key_entities", []),
                            created_at=t.get("created_at", ""),
                            metadata=t.get("metadata", {}),
                        )
                        for t in data.get("topics", [])
                    ],
                    total_turns=data.get("total_turns", 0),
                    created_at=data.get("created_at", ""),
                    last_activity_at=data.get("archived_at", ""),
                )
                ctx.touch()
                self._persist_restored(ctx)
                self._sessions[session_id] = ctx
                logger.info(f"会话 {session_id} 已从冷层恢复 ({ctx.total_turns} 轮)")
                return ctx
            except Exception as e:
                logger.warning(f"从冷层恢复会话 {session_id} 失败: {e}")
                return None

        # 2. 冷层无，尝试活跃目录
        if active_path.exists():
            try:
                with open(active_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ctx = ConversationContext.from_dict(data)
                ctx.status = "active"
                ctx.touch()
                self._persist_restored(ctx)
                self._sessions[session_id] = ctx
                logger.info(f"会话 {session_id} 已从活跃目录恢复 ({ctx.total_turns} 轮)")
                return ctx
            except Exception as e:
                logger.warning(f"从活跃目录恢复会话 {session_id} 失败: {e}")
                return None

        logger.warning(f"会话 {session_id} 在冷层和活跃目录均不存在")
        return None

    def _persist_restored(self, ctx: ConversationContext) -> None:
        """恢复后写回存储"""
        if self._redis_store:
            self._redis_store.save(ctx, ttl=self._idle_timeout)
        else:
            self._save_active(ctx)

    # ------------------------------------------------------------------
    # 会话列表
    # ------------------------------------------------------------------

    def list_active(self) -> list[str]:
        """列出所有活跃会话 ID（Redis + 内存 + 活跃目录，自动清理过期）"""
        result = set()
        if self._redis_store:
            result.update(self._redis_store.list_active())
        result.update(self._sessions.keys())
        # 从活跃目录读取，过滤过期会话
        active_dir = self._active_dir()
        if active_dir.exists():
            for p in active_dir.glob("*.json"):
                sid = p.stem
                if sid in result:
                    continue
                # 读取文件检查是否过期
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    last_activity = data.get("last_activity_at", "")
                    if last_activity:
                        from datetime import datetime, timezone
                        last = datetime.fromisoformat(last_activity)
                        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                        if elapsed > self._idle_timeout:
                            # 过期会话自动归档
                            self._auto_archive_file(p, data)
                            continue
                    result.add(sid)
                except Exception:
                    result.add(sid)  # 读取失败保守保留
        return sorted(result)

    def _auto_archive_file(self, path: Path, data: dict) -> None:
        """自动归档过期活跃会话文件"""
        try:
            from datetime import datetime, timezone
            from .context.types import ArchivedSession, TopicSnapshot

            session_id = path.stem
            warm_summaries = []
            for ts_data in data.get("warm_summaries", []):
                warm_summaries.append(TopicSnapshot(
                    topic_id=ts_data.get("topic_id", ""),
                    topic_label=ts_data.get("topic_label", ""),
                    start_turn=ts_data.get("start_turn", 0),
                    end_turn=ts_data.get("end_turn", 0),
                    summary=ts_data.get("summary", ""),
                    key_entities=ts_data.get("key_entities", []),
                    created_at=ts_data.get("created_at", ""),
                    metadata=ts_data.get("metadata", {}),
                ))

            now = datetime.now(timezone.utc).isoformat()
            duration = 0
            created_at = data.get("created_at", "")
            if created_at:
                try:
                    created = datetime.fromisoformat(created_at)
                    duration = int((datetime.now(timezone.utc) - created).total_seconds())
                except (ValueError, TypeError):
                    pass

            archived = ArchivedSession(
                session_id=session_id,
                total_turns=data.get("total_turns", 0),
                topics=warm_summaries,
                hot_messages=data.get("hot_messages", []),
                created_at=created_at,
                archived_at=now,
                duration=duration,
            )

            archive_dir = self._archive_dir()
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{session_id}.json"
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(archived.to_dict(), f, ensure_ascii=False, indent=2)
            path.unlink()  # 删除活跃目录中的文件
            logger.info(f"会话 {session_id} 已自动归档（空闲超时）")
        except Exception as e:
            logger.warning(f"自动归档会话 {path.stem} 失败: {e}")

    def list_archived(self) -> list[str]:
        """列出所有已归档会话 ID（排除活跃会话和快照）"""
        active_ids = set(self.list_active())
        archive_dir = self._archive_dir()
        if not archive_dir.exists():
            return []
        return sorted(
            p.stem for p in archive_dir.glob("*.json")
            if not p.name.endswith(".snapshot.json") and p.stem not in active_ids
        )

    def get_session_status(self, session_id: str) -> str:
        """获取会话状态"""
        if session_id in self._sessions:
            return self._sessions[session_id].status
        if self._redis_store and self._redis_store.exists(session_id):
            return "active"
        if (self._archive_dir() / f"{session_id}.json").exists():
            return "archived"
        return "unknown"

    def get_metrics(self) -> dict:
        """获取全局会话统计指标（供 4.1 监控面板使用）

        Returns:
            dict with active_sessions, total_turns, total_sessions, avg_turns
        """
        active = self.list_active()
        total_turns = 0
        total_sessions = len(active)
        for sid in active:
            ctx = self._sessions.get(sid)
            if ctx is None and self._redis_store:
                ctx = self._redis_store.load(sid)
            if ctx:
                total_turns += ctx.total_turns
        return {
            "active_sessions": len(active),
            "total_turns": total_turns,
            "total_sessions": total_sessions,
            "avg_turns": total_turns / max(1, total_sessions),
        }