"""会话生命周期测试 — AC-10 到 AC-13

AC-10: 会话关闭后可从冷层恢复，调用 restore() 后 get_summary() 结果非空
AC-11: 空闲超时应自动归档，设置 10 秒超时并等待后验证
AC-12: 多线程并发安全，并发 100 个请求无数据错乱
AC-13: 摘要处理异步且不阻塞响应，通过测量 prepare_messages 耗时验证
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from diagnosis_agent.agent.session_manager import SessionManager, SessionRedisStore
from diagnosis_agent.agent.context_manager import SimpleContextManager, AsyncContextManager
from diagnosis_agent.agent.context.types import (
    ConversationContext,
    TopicSnapshot,
    TrimInfo,
    PrepareResult,
)
from diagnosis_agent.config import reset_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session_manager():
    """每个测试前后重置 SessionManager 单例、Redis 配置并清理 Redis 数据"""
    SessionManager._instance = None
    SessionManager._persist_dir = ""
    SessionManager._redis_store = None
    reset_settings()
    # 清理 Redis 测试数据
    try:
        import redis as r
        client = r.from_url("redis://localhost:6379/0", decode_responses=True, protocol=2)
        for key in client.scan_iter("session:test-*"):
            client.delete(key)
    except Exception:
        pass
    yield
    SessionManager._instance = None
    SessionManager._persist_dir = ""
    SessionManager._redis_store = None
    try:
        import redis as r
        client = r.from_url("redis://localhost:6379/0", decode_responses=True, protocol=2)
        for key in client.scan_iter("session:test-*"):
            client.delete(key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AC-10: 冷层恢复
# ---------------------------------------------------------------------------


class TestColdLayerRestore:
    """AC-10: 会话关闭后可从冷层恢复，验证 restore() 后 get_summary() 结果非空"""

    def test_restore_after_archive_returns_warm_summaries(self, tmp_path):
        """归档后恢复，warm_summaries 非空"""
        sm = SessionManager(persist_dir=str(tmp_path))

        # 1. 创建会话并添加温层摘要
        ctx = sm._load_to_memory("test-ac10")
        ctx.total_turns = 5
        ctx.status = "active"
        ctx.hot_messages = [{"role": "user", "content": "电机抖动"}]

        snapshot = TopicSnapshot(
            topic_id="topic-1",
            topic_label="过温故障-P1A3E98",
            start_turn=1,
            end_turn=3,
            summary="确认电机过温，建议检查散热系统",
            key_entities=["P1A3E98", "IGBT"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        ctx.warm_summaries.append(snapshot)
        sm._sessions["test-ac10"] = ctx

        # 2. 归档
        result = sm.archive("test-ac10")
        assert result is not None
        assert result.total_turns == 5

        # 3. 从冷层恢复
        restored = sm.restore_from_archive("test-ac10")
        assert restored is not None, "恢复失败"
        assert restored.status == "active"
        assert len(restored.warm_summaries) == 1, "温层摘要应为 1 个"
        assert restored.warm_summaries[0].topic_label == "过温故障-P1A3E98"
        assert "散热系统" in restored.warm_summaries[0].summary
        assert restored.total_turns == 5

    def test_restore_restores_hot_messages(self, tmp_path):
        """恢复后热层消息保留（不再清空）"""
        sm = SessionManager(persist_dir=str(tmp_path))

        ctx = sm._load_to_memory("test-ac10-hot")
        ctx.total_turns = 3
        ctx.status = "active"
        ctx.hot_messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        ctx.warm_summaries = [
            TopicSnapshot(
                topic_id="t1", topic_label="通信故障", start_turn=1, end_turn=2,
                summary="U1624 通信异常", key_entities=["U1624"],
            )
        ]
        sm._sessions["test-ac10-hot"] = ctx

        sm.archive("test-ac10-hot")
        restored = sm.restore_from_archive("test-ac10-hot")

        assert restored is not None
        assert len(restored.hot_messages) == 3, "恢复后热层应保留 3 条消息"
        assert len(restored.warm_summaries) == 1

    def test_restore_nonexistent_session(self, tmp_path):
        """恢复不存在的会话返回 None"""
        sm = SessionManager(persist_dir=str(tmp_path))
        result = sm.restore_from_archive("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# AC-11: 空闲超时自动归档
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    """AC-11: 空闲超时应自动归档，设置 10 秒超时并等待后验证"""

    def test_idle_timeout_triggers_archive(self, tmp_path):
        """设置 10 秒空闲超时，修改 last_activity_at 模拟超时，验证自动归档"""
        sm = SessionManager(persist_dir=str(tmp_path))
        sm._idle_timeout = 10  # 10 秒空闲超时

        # 1. 创建活跃会话，设置 last_activity_at 为 15 秒前
        ctx = sm._load_to_memory("test-ac11")
        ctx.status = "active"
        ctx.total_turns = 3
        ctx.hot_messages = [{"role": "user", "content": "电机抖动"}]
        ctx.last_activity_at = (
            datetime.now(timezone.utc) - timedelta(seconds=15)
        ).isoformat()
        sm._sessions["test-ac11"] = ctx

        # 2. 再次访问会话，应触发空闲超时归档
        reloaded = sm._load_to_memory("test-ac11")

        # 3. 验证：应返回新建的会话
        assert reloaded.status == "created"
        assert reloaded.total_turns == 0

        # 4. 验证：旧会话已归档到冷层
        archive_dir = tmp_path / "archive"
        archive_path = archive_dir / "test-ac11.json"
        assert archive_path.exists(), "冷层归档文件应存在"

        with open(archive_path) as f:
            archived_data = json.load(f)
        assert archived_data["session_id"] == "test-ac11"
        assert archived_data["total_turns"] == 3

    def test_active_session_not_archived(self, tmp_path):
        """未超时的活跃会话不应被归档"""
        sm = SessionManager(persist_dir=str(tmp_path))
        sm._idle_timeout = 10

        ctx = sm._load_to_memory("test-ac11-active")
        ctx.status = "active"
        ctx.total_turns = 3
        ctx.hot_messages = [{"role": "user", "content": "电机抖动"}]
        ctx.last_activity_at = (
            datetime.now(timezone.utc) - timedelta(seconds=5)
        ).isoformat()  # 仅 5 秒，未超时
        sm._sessions["test-ac11-active"] = ctx

        reloaded = sm._load_to_memory("test-ac11-active")
        assert reloaded.status == "active", "未超时应保持 active"
        assert reloaded.total_turns == 3

    def test_archive_file_contains_correct_data(self, tmp_path):
        """归档文件包含正确的会话数据"""
        sm = SessionManager(persist_dir=str(tmp_path))
        sm._idle_timeout = 10

        ctx = sm._load_to_memory("test-ac11-data")
        ctx.status = "active"
        ctx.total_turns = 7
        ctx.hot_messages = [{"role": "user", "content": "全油门提速抖动"}]
        ctx.last_activity_at = (
            datetime.now(timezone.utc) - timedelta(seconds=20)
        ).isoformat()
        sm._sessions["test-ac11-data"] = ctx

        sm._load_to_memory("test-ac11-data")  # 触发归档

        archive_dir = tmp_path / "archive"
        archive_path = archive_dir / "test-ac11-data.json"
        with open(archive_path) as f:
            data = json.load(f)

        assert data["session_id"] == "test-ac11-data"
        assert data["total_turns"] == 7


# ---------------------------------------------------------------------------
# AC-12: 多线程并发安全
# ---------------------------------------------------------------------------


class TestConcurrency:
    """AC-12: 多线程并发安全，并发 100 个请求无数据错乱"""

    def test_concurrent_updates_no_data_corruption(self, tmp_path):
        """100 个并发请求同时更新同一会话，total_turns 应等于 100"""
        sm = SessionManager(persist_dir=str(tmp_path))

        # 初始化会话
        ctx = sm._load_to_memory("test-ac12")
        ctx.status = "active"
        ctx.total_turns = 0
        sm._sessions["test-ac12"] = ctx

        errors = []
        lock = threading.Lock()

        def update_session(i: int):
            try:
                # 每个线程模拟一轮对话
                msgs = [
                    {"role": "user", "content": f"query_{i}"},
                    {"role": "assistant", "content": f"answer_{i}"},
                ]
                sm.update("test-ac12", f"query_{i}", msgs)
            except Exception as e:
                with lock:
                    errors.append(f"线程 {i}: {e}")

        # 并发 100 个线程
        threads = []
        for i in range(100):
            t = threading.Thread(target=update_session, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证
        assert len(errors) == 0, f"并发错误: {errors}"

        final_ctx = sm.get_conversation_context("test-ac12")
        assert final_ctx is not None
        assert final_ctx.total_turns == 100, (
            f"轮次应为 100，实际为 {final_ctx.total_turns}"
        )
        assert final_ctx.status == "active"

    def test_concurrent_read_write_safety(self, tmp_path):
        """并发读写不产生数据错乱"""
        sm = SessionManager(persist_dir=str(tmp_path))

        ctx = sm._load_to_memory("test-ac12-rw")
        ctx.status = "active"
        ctx.total_turns = 0
        ctx.hot_messages = [{"role": "user", "content": "initial"}]
        sm._sessions["test-ac12-rw"] = ctx

        errors = []
        lock = threading.Lock()

        def reader(i: int):
            try:
                ctx = sm.get_conversation_context("test-ac12-rw")
                if ctx:
                    _ = ctx.total_turns  # 只读访问
            except Exception as e:
                with lock:
                    errors.append(f"读线程 {i}: {e}")

        def writer(i: int):
            try:
                sm.update("test-ac12-rw", f"w_{i}", [
                    {"role": "user", "content": f"w_{i}"},
                ])
            except Exception as e:
                with lock:
                    errors.append(f"写线程 {i}: {e}")

        threads = []
        for i in range(50):
            threads.append(threading.Thread(target=reader, args=(i,)))
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发错误: {errors}"

        # 最终应有 50 次写入
        final_ctx = sm.get_conversation_context("test-ac12-rw")
        assert final_ctx.total_turns == 50


# ---------------------------------------------------------------------------
# AC-13: 异步摘要不阻塞响应
# ---------------------------------------------------------------------------


class SlowSummarizer:
    """模拟慢速摘要器（用于测试异步不阻塞）"""

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.call_count = 0

    def summarize(self, messages, topic_label="", start_turn=0, end_turn=0):
        time.sleep(self.delay)  # 模拟 LLM 调用延迟
        self.call_count += 1
        return TopicSnapshot(
            topic_id=f"summary-{self.call_count}",
            topic_label=topic_label or f"异步摘要-{self.call_count}",
            start_turn=start_turn,
            end_turn=end_turn,
            summary=f"摘要内容: 共 {len(messages)} 条消息",
            key_entities=["test"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def merge_summaries(self, summaries):
        return TopicSnapshot(
            topic_id="merged",
            topic_label="合并摘要",
            start_turn=0, end_turn=0, summary="合并后摘要",
            key_entities=[], created_at="",
        )


class TestAsyncSummary:
    """AC-13: 摘要处理异步且不阻塞响应，通过测量 prepare_messages 耗时验证"""

    @staticmethod
    def _make_msg(role: str, content: str) -> dict:
        """生成 LangChain 兼容的消息格式（messages_to_dict 格式）"""
        if role == "user":
            return {
                "type": "human",
                "data": {"content": content, "type": "human", "id": None, "name": None},
            }
        else:
            return {
                "type": "ai",
                "data": {"content": content, "type": "ai", "id": None, "name": None},
            }

    @pytest.mark.asyncio
    async def test_prepare_messages_async_does_not_block(self, tmp_path):
        """异步 prepare_messages_async 耗时 < 摘要耗时（证明不阻塞）"""
        sm = SessionManager(persist_dir=str(tmp_path))
        slow_summarizer = SlowSummarizer(delay=2.0)

        acm = AsyncContextManager(
            window_size=1,
            max_tokens=8000,
            summarizer=slow_summarizer,
            session_manager=sm,
            max_workers=1,
        )

        # 创建会话，热层有 2 轮消息（超出 window_size=1，触发摘要）
        ctx = sm._load_to_memory("test-ac13")
        ctx.status = "active"
        ctx.total_turns = 2
        ctx.hot_messages = [
            self._make_msg("user", "第一轮问题"),
            self._make_msg("assistant", "第一轮回答"),
            self._make_msg("user", "第二轮问题"),
            self._make_msg("assistant", "第二轮回答"),
        ]
        sm._sessions["test-ac13"] = ctx

        # 测量 prepare_messages_async 耗时
        start = time.perf_counter()
        result = await acm.prepare_messages_async(ctx, "第三轮问题")
        elapsed = time.perf_counter() - start

        # 验证：不阻塞，耗时 < 摘要延迟（2.0 秒）
        assert elapsed < 1.0, (
            f"prepare_messages_async 耗时 {elapsed:.2f}s，"
            f"应 < 1.0s（摘要需要 {slow_summarizer.delay}s）"
        )
        assert result is not None
        assert len(result.messages) > 0

        # 等待后台摘要完成
        await asyncio.sleep(slow_summarizer.delay + 0.5)

        # 验证摘要最终完成
        assert slow_summarizer.call_count == 1, "摘要应被调用一次"

        acm.shutdown()

    @pytest.mark.asyncio
    async def test_summary_eventually_updates_warm_layer(self, tmp_path):
        """摘要最终更新温层（下次请求生效）"""
        sm = SessionManager(persist_dir=str(tmp_path))
        slow_summarizer = SlowSummarizer(delay=0.5)

        acm = AsyncContextManager(
            window_size=1,
            max_tokens=8000,
            summarizer=slow_summarizer,
            session_manager=sm,
            max_workers=1,
        )

        ctx = sm._load_to_memory("test-ac13-warm")
        ctx.status = "active"
        ctx.total_turns = 2
        ctx.hot_messages = [
            self._make_msg("user", "q1"),
            self._make_msg("assistant", "a1"),
            self._make_msg("user", "q2"),
            self._make_msg("assistant", "a2"),
        ]
        sm._sessions["test-ac13-warm"] = ctx

        # 执行异步准备
        await acm.prepare_messages_async(ctx, "q3")

        # 等待后台摘要完成
        await asyncio.sleep(1.0)

        # 验证温层已更新
        updated_ctx = sm.get_conversation_context("test-ac13-warm")
        assert updated_ctx is not None
        assert len(updated_ctx.warm_summaries) >= 1, (
            f"温层应有摘要，实际 {len(updated_ctx.warm_summaries)} 个"
        )

        acm.shutdown()

    @pytest.mark.asyncio
    async def test_no_summary_when_within_window(self, tmp_path):
        """热层未超出窗口时不触发摘要"""
        sm = SessionManager(persist_dir=str(tmp_path))
        slow_summarizer = SlowSummarizer(delay=0.5)

        acm = AsyncContextManager(
            window_size=5,  # 大窗口
            max_tokens=8000,
            summarizer=slow_summarizer,
            session_manager=sm,
            max_workers=1,
        )

        ctx = sm._load_to_memory("test-ac13-nowin")
        ctx.status = "active"
        ctx.total_turns = 1
        ctx.hot_messages = [
            self._make_msg("user", "q1"),
            self._make_msg("assistant", "a1"),
        ]
        sm._sessions["test-ac13-nowin"] = ctx

        start = time.perf_counter()
        result = await acm.prepare_messages_async(ctx, "q2")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, "窗口内不应触发摘要，耗时应在毫秒级"
        assert result.metadata.trim_info.needs_summary is False
        assert slow_summarizer.call_count == 0

        acm.shutdown()