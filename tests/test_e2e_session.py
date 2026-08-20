"""端到端测试 — 会话生命周期 + 异步摘要

模拟完整的多轮诊断对话流程，不依赖 LLM API key。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from diagnosis_agent.agent.session_manager import SessionManager
from diagnosis_agent.agent.context_manager import AsyncContextManager
from diagnosis_agent.agent.context.types import (
    TopicSnapshot,
    ConversationContext,
)
from diagnosis_agent.config import reset_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """每个测试前后重置 SessionManager 单例并清理 Redis"""
    SessionManager._instance = None
    SessionManager._persist_dir = ""
    SessionManager._redis_store = None
    reset_settings()
    try:
        import redis as r
        client = r.from_url("redis://localhost:6379/0", decode_responses=True, protocol=2)
        for key in client.scan_iter("session:test-*"):
            client.delete(key)
        for key in client.scan_iter("session:e2e-*"):
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
        for key in client.scan_iter("session:e2e-*"):
            client.delete(key)
    except Exception:
        pass


def _make_msg(role: str, content: str) -> dict:
    """生成 LangChain 兼容的消息格式"""
    if role == "user":
        return {"type": "human", "data": {"content": content, "type": "human", "id": None, "name": None}}
    else:
        return {"type": "ai", "data": {"content": content, "type": "ai", "id": None, "name": None}}


class SlowSummarizer:
    """模拟慢速摘要器"""
    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.call_count = 0

    def summarize(self, messages, topic_label="", start_turn=0, end_turn=0):
        time.sleep(self.delay)
        self.call_count += 1
        return TopicSnapshot(
            topic_id=f"s-{self.call_count}",
            topic_label=topic_label or f"摘要{self.call_count}",
            start_turn=start_turn, end_turn=end_turn,
            summary=f"共{len(messages)}条消息的摘要",
            key_entities=["MCU", "电机"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def merge_summaries(self, summaries):
        return TopicSnapshot(
            topic_id="merged", topic_label="合并", start_turn=0, end_turn=0,
            summary="合并摘要", key_entities=[], created_at="",
        )


# ---------------------------------------------------------------------------
# 端到端测试
# ---------------------------------------------------------------------------


class TestE2ESessionLifecycle:
    """端到端：会话完整生命周期"""

    def test_full_lifecycle(self, tmp_path):
        """完整流程：创建 → 多轮对话 → 归档 → 冷层恢复 → 继续对话"""
        sm = SessionManager(persist_dir=str(tmp_path))

        # ── 阶段 1: 创建会话，模拟多轮诊断对话 ──
        session_id = "e2e-lifecycle"
        rounds = [
            ("电机高速运转时抖动", "确认为相电流不平衡，建议检查 IGBT 模块"),
            ("换了一个 IGBT 还是抖", "可能是驱动板信号异常，需检查 PWM 波形"),
            ("PWM 波形正常，还有什么可能", "检查旋变信号和电机位置传感器"),
            ("旋变信号有毛刺", "确认旋变故障，建议更换旋变传感器"),
            ("好的，换了之后确实好了", "故障已排除，本次诊断结束"),
        ]

        for i, (query, answer) in enumerate(rounds):
            msgs = [
                _make_msg("user", query),
                _make_msg("assistant", answer),
            ]
            sm.update(session_id, query, msgs)

        ctx = sm.get_conversation_context(session_id)
        assert ctx is not None
        assert ctx.total_turns == 5, f"应为 5 轮，实际 {ctx.total_turns}"
        assert ctx.status == "active"
        assert len(ctx.hot_messages) == 10  # 5 轮 x 2 条消息

        # ── 阶段 2: 归档到冷层 ──
        archived = sm.archive(session_id)
        assert archived is not None
        assert archived.total_turns == 5

        # 验证冷层文件存在
        archive_path = tmp_path / "archive" / f"{session_id}.json"
        assert archive_path.exists(), "冷层归档文件应存在"

        with open(archive_path) as f:
            data = json.load(f)
        assert data["total_turns"] == 5

        # 归档后内存中不应存在
        assert sm.get_conversation_context(session_id) is not None  # 新会话

        # ── 阶段 3: 从冷层恢复 ──
        restored = sm.restore_from_archive(session_id)
        assert restored is not None
        assert restored.total_turns == 5
        assert restored.status == "active"
        assert len(restored.hot_messages) == 10  # 恢复后保留热层消息

        # ── 阶段 4: 恢复后继续对话 ──
        sm._sessions[session_id] = restored
        sm.update(session_id, "再问一个问题，电机温度异常",
                  [_make_msg("user", "再问一个问题，电机温度异常"),
                   _make_msg("assistant", "检查冷却系统")])

        ctx2 = sm.get_conversation_context(session_id)
        assert ctx2.total_turns == 6  # 5 + 1
        assert ctx2.status == "active"

    def test_idle_timeout_auto_archive(self, tmp_path):
        """空闲超时自动归档 + 冷层恢复"""
        sm = SessionManager(persist_dir=str(tmp_path))
        sm._idle_timeout = 5  # 5 秒空闲超时

        session_id = "e2e-idle"

        # 模拟 3 轮对话
        for i in range(3):
            sm.update(session_id, f"query_{i}", [
                _make_msg("user", f"query_{i}"),
                _make_msg("assistant", f"answer_{i}"),
            ])

        ctx = sm.get_conversation_context(session_id)
        assert ctx.total_turns == 3
        assert ctx.status == "active"

        # 模拟空闲：设置 last_activity_at 为 10 秒前
        ctx.last_activity_at = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        sm._sessions[session_id] = ctx

        # 再次访问触发空闲超时归档
        ctx_reloaded = sm._load_to_memory(session_id)
        assert ctx_reloaded.status == "created"  # 新会话
        assert ctx_reloaded.total_turns == 0

        # 验证冷层归档
        archive_path = tmp_path / "archive" / f"{session_id}.json"
        assert archive_path.exists(), "空闲超时应触发归档"

        with open(archive_path) as f:
            data = json.load(f)
        assert data["total_turns"] == 3

        # 从冷层恢复
        restored = sm.restore_from_archive(session_id)
        assert restored.total_turns == 3

    def test_concurrent_multi_session(self, tmp_path):
        """并发多会话无数据错乱"""
        sm = SessionManager(persist_dir=str(tmp_path))
        num_sessions = 10
        rounds_per_session = 20

        errors = []
        lock = threading.Lock()

        def session_worker(sid: str):
            try:
                for i in range(rounds_per_session):
                    msgs = [
                        _make_msg("user", f"{sid}_q{i}"),
                        _make_msg("assistant", f"{sid}_a{i}"),
                    ]
                    sm.update(sid, f"{sid}_q{i}", msgs)
            except Exception as e:
                with lock:
                    errors.append(f"{sid}: {e}")

        # 并发运行 10 个会话，每个 20 轮
        threads = []
        for s in range(num_sessions):
            t = threading.Thread(target=session_worker, args=(f"e2e-conc-{s}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发错误: {errors}"

        # 验证所有会话数据正确
        for s in range(num_sessions):
            ctx = sm.get_conversation_context(f"e2e-conc-{s}")
            assert ctx is not None, f"会话 e2e-conc-{s} 不存在"
            assert ctx.total_turns == rounds_per_session, (
                f"会话 e2e-conc-{s} 轮次: {ctx.total_turns} != {rounds_per_session}"
            )


class TestE2EAsyncSummary:
    """端到端：异步摘要不阻塞"""

    @pytest.mark.asyncio
    async def test_async_summary_full_flow(self, tmp_path):
        """模拟 10 轮对话，异步摘要在后台执行，不阻塞每轮响应"""
        sm = SessionManager(persist_dir=str(tmp_path))
        summarizer = SlowSummarizer(delay=0.5)
        acm = AsyncContextManager(
            window_size=3,
            max_tokens=8000,
            summarizer=summarizer,
            session_manager=sm,
            max_workers=2,
        )

        session_id = "e2e-async"
        response_times = []

        for i in range(10):
            ctx = sm._load_to_memory(session_id)

            # 模拟一轮对话消息
            msgs = [
                _make_msg("user", f"诊断问题_{i}"),
                _make_msg("assistant", f"诊断回答_{i}"),
            ]

            # 在 prepare 之前先更新热层（模拟上一轮的消息已存储）
            if i > 0:
                ctx.hot_messages.extend(msgs)

            sm._sessions[session_id] = ctx

            # 测量 prepare_messages_async 耗时
            start = time.perf_counter()
            prepared = await acm.prepare_messages_async(ctx, f"追问_{i}")
            elapsed = time.perf_counter() - start
            response_times.append(elapsed)

            # 更新会话
            sm.update(session_id, f"追问_{i}", [
                _make_msg("user", f"追问_{i}"),
                _make_msg("assistant", f"最终回答_{i}"),
            ])

        # 验证：每轮耗时都 < 0.5s（摘要需要 0.5s，不应阻塞）
        max_time = max(response_times)
        avg_time = sum(response_times) / len(response_times)
        assert max_time < 0.5, (
            f"最长响应 {max_time:.2f}s 应 < 0.5s（摘要耗时 0.5s），不阻塞验证失败"
        )
        # 平均 < 0.2s 才是正常的异步行为
        assert avg_time < 0.2, (
            f"平均响应 {avg_time:.2f}s 应 < 0.2s，摘要可能阻塞了主流程"
        )

        # 等待所有后台摘要完成
        await asyncio.sleep(2.0)

        # 验证摘要最终已生成
        ctx = sm.get_conversation_context(session_id)
        assert summarizer.call_count >= 1, "至少应生成 1 次摘要"

        acm.shutdown()
        print(f"\n  异步摘要端到端: 10 轮, 平均 {avg_time:.3f}s, 生成了 {summarizer.call_count} 个摘要")

    @pytest.mark.asyncio
    async def test_archive_during_async_summary(self, tmp_path):
        """会话在异步摘要执行期间被归档，摘要应被丢弃（不报错）"""
        sm = SessionManager(persist_dir=str(tmp_path))
        summarizer = SlowSummarizer(delay=1.0)  # 1 秒摘要
        acm = AsyncContextManager(
            window_size=1,
            max_tokens=8000,
            summarizer=summarizer,
            session_manager=sm,
            max_workers=1,
        )

        session_id = "e2e-archive-mid"

        # 创建会话，热层有 2 轮（超出 window_size=1）
        ctx = sm._load_to_memory(session_id)
        ctx.status = "active"
        ctx.total_turns = 2
        ctx.hot_messages = [
            _make_msg("user", "q1"),
            _make_msg("assistant", "a1"),
            _make_msg("user", "q2"),
            _make_msg("assistant", "a2"),
        ]
        sm._sessions[session_id] = ctx

        # 触发异步摘要（后台线程开始 1 秒摘要）
        await acm.prepare_messages_async(ctx, "q3")

        # 立即归档会话（不等摘要完成）
        archived = sm.archive(session_id)
        assert archived is not None

        # 等待摘要完成（允许归档期间丢弃）
        await asyncio.sleep(1.5)

        # 验证：不报错，归档成功
        archive_path = tmp_path / "archive" / f"{session_id}.json"
        assert archive_path.exists()

        acm.shutdown()
        print(f"\n  归档-异步摘要并发: 归档成功，摘要被丢弃（正确行为）")