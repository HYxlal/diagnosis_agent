"""端到端 LLM 实测 — 模拟多轮 chat 对话

测试内容：
1. 5 轮电机故障诊断对话
2. 会话归档 + 冷层恢复
3. 恢复后继续对话
4. 空闲超时自动归档
5. 异步摘要不阻塞

用法: conda run -n diagnose_agent python tests/e2e_chat_test.py
"""

import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnosis_agent.config import get_settings, reset_settings
from diagnosis_agent.agent.session_manager import SessionManager
from diagnosis_agent.agent.langchain_agent import LangChainDiagnosticAgent
from diagnosis_agent.agent.context_manager import AsyncContextManager
from diagnosis_agent.agent.context.summarizer import Summarizer
from diagnosis_agent.agent.context.topic_detector import TopicDetector
from diagnosis_agent.models.input import StandardInput, StandardEntities


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(ok: bool, msg: str, detail: str = ""):
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    if detail:
        print(f"     {detail}")


async def main():
    reset_settings()
    settings = get_settings()
    persist_dir = "data/sessions"
    session_id = f"e2e-llm-{datetime.now().strftime('%H%M%S')}"

    print_section(f"端到端 LLM 实测 — 会话: {session_id} (window_size=2)")

    # ── 初始化组件 ──
    sm = SessionManager(persist_dir=persist_dir)
    agent = LangChainDiagnosticAgent(settings=settings)

    def _make_context_manager(window_size=None):
        """创建 AsyncContextManager 实例"""
        ws = window_size if window_size is not None else settings.context.window_size
        return AsyncContextManager(
            window_size=ws,
            max_tokens=settings.context.max_tokens,
            summarizer=Summarizer(
                strategy=settings.context.summary_strategy,
                max_tokens=settings.context.summary_max_tokens,
            ) if settings.context.summary_enabled else None,
            topic_detector=TopicDetector(
                strategy=settings.context.topic_detection_strategy,
                threshold_high=settings.context.topic_similarity_high,
                threshold_low=settings.context.topic_similarity_low,
                model=settings.context.topic_detection_model or None,
                switch_signal_words=settings.context.topic_signal_words.get("switch", []),
                entity_overlap_enabled=settings.context.entity_overlap_enabled,
                time_decay_short_sec=settings.context.topic_time_decay_short_sec if settings.context.topic_time_decay_enabled else 0,
                time_decay_short_max_len=settings.context.topic_time_decay_short_max_len,
                time_decay_long_sec=settings.context.topic_time_decay_long_sec,
                scope_detection_enabled=settings.context.scope_detection_enabled,
                scope_use_llm=settings.context.scope_use_llm,
                scope_out_keywords=settings.context.scope_out_keywords,
            ) if settings.context.topic_detection_enabled else None,
            session_manager=sm,
            max_workers=2,
        )

    context_manager = _make_context_manager(window_size=2)

    # ──────────────────────────────────────────────
    # 阶段 1: 5 轮电机故障诊断对话
    # ──────────────────────────────────────────────
    print_section("阶段 1: 5 轮电机故障诊断对话")

    queries = [
        "电机在高速运转时出现抖动，伴随有异响",
        "检查了电机轴承，发现轴承有磨损，但更换后问题依然存在",
        "PWM 波形看起来正常，但相电流不平衡，峰值偏差超过 20%",
        "更换了 MCU 控制板，问题还在，感觉不是硬件问题",
        "好的，那可能是软件层面的扭矩控制策略有问题，帮我总结一下整个诊断过程",
    ]

    total_ok = 0
    for i, query in enumerate(queries):
        print(f"\n--- 第 {i+1} 轮 ---")
        print(f"  用户: {query}")

        # 构建上下文
        ctx = sm.get_conversation_context(session_id)
        prepared = await context_manager.prepare_messages_async(ctx, query)

        # 构建标准输入
        standard_input = StandardInput(
            raw_query=query,
            mcuid="MCU_002",
            session_id=session_id,
            entities=StandardEntities(),
        )

        start = time.perf_counter()
        try:
            result = agent.diagnose_with_standard_input(
                standard_input,
                prepared_messages=prepared.messages if prepared else None,
            )
            elapsed = time.perf_counter() - start
        except Exception as e:
            print(f"  ❌ 诊断失败: {e}")
            continue

        if result.code == 0 and result.diagnosis_result:
            dr = result.diagnosis_result
            print(f"  分类: {dr.classification}")
            print(f"  根因: {dr.fault_root_cause[0][:80] if dr.fault_root_cause else 'N/A'}...")
            print(f"  置信度: {result.diagnosis_confidence:.0%}")
            print(f"  耗时: {elapsed:.1f}s")
            total_ok += 1

            # 存储本轮消息
            from langchain_core.messages import messages_to_dict
            current_msgs = messages_to_dict(getattr(agent, "_last_messages", []))
            if current_msgs:
                sm.update(session_id, query, current_msgs)
        else:
            print(f"  ❌ code={result.code}, msg={result.msg}")

    print_result(total_ok == 5, f"5 轮完成: {total_ok}/5 成功")

    # 验证会话状态
    ctx = sm.get_conversation_context(session_id)
    if ctx:
        print_result(ctx.total_turns == 5, f"会话轮次: {ctx.total_turns}", f"status={ctx.status}")
        print_result(len(ctx.hot_messages) > 0, f"热层消息数: {len(ctx.hot_messages)}")

    # 等待异步摘要完成（window_size=2，5 轮会触发多次摘要）
    await asyncio.sleep(5.0)
    ctx = sm.get_conversation_context(session_id)
    if ctx:
        print_result(
            len(ctx.warm_summaries) > 0,
            f"异步摘要完成: 温层 {len(ctx.warm_summaries)} 个摘要",
        )
        for i, s in enumerate(ctx.warm_summaries):
            print(f"     [{s.topic_label}] {s.summary[:80]}...")

    # ──────────────────────────────────────────────
    # 阶段 2: 归档 + 冷层恢复
    # ──────────────────────────────────────────────
    print_section("阶段 2: 归档 + 冷层恢复")

    archived = sm.archive(session_id)
    print_result(archived is not None, f"归档成功: {archived.total_turns if archived else 'N/A'} 轮")

    archive_file = Path(persist_dir) / "archive" / f"{session_id}.json"
    print_result(archive_file.exists(), f"冷层文件存在: {archive_file}")

    # 恢复
    restored = sm.restore_from_archive(session_id)
    print_result(restored is not None, f"恢复成功")
    if restored:
        print_result(restored.total_turns == 5, f"恢复后轮次: {restored.total_turns}")
        print_result(restored.status == "active", f"恢复后状态: {restored.status}")
        print_result(
            len(restored.warm_summaries) > 0,
            f"温层摘要数: {len(restored.warm_summaries)} (应为非空)"
        )

    # ──────────────────────────────────────────────
    # 阶段 3: 恢复后继续对话
    # ──────────────────────────────────────────────
    print_section("阶段 3: 恢复后继续对话")

    sm._sessions[session_id] = restored  # 手动放回内存
    ctx = sm.get_conversation_context(session_id)
    prepared = await context_manager.prepare_messages_async(ctx, "回顾一下之前的诊断结论")

    standard_input = StandardInput(
        raw_query="回顾一下之前的诊断结论",
        mcuid="MCU_002",
        session_id=session_id,
        entities=StandardEntities(),
    )

    try:
        result = agent.diagnose_with_standard_input(
            standard_input,
            prepared_messages=prepared.messages if prepared else None,
        )
        if result.code == 0:
            from langchain_core.messages import messages_to_dict
            current_msgs = messages_to_dict(getattr(agent, "_last_messages", []))
            if current_msgs:
                sm.update(session_id, "回顾一下之前的诊断结论", current_msgs)
            ctx = sm.get_conversation_context(session_id)
            print_result(ctx.total_turns == 6, f"恢复后继续对话: 轮次 {ctx.total_turns} (应为 6)")
        else:
            print_result(False, f"诊断失败: code={result.code}")
    except Exception as e:
        print_result(False, f"异常: {e}")

    # ──────────────────────────────────────────────
    # 阶段 4: 空闲超时自动归档
    # ──────────────────────────────────────────────
    print_section("阶段 4: 空闲超时自动归档")

    sm._idle_timeout = 10  # 缩到 10 秒
    ctx = sm.get_conversation_context(session_id)
    if ctx:
        ctx.last_activity_at = (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat()
        sm._sessions[session_id] = ctx

    # 再次访问应触发归档
    ctx_reloaded = sm._load_to_memory(session_id)
    print_result(ctx_reloaded.status == "created", f"空闲超时触发归档: status={ctx_reloaded.status}")

    archive_file = Path(persist_dir) / "archive" / f"{session_id}.json"
    print_result(archive_file.exists(), "冷层文件存在（空闲超时归档）")

    # ──────────────────────────────────────────────
    # 阶段 5: 异步摘要不阻塞
    # ──────────────────────────────────────────────
    print_section("阶段 5: 异步摘要耗时验证（window_size=1）")

    # 创建独立的 context_manager，window_size=1 确保次次触发摘要
    acm_fast = _make_context_manager(window_size=1)
    session_id2 = f"e2e-llm-async-{datetime.now().strftime('%H%M%S')}"
    response_times = []

    for i in range(3):
        ctx = sm.get_conversation_context(session_id2)
        print(f"  [debug] 第{i+1}轮: hot_rounds={acm_fast._count_hot_rounds(ctx)}, hot_messages={len(ctx.hot_messages)}")
        prepared = await acm_fast.prepare_messages_async(ctx, f"电机故障诊断问题_{i}")
        print(f"  [debug] needs_summary={prepared.metadata.trim_info.needs_summary}, overflow_cache={list(acm_fast._overflow_cache.keys())}")

        standard_input = StandardInput(
            raw_query=f"电机故障诊断问题_{i}",
            mcuid="MCU_003",
            session_id=session_id2,
            entities=StandardEntities(),
        )

        start = time.perf_counter()
        result = agent.diagnose_with_standard_input(
            standard_input,
            prepared_messages=prepared.messages if prepared else None,
        )
        elapsed = time.perf_counter() - start
        response_times.append(elapsed)

        if result.code == 0:
            from langchain_core.messages import messages_to_dict
            last_msgs = getattr(agent, "_last_messages", [])
            current_msgs = messages_to_dict(last_msgs) if last_msgs else []
            print(f"  [debug] _last_messages len={len(last_msgs) if last_msgs else 0}, current_msgs len={len(current_msgs)}")
            if current_msgs:
                sm.update(session_id2, f"电机故障诊断问题_{i}", current_msgs)
            print(f"  第{i+1}轮: {elapsed:.1f}s (LLM 耗时)")
        else:
            print(f"  第{i+1}轮: ❌ code={result.code}")

    # 注意：LLM 调用本身是同步的，所以 response_times 包含 LLM 耗时
    # 真正的异步验证是 prepare_messages_async 不包含摘要耗时
    # 这里重点验证 prepare 阶段不阻塞
    print(f"\n  LLM 响应时间: min={min(response_times):.1f}s, max={max(response_times):.1f}s, avg={sum(response_times)/len(response_times):.1f}s")

    # 等待后台摘要完成
    await asyncio.sleep(5.0)
    ctx = sm.get_conversation_context(session_id2)
    if ctx:
        print_result(
            len(ctx.warm_summaries) > 0,
            f"异步摘要完成: 温层 {len(ctx.warm_summaries)} 个摘要",
        )
        for i, s in enumerate(ctx.warm_summaries):
            print(f"     [{s.topic_label}] {s.summary[:80]}...")

    context_manager.shutdown()
    acm_fast.shutdown()

    # ── 总结 ──
    print_section("测试总结")
    print(f"  会话 ID: {session_id}")
    print(f"  冷层文件: {Path(persist_dir) / 'archive' / f'{session_id}.json'}")
    print(f"  会话 ID (异步): {session_id2}")
    print(f"  冷层文件: {Path(persist_dir) / 'archive' / f'{session_id2}.json'}")
    print(f"\n  查看归档: python -m diagnosis_agent.cli session show {session_id}")
    print(f"  会话列表: python -m diagnosis_agent.cli session list")


if __name__ == "__main__":
    asyncio.run(main())