"""监控指标 — Prometheus 端点 + 终端面板

提供两种观测方式：
1. HTTP Prometheus 端点（供外部采集，如 Grafana）
2. 终端内嵌指标面板（chat 模式每轮可见，使用 rich Panel）

用法：
    from .context.metrics import metrics

    # 更新指标
    metrics.window_utilization.set(0.62)
    metrics.hot_message_count.set(12)
    metrics.degradation_trim.inc()

    # 启动 HTTP 端点（后台线程）
    metrics.start_http_server(port=9090)

    # 终端面板
    console.print(metrics.render_panel())
"""

from __future__ import annotations

import threading
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from prometheus_client import start_http_server as prom_start_http_server


class ContextMetrics:
    """上下文运行指标（单例）

    所有指标通过 prometheus_client 的 Counter/Gauge/Histogram 管理，
    自动支持 Prometheus 格式导出。
    """

    _instance: Optional["ContextMetrics"] = None

    def __new__(cls) -> "ContextMetrics":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._registry = CollectorRegistry(auto_describe=True)

        # ── 窗口指标 ──
        self.window_utilization = Gauge(
            "context_window_utilization",
            "Token 窗口利用率 (0-1)",
            registry=self._registry,
        )
        self.hot_message_count = Gauge(
            "context_hot_message_count",
            "热层消息数",
            registry=self._registry,
        )
        self.warm_summary_count = Gauge(
            "context_warm_summary_count",
            "温层摘要数",
            registry=self._registry,
        )
        self.current_window_size = Gauge(
            "context_current_window_size",
            "当前窗口大小（轮次）",
            registry=self._registry,
        )

        # ── 降级指标 ──
        self.degradation_trim = Counter(
            "context_degradation_total",
            "降级次数",
            ["step"],
            registry=self._registry,
        )
        # 确保三种标签值都初始化
        for step in ("trim", "summarize", "emergency"):
            self.degradation_trim.labels(step=step)

        # ── 话题指标 ──
        self.topic_switch_count = Counter(
            "context_topic_switch_total",
            "话题切换次数",
            registry=self._registry,
        )
        self.fast_screen_hit_rate = Gauge(
            "context_fast_screen_hit_rate",
            "快筛命中率 (L0 信号词)",
            registry=self._registry,
        )

        # ── 摘要指标 ──
        self.summary_success = Counter(
            "context_summary_total",
            "摘要执行次数",
            ["result"],
            registry=self._registry,
        )
        for result in ("success", "failure"):
            self.summary_success.labels(result=result)

        self.summary_latency_ms = Histogram(
            "context_summary_latency_ms",
            "摘要延迟 (ms)",
            buckets=[100, 250, 500, 1000, 2000, 5000],
            registry=self._registry,
        )
        self.compression_ratio = Gauge(
            "context_compression_ratio",
            "摘要压缩比 (摘要长度/原始长度)",
            registry=self._registry,
        )

        # ── 会话指标 ──
        self.active_sessions = Gauge(
            "context_active_sessions",
            "活跃会话数",
            registry=self._registry,
        )
        self.avg_turns_per_session = Gauge(
            "context_avg_turns_per_session",
            "平均每会话轮次",
            registry=self._registry,
        )

        # ── 非 Prometheus 统计（仅供面板使用） ──
        self._total_turns = 0
        self._total_sessions = 0
        self._total_topic_switches = 0
        self._total_degradations: dict[str, int] = {"trim": 0, "summarize": 0, "emergency": 0}
        self._lock = threading.Lock()

    # ── 便捷更新方法 ──

    def record_degradation(self, step: str) -> None:
        """记录一次降级事件"""
        self.degradation_trim.labels(step=step).inc()
        with self._lock:
            self._total_degradations[step] = self._total_degradations.get(step, 0) + 1

    def record_topic_switch(self) -> None:
        """记录一次话题切换"""
        self.topic_switch_count.inc()
        with self._lock:
            self._total_topic_switches += 1

    def record_summary(self, success: bool, latency_ms: float, ratio: float) -> None:
        """记录一次摘要执行"""
        label = "success" if success else "failure"
        self.summary_success.labels(result=label).inc()
        self.summary_latency_ms.observe(latency_ms)
        self.compression_ratio.set(ratio)

    def record_turn(self) -> None:
        """记录一轮对话"""
        with self._lock:
            self._total_turns += 1

    def record_session(self) -> None:
        """记录一个会话"""
        with self._lock:
            self._total_sessions += 1

    def update_session_stats(self, active_count: int, avg_turns: float) -> None:
        """更新全局会话统计"""
        self.active_sessions.set(active_count)
        self.avg_turns_per_session.set(avg_turns)

    def update_from_prepare(self, metadata) -> None:
        """从 PrepareResult.metadata 更新指标"""
        if metadata.token_usage > 0 and metadata.max_tokens > 0:
            self.window_utilization.set(metadata.token_usage / metadata.max_tokens)
        self.hot_message_count.set(metadata.hot_message_count)
        self.warm_summary_count.set(metadata.warm_summary_count)
        if metadata.trim_info and metadata.trim_info.step != "none":
            self.record_degradation(metadata.trim_info.step)

    # ── Prometheus 导出 ──

    def to_prometheus(self) -> str:
        """导出 Prometheus text format"""
        return generate_latest(self._registry).decode("utf-8")

    def start_http_server(self, port: int = 9090) -> bool:
        """在后台线程启动 HTTP 服务器暴露 /metrics 端点

        Returns:
            True 表示启动成功，False 表示端口已被占用
        """
        try:
            prom_start_http_server(port, registry=self._registry)
            return True
        except OSError:
            return False

    # ── 终端面板 ──

    def render_panel(self) -> str:
        """渲染指标面板文本（供 rich Panel 使用）

        Returns:
            Markdown 格式的指标摘要文本
        """
        util = self.window_utilization._value.get() if hasattr(self.window_utilization, '_value') else 0
        hot = self.hot_message_count._value.get() if hasattr(self.hot_message_count, '_value') else 0
        warm = self.warm_summary_count._value.get() if hasattr(self.warm_summary_count, '_value') else 0
        wsize = self.current_window_size._value.get() if hasattr(self.current_window_size, '_value') else 0

        with self._lock:
            d_trim = self._total_degradations.get("trim", 0)
            d_summary = self._total_degradations.get("summarize", 0)
            d_emergency = self._total_degradations.get("emergency", 0)
            switches = self._total_topic_switches
            sessions = self._total_sessions
            avg_turns = self._total_turns / max(1, self._total_sessions)

        lines = [
            f"窗口利用率: {util:.0%}  热层消息: {hot}  温层摘要: {warm}  窗口大小: {wsize}",
            f"降级: {d_trim}(trim) {d_summary}(summarize) {d_emergency}(emergency)",
            f"话题切换: {switches}  活跃会话: {sessions}  平均轮次: {avg_turns:.1f}",
        ]
        return "\n".join(lines)


# 全局单例
metrics = ContextMetrics()