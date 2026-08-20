"""自适应窗口管理器 — 根据 Token 利用率动态调整窗口大小

根据历史 token 利用率自动调整窗口大小，使窗口在 10 轮内收敛到目标利用率附近。
利用率 = 实际 token 使用量 / max_tokens 预算。

集成到 SimpleContextManager：
- 构造函数增加 adaptive_window 参数
- prepare() 末尾调用 record_utilization()
- prepare_from_context() 中，用 get_current_window() 动态替换 self.window_size

每个 Session 独立维护自己的窗口状态。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveWindowManager:
    """自适应窗口管理器

    根据最近 N 轮的平均 token 利用率动态调整窗口大小。
    每 adjustment_interval 轮调整一次。

    Args:
        initial_window: 初始窗口大小（从 config 读取）
        min_window: 最小窗口大小
        max_window: 最大窗口大小
        target_utilization: 目标利用率，默认 0.75（75%）
        adjustment_interval: 调整间隔（轮次），默认 10
    """

    def __init__(
        self,
        initial_window: int = 5,
        min_window: int = 2,
        max_window: int = 20,
        target_utilization: float = 0.75,
        adjustment_interval: int = 10,
    ):
        self._window = initial_window
        self._min_window = min_window
        self._max_window = max_window
        self._target = target_utilization
        self._interval = adjustment_interval
        self._history: deque[float] = deque(maxlen=adjustment_interval)
        self._total_calls = 0
        self._adjustments = 0

        logger.info(
            f"自适应窗口初始化: initial={initial_window}, "
            f"min={min_window}, max={max_window}, "
            f"target={target_utilization:.0%}, interval={adjustment_interval}"
        )

    def record_utilization(self, utilization: float) -> None:
        """记录本轮 token 利用率"""
        self._history.append(utilization)
        self._total_calls += 1

    def should_adjust(self) -> bool:
        """判断是否需要调整窗口大小

        每 interval 轮触发一次调整，且至少有足够的历史数据。
        """
        if self._total_calls < self._interval:
            return False
        return self._total_calls % self._interval == 0

    def adjust(self) -> int:
        """根据最近 interval 轮的平均利用率调整窗口大小

        调整策略：
        - 平均利用率 < 目标 85% → 扩大窗口 (+2, 上限 max_window)
        - 平均利用率 > 目标 115% → 缩小窗口 (-1, 下限 min_window)
        - 否则 → 保持

        Returns:
            调整后的窗口大小
        """
        if not self._history:
            return self._window

        avg_util = sum(self._history) / len(self._history)
        old_window = self._window

        low_threshold = self._target * 0.85
        high_threshold = self._target * 1.15

        if avg_util < low_threshold:
            # 利用率太低 → 扩大窗口以容纳更多历史
            new_window = min(self._window + 2, self._max_window)
            reason = f"利用率 {avg_util:.0%} < 目标 {low_threshold:.0%}"
        elif avg_util > high_threshold:
            # 利用率太高 → 缩小窗口以减少 token
            new_window = max(self._window - 1, self._min_window)
            reason = f"利用率 {avg_util:.0%} > 目标 {high_threshold:.0%}"
        else:
            new_window = self._window
            reason = f"利用率 {avg_util:.0%} 在目标范围内"

        self._window = new_window
        self._adjustments += 1

        logger.info(
            f"[自适应窗口] window_size: {old_window} → {new_window} "
            f"({reason}, 目标 {self._target:.0%}, 调整 #{self._adjustments})"
        )

        return self._window

    def get_current_window(self) -> int:
        """返回当前窗口大小"""
        return self._window

    def get_stats(self) -> dict:
        """获取当前状态统计"""
        return {
            "current_window": self._window,
            "min_window": self._min_window,
            "max_window": self._max_window,
            "target_utilization": self._target,
            "adjustment_interval": self._interval,
            "total_calls": self._total_calls,
            "adjustments": self._adjustments,
            "avg_utilization": sum(self._history) / max(1, len(self._history)),
        }