"""上下文管理器 — Token 预算管理

在 LLM 调用前计算消息 token 数，固定窗口裁剪，超预算时截断不报错。
窗口大小和 token 预算外部化配置。

核心流程：
1. 按 window_size 保留最近 N 轮（完整轮次，不拆分工具调用链）
2. 按 max_tokens 从最早轮次开始丢弃，直到预算内
3. 至少保留最后一轮（当前用户问题）
4. 返回 PrepareResult（裁剪后的消息 + 元数据）

分层记忆架构：
  热层 (hot)  → 完整消息 → prepare() 裁剪后发给 LLM
  温层 (warm) → 摘要     → 预留（Step 1 实现）
  冷层 (cold) → 归档     → 预留（Step 2 实现）
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from .context.types import (
    ContextMetadata,
    PrepareResult,
    TrimInfo,
    ConversationContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 分词器（延迟初始化，首次使用时才加载 tiktoken）
# ---------------------------------------------------------------------------

_enc = None


def _get_encoding():
    """获取 tiktoken 编码器（延迟加载）"""
    global _enc
    if _enc is None:
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken 未安装，使用字符数 / 4 估算 token 数")
            _enc = None
    return _enc


def count_tokens(text: str) -> int:
    """计算文本的 token 数

    优先使用 tiktoken cl100k_base 编码计算。
    tiktoken 不可用时，用 len(text) // 4 估算。
    """
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 消息工具函数
# ---------------------------------------------------------------------------


def _is_user_message(msg: Any) -> bool:
    """判断消息是否为用户消息（轮次边界）

    支持两种格式：
    - LangChain BaseMessage 子类（HumanMessage）
    - dict 格式（{"role": "user", ...}）
    """
    if isinstance(msg, HumanMessage):
        return True
    if isinstance(msg, dict) and msg.get("role") == "user":
        return True
    return False


def _get_message_content(msg: Any) -> str:
    """获取消息的文本内容

    支持 BaseMessage 和 dict 两种格式。
    """
    if isinstance(msg, BaseMessage):
        content = msg.content
    elif isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = ""
    return str(content) if content is not None else ""


def _find_round_boundaries(messages: list) -> list[int]:
    """找到所有用户消息的索引，作为轮次边界

    返回递增的索引列表，每个索引对应一轮的开始。
    """
    return [i for i, msg in enumerate(messages) if _is_user_message(msg)]


# ---------------------------------------------------------------------------
# 上下文管理器
# ---------------------------------------------------------------------------


class SimpleContextManager:
    """Phase 1: 最简单的上下文管理器

    在 LLM 调用前对消息列表做预算管理，防止上下文窗口溢出。

    Args:
        window_size: 热层窗口大小（轮次），默认 5
        max_tokens: 消息列表总 token 预算上限，默认 8000
    """

    def __init__(self, window_size: int = 5, max_tokens: int = 8000):
        self.window_size = window_size
        self.max_tokens = max_tokens

    def prepare(self, messages: list) -> PrepareResult:
        """裁剪消息到预算内，返回 PrepareResult

        裁剪策略（以"轮"为最小单位）：
        1. 总 token 数 <= max_tokens 且轮数 <= window_size → 原样返回
        2. 超轮数 → 按 window_size 保留最近 N 轮
        3. 超预算 → 从最早轮次开始丢弃，直到 token 数回归预算内
        4. 至少保留最后一轮（当前用户问题）
        5. 不报错，只打 warning 日志
        6. 不会把 tool_calls 和对应的 ToolMessage 拆开
        """
        trim_info = TrimInfo()
        original_count = len(messages)

        if not messages:
            return PrepareResult(
                messages=[],
                metadata=self._build_metadata(
                    session_id="",
                    messages=[],
                    trim_info=trim_info,
                    turn_count=0,
                ),
            )

        # 1. 按 window_size 截断
        before_window = len(messages)
        messages = self._truncate_by_window(messages)
        if len(messages) < before_window:
            trim_info.step = "trim"
            trim_info.trimmed_turns = self._count_rounds_dropped(
                messages, before_window - len(messages)
            )

        # 2. 按 token 预算截断
        before_tokens = len(messages)
        messages = self._truncate_by_tokens(messages)
        if len(messages) < before_tokens:
            trim_info.step = "trim"
            # 本轮可能是 token 裁剪额外丢弃的轮次
            extra_dropped = self._count_rounds_dropped(
                messages, before_tokens - len(messages)
            )
            trim_info.trimmed_turns += extra_dropped

        # 计算实际 token 用量
        token_usage = self._count_tokens(messages)

        dropped = original_count - len(messages)
        if dropped > 0:
            logger.warning(
                f"上下文裁剪: 丢弃 {dropped} 条消息, "
                f"保留 {len(messages)} 条 ({token_usage}/{self.max_tokens} tokens)"
            )

        total_turns = len(_find_round_boundaries(messages))

        return PrepareResult(
            messages=messages,
            metadata=self._build_metadata(
                session_id="",
                messages=messages,
                trim_info=trim_info,
                turn_count=total_turns,
            ),
        )

    def prepare_from_context(
        self, ctx: ConversationContext, query: str
    ) -> PrepareResult:
        """从 ConversationContext 构建消息列表并裁剪

        1. 从热层取历史消息
        2. 追加当前用户消息
        3. 裁剪到预算内
        4. 返回 PrepareResult
        """
        from langchain_core.messages import messages_from_dict

        messages: list = []
        if ctx.hot_messages:
            messages.extend(messages_from_dict(ctx.hot_messages))
        messages.append({"role": "user", "content": query})

        result = self.prepare(messages)
        # 填充 session_id
        result.metadata.session_id = ctx.session_id
        result.metadata.total_turns = ctx.total_turns
        result.metadata.warm_summary_count = len(ctx.warm_summaries)
        if ctx.current_topic:
            result.metadata.current_topic = ctx.current_topic.topic_label
        result.metadata.archived_topic_count = ctx.archived_topic_count

        return result

    def _truncate_by_window(self, messages: list) -> list:
        """按 window_size 保留最近 N 轮"""
        boundaries = _find_round_boundaries(messages)
        if len(boundaries) <= self.window_size:
            return messages

        keep_from = boundaries[-self.window_size]
        dropped = len(boundaries) - self.window_size
        logger.warning(
            f"上下文窗口裁剪: 丢弃 {dropped} 轮历史 "
            f"(window_size={self.window_size})"
        )
        return messages[keep_from:]

    def _truncate_by_tokens(self, messages: list) -> list:
        """按 token 预算从最早轮次开始丢弃"""
        total = self._count_tokens(messages)
        if total <= self.max_tokens:
            return messages

        boundaries = _find_round_boundaries(messages)
        while len(boundaries) > 1:
            second_round_start = boundaries[1]
            messages = messages[second_round_start:]
            total = self._count_tokens(messages)
            boundaries = _find_round_boundaries(messages)
            if total <= self.max_tokens:
                logger.warning(
                    f"Token 预算裁剪: 丢弃历史直到 {total}/{self.max_tokens} tokens"
                )
                break

        if total > self.max_tokens:
            logger.warning(
                f"单轮消息已超 token 预算: {total}/{self.max_tokens}，未裁剪"
            )

        return messages

    def _count_tokens(self, messages: list) -> int:
        """计算消息列表总 token 数"""
        total = 0
        for msg in messages:
            content = _get_message_content(msg)
            total += count_tokens(content)
        return total

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        session_id: str,
        messages: list,
        trim_info: TrimInfo,
        turn_count: int,
    ) -> ContextMetadata:
        """构建上下文元数据"""
        return ContextMetadata(
            session_id=session_id,
            total_turns=turn_count,
            hot_message_count=len(messages),
            warm_summary_count=0,
            archived_topic_count=0,
            current_topic=None,
            topic_changed=False,
            trim_info=trim_info,
            token_usage=self._count_tokens(messages),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _count_rounds_dropped(self, messages: list, dropped_count: int) -> int:
        """估算丢弃的轮次（按平均每条消息估算）"""
        # 精确计算丢弃的轮次数
        boundaries = _find_round_boundaries(messages)
        if not boundaries:
            return 0
        # 粗略估算：轮次数 ≈ 用户消息数
        average_messages_per_round = max(1, len(messages) // max(1, len(boundaries)))
        return max(1, dropped_count // average_messages_per_round)