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
import os
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .context.types import (
    ContextMetadata,
    PrepareResult,
    TrimInfo,
    ConversationContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 分词器（延迟初始化，直接加载本地 BPE 文件，不依赖网络）
# ---------------------------------------------------------------------------

_enc = None
_enc_loaded = False
_BPE_CACHE_PATH = os.path.expanduser("~/.cache/tiktoken/cl100k_base.tiktoken")


def _get_encoding():
    """获取 tiktoken 编码器（延迟加载，加载本地缓存文件）

    直接用 load_tiktoken_bpe 加载本地文件，绕过 tiktoken.get_encoding()
    的网络请求（当前环境 Azure 无法连通）。

    BPE 文件缓存路径：~/.cache/tiktoken/cl100k_base.tiktoken
    完整文件约 1.68MB，加载耗时约 0.12s。
    """
    global _enc, _enc_loaded
    if _enc_loaded:
        return _enc
    _enc_loaded = True

    try:
        from tiktoken.load import load_tiktoken_bpe
        from tiktoken.core import Encoding

        if not os.path.exists(_BPE_CACHE_PATH):
            logger.warning(f"tiktoken BPE 文件不存在: {_BPE_CACHE_PATH}，使用字符数估算")
            return None

        mergeable_ranks = load_tiktoken_bpe(_BPE_CACHE_PATH)
        _enc = Encoding(
            name="cl100k_base",
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={
                "<|endoftext|>": 100256,
                "<|fim_prefix|>": 100257,
                "<|fim_middle|>": 100258,
                "<|fim_suffix|>": 100259,
                "<|endofprompt|>": 100260,
            },
        )
        logger.info(f"tiktoken 加载成功 ({len(mergeable_ranks)} tokens)")
    except ImportError:
        logger.warning("tiktoken 未安装，使用字符数估算 token 数")
    except Exception as e:
        logger.warning(f"tiktoken 加载失败: {e}，使用字符数估算 token 数")

    return _enc


def count_tokens(text: str) -> int:
    """计算文本的 token 数

    优先使用 tiktoken cl100k_base 编码计算。
    tiktoken 不可用时，用 len(text) // 4 估算。
    """
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    return max(0, len(text) // 4)


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

    三步降级流程（prepare_from_context）：
    步骤1: 热层→温层 — 超出窗口的旧消息生成 LLM 摘要
    步骤2: 温层合并 — 温层摘要过多时合并压缩
    步骤3: 紧急截断 — 保留最近 2 轮 + 摘要

    Args:
        window_size: 热层窗口大小（轮次），默认 5
        max_tokens: 消息列表总 token 预算上限，默认 8000
        summarizer: 摘要生成器（可选，Step 1 注入）
        topic_detector: 话题检测器（可选，Step 1 注入）
    """

    def __init__(
        self,
        window_size: int = 5,
        max_tokens: int = 8000,
        summarizer=None,
        topic_detector=None,
    ):
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.topic_detector = topic_detector

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

        三步降级 + 话题感知 + 摘要注入：

        1. 话题检测：检测查询是否切换话题
           - 话题切换 → 旧话题生成摘要 → 移入温层
           - 更新 current_topic
        2. 三步降级：
           步骤1: 热层超窗口 → LLM 摘要旧消息 → 移入温层
           步骤2: 温层超预算 → LLM 合并摘要
           步骤3: 紧急截断 → 保留最近 2 轮
        3. 构建消息：
           [SystemMessage(摘要)] + hot_messages + [query]
        4. 返回 PrepareResult
        """
        from langchain_core.messages import messages_from_dict
        from datetime import datetime, timezone

        trim_info = TrimInfo()
        topic_changed = False

        # ── 话题检测 ──
        if self.topic_detector and ctx.total_turns > 0:
            decision = self.topic_detector.detect(ctx, query)
            if not decision.is_in_scope:
                logger.warning(f"话题检测器判定不在电驱范围内: {query[:100]}")
                # 不中断流程，继续构建消息（is_in_scope 仅作参考，不阻断）

            if decision.decision == "different":
                topic_changed = True
                logger.info(f"话题切换: {decision.new_topic_label}")
                # 步骤1: 生成旧话题摘要
                if self.summarizer and ctx.current_topic:
                    self._summarize_current_topic(ctx)
                # 步骤2: 归档旧话题（完整消息快照存入冷层）
                if ctx.hot_messages:
                    self._archive_topic_snapshot(ctx)
                # 步骤3: 清空热层
                ctx.hot_messages = []
                # 步骤4: 创建新话题
                from .context.types import TopicSnapshot
                ctx.current_topic = TopicSnapshot(
                    topic_id=ctx.session_id + "-" + str(ctx.total_turns),
                    topic_label=decision.new_topic_label or f"话题-{ctx.total_turns + 1}",
                    start_turn=ctx.total_turns,
                    end_turn=ctx.total_turns,
                    summary="",
                    key_entities=[],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                # 步骤5: 旧摘要已存入 warm_summaries，后续构建消息时注入 SystemMessage

        # ── 三步降级（两层触发：window_size + token 预算） ──
        # 步骤1: 热层超 window_size → 溢出消息生成摘要，追加到温层
        hot_rounds = self._count_hot_rounds(ctx)
        if hot_rounds > self.window_size:
            self._step1_hot_to_warm(ctx)
            trim_info.step = "summarize"
            trim_info.summarized_turns = hot_rounds - self.window_size

        # ── 构建消息（初始版本，用于检查 token 预算） ──
        messages: list = []
        if ctx.warm_summaries:
            summary_text = self._build_summary_text(ctx)
            if summary_text:
                messages.append(SystemMessage(content=summary_text))
        if ctx.hot_messages:
            messages.extend(messages_from_dict(ctx.hot_messages))
        messages.append({"role": "user", "content": query})

        # 检查 token 预算
        token_usage = self._count_tokens(messages)
        if token_usage > self.max_tokens:
            # 步骤2: 温层摘要合并（多个摘要 → 一个精炼摘要）
            if ctx.warm_summaries and len(ctx.warm_summaries) > 1:
                self._step2_merge_warm(ctx)
                trim_info.step = "summarize"

                # 重建消息
                messages = []
                if ctx.warm_summaries:
                    summary_text = self._build_summary_text(ctx)
                    if summary_text:
                        messages.append(SystemMessage(content=summary_text))
                if ctx.hot_messages:
                    messages.extend(messages_from_dict(ctx.hot_messages))
                messages.append({"role": "user", "content": query})

                # 重新检查
                token_usage = self._count_tokens(messages)

            # 步骤3: 仍超标 → 紧急截断（保留最近 2 轮 + 摘要）
            if token_usage > self.max_tokens:
                messages = self._step3_emergency_truncate(messages)
                if trim_info.step != "summarize":
                    trim_info.step = "emergency"
                trim_info.trimmed_turns = self._count_rounds_dropped(messages, 0)

        # 最终裁剪
        result = self.prepare(messages)
        # 合并 trim_info
        if result.metadata.trim_info.step == "none":
            result.metadata.trim_info = trim_info

        # 填充元数据
        result.metadata.session_id = ctx.session_id
        result.metadata.total_turns = ctx.total_turns
        result.metadata.warm_summary_count = len(ctx.warm_summaries)
        result.metadata.archived_topic_count = ctx.archived_topic_count
        result.metadata.topic_changed = topic_changed
        if ctx.current_topic:
            result.metadata.current_topic = ctx.current_topic.topic_label

        return result

    # ------------------------------------------------------------------
    # 三步降级
    # ------------------------------------------------------------------

    def _step1_hot_to_warm(self, ctx: ConversationContext) -> None:
        """步骤1: 热层→温层（追加摘要）

        将超出 window_size 的最早轮次消息移出热层，
        生成摘要后追加到温层列表（不合并）。

        每次热层溢出只追加一个新摘要，不合并旧摘要。
        温层摘要数量会逐步增长，直到步骤2按 token 预算合并。
        """
        boundaries = self._find_hot_round_boundaries(ctx)
        if len(boundaries) <= self.window_size:
            return

        keep_from = boundaries[-self.window_size]
        overflow = ctx.hot_messages[:keep_from]

        if not overflow:
            return

        logger.info(
            f"步骤1: 热层→温层 — 将 {len(overflow)} 条消息压缩为摘要 "
            f"(保留最近 {self.window_size} 轮，温层 {len(ctx.warm_summaries) + 1} 个摘要)"
        )

        if self.summarizer:
            from .context.types import TopicSnapshot
            new_snapshot = self.summarizer.summarize(
                overflow,
                topic_label=f"对话摘要-{ctx.total_turns - self.window_size + 1}-{ctx.total_turns}轮",
                start_turn=ctx.total_turns - self.window_size,
                end_turn=ctx.total_turns,
            )
            ctx.warm_summaries.append(new_snapshot)

        # 从热层移除
        ctx.hot_messages = ctx.hot_messages[keep_from:]

    def _step2_merge_warm(self, ctx: ConversationContext) -> None:
        """步骤2: 温层摘要合并

        按 token 预算触发：温层摘要过多时合并为一个精炼摘要。
        合并后减少 LLM 调用次数，控制 SystemMessage 长度。
        """
        if not ctx.warm_summaries or len(ctx.warm_summaries) <= 1:
            return

        total = self._count_tokens_from_summaries(ctx.warm_summaries)
        logger.info(
            f"步骤2: 温层合并 — 合并 {len(ctx.warm_summaries)} 个摘要 "
            f"({total} tokens)"
        )

        if self.summarizer:
            merged = self.summarizer.merge_summaries(ctx.warm_summaries)
            ctx.warm_summaries = [merged]

    def _step3_emergency_truncate(self, messages: list) -> list:
        """步骤3: 紧急截断

        保留最近 2 轮 + 摘要 SystemMessage。
        这是最后兜底，确保不超 token 预算。
        """
        logger.warning("步骤3: 紧急截断 — 仅保留最近 2 轮")

        boundaries = _find_round_boundaries(messages)
        if len(boundaries) <= 2:
            return messages

        # 保留 SystemMessage（摘要） + 最近 2 轮
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append(msg)
                break

        keep_from = boundaries[-2] if len(boundaries) >= 2 else boundaries[0]
        result.extend(messages[keep_from:])
        return result

    # ------------------------------------------------------------------
    # 摘要相关
    # ------------------------------------------------------------------

    def _summarize_current_topic(self, ctx: ConversationContext) -> None:
        """步骤1: 将当前话题的全量消息生成摘要，存入温层"""
        if not self.summarizer or not ctx.current_topic:
            return

        start_turn = ctx.current_topic.start_turn
        end_turn = ctx.current_topic.end_turn

        topic_msgs = ctx.hot_messages[:]
        if not topic_msgs:
            return

        snapshot = self.summarizer.summarize(
            topic_msgs,
            topic_label=ctx.current_topic.topic_label,
            start_turn=start_turn,
            end_turn=end_turn,
        )
        ctx.warm_summaries.append(snapshot)
        logger.info(f"话题摘要已生成: {snapshot.topic_label}")

    def _archive_topic_snapshot(self, ctx: ConversationContext) -> None:
        """步骤2: 归档旧话题完整消息到冷层

        将当前热层的完整消息写入冷层归档文件。
        文件路径: data/sessions/archive/{session_id}_{topic_id}.snapshot.json
        """
        import json
        from pathlib import Path

        if not ctx.hot_messages or not ctx.current_topic:
            return

        try:
            archive_dir = Path("data/sessions/archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            path = archive_dir / f"{ctx.session_id}_{ctx.current_topic.topic_id}.snapshot.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": ctx.session_id,
                    "topic_id": ctx.current_topic.topic_id,
                    "topic_label": ctx.current_topic.topic_label,
                    "start_turn": ctx.current_topic.start_turn,
                    "end_turn": ctx.current_topic.end_turn,
                    "messages": ctx.hot_messages,
                    "total_messages": len(ctx.hot_messages),
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"话题快照已归档: {path}")
        except OSError as e:
            logger.warning(f"归档话题快照失败: {e}")

    def _build_summary_text(self, ctx: ConversationContext) -> str:
        """构建温层摘要 SystemMessage 文本"""
        if not ctx.warm_summaries:
            return ""

        lines = ["[历史会话摘要]"]
        for i, s in enumerate(ctx.warm_summaries):
            entities = ", ".join(s.key_entities) if s.key_entities else "无"
            todos = s.metadata.get("todos", [])
            if isinstance(todos, list) and todos:
                todo_text = "；待办: " + "、".join(todos)
            else:
                todo_text = ""
            lines.append(
                f"{i + 1}. [{s.topic_label}] 实体: {entities} | "
                f"结论: {s.summary}{todo_text}"
            )

        return "\n".join(lines)

    def _count_tokens_from_summaries(
        self, summaries: list["TopicSnapshot"]
    ) -> int:
        """计算温层摘要总 token 数"""
        total = 0
        for s in summaries:
            total += count_tokens(s.summary)
            total += count_tokens(s.topic_label)
            for e in s.key_entities:
                total += count_tokens(e)
        return total

    # ------------------------------------------------------------------
    # 热层工具方法
    # ------------------------------------------------------------------

    def _count_hot_rounds(self, ctx: ConversationContext) -> int:
        """计算热层中的轮次数"""
        return len(self._find_hot_round_boundaries(ctx))

    def _find_hot_round_boundaries(self, ctx: ConversationContext) -> list[int]:
        """找到热层消息中用户消息的索引"""
        boundaries = []
        for i, msg in enumerate(ctx.hot_messages):
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            msg_type = msg.get("type", "") if isinstance(msg, dict) else ""
            if role == "user" or msg_type == "human":
                boundaries.append(i)
        return boundaries

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