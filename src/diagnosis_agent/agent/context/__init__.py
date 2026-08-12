"""上下文管理模块 — 分层记忆架构

热层 (hot)  → 最近 N 轮完整消息
温层 (warm) → LLM 摘要 + 话题感知
冷层 (cold) → 归档文件
"""

from .types import (
    TopicSnapshot,
    TrimInfo,
    ContextMetadata,
    PrepareResult,
    ConversationContext,
    ArchivedSession,
)
from .summarizer import Summarizer
from .topic_detector import TopicDetector, TopicDecision

__all__ = [
    "TopicSnapshot",
    "TrimInfo",
    "ContextMetadata",
    "PrepareResult",
    "ConversationContext",
    "ArchivedSession",
    "Summarizer",
    "TopicDetector",
    "TopicDecision",
]