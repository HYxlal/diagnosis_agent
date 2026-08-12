"""上下文管理 — 数据模型

定义分层记忆架构中所有核心类型，对齐 IContextManager 接口协议。

层次结构：
  热层 (hot)  → 最近 N 轮完整消息（LangChain messages_to_dict 格式）
  温层 (warm) → 旧对话摘要（TopicSnapshot 列表）
  冷层 (cold) → 已归档话题（ArchivedSession，磁盘文件）

内部消息格式：LangChain messages_to_dict(messages) 的 dict 列表，
与 messages_from_dict() 兼容。后续对外接口可做协议 Message 格式转换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# 话题快照
# ---------------------------------------------------------------------------


@dataclass
class TopicSnapshot:
    """话题快照 — 温层摘要单元

    当对话切换到新话题时，旧话题的完整消息被压缩为摘要，
    存入 ConversationContext.warm_summaries。
    """

    topic_id: str                                    # 话题唯一标识
    topic_label: str                                 # 话题标签（如"过温故障-P1A3E98"）
    start_turn: int                                  # 起始轮次
    end_turn: int                                    # 结束轮次
    summary: str                                     # 摘要文本
    key_entities: list[str] = field(default_factory=list)  # 关键实体（DTC/车型/部件）
    created_at: str = ""                             # 创建时间
    metadata: dict = field(default_factory=dict)     # 扩展元数据


# ---------------------------------------------------------------------------
# 裁剪信息
# ---------------------------------------------------------------------------


@dataclass
class TrimInfo:
    """裁剪步骤信息 — 记录每次 prepare() 的裁剪行为"""

    step: str = "none"            # 裁剪步骤: none | trim | summarize | emergency
    trimmed_turns: int = 0        # 被裁剪的轮次
    summarized_turns: int = 0     # 被摘要的轮次（预留，Step 1 实现）


# ---------------------------------------------------------------------------
# 上下文元数据
# ---------------------------------------------------------------------------


@dataclass
class ContextMetadata:
    """prepare() 返回的元数据 — 描述当前会话状态"""

    session_id: str = ""
    total_turns: int = 0
    hot_message_count: int = 0
    warm_summary_count: int = 0
    archived_topic_count: int = 0
    current_topic: str | None = None
    topic_changed: bool = False
    trim_info: TrimInfo = field(default_factory=TrimInfo)
    token_usage: int = 0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# prepare() 返回结果
# ---------------------------------------------------------------------------


@dataclass
class PrepareResult:
    """prepare() 返回结果

    messages: 裁剪后的消息列表（LangChain 格式，可直接拼入 Agent invoke）
    metadata: 上下文元数据（供上层决策：是否摘要、是否归档等）
    """

    messages: list = field(default_factory=list)
    metadata: ContextMetadata = field(default_factory=ContextMetadata)


# ---------------------------------------------------------------------------
# 会话上下文（持久化核心）
# ---------------------------------------------------------------------------


@dataclass
class ConversationContext:
    """会话上下文 — 核心持久化对象

    一次会话的完整状态，序列化为 JSON 文件存储。

    热层消息格式：LangChain messages_to_dict() 的 dict 列表，
    内部使用 messages_from_dict() 恢复为 BaseMessage。
    """

    session_id: str = ""
    # 热层：最近 N 轮完整消息
    hot_messages: list[dict] = field(default_factory=list)
    # 温层：旧对话摘要
    warm_summaries: list[TopicSnapshot] = field(default_factory=list)
    # 冷层：已归档话题数（归档文件独立存储）
    archived_topic_count: int = 0
    # 当前话题
    current_topic: TopicSnapshot | None = None
    # 总轮次
    total_turns: int = 0
    # 时间戳
    created_at: str = ""
    last_activity_at: str = ""
    # 扩展元数据
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为 dict（用于 JSON 持久化）"""
        return {
            "session_id": self.session_id,
            "hot_messages": self.hot_messages,
            "warm_summaries": [
                {
                    "topic_id": ts.topic_id,
                    "topic_label": ts.topic_label,
                    "start_turn": ts.start_turn,
                    "end_turn": ts.end_turn,
                    "summary": ts.summary,
                    "key_entities": ts.key_entities,
                    "created_at": ts.created_at,
                    "metadata": ts.metadata,
                }
                for ts in self.warm_summaries
            ],
            "archived_topic_count": self.archived_topic_count,
            "current_topic": (
                {
                    "topic_id": self.current_topic.topic_id,
                    "topic_label": self.current_topic.topic_label,
                    "start_turn": self.current_topic.start_turn,
                    "end_turn": self.current_topic.end_turn,
                    "summary": self.current_topic.summary,
                    "key_entities": self.current_topic.key_entities,
                    "created_at": self.current_topic.created_at,
                    "metadata": self.current_topic.metadata,
                }
                if self.current_topic else None
            ),
            "total_turns": self.total_turns,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """从 dict 反序列化"""
        ctx = cls(
            session_id=data.get("session_id", ""),
            hot_messages=data.get("hot_messages", []),
            archived_topic_count=data.get("archived_topic_count", 0),
            total_turns=data.get("total_turns", 0),
            created_at=data.get("created_at", ""),
            last_activity_at=data.get("last_activity_at", ""),
            metadata=data.get("metadata", {}),
        )
        # 温层摘要
        for ts_data in data.get("warm_summaries", []):
            ctx.warm_summaries.append(TopicSnapshot(
                topic_id=ts_data.get("topic_id", ""),
                topic_label=ts_data.get("topic_label", ""),
                start_turn=ts_data.get("start_turn", 0),
                end_turn=ts_data.get("end_turn", 0),
                summary=ts_data.get("summary", ""),
                key_entities=ts_data.get("key_entities", []),
                created_at=ts_data.get("created_at", ""),
                metadata=ts_data.get("metadata", {}),
            ))
        # 当前话题
        ct_data = data.get("current_topic")
        if ct_data:
            ctx.current_topic = TopicSnapshot(
                topic_id=ct_data.get("topic_id", ""),
                topic_label=ct_data.get("topic_label", ""),
                start_turn=ct_data.get("start_turn", 0),
                end_turn=ct_data.get("end_turn", 0),
                summary=ct_data.get("summary", ""),
                key_entities=ct_data.get("key_entities", []),
                created_at=ct_data.get("created_at", ""),
                metadata=ct_data.get("metadata", {}),
            )
        return ctx

    @classmethod
    def create_new(cls, session_id: str) -> "ConversationContext":
        """创建新会话"""
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            session_id=session_id,
            created_at=now,
            last_activity_at=now,
        )

    def touch(self) -> None:
        """更新最后活动时间"""
        self.last_activity_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 归档记录
# ---------------------------------------------------------------------------


@dataclass
class ArchivedSession:
    """归档记录 — 冷层持久化对象

    会话关闭时，将 ConversationContext 压缩为归档记录。
    归档文件存储在 data/sessions/archive/{session_id}.json。
    """

    session_id: str
    total_turns: int
    topics: list[TopicSnapshot] = field(default_factory=list)
    created_at: str = ""
    archived_at: str = ""
    duration: int = 0         # 会话时长（秒）
    user_id: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_turns": self.total_turns,
            "topics": [
                {
                    "topic_id": ts.topic_id,
                    "topic_label": ts.topic_label,
                    "start_turn": ts.start_turn,
                    "end_turn": ts.end_turn,
                    "summary": ts.summary,
                    "key_entities": ts.key_entities,
                    "created_at": ts.created_at,
                    "metadata": ts.metadata,
                }
                for ts in self.topics
            ],
            "created_at": self.created_at,
            "archived_at": self.archived_at,
            "duration": self.duration,
            "user_id": self.user_id,
        }