"""测试 SimpleContextManager — token 预算管理 + PrepareResult"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from diagnosis_agent.agent.context_manager import (
    SimpleContextManager,
    _is_user_message,
    _get_message_content,
    _find_round_boundaries,
    count_tokens,
)
from diagnosis_agent.agent.context.types import (
    PrepareResult,
    ContextMetadata,
    TrimInfo,
    ConversationContext,
    TopicSnapshot,
)


class TestCountTokens:
    """测试 token 计数"""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_chinese_text(self):
        assert count_tokens("你好世界") > 0

    def test_english_text(self):
        assert count_tokens("hello world") > 0


class TestMessageHelpers:
    """测试消息工具函数"""

    def test_is_user_message_human_message(self):
        assert _is_user_message(HumanMessage(content="hi")) is True

    def test_is_user_message_ai_message(self):
        assert _is_user_message(AIMessage(content="hi")) is False

    def test_is_user_message_tool_message(self):
        assert _is_user_message(ToolMessage(content="{}", tool_call_id="1")) is False

    def test_is_user_message_dict_user(self):
        assert _is_user_message({"role": "user", "content": "hi"}) is True

    def test_is_user_message_dict_assistant(self):
        assert _is_user_message({"role": "assistant", "content": "hi"}) is False

    def test_get_content_from_base_message(self):
        assert _get_message_content(HumanMessage(content="hello")) == "hello"

    def test_get_content_from_dict(self):
        assert _get_message_content({"role": "user", "content": "world"}) == "world"

    def test_get_content_none(self):
        assert _get_message_content(HumanMessage(content="")) == ""

    def test_find_round_boundaries(self):
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            {"role": "user", "content": "q3"},
        ]
        assert _find_round_boundaries(messages) == [0, 2, 4]


class TestSimpleContextManager:
    """测试核心上下文管理器"""

    def test_prepare_empty(self):
        cm = SimpleContextManager(window_size=5, max_tokens=8000)
        result = cm.prepare([])
        assert isinstance(result, PrepareResult)
        assert result.messages == []
        assert result.metadata.token_usage == 0

    def test_prepare_within_budget(self):
        """预算内不裁剪"""
        cm = SimpleContextManager(window_size=5, max_tokens=8000)
        messages = [
            HumanMessage(content="短问题"),
            AIMessage(content="短回答"),
        ]
        result = cm.prepare(messages)
        assert result.messages == messages
        assert result.metadata.trim_info.step == "none"

    def test_prepare_truncate_by_window(self):
        """超 window_size 时裁剪最早轮次"""
        cm = SimpleContextManager(window_size=2, max_tokens=8000)
        messages = [
            # 第 1 轮（应被裁剪）
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            # 第 2 轮（应被裁剪）
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            # 第 3 轮（保留）
            HumanMessage(content="q3"),
            AIMessage(content="a3"),
            # 第 4 轮（保留）
            HumanMessage(content="q4"),
            AIMessage(content="a4"),
        ]
        result = cm.prepare(messages)
        assert len(result.messages) == 4
        assert _get_message_content(result.messages[0]) == "q3"
        assert result.metadata.trim_info.trimmed_turns > 0

    def test_prepare_truncate_by_tokens(self):
        """超 token 预算时从最早开始丢弃"""
        cm = SimpleContextManager(window_size=5, max_tokens=50)
        long_text = "这是一段很长的文本" * 50
        messages = [
            HumanMessage(content=long_text),
            AIMessage(content=long_text),
            HumanMessage(content="当前问题"),
            AIMessage(content="短回答"),
        ]
        result = cm.prepare(messages)
        assert len(result.messages) == 2
        assert _get_message_content(result.messages[0]) == "当前问题"
        assert result.metadata.trim_info.trimmed_turns > 0

    def test_preserve_last_round(self):
        """确保最后一轮始终保留"""
        cm = SimpleContextManager(window_size=1, max_tokens=10)
        short_text = "测试" * 5
        messages = [
            HumanMessage(content=short_text),
            AIMessage(content=short_text),
            HumanMessage(content="最终问题"),
        ]
        result = cm.prepare(messages)
        assert len(result.messages) >= 1
        assert _get_message_content(result.messages[-1]) == "最终问题"

    def test_tool_calls_not_split(self):
        """确保 tool_calls 和 ToolMessage 不会被拆开"""
        cm = SimpleContextManager(window_size=2, max_tokens=8000)
        messages = [
            # 第 1 轮（将被裁剪）
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            # 第 2 轮 - 含工具调用（保留）
            HumanMessage(content="q2 with tool"),
            AIMessage(
                content="我需要调用工具",
                tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
            ),
            ToolMessage(content="工具返回结果", tool_call_id="call_1"),
            AIMessage(content="基于工具结果的分析"),
            # 第 3 轮（当前）
            {"role": "user", "content": "q3"},
        ]
        result = cm.prepare(messages)
        assert len(result.messages) == 5
        assert _get_message_content(result.messages[2]) == "工具返回结果"

    def test_mixed_dict_and_base_message(self):
        """混合 dict 和 BaseMessage 格式的消息列表"""
        cm = SimpleContextManager(window_size=3, max_tokens=8000)
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
        result = cm.prepare(messages)
        assert len(result.messages) == 5
        assert _get_message_content(result.messages[-1]) == "q3"

    def test_prepare_result_metadata(self):
        """验证 PrepareResult 元数据"""
        cm = SimpleContextManager(window_size=5, max_tokens=8000)
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="answer"),
        ]
        result = cm.prepare(messages)
        assert isinstance(result.metadata, ContextMetadata)
        assert isinstance(result.metadata.trim_info, TrimInfo)
        assert result.metadata.hot_message_count == 2
        assert result.metadata.token_usage > 0
        assert result.metadata.timestamp != ""

    def test_prepare_from_context(self):
        """从 ConversationContext 准备消息"""
        cm = SimpleContextManager(window_size=5, max_tokens=8000)
        from langchain_core.messages import messages_to_dict

        ctx = ConversationContext.create_new("test-session")
        ctx.hot_messages = messages_to_dict([
            HumanMessage(content="历史问题"),
            AIMessage(content="历史回答"),
        ])
        ctx.total_turns = 1

        result = cm.prepare_from_context(ctx, "当前问题")
        assert len(result.messages) == 3  # 历史2 + 当前1
        assert result.metadata.session_id == "test-session"
        assert result.metadata.total_turns == 1


class TestConversationContext:
    """测试 ConversationContext 模型"""

    def test_create_new(self):
        ctx = ConversationContext.create_new("sess-1")
        assert ctx.session_id == "sess-1"
        assert ctx.total_turns == 0
        assert ctx.hot_messages == []
        assert ctx.warm_summaries == []
        assert ctx.created_at != ""
        assert ctx.last_activity_at != ""

    def test_touch(self):
        ctx = ConversationContext.create_new("sess-1")
        old_time = ctx.last_activity_at
        ctx.touch()
        assert ctx.last_activity_at != old_time

    def test_roundtrip(self):
        """序列化/反序列化往返测试"""
        ctx = ConversationContext.create_new("sess-1")
        ctx.total_turns = 3
        ctx.hot_messages = [
            {"type": "human", "data": {"content": "test"}}
        ]
        ctx.warm_summaries = [
            TopicSnapshot(
                topic_id="t1",
                topic_label="过温故障",
                start_turn=1,
                end_turn=2,
                summary="电机过温",
                key_entities=["P1A3E98"],
            )
        ]
        ctx.current_topic = TopicSnapshot(
            topic_id="t2",
            topic_label="通信故障",
            start_turn=3,
            end_turn=3,
            summary="",
            key_entities=[],
        )

        data = ctx.to_dict()
        restored = ConversationContext.from_dict(data)
        assert restored.session_id == "sess-1"
        assert restored.total_turns == 3
        assert restored.hot_messages == ctx.hot_messages
        assert len(restored.warm_summaries) == 1
        assert restored.warm_summaries[0].topic_label == "过温故障"
        assert restored.current_topic.topic_label == "通信故障"

    def test_from_dict_minimal(self):
        """最小数据反序列化"""
        ctx = ConversationContext.from_dict({})
        assert ctx.session_id == ""
        assert ctx.hot_messages == []
        assert ctx.total_turns == 0

    def test_archived_topic_count(self):
        ctx = ConversationContext.create_new("sess-1")
        ctx.archived_topic_count = 5
        data = ctx.to_dict()
        restored = ConversationContext.from_dict(data)
        assert restored.archived_topic_count == 5