"""话题检测器 — 多级话题切换检测

层级:
  L0: 信号词规则 (零成本, 零延迟)
        ├── 新话题信号词 → 直接判 different
        └── 延续信号词   → 辅助判 same (暂未启用)
  L1: 实体重叠检测 (毫秒级)
        ├── key_entities 与 query 字符串匹配
        └── 重叠率 > 50% → 判 same
  L2: Embedding 快筛 (毫秒级)
        ├── 相似度 ≥ high → 判 same
        ├── 相似度 ≤ low  → 判 different
        └── 模糊区间 → 进入 L3
  L3: LLM 精判 (200-500ms)
        └── 返回 {decision, confidence, new_topic_label, is_in_scope}

信号词和实体重叠均外部化配置在 config.yaml 中。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from ...config import get_settings
from ...utils.llm_factory import create_llm, create_embedding
from ...prompts.topic_detector import TOPIC_JUDGE_PROMPT, SCOPE_CHECK_PROMPT
from .types import ConversationContext, TopicSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TopicDecision:
    """话题检测结果"""
    decision: str = "same"               # "same" | "different"
    confidence: float = 1.0              # 置信度
    new_topic_label: str = ""            # 新话题标签
    is_in_scope: bool = True             # 是否在电驱范围内
    method: str = "embedding"            # 检测方法: signal_word | entity_overlap | embedding | llm | fallback


class TopicDetector:
    """多级话题检测器

    Args:
        strategy: 检测策略 — "embedding+llm" | "embedding" | "llm" | "rule"
        threshold_high: L2 高阈值（≥此值判 same），默认 0.6
        threshold_low: L2 低阈值（≤此值判 diff），默认 0.27
        model: LLM 模型名，默认跟随 settings.llm.model
        switch_signal_words: 新话题信号词列表
        continue_signal_words: 延续信号词列表（暂未启用）
        entity_overlap_enabled: 是否启用实体重叠检测
    """

    def __init__(
        self,
        strategy: str = "embedding+llm",
        threshold_high: float = 0.6,
        threshold_low: float = 0.27,
        model: str | None = None,
        switch_signal_words: list[str] | None = None,
        continue_signal_words: list[str] | None = None,
        entity_overlap_enabled: bool = False,
        time_decay_short_sec: int = 30,
        time_decay_short_max_len: int = 20,
        time_decay_long_sec: int = 1800,
        scope_detection_enabled: bool = False,
        scope_out_keywords: list[str] | None = None,
        scope_use_llm: bool = True,
    ):
        self.strategy = strategy
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.model = model
        self.switch_signal_words = switch_signal_words or []
        self.continue_signal_words = continue_signal_words or []
        self.entity_overlap_enabled = entity_overlap_enabled
        self.time_decay_short_sec = time_decay_short_sec
        self.time_decay_short_max_len = time_decay_short_max_len
        self.time_decay_long_sec = time_decay_long_sec
        self.scope_detection_enabled = scope_detection_enabled
        self.scope_out_keywords = scope_out_keywords or []
        self.scope_use_llm = scope_use_llm
        self._embedding = None
        self._last_query: str = ""
        self._last_embedding: list[float] | None = None

    def detect(
        self, ctx: ConversationContext, query: str
    ) -> TopicDecision:
        """检测话题是否切换（多级检测）

        Args:
            ctx: 当前会话上下文
            query: 用户当前问题

        Returns:
            TopicDecision 对象
        """
        # 首轮或无历史：scope 检查 + 返回 same
        if not ctx or ctx.total_turns == 0 or not ctx.hot_messages:
            if self.scope_detection_enabled:
                in_scope = self._scope_check(query)
                if in_scope is False:
                    return TopicDecision(
                        decision="different",
                        is_in_scope=False,
                        method="scope",
                    )
            return TopicDecision()

        # 获取上一轮用户消息
        prev_query = self._get_prev_query(ctx)
        if not prev_query:
            return TopicDecision()

        # 逐级检测，每层返回前统一做 scope 检查
        decision = self._detect_layers(ctx, prev_query, query)
        return self._apply_scope_check(decision, query)

    def _detect_layers(self, ctx: ConversationContext, prev_query: str, query: str) -> TopicDecision:
        """执行 L0 ~ L3 的多级话题检测"""
        # L0: 信号词
        if self.switch_signal_words:
            decision = self._signal_word_check(query)
            if decision is not None:
                return decision

        # L0.5: 时间衰减
        if self.time_decay_short_sec > 0:
            decision = self._time_decay_check(ctx, query)
            if decision is not None:
                return decision

        # L1: 实体重叠
        if self.entity_overlap_enabled:
            decision = self._entity_overlap_check(ctx, query)
            if decision is not None:
                return decision

        # L2: Embedding 快筛
        if self.strategy in ("embedding", "embedding+llm"):
            decision = self._fast_check(prev_query, query)
            if decision is not None:
                return decision

        # L3: LLM 精判
        if self.strategy in ("llm", "embedding+llm"):
            decision = self._llm_judge(ctx, query)
            if decision is not None:
                return decision

        return TopicDecision(method="fallback")

    # ------------------------------------------------------------------
    # L0: 信号词检测
    # ------------------------------------------------------------------

    def _signal_word_check(self, query: str) -> Optional[TopicDecision]:
        """信号词检测

        新话题信号词（switch）→ 用户明确表示切换话题 → 直接判 different。
        延续信号词（continue）→ 暂不启用。
        """
        for word in self.switch_signal_words:
            if word in query:
                logger.info(
                    f"信号词检测: different (命中新话题信号词: '{word}')"
                )
                return TopicDecision(
                    decision="different",
                    confidence=0.85,
                    new_topic_label=query[:30],
                    is_in_scope=True,
                    method="signal_word",
                )

        return None

    # ------------------------------------------------------------------
    # L0.5: 时间衰减检测
    # ------------------------------------------------------------------

    def _time_decay_check(
        self, ctx: ConversationContext, query: str
    ) -> Optional[TopicDecision]:
        """时间衰减检测

        短间隔（<short_sec）+ 短消息（<short_max_len）→ 大概率追问 → 判 same。
        长间隔（>long_sec）→ 用户可能已离开很久 → 判 different。

        阈值外部化配置在 config.yaml。
        """
        if not ctx.last_activity_at:
            return None

        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            last = datetime.fromisoformat(ctx.last_activity_at)
            elapsed = (now - last).total_seconds()
        except (ValueError, OSError):
            return None

        # 短间隔 + 短消息 → 快速追问，延续
        if elapsed < self.time_decay_short_sec and len(query) < self.time_decay_short_max_len:
            logger.info(
                f"时间衰减: same (elapsed={elapsed:.0f}s < {self.time_decay_short_sec}s, "
                f"msg_len={len(query)} < {self.time_decay_short_max_len})"
            )
            return TopicDecision(
                decision="same",
                confidence=0.8,
                is_in_scope=True,
                method="time_decay",
            )

        # 长间隔 → 长时间未活动，可能切换话题
        if elapsed > self.time_decay_long_sec:
            logger.info(
                f"时间衰减: different (elapsed={elapsed:.0f}s > {self.time_decay_long_sec}s)"
            )
            return TopicDecision(
                decision="different",
                confidence=0.7,
                is_in_scope=True,
                method="time_decay",
            )

        return None

    # ------------------------------------------------------------------
    # Scope 领域检测（关键词规则，零延迟）
    # ------------------------------------------------------------------

    def _scope_check(self, query: str) -> Optional[bool]:
        """Scope 检测: 关键词快筛 + LLM 精判兜底

        只在首轮和话题切换后调用。

        Returns:
            True  → 在范围内
            False → 不在范围内
            None  → 无法判断（宽松处理，判在范围内）
        """
        # 1. 关键词快筛（零成本，仅排除明显不在范围内的）
        for kw in self.scope_out_keywords:
            if kw in query:
                logger.info(f"Scope 检测: 不在范围内 (关键词: '{kw}')")
                return False

        # 2. LLM 精判兜底（不做关键词 in 判断，太粗糙）
        if self.scope_use_llm:
            result = self._scope_llm_check(query)
            if result is not None:
                return result

        # 3. LLM 失败 → 宽松处理
        return None

    def _scope_llm_check(self, query: str) -> Optional[bool]:
        """LLM 精判 scope（scope-only prompt，不涉及话题比较）"""
        model = self.model or get_settings().context.topic_detection_model or get_settings().llm.model
        try:
            llm = create_llm(
                model=model,
                temperature=0.1,
                max_tokens=64,
            )
            prompt = SCOPE_CHECK_PROMPT.format(current_query=query[:500])
            response = llm.invoke(prompt)
            data = self._parse_json(response.content)
            logger.info(f"Scope LLM 精判: {data}")
            if data and "is_in_scope" in data:
                return bool(data["is_in_scope"])
        except Exception as e:
            logger.warning(f"Scope LLM 精判失败: {e}")
        return None

    def _apply_scope_check(
        self, decision: TopicDecision, query: str
    ) -> TopicDecision:
        """检查新话题是否在领域内

        different → 完整 scope 检查（关键词 + LLM）
        same      → 仅 out 关键词快筛（零成本）
        """
        if not self.scope_detection_enabled:
            return decision
        if not decision.is_in_scope:
            return decision

        if decision.decision == "different":
            # 完整 scope 检查
            in_scope = self._scope_check(query)
            if in_scope is False:
                decision.is_in_scope = False
                decision.method = "scope"
                logger.info(f"Scope 检测: 切换后不在范围内")
        else:
            # same 场景仅做 out 关键词快筛
            for kw in self.scope_out_keywords:
                if kw in query:
                    decision.is_in_scope = False
                    decision.method = "scope"
                    logger.info(f"Scope 检测: 不在范围内 (关键词: '{kw}')")
                    break

        return decision

    # ------------------------------------------------------------------
    # L1: 实体重叠检测
    # ------------------------------------------------------------------

    def _entity_overlap_check(
        self, ctx: ConversationContext, query: str
    ) -> Optional[TopicDecision]:
        """实体重叠检测

        用 ctx.current_topic.key_entities 与 query 做字符串匹配。
        重叠率 > 50% → 判 same（用户还在讨论相同实体）。
        没有实体信息或重叠率低 → 不阻断，交给后续层级。
        """
        if not ctx.current_topic or not ctx.current_topic.key_entities:
            return None

        history_entities = ctx.current_topic.key_entities
        if not history_entities:
            return None

        # 统计历史实体在 query 中出现的次数
        matched = [e for e in history_entities if e in query]
        overlap_ratio = len(matched) / len(history_entities)

        if matched:
            logger.info(
                f"实体重叠: {len(matched)}/{len(history_entities)} = {overlap_ratio:.0%} "
                f"(匹配: {matched})"
            )

        if overlap_ratio > 0.5:
            logger.info(
                f"实体重叠检测: same (overlap={overlap_ratio:.0%})"
            )
            return TopicDecision(
                decision="same",
                confidence=overlap_ratio,
                is_in_scope=True,
                method="entity_overlap",
            )

        return None

    # ------------------------------------------------------------------
    # L2: Embedding 快筛
    # ------------------------------------------------------------------

    def _fast_check(self, prev_query: str, query: str) -> Optional[TopicDecision]:
        """Embedding 快筛

        计算上一轮 query 和当前 query 的余弦相似度。
        阈值判断后直接返回决策，不进入 LLM。
        """
        try:
            emb1 = self._get_embedding(prev_query)
            emb2 = self._get_embedding(query)
        except Exception as e:
            logger.warning(f"Embedding 快筛失败: {e}")
            return None

        if emb1 is None or emb2 is None:
            return None

        similarity = self._cosine_similarity(emb1, emb2)
        logger.info(f"话题相似度: {similarity:.4f}")

        if similarity >= self.threshold_high:
            logger.info(
                f"Embedding 快筛: same (similarity={similarity:.4f} >= high={self.threshold_high})"
            )
            return TopicDecision(
                decision="same",
                confidence=similarity,
                is_in_scope=True,
                method="embedding",
            )

        if similarity <= self.threshold_low:
            logger.info(
                f"Embedding 快筛: different (similarity={similarity:.4f} <= low={self.threshold_low})"
            )
            return TopicDecision(
                decision="different",
                confidence=1.0 - similarity,
                is_in_scope=True,
                method="embedding",
            )

        # 模糊区间 → 进入阶段2
        logger.info(
            f"Embedding 快筛: 模糊区间 (similarity={similarity:.4f}, "
            f"high={self.threshold_high}, low={self.threshold_low})，进入 LLM 精判"
        )
        return None

    # ------------------------------------------------------------------
    # L3: LLM 精判
    # ------------------------------------------------------------------

    def _llm_judge(self, ctx: ConversationContext, query: str) -> Optional[TopicDecision]:
        """LLM 精判

        输入上一轮摘要 + 当前 query，输出话题决策。
        同时判断是否在电驱范围内（融合 InputRouter out_of_scope）。
        """
        previous_summary = self._build_previous_summary(ctx)
        if not previous_summary:
            return None

        model = self.model or get_settings().context.topic_detection_model or get_settings().llm.model
        try:
            llm = create_llm(
                model=model,
                temperature=0.1,
                max_tokens=256,
            )
            prompt = TOPIC_JUDGE_PROMPT.format(
                previous_summary=previous_summary[:2000],
                current_query=query[:500],
            )
            response = llm.invoke(prompt)
            data = self._parse_json(response.content)
            logger.info(f"LLM 话题精判结果: {data}")
            if not data:
                return None

            decision_str = data.get("decision", "same")
            is_in_scope = data.get("is_in_scope", True)
            confidence = float(data.get("confidence", 0.5))

            # 如果不在电驱范围内，统一标记为话题切换
            if not is_in_scope:
                return TopicDecision(
                    decision="different",
                    confidence=confidence,
                    new_topic_label= data.get("new_topic_label", ""),
                    is_in_scope=False,
                    method="llm",
                )

            if decision_str == "different":
                return TopicDecision(
                    decision="different",
                    confidence=confidence,
                    new_topic_label=data.get("new_topic_label", ""),
                    is_in_scope=True,
                    method="llm",
                )
            else:
                return TopicDecision(
                    decision="same",
                    confidence=confidence,
                    is_in_scope=True,
                    method="llm",
                )

        except Exception as e:
            logger.warning(f"LLM 话题精判失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """获取文本的 embedding 向量"""
        if self._embedding is None:
            self._embedding = create_embedding()
        try:
            result = self._embedding.embed_query(text)
            logger.debug(f"Embedding 计算成功: text_len={len(text)}, emb_dim={len(result) if result else 0}")
            return result
        except Exception as e:
            logger.warning(f"Embedding 计算失败: {e}")
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_prev_query(self, ctx: ConversationContext) -> str:
        """从热层中提取上一轮用户查询

        支持两种序列化格式：
        - { "role": "user", "content": "..." }
        - { "type": "human", "data": { "content": "..." } }
        """
        for msg in reversed(ctx.hot_messages):
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            msg_type = msg.get("type", "")

            if role == "user" or msg_type == "human":
                content = msg.get("content", "")
                if not content and "data" in msg:
                    content = msg["data"].get("content", "")
                return str(content) if content else ""

        return ""

    def _build_previous_summary(self, ctx: ConversationContext) -> str:
        """构建上一轮的摘要（用于 LLM 精判）"""
        parts = []

        # 当前话题摘要
        if ctx.current_topic:
            parts.append(f"当前话题: {ctx.current_topic.topic_label}")
            parts.append(f"摘要: {ctx.current_topic.summary}")
            if ctx.current_topic.key_entities:
                parts.append(f"关键实体: {', '.join(ctx.current_topic.key_entities)}")

        # 最近一轮用户消息
        prev_query = self._get_prev_query(ctx)
        if prev_query:
            parts.append(f"上一轮用户问题: {prev_query}")

        # 温层摘要（最近一条）
        if ctx.warm_summaries:
            latest = ctx.warm_summaries[-1]
            parts.append(f"历史话题: {latest.topic_label} - {latest.summary}")

        return "\n".join(parts) if parts else ""

    def _parse_json(self, content: str) -> Optional[dict]:
        """解析 LLM 返回的 JSON"""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # 尝试从代码块提取
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return None