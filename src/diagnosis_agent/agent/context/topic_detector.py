"""话题检测器 — 两阶段话题切换检测

阶段1: Embedding 快筛 (毫秒级)
    ├── 相似度 ≥ 0.8 → 同一话题，跳过 LLM
    ├── 相似度 ≤ 0.4 → 话题切换，跳过 LLM
    └── 0.4 < 相似度 < 0.8 → 进入阶段2

阶段2: LLM 精判 (200-500ms)
    └── 返回 {decision, confidence, new_topic_label, is_in_scope}

融合 InputRouter 的 out_of_scope 判断：
    话题切换时，同时判断新话题是否在电驱领域内。
    如果不在，is_in_scope=false，上层决定是否继续。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from ...config import get_settings
from ...utils.llm_factory import create_llm, create_embedding
from ...prompts.topic_detector import TOPIC_JUDGE_PROMPT
from .types import ConversationContext, TopicSnapshot

logger = logging.getLogger(__name__)

# 阶段1：Embedding 相似度阈值
THRESHOLD_SAME = 0.7       # ≥ 此值 → 同一话题
THRESHOLD_DIFFERENT = 0.3  # ≤ 此值 → 话题切换

# 阶段2：LLM 精判 Prompt
# Prompt 已提取到 prompts/topic_detector.py


@dataclass
class TopicDecision:
    """话题检测结果"""
    decision: str = "same"               # "same" | "different"
    confidence: float = 1.0              # 置信度
    new_topic_label: str = ""            # 新话题标签
    is_in_scope: bool = True             # 是否在电驱范围内
    method: str = "embedding"            # 检测方法: embedding | llm | fallback


class TopicDetector:
    """两阶段话题检测器

    Args:
        strategy: 检测策略 — "embedding+llm" | "embedding" | "llm" | "rule"
        similarity_threshold: 阶段1 高阈值，默认 0.8
    """

    def __init__(
        self,
        strategy: str = "embedding+llm",
        similarity_threshold: float = 0.8,
    ):
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self._embedding = None
        self._last_query: str = ""
        self._last_embedding: list[float] | None = None

    def detect(
        self, ctx: ConversationContext, query: str
    ) -> TopicDecision:
        """检测话题是否切换

        Args:
            ctx: 当前会话上下文
            query: 用户当前问题

        Returns:
            TopicDecision 对象
        """
        # 首轮或无历史：直接返回 same
        if not ctx or ctx.total_turns == 0 or not ctx.hot_messages:
            return TopicDecision()

        # 获取上一轮用户消息
        prev_query = self._get_prev_query(ctx)
        if not prev_query:
            return TopicDecision()

        # 阶段1: Embedding 快筛
        if self.strategy in ("embedding", "embedding+llm"):
            decision = self._fast_check(prev_query, query)
            if decision is not None:
                return decision

        # 阶段2: LLM 精判
        if self.strategy in ("llm", "embedding+llm"):
            decision = self._llm_judge(ctx, query)
            if decision is not None:
                return decision

        # 回退
        return TopicDecision(method="fallback")

    # ------------------------------------------------------------------
    # 阶段1: Embedding 快筛
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

        if similarity >= THRESHOLD_SAME:
            return TopicDecision(
                decision="same",
                confidence=similarity,
                is_in_scope=True,
                method="embedding",
            )

        if similarity <= THRESHOLD_DIFFERENT:
            return TopicDecision(
                decision="different",
                confidence=1.0 - similarity,
                is_in_scope=True,  # 需要 LLM 再确认
                method="embedding",
            )

        # 模糊区间 → 进入阶段2
        return None

    # ------------------------------------------------------------------
    # 阶段2: LLM 精判
    # ------------------------------------------------------------------

    def _llm_judge(self, ctx: ConversationContext, query: str) -> Optional[TopicDecision]:
        """LLM 精判

        输入上一轮摘要 + 当前 query，输出话题决策。
        同时判断是否在电驱范围内（融合 InputRouter out_of_scope）。
        """
        previous_summary = self._build_previous_summary(ctx)
        if not previous_summary:
            return None

        try:
            llm = create_llm(
                model="qwen-turbo",
                temperature=0.1,
                max_tokens=256,
            )
            prompt = TOPIC_JUDGE_PROMPT.format(
                previous_summary=previous_summary[:2000],
                current_query=query[:500],
            )
            response = llm.invoke(prompt)
            data = self._parse_json(response.content)
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
            # embed_query 返回单个向量
            result = self._embedding.embed_query(text)
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
        """从热层中提取上一轮用户查询"""
        for msg in reversed(ctx.hot_messages):
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            if role == "user":
                content = msg.get("data", {}).get("content", "") if isinstance(msg, dict) else ""
                return str(content) if content else ""
            # 兼容旧格式
            if isinstance(msg, dict) and msg.get("type") == "human":
                content = msg.get("data", {}).get("content", "")
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

