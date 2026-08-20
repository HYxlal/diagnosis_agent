"""知识沉淀核心提取器

实现对话知识提取、审核队列管理、Neo4j 写入和持久化恢复。
与 GraphWriter（图写入）和 ManualEditManager（编辑保护）解耦协作。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from ..utils.llm_factory import create_llm
from .models import (
    ConversationKnowledge,
    ExtractedEntity,
    ExtractedRelationship,
    KnowledgeStats,
)
from ..prompts.knowledge_extraction import KNOWLEDGE_EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)

# 实体和关系类型约束
ENTITY_TYPES = ["现象", "根因", "对策", "电驱代号", "车辆类型", "仪表指示灯", "故障DTC", "故障场景"]
RELATIONSHIP_TYPES = ["由...引起", "导致", "对应对策", "适用于", "发生于", "关联DTC", "亮起", "配备", "出现于", "排除", "关联", "互斥", "并存"]

# 默认分隔符
_DEFAULT_TUPLE = ":"
_DEFAULT_RECORD = ")"
_DEFAULT_COMPLETION = "返回空列表"


class ConversationKnowledgeExtractor:
    """对话知识提取与审核管理器

    从对话中自动提取实体关系，管理审核队列，写入 Neo4j 知识图谱。
    图写入和编辑保护分别委托给 GraphWriter 和 ManualEditManager。
    """

    def __init__(
        self,
        llm=None,
        graph_writer=None,
        edit_manager=None,
        persistence_path: str = None,
        tuple_delimiter: str = None,
        record_delimiter: str = None,
        completion_delimiter: str = None,
    ):
        self._llm = llm or create_llm(temperature=0.0, max_tokens=2048)
        self._graph_writer = graph_writer
        self._edit_manager = edit_manager
        self._persistence_path = Path(
            persistence_path or "cache/conversation_knowledge.json"
        )
        self._pending_reviews: list[ConversationKnowledge] = []
        self._processed_knowledge: dict[str, ConversationKnowledge] = {}
        self._stats = KnowledgeStats()

        # 分隔符配置（可配置化，不同项目可自定义）
        self._tuple_delimiter = tuple_delimiter or _DEFAULT_TUPLE
        self._record_delimiter = record_delimiter or _DEFAULT_RECORD
        self._completion_delimiter = completion_delimiter or _DEFAULT_COMPLETION

        # 动态编译正则（基于分隔符，自动转义正则特殊字符）
        self._entity_pattern = self._build_entity_pattern()
        self._relationship_pattern = self._build_relationship_pattern()

        self._load_from_file()

    def _build_entity_pattern(self) -> re.Pattern:
        """基于 tuple_delimiter 和 record_delimiter 动态构建实体正则"""
        td = re.escape(self._tuple_delimiter)
        rd = re.escape(self._record_delimiter)
        return re.compile(
            rf'\("entity"\s*{td}\s*"([^"]*)"\s*{td}\s*"([^"]*)"\s*{td}\s*"([^"]*)"{rd}'
        )

    def _build_relationship_pattern(self) -> re.Pattern:
        """基于 tuple_delimiter 和 record_delimiter 动态构建关系正则"""
        td = re.escape(self._tuple_delimiter)
        rd = re.escape(self._record_delimiter)
        return re.compile(
            rf'\("relationship"\s*{td}\s*"([^"]*)"\s*{td}\s*"([^"]*)"\s*{td}\s*"([^"]*)"\s*{td}\s*"([^"]*)"\s*{td}\s*([\d.]+){rd}'
        )

    # ========================================================================
    # FR-1.1：对话格式化
    # ========================================================================

    def format_conversation(self, messages: list[dict]) -> str:
        """将消息列表转为 LLM 可读的文本格式"""
        lines = []
        for msg in messages:
            role = msg.get("role") or msg.get("type", "unknown")
            content = msg.get("content") or (msg.get("data", {}).get("content", ""))
            if role in ("user", "human"):
                lines.append(f"用户: {content}")
            elif role in ("assistant", "ai"):
                lines.append(f"助手: {content}")
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ========================================================================
    # FR-1.2：LLM 实体关系提取
    # ========================================================================

    def extract_from_conversation(
        self, messages: list[dict], conversation_id: str = None
    ) -> ConversationKnowledge:
        """1. 格式化 → 2. LLM 提取 → 3. 正则解析 → 4. 封装"""
        conversation_text = self.format_conversation(messages)

        prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_EXTRACTION_SYSTEM),
            ("human", "对话内容：\n{conversation}\n\n请提取实体和关系："),
        ])

        try:
            messages_list = prompt.format_messages(
                conversation=conversation_text,
                entity_types=", ".join(ENTITY_TYPES),
                relationship_types=", ".join(RELATIONSHIP_TYPES),
                tuple_delimiter=self._tuple_delimiter,
                record_delimiter=self._record_delimiter,
                completion_delimiter=self._completion_delimiter,
            )
            response = self._llm.invoke(messages_list)
            raw_text = response.content if hasattr(response, "content") else str(response)
            raw_text = raw_text or ""
        except Exception as e:
            logger.error(f"LLM 知识提取失败: {e}")
            self._stats.errors += 1
            return ConversationKnowledge(
                review_comment=f"LLM 提取失败: {e}",
            )

        entities = self._parse_entities(raw_text)
        relationships = self._parse_relationships(raw_text)

        knowledge = ConversationKnowledge(
            conversation_id=conversation_id or "",
            extracted_entities=entities,
            extracted_relationships=relationships,
            conversation_context=conversation_text,
        )
        self._stats.extractions += 1
        self._stats.entities_extracted += len(entities)
        self._stats.relationships_extracted += len(relationships)
        return knowledge

    # ========================================================================
    # FR-1.3：正则解析（基于可配置分隔符）
    # ========================================================================

    def _parse_entities(self, text: str) -> list[ExtractedEntity]:
        entities = []
        for m in self._entity_pattern.finditer(text):
            entities.append(ExtractedEntity(
                entity_name=m.group(1),
                entity_type=m.group(2),
                description=m.group(3),
            ))
        return entities

    def _parse_relationships(self, text: str) -> list[ExtractedRelationship]:
        rels = []
        for m in self._relationship_pattern.finditer(text):
            try:
                rels.append(ExtractedRelationship(
                    source_id=m.group(1),
                    target_id=m.group(2),
                    relation_type=m.group(3),
                    description=m.group(4),
                    weight=float(m.group(5)),
                ))
            except ValueError:
                pass
        return rels

    # ========================================================================
    # FR-4：一站式提取提交
    # ========================================================================

    def extract_and_submit(
        self, messages: list[dict], conversation_id: str = None
    ) -> str:
        """提取 → 提交审核，返回 knowledge_id 或空字符串"""
        knowledge = self.extract_from_conversation(messages, conversation_id)
        if knowledge.extracted_entities or knowledge.extracted_relationships:
            return self.submit_for_review(knowledge)
        return ""

    # ========================================================================
    # FR-2：审核队列管理
    # ========================================================================

    def submit_for_review(self, knowledge: ConversationKnowledge) -> str:
        """提交待审核，自动持久化"""
        knowledge.status = "pending"
        self._pending_reviews.append(knowledge)
        self._processed_knowledge[knowledge.knowledge_id] = knowledge
        self._stats.submitted_for_review += 1
        self._save_to_file()
        logger.info(f"知识已提交审核: {knowledge.knowledge_id}")
        return knowledge.knowledge_id

    def get_pending_reviews(self) -> list[ConversationKnowledge]:
        return sorted(
            [k for k in self._pending_reviews if k.status == "pending"],
            key=lambda k: k.created_at,
        )

    def review_knowledge(
        self, knowledge_id: str, approved: bool, reviewer: str, comment: str = None
    ) -> bool:
        """审核通过/拒绝"""
        knowledge = self._processed_knowledge.get(knowledge_id)
        if not knowledge:
            return False
        knowledge.status = "approved" if approved else "rejected"
        knowledge.reviewer = reviewer
        knowledge.review_time = datetime.now().isoformat()
        knowledge.review_comment = comment
        if approved:
            self._stats.approved_knowledge += 1
        else:
            self._stats.rejected_knowledge += 1
        self._pending_reviews = [
            k for k in self._pending_reviews if k.knowledge_id != knowledge_id
        ]
        self._save_to_file()
        logger.info(
            f"审核{'通过' if approved else '拒绝'}: {knowledge_id} by {reviewer}"
        )
        return True

    def clear_pending_reviews(self) -> None:
        self._pending_reviews.clear()
        self._save_to_file()

    # ========================================================================
    # FR-3：知识写入 Neo4j
    # ========================================================================

    def write_approved_knowledge(self, knowledge_id: str) -> bool:
        """将审核通过的知识写入 Neo4j（委托给 GraphWriter）"""
        knowledge = self._processed_knowledge.get(knowledge_id)
        if not knowledge or knowledge.status != "approved":
            logger.warning(f"知识 {knowledge_id} 未审核通过，无法写入")
            return False

        if not knowledge.extracted_entities and not knowledge.extracted_relationships:
            return False

        if self._graph_writer is None:
            logger.warning("GraphWriter 不可用，跳过知识写入")
            return False

        try:
            node_count, merged_count = self._graph_writer.write(knowledge)
            if node_count > 0 or merged_count > 0:
                self._stats.merged_entities += merged_count
                logger.info(
                    f"知识已写入 Neo4j: {knowledge_id}, "
                    f"写入 {node_count} 节点, 合并 {merged_count} 实体"
                )
                return True
            logger.info(f"知识 {knowledge_id} 无新建实体，写入跳过")
            return True
        except Exception as e:
            logger.error(f"Neo4j 写入失败: {e}")
            self._stats.errors += 1
            return False

    # ========================================================================
    # FR-5：持久化与恢复
    # ========================================================================

    def _save_to_file(self) -> None:
        """持久化到 JSON 文件"""
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pending_reviews": [k.model_dump() for k in self._pending_reviews],
            "processed_knowledge": {
                k: v.model_dump()
                for k, v in self._processed_knowledge.items()
            },
            "stats": self._stats.model_dump(),
            "saved_at": datetime.now().isoformat(),
        }
        try:
            self._persistence_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"知识持久化失败: {e}")

    def _load_from_file(self) -> None:
        """启动恢复"""
        if not self._persistence_path.exists():
            return
        try:
            data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
            self._pending_reviews = [
                ConversationKnowledge(**k)
                for k in data.get("pending_reviews", [])
            ]
            self._processed_knowledge = {
                k: ConversationKnowledge(**v)
                for k, v in data.get("processed_knowledge", {}).items()
            }
            # 合并恢复统计（而非覆盖），与文档要求一致
            saved_stats = data.get("stats", {})
            if saved_stats:
                self._stats = KnowledgeStats(
                    extractions=max(self._stats.extractions, saved_stats.get("extractions", 0)),
                    entities_extracted=max(self._stats.entities_extracted, saved_stats.get("entities_extracted", 0)),
                    relationships_extracted=max(self._stats.relationships_extracted, saved_stats.get("relationships_extracted", 0)),
                    submitted_for_review=max(self._stats.submitted_for_review, saved_stats.get("submitted_for_review", 0)),
                    approved_knowledge=max(self._stats.approved_knowledge, saved_stats.get("approved_knowledge", 0)),
                    rejected_knowledge=max(self._stats.rejected_knowledge, saved_stats.get("rejected_knowledge", 0)),
                    errors=max(self._stats.errors, saved_stats.get("errors", 0)),
                    merged_entities=max(self._stats.merged_entities, saved_stats.get("merged_entities", 0)),
                )
            logger.info(
                f"知识恢复完成: {len(self._pending_reviews)} 条待审核, "
                f"{self._stats.extractions} 次提取"
            )
        except Exception as e:
            logger.error(f"知识恢复失败: {e}")

    # ========================================================================
    # FR-6：统计信息
    # ========================================================================

    def get_knowledge_stats(self) -> KnowledgeStats:
        return self._stats

    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[ConversationKnowledge]:
        return self._processed_knowledge.get(knowledge_id)

    def display_pending_reviews(self) -> None:
        """打印待审核列表"""
        pending = self.get_pending_reviews()
        if not pending:
            print("暂无待审核的知识")
            return
        for i, k in enumerate(pending, 1):
            print(f"\n{'='*40}")
            print(f"序号: {i}")
            print(f"知识ID: {k.knowledge_id}")
            print(f"对话ID: {k.conversation_id}")
            print(f"实体数: {len(k.extracted_entities)}")
            print(f"关系数: {len(k.extracted_relationships)}")
            print(f"创建时间: {k.created_at}")
            if k.extracted_entities:
                print(f"实体: {', '.join(e.entity_name for e in k.extracted_entities[:3])}")
            if k.extracted_relationships:
                print(f"关系: {', '.join(f'{r.source_id}→{r.relation_type}→{r.target_id}' for r in k.extracted_relationships[:3])}")
            print(f"对话摘要: {k.conversation_context[:100]}...")