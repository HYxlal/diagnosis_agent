"""知识沉淀数据模型

与飞书需求文档 5.1-5.5 节完全对齐。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """LLM 提取的实体（FR-5.1）"""

    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_name: str
    entity_type: str
    description: str


class ExtractedRelationship(BaseModel):
    """LLM 提取的关系（FR-5.2）"""

    source_id: str  # 源实体名称
    target_id: str  # 目标实体名称
    relation_type: str
    description: str
    weight: float = 0.5


class ConversationKnowledge(BaseModel):
    """对话知识（FR-5.3）"""

    knowledge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)
    extracted_relationships: list[ExtractedRelationship] = Field(default_factory=list)
    conversation_context: str = ""
    status: str = "pending"  # pending / approved / rejected
    reviewer: Optional[str] = None
    review_time: Optional[str] = None
    review_comment: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeStats(BaseModel):
    """统计信息（FR-6）"""

    extractions: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    submitted_for_review: int = 0
    approved_knowledge: int = 0
    rejected_knowledge: int = 0
    errors: int = 0
    merged_entities: int = 0