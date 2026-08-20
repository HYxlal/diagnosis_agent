"""图写入器

封装 Neo4j 知识写入流程：实体去重 → 节点/关系转换 → 写入 Neo4j。
依赖 ManualEditManager 保护人工编辑节点。

实体类型 → Neo4j 标签映射（与 retrieval/cypher_builder.py 对齐）：
  现象 → Fault
  根因 → RootCause
  对策 → Solution
  故障DTC → DTC
  电驱代号 → MotorType
  车辆类型 → VehicleType
  仪表指示灯 → Indicator
  故障场景 → Scenario
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import (
    GraphDocument,
    Node,
    Relationship,
)

from .models import ConversationKnowledge, ExtractedEntity
from .edit_manager import ManualEditManager

logger = logging.getLogger(__name__)

# 实体类型 → Neo4j 节点标签
ENTITY_TYPE_TO_LABEL = {
    "现象": "Fault",
    "根因": "RootCause",
    "对策": "Solution",
    "故障DTC": "DTC",
    "电驱代号": "MotorType",
    "车辆类型": "VehicleType",
    "仪表指示灯": "Indicator",
    "故障场景": "Scenario",
}

# 所有可查询的标签列表（用于去重查询）
_ALL_LABELS = list(ENTITY_TYPE_TO_LABEL.values())


class GraphWriter:
    """图写入器

    封装从 ConversationKnowledge 到 Neo4j 的完整写入流程：
    1. 实体去重（查询已存在实体）
    2. 保护人工编辑节点（跳过已手动编辑的实体）
    3. 构建 GraphDocument（实体→Node，关系→Relationship）
    4. 写入 Neo4j + 标记人工编辑

    与 ManualEditManager 协作，确保人工编辑的节点不被自动提取覆盖。
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        edit_manager: ManualEditManager,
        base_entity_label: bool = False,
        include_source: bool = True,
    ):
        self._graph = graph
        self._edit_manager = edit_manager
        self._base_entity_label = base_entity_label
        self._include_source = include_source

    def write(self, knowledge: ConversationKnowledge) -> tuple[int, int]:
        """将审核通过的知识写入 Neo4j

        Returns:
            (写入节点数, 合并实体数)
        """
        if not knowledge.extracted_entities and not knowledge.extracted_relationships:
            return 0, 0

        # 1. 实体去重 + 保护手动编辑节点
        entity_nodes, merged_count, skipped_count = self._build_entity_nodes(knowledge)

        if not entity_nodes:
            logger.info("无新建实体节点（全部已存在且受保护），跳过写入")
            return 0, merged_count

        # 2. 构建关系
        graph_relationships = self._build_relationships(
            knowledge, entity_nodes
        )

        # 3. 构建 GraphDocument
        graph_doc = GraphDocument(
            nodes=list(entity_nodes.values()),
            relationships=graph_relationships,
        )

        # 4. 写入 Neo4j
        self._graph.add_graph_documents(
            [graph_doc],
            baseEntityLabel=self._base_entity_label,
            include_source=self._include_source,
        )

        # 5. 标记人工编辑
        for node_id_str in entity_nodes:
            self._edit_manager.mark_manual_edit(
                entity_id=node_id_str,
                edited_by=knowledge.reviewer or "system",
                edit_comment=f"从对话中提取的知识: {knowledge.conversation_context[:100]}...",
            )

        node_count = len(entity_nodes)
        logger.info(
            f"GraphWriter 写入: {node_count} 个节点, "
            f"{len(graph_relationships)} 条关系, "
            f"合并 {merged_count} 个已存在实体, "
            f"跳过 {skipped_count} 个受保护节点"
        )
        return node_count, merged_count

    # ------------------------------------------------------------------
    # 实体节点构建
    # ------------------------------------------------------------------

    def _build_entity_nodes(
        self, knowledge: ConversationKnowledge
    ) -> tuple[dict[str, Node], int, int]:
        """构建实体节点，处理去重和手动编辑保护

        Returns:
            (entity_nodes, merged_count, skipped_count)
        """
        entity_nodes: dict[str, Node] = {}
        merged_count = 0
        skipped_count = 0

        for ent in knowledge.extracted_entities:
            existing_id = self._find_existing_entity(ent.entity_name)

            if existing_id:
                merged_count += 1
                # 检查是否被人工编辑过，保护手动编辑节点
                if self._edit_manager.is_manually_edited(existing_id):
                    logger.info(
                        f"实体 {ent.entity_name} (ID={existing_id}) 已被人工编辑，跳过覆盖"
                    )
                    skipped_count += 1
                    continue

            node_id = existing_id or ent.entity_name
            label = ENTITY_TYPE_TO_LABEL.get(ent.entity_type, "__Entity__")
            entity_nodes[ent.entity_name] = Node(
                id=node_id,
                type=label,
                properties={
                    "name": ent.entity_name,
                    "entity_type": ent.entity_type,
                    "description": ent.description,
                    "source": "conversation",
                    "conversation_id": knowledge.conversation_id,
                    "knowledge_id": knowledge.knowledge_id,
                },
            )

        return entity_nodes, merged_count, skipped_count

    # ------------------------------------------------------------------
    # 关系构建
    # ------------------------------------------------------------------

    def _build_relationships(
        self,
        knowledge: ConversationKnowledge,
        entity_nodes: dict[str, Node],
    ) -> list[Relationship]:
        """构建关系列表"""
        graph_relationships: list[Relationship] = []

        for rel in knowledge.extracted_relationships:
            source_node = entity_nodes.get(rel.source_id)
            if not source_node:
                continue

            target_node = entity_nodes.get(rel.target_id)
            if not target_node:
                # 跨知识实体：在图中查找，用已有标签或映射
                existing_id = self._find_existing_entity(rel.target_id) or rel.target_id
                target_label = self._find_existing_label(rel.target_id) or "__Entity__"
                target_node = Node(id=existing_id, type=target_label, properties={})

            graph_relationships.append(Relationship(
                source=source_node,
                target=target_node,
                type=rel.relation_type,
                properties={
                    "description": rel.description,
                    "weight": rel.weight,
                    "source": "conversation",
                    "knowledge_id": knowledge.knowledge_id,
                },
            ))

        return graph_relationships

    # ------------------------------------------------------------------
    # 实体去重查询
    # ------------------------------------------------------------------

    def _find_existing_entity(self, entity_name: str) -> Optional[str]:
        """在 Neo4j 中查询已存在的同名实体

        匹配策略（从严到宽）：
        1. 精确 ID 匹配
        2. 精确 name 匹配
        3. 不包含 description CONTAINS（太宽泛，容易误匹配）
        """
        labels_union = " OR ".join(f"e:{label}" for label in _ALL_LABELS)
        query = f"""
        MATCH (e)
        WHERE ({labels_union})
          AND (e.id = $entity_name OR e.name = $entity_name)
        RETURN e.id AS entity_id LIMIT 1
        """
        try:
            result = self._graph.query(query, params={"entity_name": entity_name})
            if result and result[0].get("entity_id"):
                return result[0]["entity_id"]
        except Exception:
            logger.warning(f"实体去重查询失败: {entity_name}", exc_info=True)
        return None

    def _find_existing_label(self, entity_name: str) -> Optional[str]:
        """查询已存在实体的标签名"""
        labels_union = " OR ".join(f"e:{label}" for label in _ALL_LABELS)
        query = f"""
        MATCH (e)
        WHERE ({labels_union})
          AND (e.id = $entity_name OR e.name = $entity_name)
        RETURN [l IN labels(e) WHERE l <> '__Entity__'] AS labels
        LIMIT 1
        """
        try:
            result = self._graph.query(query, params={"entity_name": entity_name})
            if result and result[0].get("labels"):
                return result[0]["labels"][0]
        except Exception:
            pass
        return None