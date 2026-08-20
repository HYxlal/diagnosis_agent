"""手动编辑管理器

保护人工编辑过的 Neo4j 节点不被自动提取覆盖。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


class ManualEditManager:
    """手动编辑标记管理器

    职责：
    - 标记节点为人工编辑（写入后调用）
    - 查询节点的编辑状态（写入前检查）
    - 保护人工编辑节点不被自动提取覆盖

    使用 Neo4j 节点属性 manual_edit 作为标记字段。
    """

    def __init__(self, graph: Neo4jGraph):
        self._graph = graph

    def mark_manual_edit(
        self,
        entity_id: str,
        edited_by: str = "system",
        edit_comment: str = "",
    ) -> bool:
        """标记实体节点为人工编辑

        在 Neo4j 中设置节点属性：manual_edit=True, edited_by, edit_time, edit_comment。
        """
        query = """
        MATCH (e {id: $entity_id})
        SET e.manual_edit = true,
            e.edited_by = $edited_by,
            e.edit_time = $edit_time,
            e.edit_comment = $edit_comment
        RETURN e.id AS entity_id
        """
        try:
            result = self._graph.query(
                query,
                params={
                    "entity_id": entity_id,
                    "edited_by": edited_by,
                    "edit_time": datetime.now().isoformat(),
                    "edit_comment": edit_comment,
                },
            )
            if result and result[0].get("entity_id"):
                logger.info(f"已标记人工编辑: {entity_id}")
                return True
            logger.warning(f"标记人工编辑失败，实体不存在: {entity_id}")
            return False
        except Exception as e:
            logger.error(f"标记人工编辑异常: {e}")
            return False

    def is_manually_edited(self, entity_id: str) -> bool:
        """检查实体节点是否已被人工编辑"""
        query = """
        MATCH (e {id: $entity_id})
        WHERE e.manual_edit = true
        RETURN e.id AS entity_id LIMIT 1
        """
        try:
            result = self._graph.query(query, params={"entity_id": entity_id})
            return bool(result and result[0].get("entity_id"))
        except Exception as e:
            logger.warning(f"查询编辑状态异常: {e}")
            return False

    def get_edit_info(self, entity_id: str) -> Optional[dict[str, Any]]:
        """获取实体节点的编辑信息"""
        query = """
        MATCH (e {id: $entity_id})
        RETURN e.manual_edit AS manual_edit,
               e.edited_by AS edited_by,
               e.edit_time AS edit_time,
               e.edit_comment AS edit_comment
        LIMIT 1
        """
        try:
            result = self._graph.query(query, params={"entity_id": entity_id})
            if result:
                return dict(result[0])
        except Exception as e:
            logger.warning(f"查询编辑信息异常: {e}")
        return None