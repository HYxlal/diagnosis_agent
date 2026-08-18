"""Agent 可用工具定义

使用 LangChain StructuredTool 动态创建工具，支持实例方法绑定。

工具清单：
1. search_similar_incidents — 语义检索相似工单
2. filter_by_vehicle_type   — 按车型精确过滤
3. get_incident_detail      — 获取工单详情
4. convert_working_condition_file — 工况文件转换接口（预留，工具对接）
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from ..config import Settings
from ..models.incident import IncidentRecord
from ..retrieval.langchain_retrievers import ChromaVectorRetriever, document_to_record
from ..storage.vector_store import SearchResult
from ..tools.can_converter_tool import build_can_converter_tool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 输入模型定义
# ------------------------------------------------------------------

class SearchSimilarIncidentsInput(BaseModel):
    """search_similar_incidents 工具输入模型"""
    query: str = Field(description="故障描述文本（必填）")
    vehicle_type: Optional[str] = Field(default=None, description="车型（可选，用于过滤）")
    top_k: Optional[int] = Field(default=5, description="返回数量（可选，默认5）")


class GetIncidentDetailInput(BaseModel):
    """get_incident_detail 工具输入模型"""
    record_id: str = Field(description="记录ID（必填）")


class ConvertWorkingConditionFileInput(BaseModel):
    """convert_working_condition_file 工具输入模型"""
    file_path: str = Field(description="工况文件路径（必填）")


class QueryFaultGraphInput(BaseModel):
    """query_fault_graph 工具输入模型

    字段直接对齐 StandardInput.entities：
    mcuid / dtc_code / project / component / working_condition / software_version
    """
    mcuid: Optional[str] = Field(
        default=None, description="MCU 标识（可选，如 MCU_001）"
    )
    dtc_code: Optional[list[str]] = Field(
        default=None,
        description="DTC 码列表（可选，如 ['P1A3E98', 'U1624']）",
    )
    project: Optional[str] = Field(
        default=None,
        description="项目型号/车型代号（可选，如 H37A）",
    )
    component: Optional[str] = Field(
        default=None,
        description="涉及部件（可选，如 IGBT/MCU）",
    )
    working_condition: Optional[str] = Field(
        default=None,
        description="工作条件描述（可选，如 爬坡满载、低温冷启动）",
    )
    software_version: Optional[str] = Field(
        default=None,
        description="软件版本信息（可选，如 H37A3621830AW）",
    )
    depth: Optional[int] = Field(
        default=1, description="关系扩展深度（1=直接关系，2=两度关系）"
    )
    top_k: Optional[int] = Field(
        default=10, description="返回数量（可选，默认10）"
    )

    @field_validator('dtc_code', mode='before')
    @classmethod
    def _parse_dtc(cls, v: Any) -> Optional[list[str]]:
        """兼容 LLM 错误传参：将 JSON 字符串或逗号分隔字符串转为 list"""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            import json as _json
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (_json.JSONDecodeError, ValueError):
                pass
            return [s.strip() for s in v.split(",") if s.strip()]
        return None


# ------------------------------------------------------------------
# 工具集类
# ------------------------------------------------------------------

class DiagnosticTools:
    """诊断工具集

    封装为可被 LangChain Agent 调用的工具。
    使用 StructuredTool 动态创建工具，支持实例方法绑定。
    """

    def __init__(
        self,
        retriever,
        settings: Optional[Settings] = None,
    ):
        self.retriever = retriever
        self.settings = settings

        self._default_search_top_k = (
            settings.tools.search_top_k if settings else 5
        )

        self._working_condition_converter: Optional[Callable[[str], dict[str, Any]]] = None

        # 检测 Neo4j 可用性（需实际连通）
        self._neo4j_available = False
        try:
            neo4j_attr = getattr(retriever, "neo4j", None)
            if neo4j_attr is not None and neo4j_attr.available:
                # 可用性检查：发一条简单查询验证连通性
                driver = getattr(neo4j_attr, '_driver', None)
                if driver is not None:
                    with driver.session(database="neo4j") as session:
                        session.run("RETURN 1")
                        self._neo4j_available = True
        except Exception:
            logger.info("Neo4j 连接不可达，query_fault_graph 工具已移除")

    def register_working_condition_converter(self, converter: Callable[[str], dict[str, Any]]) -> None:
        """注册工况文件转换工具"""
        self._working_condition_converter = converter
        logger.info("工况文件转换工具已注册")

    # ------------------------------------------------------------------
    # 工具实现方法（内部方法）
    # ------------------------------------------------------------------

    def _search_similar_incidents_impl(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """检索相似工单实现

        优先用 HybridRetriever.semantic_search（走纯 Chroma 语义检索），
        兼容旧 ChromaVectorRetriever 的 search_with_filters / invoke。
        """
        k = top_k or self._default_search_top_k

        if hasattr(self.retriever, 'semantic_search'):
            # HybridRetriever 路径
            docs = self.retriever.semantic_search(
                query=query,
                vehicle_type=vehicle_type,
                top_k=k,
            )
        elif hasattr(self.retriever, 'search_with_filters'):
            # 旧 ChromaVectorRetriever 路径
            docs = self.retriever.search_with_filters(
                query=query,
                vehicle_type=vehicle_type,
                top_k=k,
            )
        else:
            docs = self.retriever.invoke(query)[:k]

        return [self._doc_to_dict(doc) for doc in docs]

    def _get_incident_detail_impl(self, record_id: str) -> Optional[dict]:
        """获取工单详情实现"""
        try:
            # HybridRetriever 路径：通过 chroma 属性拿 store
            store = None
            if hasattr(self.retriever, 'chroma'):
                store = getattr(self.retriever.chroma, 'store', None)
            elif hasattr(self.retriever, 'store'):
                store = self.retriever.store

            if store and hasattr(store, '_collection'):
                result = store._collection.get(ids=[record_id])
                if result["ids"] and result["metadatas"]:
                    record = IncidentRecord.from_dict(result["metadatas"][0])
                    return record.to_dict()
        except Exception as e:
            logger.error(f"获取工单详情失败: {e}")
        return None

    def _convert_working_condition_file_impl(self, file_path: str) -> dict:
        """工况文件转换实现"""
        if self._working_condition_converter is None:
            return {
                "error": "工况文件转换工具未注册",
                "description": "",
                "hint": "请联系工具注册工况文件转换工具（register_working_condition_converter）",
            }

        try:
            result = self._working_condition_converter(file_path)
            logger.info(f"工况文件转换完成: {file_path}")
            return result
        except Exception as e:
            logger.error(f"工况文件转换失败: {e}")
            return {"error": str(e), "description": ""}

    def _query_fault_graph_impl(
        self,
        mcuid: Optional[str] = None,
        dtc_code: Optional[list[str]] = None,
        project: Optional[str] = None,
        component: Optional[str] = None,
        working_condition: Optional[str] = None,
        software_version: Optional[str] = None,
        depth: Optional[int] = 1,
        top_k: Optional[int] = 10,
    ) -> list[dict]:
        """图查询实现：直接调 Neo4j 召回，不精排

        Agent 主动发起点结构化查询时调用，返回结构化命中的全量候选
        （不像预检索走精排只给 top-K）。

        字段对齐 StandardInput.entities。
        """
        neo4j_retriever = getattr(self.retriever, "neo4j", None)
        if neo4j_retriever is None:
            return [{"error": "Neo4j 检索器不可用"}]
        if not neo4j_retriever.available:
            return [{"error": "Neo4j 不可用（未配置或连接失败）"}]

        # 当前 Neo4j schema 尚未按新字段重构，
        # 除 dtc_code 外其余字段统一作为 keyword 模糊匹配。
        # 与 HybridRetriever._neo4j_recall 统一用 SearchCondition.to_keyword()。
        from ..retrieval.search_condition import SearchCondition
        cond = SearchCondition(
            mcuid=mcuid,
            dtc_code=dtc_code or [],
            project=project,
            component=component,
            working_condition=working_condition,
            software_version=software_version,
        )
        keyword = cond.to_keyword() or None

        candidates = neo4j_retriever.structured_recall(
            mcuid=mcuid,
            dtc_codes=dtc_code or None,
            keyword=keyword,
            depth=depth,
            limit=top_k,
        )

        return [
            {
                "fault_id": c.fault_id,
                "description": c.description,
                "root_cause": c.root_cause,
                "solution": c.solution,
                "dtc_code": c.dtc_codes,
                "motor_code": c.motor_code,
                "vehicle_type": c.vehicle_type,
                "indicators": c.indicators,
                "scenario": c.scenario,
                "source": "neo4j",
            }
            for c in candidates
        ]

    @staticmethod
    def _doc_to_dict(doc) -> dict:
        """将 Document 转为字典"""
        record = document_to_record(doc)
        d = record.to_dict()
        d["record_id"] = doc.metadata.get("id", "")
        d["similarity"] = round(doc.metadata.get("score", 0.0), 4)
        return d

    # ------------------------------------------------------------------
    # 工具列表生成（使用 StructuredTool 绑定实例方法）
    # ------------------------------------------------------------------

    def get_tool_list(self) -> list:
        """返回工具列表（供 create_agent 使用）

        使用 StructuredTool.from_function 动态创建工具，
        正确绑定实例方法。

        工具集：
        - search_similar_incidents：语义检索（模糊匹配故障现象）
        - query_fault_graph：结构化图查询（精确匹配 DTC/电驱代号/场景等，可扩展图关系）
        - can_converter：CAN 报文文件转 CSV/Excel（结合 DBC 解码）
        - get_incident_detail：工单详情
        - convert_working_condition_file：工况文件转换
        """
        tools = [
            StructuredTool.from_function(
                func=self._search_similar_incidents_impl,
                name="search_similar_incidents",
                description="检索与当前故障相似的历史工单。参数：query（故障描述）、vehicle_type（车型，可选）、top_k（返回数量，可选）。",
                args_schema=SearchSimilarIncidentsInput,
            ),
        ]
        if self._neo4j_available:
            tools.append(
                StructuredTool.from_function(
                    func=self._query_fault_graph_impl,
                    name="query_fault_graph",
                    description=(
                        "按结构化字段精确查询故障知识图谱。"
                        "参数：mcuid（MCU标识）、dtc_code（DTC码，多个逗号分隔）、"
                        "project（项目/车型代号）、component（涉及部件）、"
                        "working_condition（工作条件）、software_version（软件版本）、"
                        "depth（关系扩展深度1-2）、top_k（返回数量）。"
                        "适合用确定的结构化字段（如 DTC、MCU标识）做精确召回。"
                    ),
                    args_schema=QueryFaultGraphInput,
                )
            )
        tools.append(build_can_converter_tool())
        tools.extend([
            StructuredTool.from_function(
                func=self._get_incident_detail_impl,
                name="get_incident_detail",
                description="根据记录ID获取详细信息。参数：record_id（记录ID）。",
                args_schema=GetIncidentDetailInput,
            ),
            StructuredTool.from_function(
                func=self._convert_working_condition_file_impl,
                name="convert_working_condition_file",
                description="将工况文件转换为结构化数据或自然语言描述。参数：file_path（工况文件路径）。",
                args_schema=ConvertWorkingConditionFileInput,
            ),
        ])
        return tools