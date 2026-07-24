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

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..config import Settings
from ..retrieval.hybrid import HybridRetriever
from ..storage.vector_store import SearchResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 输入模型定义
# ------------------------------------------------------------------

class SearchSimilarIncidentsInput(BaseModel):
    """search_similar_incidents 工具输入模型"""
    query: str = Field(description="故障描述文本（必填）")
    vehicle_type: Optional[str] = Field(default=None, description="车型（可选，用于过滤）")
    top_k: Optional[int] = Field(default=5, description="返回数量（可选，默认5）")


class FilterByVehicleTypeInput(BaseModel):
    """filter_by_vehicle_type 工具输入模型"""
    vehicle_type: str = Field(description="车型名称（必填）")
    top_k: Optional[int] = Field(default=10, description="返回数量（可选，默认10）")


class GetIncidentDetailInput(BaseModel):
    """get_incident_detail 工具输入模型"""
    record_id: str = Field(description="记录ID（必填）")


class ConvertWorkingConditionFileInput(BaseModel):
    """convert_working_condition_file 工具输入模型"""
    file_path: str = Field(description="工况文件路径（必填）")


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
        retriever: HybridRetriever,
        settings: Optional[Settings] = None,
    ):
        self.retriever = retriever
        self.settings = settings

        self._default_search_top_k = (
            settings.tools.search_top_k if settings else 5
        )
        self._default_filter_top_k = (
            settings.tools.filter_top_k if settings else 10
        )

        self._working_condition_converter: Optional[Callable[[str], dict[str, Any]]] = None

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
        """检索相似工单实现"""
        k = top_k or self._default_search_top_k
        results = self.retriever.retrieve(query=query, vehicle_type=vehicle_type, top_k=k)
        return [self._to_dict(r) for r in results]

    def _filter_by_vehicle_type_impl(self, vehicle_type: str, top_k: Optional[int] = None) -> list[dict]:
        """按车型过滤实现"""
        k = top_k or self._default_filter_top_k
        results = self.retriever.filter_retriever.filter_by(vehicle_type=vehicle_type, top_k=k)
        return [self._to_dict(r) for r in results]

    def _get_incident_detail_impl(self, record_id: str) -> Optional[dict]:
        """获取工单详情实现"""
        record = self.retriever.store.get_by_id(record_id)
        if record:
            return record.to_dict()
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

    @staticmethod
    def _to_dict(result: SearchResult) -> dict:
        """将 SearchResult 转为字典"""
        record = result.record
        if record:
            d = record.to_dict()
        else:
            d = result.metadata.copy()
        d["record_id"] = result.id
        d["similarity"] = round(result.score, 4)
        return d

    # ------------------------------------------------------------------
    # 工具列表生成（使用 StructuredTool 绑定实例方法）
    # ------------------------------------------------------------------

    def get_tool_list(self) -> list:
        """返回工具列表（供 create_react_agent 使用）

        使用 StructuredTool.from_function 动态创建工具，
        正确绑定实例方法。
        """
        return [
            StructuredTool.from_function(
                func=self._search_similar_incidents_impl,
                name="search_similar_incidents",
                description="检索与当前故障相似的历史工单。参数：query（故障描述）、vehicle_type（车型，可选）、top_k（返回数量，可选）。",
                args_schema=SearchSimilarIncidentsInput,
            ),
            StructuredTool.from_function(
                func=self._filter_by_vehicle_type_impl,
                name="filter_by_vehicle_type",
                description="按车型精确过滤历史工单。参数：vehicle_type（车型名称）、top_k（返回数量，可选）。",
                args_schema=FilterByVehicleTypeInput,
            ),
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
        ]