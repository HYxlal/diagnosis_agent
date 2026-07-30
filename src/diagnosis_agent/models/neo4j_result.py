"""Neo4j 查询中间结果模型

封装 fault_knowledge_graph 项目返回的 Fault 节点 + 关系数据，
展平为扁平结构，供精排层使用。

设计说明：
- Neo4j 原始返回是嵌套结构（Fault 节点 + paths 关系列表），
  上层精排/Agent 不应直接处理这种嵌套，故在召回层展平。
- 精排阶段填充 semantic_score / structural_match_ratio / final_score。
- to_document() 把候选转成 LangChain Document，与 Chroma 路径对齐。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


class FaultCandidate:
    """Neo4j 召回的单条候选故障

    从同事项目 multi_condition_query(output_format="json") 的原始记录展平而来。
    原始结构形如 {"f": {...Fault 属性...}, "paths": [[{...}, "HAS_DTC", {...}], ...]}，
    本类负责把 Fault 自身属性 + paths 里的关系目标属性都拍平到顶层字段。
    """

    __slots__ = (
        "fault_id",
        "description",
        "root_cause",
        "solution",
        "dtc_codes",
        "motor_code",
        "vehicle_type",
        "indicators",
        "scenario",
        "raw",
        "semantic_score",
        "structural_match_ratio",
        "final_score",
        "source",
    )

    def __init__(
        self,
        fault_id: str = "",
        description: str = "",
        root_cause: str = "",
        solution: str = "",
        dtc_codes: list[str] | None = None,
        motor_code: str = "",
        vehicle_type: str = "",
        indicators: list[str] | None = None,
        scenario: str = "",
        raw: dict | None = None,
        semantic_score: float = 0.0,
        structural_match_ratio: float = 0.0,
        final_score: float = 0.0,
        source: str = "neo4j",
    ):
        self.fault_id = fault_id
        self.description = description
        self.root_cause = root_cause
        self.solution = solution
        self.dtc_codes = dtc_codes or []
        self.motor_code = motor_code
        self.vehicle_type = vehicle_type
        self.indicators = indicators or []
        self.scenario = scenario
        self.raw = raw or {}
        self.semantic_score = semantic_score
        self.structural_match_ratio = structural_match_ratio
        self.final_score = final_score
        self.source = source

    @classmethod
    def from_neo4j_record(cls, record: dict) -> FaultCandidate:
        """从 Neo4j 单条原始记录展平为 FaultCandidate

        原始记录结构（来自 query_builder.build 的 Cypher）：
            {
              "f": {Fault 节点属性，含 description/root_cause/solution/id/...},
              "paths": [[source_data, rel_type, target_data], ...]
            }
        其中 paths 每项是 [起点属性dict, 关系类型, 终点属性dict]。
        起点恒为 Fault 自身，终点按关系类型映射到 DTC/MotorType/Indicator/Scenario 等。
        """
        fault_node = record.get("f") or {}
        if not isinstance(fault_node, dict):
            fault_node = {}

        fault_id = str(fault_node.get("id") or "")
        description = str(fault_node.get("description") or "")
        root_cause = str(fault_node.get("root_cause") or "")
        solution = str(fault_node.get("solution") or "")

        dtc_codes: list[str] = []
        motor_code = ""
        vehicle_type = ""
        indicators: list[str] = []
        scenario = ""

        # 关系类型 → 终点取值字段
        # HAS_DTC → DTC.code（优先）或 DTC.description
        # OCCURS_ON → MotorType.code
        # OCCURS_ON_VEHICLE → VehicleType.type
        # SHOWS_INDICATOR → Indicator.name
        # OCCURS_IN_SCENARIO → Scenario 拼成 "category-subcategory-detail"
        for path_item in record.get("paths") or []:
            if not isinstance(path_item, list) or len(path_item) != 3:
                continue
            target_data = path_item[2]
            rel_type = path_item[1]
            if not isinstance(target_data, dict):
                continue
            if rel_type == "HAS_DTC":
                code = target_data.get("code")
                desc = target_data.get("description")
                dtc_codes.append(str(code or desc or ""))
            elif rel_type == "OCCURS_ON":
                motor_code = str(target_data.get("code") or motor_code)
            elif rel_type == "OCCURS_ON_VEHICLE":
                vehicle_type = str(target_data.get("type") or vehicle_type)
            elif rel_type == "SHOWS_INDICATOR":
                name = target_data.get("name")
                if name:
                    indicators.append(str(name))
            elif rel_type == "OCCURS_IN_SCENARIO":
                category = target_data.get("category") or ""
                subcategory = target_data.get("subcategory") or ""
                detail = target_data.get("detail") or ""
                scenario = "-".join(
                    p for p in [category, subcategory, detail] if p
                ) or str(category or subcategory or detail)

        return cls(
            fault_id=fault_id,
            description=description,
            root_cause=root_cause,
            solution=solution,
            dtc_codes=dtc_codes,
            motor_code=motor_code,
            vehicle_type=vehicle_type,
            indicators=indicators,
            scenario=scenario,
            raw=record,
        )

    def to_document(self) -> Document:
        """转为 LangChain Document，与 Chroma 路径输出格式对齐

        page_content 用 description（供精排算 embedding 和 Agent 阅读）；
        metadata 填展平后的结构化字段 + source 标签 + score。
        """
        metadata = {
            "id": self.fault_id,
            "root_cause": self.root_cause,
            "solution": self.solution,
            "dtc_code": ", ".join(self.dtc_codes) if self.dtc_codes else "",
            "drive_code": self.motor_code,
            "vehicle_type": self.vehicle_type,
            "dashboard_indicator": ", ".join(self.indicators) if self.indicators else "",
            "fault_scenario": self.scenario,
            "score": self.final_score,
            "source": self.source,
        }
        return Document(page_content=self.description, metadata=metadata)
