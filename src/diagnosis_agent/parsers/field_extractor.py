"""字段提取模块

使用 LLM 将自然语言输入分类到 8 个标准字段：
1. problem_description - 问题描述
2. root_cause - 根本原因
3. countermeasure - 对策/解决措施
4. drive_code - 驱动代码
5. vehicle_type - 车型
6. dashboard_indicator - 仪表盘指示
7. dtc_code - DTC 故障码
8. fault_scenario - 故障场景
"""

from __future__ import annotations

import json
import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..config import get_settings
from ..models.incident import IncidentRecord
from ..utils.llm_factory import create_llm

logger = logging.getLogger(__name__)


class FieldExtractor:
    """字段提取器

    使用 LLM 将自然语言输入分类到 8 个标准字段。
    """

    def __init__(self, llm=None):
        self.settings = get_settings()
        self.llm = llm or create_llm(
            model=self.settings.llm.model,
            temperature=0.0,
            max_tokens=2048,
        )
        self.chain = self._build_chain()

    def _build_chain(self):
        """构建结构化输出链"""
        parser = JsonOutputParser(pydantic_object=IncidentRecord)

        system_prompt = """你是一个专业的车辆故障诊断字段提取器。

任务：从用户输入中提取以下字段并输出 JSON：

必须提取的字段（即使只是重复用户输入）：
- problem_description: 用户描述的故障现象

可选字段（只提取明确提到的）：
- root_cause: 根本原因（如果用户提到或能明确推断）
- countermeasure: 解决措施（如果用户提到或能明确推断）
- drive_code: 电驱代号，如 EDU、iDD
- vehicle_type: 车型，如乘用车、商用车、SUV
- dashboard_indicator: 仪表指示灯，如电机故障红灯、动力系统故障黄灯
- dtc_code: 故障码，如 P0300、L160A
- fault_scenario: 故障场景，如行驶中、启动时、充电时

输入示例：
用户输入："电机故障红灯, 动力系统故障黄灯,IGBT-B相下部Dsat故障,硬件ASC故障动力电池故障,高压回路断路"

输出示例：
{
  "problem_description": "电机故障红灯，动力系统故障黄灯，IGBT-B相下部Dsat故障，硬件ASC故障，动力电池故障，高压回路断路",
  "root_cause": "",
  "countermeasure": "",
  "drive_code": "",
  "vehicle_type": "",
  "dashboard_indicator": "电机故障红灯，动力系统故障黄灯",
  "dtc_code": "",
  "fault_scenario": ""
}

规则：
- problem_description 必须包含用户输入的所有故障描述
- 其他字段只提取明确提到的信息
- 不确定的字段留空字符串 ""
- 直接输出 JSON，不要包含任何额外内容"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{{input}}"),
        ], template_format="jinja2")

        return prompt | self.llm | parser

    def extract(self, text: str) -> IncidentRecord:
        """提取字段

        Args:
            text: 用户输入的自然语言描述

        Returns:
            IncidentRecord: 包含 8 个标准字段的记录
        """
        if not text or not text.strip():
            return IncidentRecord()

        try:
            result = self.chain.invoke({"input": text})
            logger.debug(f"字段提取 LLM 返回: {str(result)[:500]}")

            if isinstance(result, dict):
                record = IncidentRecord(**result)
            elif isinstance(result, IncidentRecord):
                record = result
            else:
                record = IncidentRecord()

            logger.info(
                f"字段提取完成: problem_description='{record.problem_description[:30]}...', "
                f"vehicle_type='{record.vehicle_type}', "
                f"dtc_code='{record.dtc_code}', "
                f"dashboard_indicator='{record.dashboard_indicator}'"
            )

            return record
        except Exception as e:
            logger.warning(f"字段提取失败: {e}")
            return IncidentRecord(
                problem_description=text,
            )


def extract_fields(text: str) -> IncidentRecord:
    """便捷函数：提取字段"""
    extractor = FieldExtractor()
    return extractor.extract(text)