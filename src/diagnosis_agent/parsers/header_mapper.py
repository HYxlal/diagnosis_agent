"""LLM 表头映射器

当输入文件的表头与标准 8 列不完全匹配时，
使用 LLM 自动映射到标准列名。

核心函数：map_headers_to_standard()
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..config import Settings
from ..models.incident import COLUMN_CN_MAP, INCIDENT_COLUMNS

logger = logging.getLogger(__name__)


def _try_exact_match(headers: list[str]) -> dict[str, str]:
    """尝试精确匹配 + 中文映射（不需要 LLM）

    Returns:
        映射字典 {原始表头: 标准列名}
    """
    mapping: dict[str, str] = {}
    for col in headers:
        col_str = str(col).strip()
        if col_str in COLUMN_CN_MAP:
            mapping[col_str] = COLUMN_CN_MAP[col_str]
        elif col_str.lower() in INCIDENT_COLUMNS:
            mapping[col_str] = col_str.lower()
    return mapping


def map_headers_to_standard(
    headers: list[str],
    settings: Optional[Settings] = None,
) -> dict[str, str]:
    """将任意表头映射到标准 8 列

    先尝试精确匹配（中文映射 + 英文大小写归一化），
    若匹配不完全则调用 LLM 进行智能映射。

    Args:
        headers: 原始表头列表
        settings: 配置（包含 LLM 配置），为 None 时使用全局配置

    Returns:
        映射字典 {原始表头: 标准列名}

    Raises:
        RuntimeError: 当 LLM 不可用或映射失败且精确匹配不完全时
    """
    if not headers:
        return {}

    # 第一层：精确匹配
    exact_mapping = _try_exact_match(headers)

    # 检查是否所有标准列都已通过精确匹配覆盖
    mapped_standard = set(exact_mapping.values())
    all_standard = set(INCIDENT_COLUMNS)

    if all_standard.issubset(mapped_standard):
        logger.info("表头精确匹配成功，覆盖全部 8 列，无需 LLM 映射")
        return exact_mapping

    # 第二层：LLM 智能映射
    unmapped_headers = [h for h in headers if str(h).strip() not in exact_mapping]
    logger.info(
        f"精确匹配不完全（已映射 {len(mapped_standard)}/8 列），"
        f"调用 LLM 映射剩余表头: {unmapped_headers}"
    )
    return _llm_map_headers(headers, exact_mapping, settings)


def _llm_map_headers(
    headers: list[str],
    exact_mapping: dict[str, str],
    settings: Optional[Settings] = None,
) -> dict[str, str]:
    """使用 LLM 映射表头

    构建 prompt 让 LLM 将原始表头映射到标准 8 列，
    然后合并精确匹配结果。
    """
    if settings is None:
        from ..config import get_settings
        settings = get_settings()

    llm_config = settings.llm

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=llm_config.model,
            temperature=0.0,
            max_tokens=1024,
            api_key=llm_config.api_key,
            base_url=llm_config.api_base,
        )

        prompt = f"""你是表头映射专家。请将以下输入文件的表头映射到标准列名。

## 标准 8 列

{json.dumps(INCIDENT_COLUMNS, ensure_ascii=False, indent=2)}

## 各列含义

- problem_description: 问题描述（故障的主要描述文本）
- root_cause: 根本原因（故障的根因分析）
- countermeasure: 对策/解决措施（修复方案）
- drive_code: 驱动代码（驱动程序/软件版本代码）
- vehicle_type: 车型（车辆型号/平台）
- dashboard_indicator: 仪表盘指示（仪表盘上的指示灯/提示）
- dtc_code: DTC 故障码（Diagnostic Trouble Code）
- fault_scenario: 故障场景（故障发生的场景/条件描述）

## 输入表头

{json.dumps(headers, ensure_ascii=False)}

## 已通过精确匹配的映射

{json.dumps(exact_mapping, ensure_ascii=False)}

## 要求

请输出一个 JSON 对象，key 为原始表头（必须与输入表头完全一致），value 为对应的标准列名。
只输出 JSON，不要任何其他文字。
未匹配到的表头可以忽略，但 8 个标准列应尽量都有对应的映射。
如果某个原始表头确实无法映射到任何标准列，不要包含它。"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content if hasattr(response, "content") else str(response)

        # 解析 JSON 响应
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError(f"LLM 响应无法解析为 JSON: {content[:200]}")

        llm_mapping = json.loads(json_match.group())

        # 合并：精确匹配优先，LLM 补充
        merged = dict(llm_mapping)
        merged.update(exact_mapping)  # 精确匹配覆盖 LLM 结果

        logger.info(f"LLM 表头映射完成: {merged}")
        return merged

    except Exception as e:
        logger.error(f"LLM 表头映射失败: {e}")
        if exact_mapping:
            logger.warning(f"回退到精确匹配结果（部分映射）: {exact_mapping}")
            return exact_mapping
        raise RuntimeError(f"LLM 表头映射失败且无精确匹配可用: {e}") from e


def apply_header_mapping(
    df,
    mapping: dict[str, str],
) -> "pd.DataFrame":
    """将映射应用到 DataFrame 的列名上

    Args:
        df: pandas DataFrame
        mapping: {原始表头: 标准列名} 映射

    Returns:
        重命名后的 DataFrame
    """
    import pandas as pd

    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in mapping:
            rename_map[col] = mapping[col_str]

    if rename_map:
        df = df.rename(columns=rename_map)
    return df
