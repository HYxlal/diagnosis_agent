"""CSV/XLSX 解析共享逻辑

抽取两种格式解析器中重复的表头映射 + 记录清洗代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..models.incident import INCIDENT_COLUMNS
from ..models.input import InputType, ParsedInput
from .header_mapper import apply_header_mapping, map_headers_to_standard


def build_parsed_input_from_df(
    df: "pd.DataFrame",
    file_path: str | Path,
    input_type: InputType,
) -> ParsedInput:
    """从 DataFrame 构建 ParsedInput

    统一流程：表头映射 → 提取标准 8 列 → 记录清洗 → 构建 ParsedInput。
    供 csv_parser / xlsx_parser 共用。

    Args:
        df: 原始 DataFrame
        file_path: 源文件路径
        input_type: 输入类型（CSV / XLSX）

    Returns:
        ParsedInput 实例

    Raises:
        ValueError: 表头映射后无任何标准列匹配
    """
    file_path = Path(file_path)

    # 表头映射：精确匹配 → LLM 智能映射
    headers = [str(c).strip() for c in df.columns.tolist()]
    mapping = map_headers_to_standard(headers)
    df = apply_header_mapping(df, mapping)

    # 只保留标准 8 列
    available_cols = [c for c in INCIDENT_COLUMNS if c in df.columns]
    if not available_cols:
        raise ValueError(
            f"{input_type.value.upper()} 表头映射后无任何标准列匹配。原始表头: {headers}"
        )
    df = df[available_cols]

    # 转为记录列表（NaN → 空字符串，其他值 strip）
    clean_records: list[dict] = []
    for _, row in df.iterrows():
        rec: dict = {}
        for col in INCIDENT_COLUMNS:
            val = row.get(col)
            if pd.isna(val):
                rec[col] = ""
            else:
                rec[col] = str(val).strip() if not isinstance(val, (int, float)) else val
        clean_records.append(rec)

    # 提取第一条记录的 problem_description 作为描述
    description = ""
    if clean_records:
        first = clean_records[0]
        description = str(first.get("problem_description", ""))

    return ParsedInput(
        input_type=input_type,
        description=description,
        bulk_records=clean_records,
        source_file=str(file_path),
        raw_input=f"[{input_type.value} file: {file_path.name}, {len(clean_records)} rows]",
    )
