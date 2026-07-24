"""XLSX 解析器"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..models.incident import INCIDENT_COLUMNS
from ..models.input import InputType, ParsedInput
from .header_mapper import apply_header_mapping, map_headers_to_standard


def parse_xlsx(file_path: str | Path, sheet_name: Optional[str] = None) -> ParsedInput:
    """解析 XLSX 文件

    流程：
    1. 读取 XLSX
    2. LLM 表头映射（精确匹配优先，不完全时调用 LLM）
    3. 提取标准 8 列数据
    4. 返回 ParsedInput

    Args:
        file_path: XLSX 文件路径
        sheet_name: 工作表名称（可选）

    Returns:
        ParsedInput 实例
    """
    file_path = Path(file_path)
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 处理 pandas 3.x 返回 dict 的情况（当文件包含多个 sheet 时）
    if isinstance(df, dict):
        if df:
            df = next(iter(df.values()))
        else:
            raise ValueError("Excel 文件为空")

    # 表头映射：精确匹配 → LLM 智能映射
    headers = [str(c).strip() for c in df.columns.tolist()]
    mapping = map_headers_to_standard(headers)
    df = apply_header_mapping(df, mapping)

    # 只保留标准 8 列
    available_cols = [c for c in INCIDENT_COLUMNS if c in df.columns]
    if not available_cols:
        raise ValueError(
            f"XLSX 表头映射后无任何标准列匹配。原始表头: {headers}"
        )
    df = df[available_cols]

    # 转为记录列表
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
        input_type=InputType.XLSX,
        description=description,
        bulk_records=clean_records,
        source_file=str(file_path),
        raw_input=f"[xlsx file: {file_path.name}, {len(clean_records)} rows]",
    )
