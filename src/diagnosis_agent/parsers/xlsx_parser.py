"""XLSX 解析器"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..models.input import InputType, ParsedInput
from ._records import build_parsed_input_from_df


def parse_xlsx(file_path: str | Path, sheet_name: Optional[str] = None) -> ParsedInput:
    """解析 XLSX 文件

    流程：
    1. 读取 XLSX
    2. 交由共享逻辑完成表头映射、记录清洗、ParsedInput 构建

    Args:
        file_path: XLSX 文件路径
        sheet_name: 工作表名称（可选）

    Returns:
        ParsedInput 实例
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # pandas 3.x 在某些情况下返回 dict（多 sheet 时）
    if isinstance(df, dict):
        if df:
            df = next(iter(df.values()))
        else:
            raise ValueError("Excel 文件为空")

    return build_parsed_input_from_df(df, file_path, InputType.XLSX)
