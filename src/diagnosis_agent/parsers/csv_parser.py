"""CSV 解析器"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..models.input import InputType, ParsedInput
from ._records import build_parsed_input_from_df


def parse_csv(file_path: str | Path, encoding: str = "utf-8") -> ParsedInput:
    """解析 CSV 文件

    流程：
    1. 读取 CSV
    2. 交由共享逻辑完成表头映射、记录清洗、ParsedInput 构建

    Args:
        file_path: CSV 文件路径
        encoding: 文件编码

    Returns:
        ParsedInput 实例
    """
    df = pd.read_csv(file_path, encoding=encoding)
    return build_parsed_input_from_df(df, file_path, InputType.CSV)
