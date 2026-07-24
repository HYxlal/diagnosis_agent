"""统一输入解析器

根据输入类型自动路由到对应的解析器。
支持：xlsx / csv / 自然语言 / 混合输入 / 工况文件
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..models.input import InputType, ParsedInput
from .csv_parser import parse_csv
from .nl_parser import parse_mixed, parse_natural_language
from .xlsx_parser import parse_xlsx

_WORKING_CONDITION_EXTENSIONS = {".asc", ".blf", ".mdf"}


def detect_input_type(text: str, file_path: Optional[str] = None) -> InputType:
    """自动检测输入类型

    Args:
        text: 输入文本
        file_path: 可选的文件路径

    Returns:
        InputType 枚举
    """
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext == ".xlsx":
            return InputType.XLSX
        elif ext == ".csv":
            return InputType.CSV

    # 如果没有文件，纯文本
    return InputType.NATURAL_LANGUAGE


def parse_input(
    text: Optional[str] = None,
    file_path: Optional[str] = None,
) -> ParsedInput:
    """统一解析入口

    Args:
        text: 自然语言文本（可选）
        file_path: 文件路径（可选，支持 .xlsx / .csv）

    Returns:
        ParsedInput 实例

    Raises:
        ValueError: 未提供任何输入
        FileNotFoundError: 文件不存在
    """
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext == ".xlsx":
            result = parse_xlsx(path)
        elif ext == ".csv":
            result = parse_csv(path)
        elif ext in _WORKING_CONDITION_EXTENSIONS:
            result = _parse_working_condition_file(path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

        # 如果同时提供了文本，合并为混合输入
        if text and text.strip():
            result.description = f"{result.description}\n---\n{text}"
            result.input_type = InputType.MIXED
            result.raw_input += f" + nl: {text[:100]}"

        return result

    if text:
        # 纯文本输入 — 由 LLM 处理语义解析
        return parse_natural_language(text)

    raise ValueError("必须提供 text 或 file_path 参数")


def _parse_working_condition_file(path: Path) -> ParsedInput:
    """解析工况文件（.dat/.log/.txt/.bin/.raw）

    Args:
        path: 工况文件路径

    Returns:
        ParsedInput 实例，input_type=MIXED，source_file=文件路径，
        description=文件内容预览（前5000字符），bulk_records=[]
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        try:
            with open(path, "rb") as f:
                content = f.read().decode("utf-8", errors="ignore")
        except Exception:
            content = ""

    description = content[:5000] if content else ""

    return ParsedInput(
        input_type=InputType.MIXED,
        description=description,
        bulk_records=[],
        source_file=str(path),
        raw_input=f"[working_condition_file: {path.name}, size={len(content)}]",
    )
