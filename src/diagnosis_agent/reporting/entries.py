"""可录入数据库条目生成器

第二层输出：生成可录入数据库的结构化条目。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from ..models.diagnosis import DatabaseEntry, DiagnosticOutput


def generate_database_entry_csv(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
    filename: str | None = None,
) -> Path:
    """生成可录入数据库的 CSV 条目文件

    Args:
        output: 诊断输出
        output_dir: 输出目录
        filename: 文件名

    Returns:
        CSV 文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"db_entry_{output.database_entry.diagnosis_id}.csv"

    filepath = output_dir / filename

    entry = output.database_entry
    entry_dict = entry.to_dict()
    entry_dict.pop('similar_record_ids', None)

    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(entry_dict.keys()))
        writer.writeheader()
        writer.writerow(entry_dict)

    return filepath


def generate_database_entry_json(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
    filename: str | None = None,
) -> Path:
    """生成可录入数据库的 JSON 条目文件

    Args:
        output: 诊断输出
        output_dir: 输出目录
        filename: 文件名

    Returns:
        JSON 文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"db_entry_{output.database_entry.diagnosis_id}.json"

    filepath = output_dir / filename

    entry = output.database_entry
    entry_dict = entry.to_dict()
    entry_dict.pop('similar_record_ids', None)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry_dict, f, ensure_ascii=False, indent=2)

    return filepath


def generate_both(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
) -> dict[str, Path]:
    """同时生成 CSV 和 JSON 格式的数据库条目

    Args:
        output: 诊断输出
        output_dir: 输出目录

    Returns:
        文件路径字典 {"csv": ..., "json": ...}
    """
    return {
        "csv": generate_database_entry_csv(output, output_dir),
        "json": generate_database_entry_json(output, output_dir),
    }
