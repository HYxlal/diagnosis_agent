"""conftest.py — pytest 配置和 fixtures"""

import sys
from pathlib import Path

import pytest

# 添加 src 到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def sample_record_dict():
    """样例记录字典 — 新 8 列"""
    return {
        "problem_description": "车辆启动后发动机故障灯常亮，怠速不稳",
        "root_cause": "进气压力传感器线路接触不良",
        "countermeasure": "更换进气压力传感器线束",
        "drive_code": "DRV-001",
        "vehicle_type": "SUV-X1",
        "dashboard_indicator": "发动机故障灯",
        "dtc_code": "P0107",
        "fault_scenario": "冷启动后低速行驶",
    }


@pytest.fixture
def sample_csv_path(tmp_path):
    """生成临时样例 CSV 文件"""
    import csv

    csv_path = tmp_path / "sample_incidents_small.csv"
    headers = [
        "problem_description",
        "root_cause",
        "countermeasure",
        "drive_code",
        "vehicle_type",
        "dashboard_indicator",
        "dtc_code",
        "fault_scenario",
    ]
    rows = [
        {
            "problem_description": f"故障描述 {i}",
            "root_cause": f"原因 {i}",
            "countermeasure": f"对策 {i}",
            "drive_code": f"DRV-{i:03d}",
            "vehicle_type": "SUV-X1" if i % 2 == 0 else "SEDAN-A2",
            "dashboard_indicator": "发动机故障灯",
            "dtc_code": f"P010{i % 10}",
            "fault_scenario": f"场景 {i}",
        }
        for i in range(10)
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
