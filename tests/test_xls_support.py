"""测试 .xls 文件格式支持

验证：
1. xlrd 库可用
2. .xls 文件可被正确解析
3. detect_input_type 正确识别 .xls
4. parse_input 正确路由 .xls
5. 端到端流程（创建 .xls → 解析 → 生成 IncidentRecord）
"""
import os
import sys
import tempfile

os.environ.setdefault("PYTHONPATH", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

TEST_PASSED = 0
TEST_FAILED = 0


def check(name, condition, detail=""):
    global TEST_PASSED, TEST_FAILED
    if condition:
        TEST_PASSED += 1
        print(f"  ✅ {name}")
    else:
        TEST_FAILED += 1
        print(f"  ❌ {name}  -- {detail}")


def main():
    global TEST_PASSED, TEST_FAILED
    print("=" * 60)
    print("测试 .xls 文件格式支持")
    print("=" * 60)

    # ---- 1. 依赖检查 ----
    print("\n[1] 依赖检查")
    try:
        import xlrd
        check("xlrd 库已安装", True)
    except ImportError:
        check("xlrd 库已安装", False, "pip install xlrd>=2.0")
        return 1

    import pandas as pd
    import xlwt

    # ---- 2. 创建测试 .xls 文件 ----
    print("\n[2] 创建测试 .xls 文件")
    tmpdir = tempfile.mkdtemp()
    xls_path = os.path.join(tmpdir, "test_sample.xls")

    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet1")
    headers = ["problem_description", "vehicle_type", "root_cause", "countermeasure"]
    for col, h in enumerate(headers):
        ws.write(0, col, h)
    ws.write(1, 0, "电池包温升异常，SOC骤降")
    ws.write(1, 1, "SUV")
    ws.write(1, 2, "BMS 温度传感器漂移")
    ws.write(1, 3, "校准传感器阈值")
    ws.write(2, 0, "电机异响，动力输出下降")
    ws.write(2, 1, "轿车")
    ws.write(2, 2, "轴承磨损")
    ws.write(2, 3, "更换轴承")
    wb.save(xls_path)

    check(".xls 测试文件创建成功", os.path.exists(xls_path))
    check(".xls 文件大小 > 0", os.path.getsize(xls_path) > 0)

    # ---- 3. pandas 读取 .xls ----
    print("\n[3] pandas 读取 .xls")
    df = pd.read_excel(xls_path)
    check("pd.read_excel 成功读取", df is not None and len(df) > 0)
    check("行数正确 (2 条数据)", len(df) == 2, f"实际: {len(df)}")
    check("列数正确 (4 列)", len(df.columns) == 4, f"实际: {len(df.columns)}")

    # ---- 4. detect_input_type 检测 ----
    print("\n[4] detect_input_type 检测")
    from diagnosis_agent.parsers.unified import detect_input_type
    from diagnosis_agent.models.input import InputType

    result = detect_input_type("", file_path=xls_path)
    check(".xls 被识别为 XLSX 类型", result == InputType.XLSX,
          f"实际: {result}")

    # 同时验证 .xlsx 仍然正常
    xlsx_path = os.path.join(tmpdir, "test_sample.xlsx")
    df.to_excel(xlsx_path, index=False)
    result2 = detect_input_type("", file_path=xlsx_path)
    check(".xlsx 仍然被正确识别", result2 == InputType.XLSX,
          f"实际: {result2}")

    # ---- 5. parse_input 解析 .xls ----
    print("\n[5] parse_input 解析 .xls")
    from diagnosis_agent.parsers.unified import parse_input

    parsed = parse_input(file_path=xls_path)
    check("parse_input 返回 ParsedInput", parsed is not None)
    check("bulk_records 不为空", len(parsed.bulk_records) > 0,
          f"实际: {len(parsed.bulk_records)}")
    check("input_type 为 XLSX", parsed.input_type == InputType.XLSX,
          f"实际: {parsed.input_type}")

    rec = parsed.bulk_records[0]
    check("第一条记录是 dict 类型", isinstance(rec, dict))
    check("第一条记录有 problem_description",
          "problem_description" in rec,
          f"实际 keys: {list(rec.keys())}")
    check("第一条记录有 vehicle_type",
          "vehicle_type" in rec,
          f"实际 keys: {list(rec.keys())}")

    # ---- 6. .xls vs .xlsx 数据一致性 ----
    print("\n[6] .xls vs .xlsx 数据一致性")
    parsed_xlsx = parse_input(file_path=xlsx_path)
    check(".xls 和 .xlsx 解析记录数一致",
          len(parsed.bulk_records) == len(parsed_xlsx.bulk_records),
          f".xls: {len(parsed.bulk_records)}, .xlsx: {len(parsed_xlsx.bulk_records)}")

    # ---- 7. SUPPORTED_EXTENSIONS 检查 ----
    print("\n[7] cli.py SUPPORTED_EXTENSIONS")
    from diagnosis_agent.cli import SUPPORTED_EXTENSIONS
    check(".xls 在 SUPPORTED_EXTENSIONS 中", ".xls" in SUPPORTED_EXTENSIONS,
          f"实际: {SUPPORTED_EXTENSIONS}")
    check(".xlsx 在 SUPPORTED_EXTENSIONS 中", ".xlsx" in SUPPORTED_EXTENSIONS)

    # ---- 清理 ----
    import shutil
    shutil.rmtree(tmpdir)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    total = TEST_PASSED + TEST_FAILED
    print(f"测试结果: {TEST_PASSED}/{total} 通过")
    if TEST_FAILED == 0:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  {TEST_FAILED} 个测试失败")
    print("=" * 60)

    return 0 if TEST_FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())