"""Markdown 报告生成器

第一层输出：生成人类可读的 Markdown 诊断报告。
包含：思维链、工具调用过程、推断过程叙述、结构化表格、相似工况索引。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..models.diagnosis import DiagnosticOutput, DiagnosticReport


def generate_markdown_report(
    output: DiagnosticOutput,
    output_dir: str | Path = "output",
    filename: str | None = None,
) -> Path:
    """生成 Markdown 诊断报告文件

    Args:
        output: 诊断输出（含报告 + 数据条目）
        output_dir: 输出目录
        filename: 文件名，默认按诊断ID命名

    Returns:
        报告文件路径
    """
    report = output.report

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"diagnosis_{report.diagnosis_id}.md"

    filepath = output_dir / filename

    lines: list[str] = []

    # ============ 标题 ============
    lines.append("# 车辆故障诊断报告")
    lines.append("")
    lines.append(f"**诊断ID**: {report.diagnosis_id}")
    lines.append(f"**诊断时间**: {report.diagnosis_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**是否基于相似工况**: {'是' if report.has_similar_cases else '否'}")
    lines.append("")

    # ============ 输入摘要 ============
    lines.append("## 一、输入摘要")
    lines.append("")
    lines.append(f"""
```
{report.input_summary}
```""")
    lines.append("")

    # ============ ReAct 思维链 ============
    if report.react_steps:
        lines.append("## 二、思维链（ReAct 推理过程）")
        lines.append("")

        for step in report.react_steps:
            lines.append(f"### 步骤 {step.step}")
            lines.append("")
            lines.append(f"**Thought**:")
            lines.append(f"> {step.thought}")
            lines.append("")

            if step.action:
                lines.append(f"**Action**: `{step.action}`")
                lines.append("")

                if step.action_input:
                    import json
                    lines.append("**Action Input**:")
                    lines.append(f"""```json
{json.dumps(step.action_input, ensure_ascii=False, indent=2)}
```""")
                    lines.append("")

            if step.observation:
                lines.append("**Observation**:")
                lines.append(f"""```
{step.observation[:1500]}
```""")
                lines.append("")

            lines.append(f"*时间: {step.timestamp.strftime('%H:%M:%S.%f')[:-3]}*")
            lines.append("")
            lines.append("---")
            lines.append("")

    # ============ 工具调用记录 ============
    if report.tool_calls:
        lines.append("## 三、工具调用记录")
        lines.append("")
        lines.append("| # | 工具名称 | 参数 | 耗时(ms) | 时间 |")
        lines.append("|---|----------|------|----------|------|")

        import json
        for i, tc in enumerate(report.tool_calls, 1):
            params_str = json.dumps(tc.parameters, ensure_ascii=False)
            if len(params_str) > 80:
                params_str = params_str[:77] + "..."
            lines.append(
                f"| {i} | {tc.tool_name} | `{params_str}` | "
                f"{tc.duration_ms:.1f} | "
                f"{tc.timestamp.strftime('%H:%M:%S.%f')[:-3]} |"
            )

        lines.append("")

        # 工具调用详情
        for i, tc in enumerate(report.tool_calls, 1):
            lines.append(f"### 调用 {i}: {tc.tool_name}")
            lines.append("")
            lines.append("**参数**:")
            lines.append(f"""```json
{json.dumps(tc.parameters, ensure_ascii=False, indent=2)}
```""")
            lines.append("")

            result_str = json.dumps(tc.result, ensure_ascii=False, indent=2, default=str) if isinstance(tc.result, (list, dict)) else str(tc.result)
            if len(result_str) > 1000:
                result_str = result_str[:997] + "..."
            lines.append("**返回结果**:")
            lines.append(f"""```json
{result_str}
```""")
            lines.append(f"*耗时: {tc.duration_ms:.1f}ms*")
            lines.append("")

    # ============ 推断过程叙述 ============
    if report.reasoning_narrative:
        lines.append("## 四、推断过程叙述")
        lines.append("")
        lines.append(report.reasoning_narrative)
        lines.append("")

    # ============ 诊断发现 ============
    if report.findings:
        lines.append("## 五、诊断发现")
        lines.append("")
        for i, f in enumerate(report.findings, 1):
            lines.append(f"### 发现 {i}: {f.title}")
            lines.append(f"- **描述**: {f.description}")
            lines.append(f"- **置信度**: {f.confidence:.0%}")
            if f.evidence:
                lines.append(f"- **证据**:")
                for ev in f.evidence:
                    lines.append(f"  - {ev}")
            lines.append("")

    # ============ 推荐对策 ============
    lines.append("## 六、推荐对策")
    lines.append("")
    lines.append(f"**{report.recommended_countermeasure}**")
    lines.append("")

    # ============ 相似工况索引 ============
    if report.similar_cases:
        lines.append("## 七、相似工况索引")
        lines.append("")
        lines.append("| # | 记录ID | 问题描述 | 车型 | DTC码 | 相似度 |")
        lines.append("|---|--------|----------|------|-------|--------|")
        for i, case in enumerate(report.similar_cases, 1):
            desc = case.problem_description[:50] + "..." if len(case.problem_description) > 50 else case.problem_description
            lines.append(
                f"| {i} | {case.record_id} | {desc} | "
                f"{case.vehicle_type} | {case.dtc_code} | "
                f"{case.similarity:.2%} |"
            )
        lines.append("")

        # 相似工况详情
        for i, case in enumerate(report.similar_cases, 1):
            lines.append(f"### 相似工况 {i} (ID: {case.record_id}, 相似度: {case.similarity:.2%})")
            lines.append(f"- **问题描述**: {case.problem_description}")
            lines.append(f"- **根本原因**: {case.root_cause}")
            lines.append(f"- **对策**: {case.countermeasure}")
            lines.append(f"- **驱动代码**: {case.drive_code}")
            lines.append(f"- **车型**: {case.vehicle_type}")
            lines.append(f"- **仪表盘指示**: {case.dashboard_indicator}")
            lines.append(f"- **DTC码**: {case.dtc_code}")
            lines.append(f"- **故障场景**: {case.fault_scenario}")
            lines.append("")

    # ============ 结构化数据条目 ============
    lines.append("## 八、数据库条目（结构化输出）")
    lines.append("")
    entry = output.database_entry
    lines.append("| 字段 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| diagnosis_id | {entry.diagnosis_id} |")
    lines.append(f"| diagnosis_time | {entry.diagnosis_time.strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| 问题描述 | {entry.problem_description[:100]}... |")
    lines.append(f"| 根因 | {entry.root_cause[:100]}... |")
    lines.append(f"| 对策 | {entry.countermeasure[:100]}... |")
    lines.append(f"| 电驱代号 | {entry.drive_code} |")
    lines.append(f"| 车辆类型 | {entry.vehicle_type} |")
    lines.append(f"| 仪表指示灯 | {entry.dashboard_indicator} |")
    lines.append(f"| 故障DTC | {entry.dtc_code} |")
    lines.append(f"| 故障场景 | {entry.fault_scenario} |")
    lines.append(f"| diagnostic_confidence | {entry.diagnostic_confidence:.0%} |")
    lines.append(f"| based_on_similar | {entry.based_on_similar} |")
    lines.append(f"| similar_record_ids | {', '.join(entry.similar_record_ids) or 'N/A'} |")
    lines.append("")

    # ============ 页脚 ============
    lines.append("---")
    lines.append(f"*由 diagnosis_agent v{report.agent_version} 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath
