"""主流程编排 + CLI 入口

串联全链路：输入解析 → 意图路由 → 检索 → 推理 → 报告生成

支持两种输入模式：
1. 传统模式：文本/文件输入，内部转换为 ParsedInput
2. 标准接口模式：JSON 输入，使用 StandardInput/StandardOutput

CLI 启动时统一展示模型配置状态。
支持批量导入和批量诊断。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .agent.input_router import InputRouter
from .agent.langchain_agent import LangChainDiagnosticAgent
from .config import Settings, get_settings, reset_settings
from .models.converter import diagnostic_output_to_standard
from .models.input import InputIntent, InputType, StandardEntities, StandardInput
from .parsers.unified import parse_input
from .reporting.entries import generate_both as generate_db_entries
from .reporting.markdown import generate_markdown_report
from .storage.chroma_store import ChromaVectorStore

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
PROCESSED_DIR_NAME = "processed"

app = typer.Typer(
    name="diagnosis-agent",
    help="车辆故障诊断 Agent — 基于向量检索 + LLM 推理",
    no_args_is_help=True,
)

console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# 模型配置状态展示
# ---------------------------------------------------------------------------

def _print_model_status(settings: Settings) -> None:
    """CLI 启动时统一展示所有模型的配置状态

    展示内容：
    - 主 LLM 模型（名称 / API key 状态 / 可用性）
    - Embedding 模型（名称 / API key 状态）
    - InputRouter 模型（名称 / 启用状态）
    - 向量存储类型
    """
    table = Table(
        title="⚙️  模型配置状态",
        show_header=True,
        header_style="bold cyan",
        title_style="bold blue",
    )
    table.add_column("组件", style="bold", width=18)
    table.add_column("模型", width=25)
    table.add_column("状态", justify="center", width=10)
    table.add_column("备注", width=30)

    # --- 主 LLM ---
    llm_key_set = bool(settings.llm.api_key)
    llm_status = "[green]✅ 已配置[/green]" if llm_key_set else "[red]❌ 未配置[/red]"
    llm_note = "诊断推理核心模型" if llm_key_set else "请设置 DASHSCOPE_API_KEY"
    table.add_row(
        "LLM (推理)",
        settings.llm.model,
        llm_status,
        llm_note,
    )

    # --- Embedding ---
    emb_key_set = bool(settings.embedding.api_key)
    emb_status = "[green]✅ 已配置[/green]" if emb_key_set else "[red]❌ 未配置[/red]"
    emb_note = "向量编码模型" if emb_key_set else "回退到 ChromaDB 默认 embedding"
    table.add_row(
        "Embedding",
        settings.embedding.model,
        emb_status,
        emb_note,
    )

    # --- InputRouter ---
    if settings.input_router.enabled:
        # InputRouter 复用主 LLM 的 API key
        router_status = "[green]✅ 已启用[/green]" if llm_key_set else "[yellow]⚠️ 回退模式[/yellow]"
        router_note = "轻量意图分类" if llm_key_set else "LLM 不可用，使用规则匹配"
        table.add_row(
            "InputRouter",
            settings.input_router.model,
            router_status,
            router_note,
        )
    else:
        table.add_row(
            "InputRouter",
            settings.input_router.model,
            "[dim]⏸️ 已禁用[/dim]",
            "所有输入走默认诊断流程",
        )

    # --- 向量存储 ---
    table.add_row(
        "向量存储",
        settings.vector_store.type,
        "[blue]📦 本地[/blue]",
        f"集合: {settings.vector_store.collection_name}",
    )

    console.print(table)
    console.print()


def _build_components(settings: Settings):
    """构建存储"""
    store = ChromaVectorStore(
        persist_dir=settings.vector_store.persist_dir,
        collection_name=settings.vector_store.collection_name,
        embedding_model=settings.embedding.model,
        api_key=settings.embedding.api_key or None,
        api_base=settings.embedding.api_base or None,
    )
    return store, settings


@app.command()
def diagnose(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="故障描述文本"),
    file: Optional[List[str]] = typer.Option(None, "--file", "-f", help="输入文件路径 (CSV/XLSX)，支持多个文件"),
    files: Optional[str] = typer.Option(None, "--files", help="包含多个文件的目录路径"),
    json_input: Optional[str] = typer.Option(None, "--json-input", help="标准输入JSON文件路径（平台Agent传入的格式）"),
    output_dir: str = typer.Option("output", "--output", "-o", help="报告输出目录"),
    generate_md: bool = typer.Option(False, "--generate-md", "-g", help="生成Markdown报告（调试用）"),
    std_output: bool = typer.Option(False, "--std-output", help="输出标准JSON格式到控制台"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """执行故障诊断

    支持两种输入模式：
    1. 传统模式：使用 --text/-f/--files 传入文本或文件
    2. 标准接口模式：使用 --json-input 传入标准JSON文件

    输出可选：
    - 标准JSON（默认）：使用 --std-output 输出到控制台
    - Markdown报告（--generate-md）：调试用，生成详细报告
    """

    _setup_logging(verbose)
    reset_settings()
    settings = get_settings()

    console.print(Panel.fit("🔍 故障诊断 Agent", style="bold blue"))
    _print_model_status(settings)

    # 标准接口模式：JSON输入
    if json_input:
        console.print(f"[dim]使用标准接口模式，加载JSON输入: {json_input}[/dim]")

        # 加载并解析 JSON
        try:
            with open(json_input, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            error_output = {"code": -1, "msg": f"JSON文件不存在: {json_input}"}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            return error_output
        except json.JSONDecodeError as e:
            error_output = {"code": -1, "msg": f"JSON解析失败: {e}"}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            return error_output

        # 创建 StandardInput 并验证
        try:
            standard_input = StandardInput(**raw_data)
        except Exception as e:
            error_output = {"code": -1, "msg": f"入参缺失关键信息无法诊断: {e}"}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            return error_output

        # 执行标准接口诊断
        try:
            agent = LangChainDiagnosticAgent(settings=settings)
            standard_output = agent.diagnose_with_standard_input(standard_input)
        except Exception as e:
            error_output = {"code": -2, "msg": f"Agent输出异常: {e}"}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            return error_output

        # 输出结果
        if standard_output.code == 0:
            console.print(Panel.fit("✅ 诊断完成", style="bold green"))
            console.print(f"  状态码: [cyan]{standard_output.code}[/cyan]")
            console.print(f"  状态: {standard_output.msg}")

            if standard_output.diagnosis_result:
                result = standard_output.diagnosis_result
                console.print(f"  诊断置信度: {standard_output.diagnosis_confidence:.0%}")
                console.print(f"  根因数量: {len(result.fault_root_cause)}")
                console.print(f"  解决方案数量: {len(result.solution)}")

            console.print()
            console.print(f"  📄 标准输出JSON:")
            console.print(json.dumps(standard_output.model_dump(), ensure_ascii=False, indent=2))
        else:
            # 错误状态只输出 code 和 msg
            console.print(Panel.fit("⚠️ 诊断失败", style="bold red"))
            error_output = {"code": standard_output.code, "msg": standard_output.msg}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            return error_output

        # 生成 Markdown 报告（-g / --generate-md 开关）
        if generate_md:
            internal_output = agent._last_diagnostic_output
            if internal_output:
                md_path = generate_markdown_report(internal_output, output_dir=output_dir)
                console.print(f"  📄 Markdown 报告: {md_path}")
            else:
                console.print("[yellow]  ⚠️ 无法生成Markdown报告: 内部诊断输出不可用[/yellow]")

        # 保存到文件
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_output_path = output_path / f"diagnosis_output_{standard_input.mcuid}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(standard_output.model_dump(), ensure_ascii=False, indent=2))
        console.print(f"\n  💾 已保存到: {json_output_path}")

        return

    # 传统模式：文本/文件输入
    if not text and not file and not files:
        console.print("[red]请指定输入：--text/-f/--files 或 --json-input[/red]")
        raise typer.Exit(1)

    # 收集所有文件
    all_files = []
    if file:
        all_files.extend(file)
    if files:
        dir_path = Path(files)
        if not dir_path.exists():
            console.print(f"[red]目录不存在: {files}[/red]")
            raise typer.Exit(1)
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(dir_path.glob(f"*{ext}"))

    # 解析输入
    if all_files:
        if len(all_files) == 1:
            file_path = str(all_files[0])
            try:
                parsed = parse_input(text=text, file_path=file_path)
            except Exception as e:
                console.print(f"[red]输入解析失败: {e}[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[yellow]检测到多个文件，将合并处理: {len(all_files)} 个[/yellow]")
            parsed = None
            for idx, f in enumerate(all_files):
                try:
                    current = parse_input(text=text if idx == 0 else None, file_path=str(f))
                    if parsed is None:
                        parsed = current
                    else:
                        parsed.bulk_records.extend(current.bulk_records)
                        if current.description and current.description not in parsed.description:
                            parsed.description += f"\n---\n{current.description}"
                except Exception as e:
                    console.print(f"[yellow]跳过文件 {f}: {e}[/yellow]")
            if parsed is None:
                console.print("[red]所有文件解析失败[/red]")
                raise typer.Exit(1)
            parsed.input_type = InputType.MIXED
    else:
        try:
            parsed = parse_input(text=text)
        except Exception as e:
            console.print(f"[red]输入解析失败: {e}[/red]")
            raise typer.Exit(1)

    # 意图路由
    router = InputRouter(settings)
    parsed = router.route(parsed)

    console.print(f"  输入类型: {parsed.input_type.value}")
    console.print(f"  意图: [cyan]{parsed.intent.value}[/cyan]")
    console.print(f"  描述: {parsed.description[:100]}...")
    if parsed.is_bulk():
        console.print(f"  批量记录: {len(parsed.bulk_records)} 条")
    if parsed.search_query:
        console.print(f"  检索 query: {parsed.search_query[:80]}...")
    console.print()

    # 执行诊断
    try:
        agent = LangChainDiagnosticAgent(settings=settings)
        output = agent.diagnose(parsed)
    except Exception as e:
        console.print(f"[red]诊断失败: {e}[/red]")
        raise typer.Exit(1)

    # 生成报告（根据开关）
    if generate_md:
        md_path = generate_markdown_report(output, output_dir=output_dir)
        console.print(f"  📄 Markdown 报告: {md_path}")

    if std_output:
        # 构造一个临时的StandardInput用于转换
        temp_standard_input = StandardInput(
            raw_query=parsed.description,
            mcuid="CLI",
            entities=StandardEntities(),
        )
        standard_output = diagnostic_output_to_standard(output, temp_standard_input)

        console.print(Panel.fit("📋 标准JSON输出", style="bold blue"))
        console.print(json.dumps(standard_output.model_dump(), ensure_ascii=False, indent=2))

    # 生成数据库条目（始终生成）
    db_paths = generate_db_entries(output, output_dir=output_dir)

    console.print(Panel.fit("✅ 诊断完成", style="bold green"))
    console.print(f"  诊断ID: {output.report.diagnosis_id}")
    console.print(f"  找到相似工况: {output.report.has_similar_cases}")
    console.print(f"  推荐对策: {output.report.recommended_countermeasure[:100]}...")
    console.print(f"  置信度: {output.database_entry.diagnostic_confidence:.0%}")
    console.print()
    if generate_md:
        console.print(f"  📄 Markdown 报告: [dim]已生成[/dim]")
    console.print(f"  📊 CSV 条目: {db_paths['csv']}")
    console.print(f"  📊 JSON 条目: {db_paths['json']}")


@app.command()
def search(
    query: str = typer.Option(..., "--query", "-q", help="查询文本"),
    vehicle_type: Optional[str] = typer.Option(None, "--vehicle-type", help="车型过滤"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="返回数量"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """检索相似工单"""

    _setup_logging(verbose)
    reset_settings()
    settings = get_settings()

    from .retrieval.langchain_retrievers import create_chroma_retriever

    retriever = create_chroma_retriever()

    docs = retriever.search_with_filters(
        query=query,
        vehicle_type=vehicle_type,
        top_k=top_k,
    )

    console.print(Panel.fit(f"🔍 检索结果 ({len(docs)} 条)", style="bold blue"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=20)
    table.add_column("问题描述", width=40)
    table.add_column("根本原因", width=30)
    table.add_column("车型", width=15)
    table.add_column("相似度", justify="right", width=10)

    for doc in docs:
        table.add_row(
            doc.metadata.get("id", "")[:20],
            doc.metadata.get("problem_description", "")[:40],
            doc.metadata.get("root_cause", "")[:30],
            doc.metadata.get("vehicle_type", "")[:15],
            f"{doc.metadata.get('score', 0.0):.2f}",
        )

    console.print(table)


@app.command()
def load_data(
    file: Optional[List[str]] = typer.Option(None, "--file", "-f", help="数据文件路径 (CSV/XLSX)，支持多个文件"),
    files: Optional[str] = typer.Option(None, "--files", help="包含多个文件的目录路径"),
    all: bool = typer.Option(False, "--all", help="导入 data/samples 目录下所有文件"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """加载批量数据到向量库（支持批量导入和默认路径）

    默认路径: data/samples/
    导入后文件会自动移动到 data/samples/processed/ 避免重复导入
    """

    _setup_logging(verbose)
    reset_settings()
    settings = get_settings()

    console.print(Panel.fit("📥 数据加载", style="bold blue"))
    _print_model_status(settings)

    # 默认路径
    default_path = Path(settings.paths.samples_dir)

    # 收集所有文件
    all_files = []
    if file:
        all_files.extend(file)
    if files:
        dir_path = Path(files)
        if not dir_path.exists():
            console.print(f"[red]目录不存在: {files}[/red]")
            raise typer.Exit(1)
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(dir_path.glob(f"*{ext}"))
    if all:
        if not default_path.exists():
            console.print(f"[red]默认目录不存在: {default_path}[/red]")
            raise typer.Exit(1)
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(default_path.glob(f"*{ext}"))

    # 如果没有指定文件，自动使用默认路径
    if not all_files:
        if default_path.exists():
            found_files = []
            for ext in SUPPORTED_EXTENSIONS:
                found_files.extend(default_path.glob(f"*{ext}"))
            if found_files:
                console.print(f"[yellow]未指定文件，自动扫描 {default_path} 目录[/yellow]")
                all_files = found_files
            else:
                console.print(f"[yellow]{default_path} 目录下无 xlsx/csv 文件[/yellow]")
                raise typer.Exit(0)
        else:
            console.print("[red]未指定文件且默认目录不存在[/red]")
            raise typer.Exit(1)

    # 创建已处理目录
    processed_dir = default_path / PROCESSED_DIR_NAME
    processed_dir.mkdir(exist_ok=True)

    # 统计变量
    total_files = len(all_files)
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_records = 0

    console.print(f"\n  找到 {total_files} 个文件待处理")
    console.print()

    # 构建组件（只构建一次）
    store, _ = _build_components(settings)
    from .models.incident import IncidentRecord

    # 逐个处理文件
    for idx, file_path in enumerate(all_files, 1):
        file_path = Path(file_path)

        # 检查是否已处理（在 processed 目录中）
        processed_path = processed_dir / file_path.name
        if processed_path.exists():
            console.print(f"  [{idx}/{total_files}] [dim]跳过[/dim] {file_path.name}（已处理）")
            skip_count += 1
            continue

        console.print(f"  [{idx}/{total_files}] 处理 {file_path.name}...")

        # 解析文件
        try:
            parsed = parse_input(file_path=str(file_path))
        except Exception as e:
            console.print(f"  [yellow]解析失败: {e}[/yellow]")
            fail_count += 1
            continue

        if not parsed.is_bulk():
            console.print(f"  [yellow]无有效记录[/yellow]")
            fail_count += 1
            continue

        # 构建 IncidentRecord 列表
        records = []
        for rec_dict in parsed.bulk_records:
            try:
                record = IncidentRecord.from_dict(rec_dict)
                records.append(record)
            except Exception as e:
                console.print(f"    [yellow]跳过记录: {e}[/yellow]")

        # 加载到向量库
        count = store.add_records(records)
        if count > 0:
            total_records += count
            success_count += 1
            console.print(f"    [green]成功加载 {count} 条记录[/green]")

            # 移动到已处理目录
            try:
                shutil.move(str(file_path), str(processed_path))
                console.print(f"    [dim]已移动到 {PROCESSED_DIR_NAME}/[/dim]")
            except Exception as e:
                console.print(f"    [yellow]移动文件失败: {e}[/yellow]")
        else:
            fail_count += 1
            console.print(f"    [yellow]未加载任何记录[/yellow]")

    # 汇总
    console.print()
    console.print(Panel.fit("📊 导入完成", style="bold blue"))
    console.print(f"  总计: {total_files} 个文件")
    console.print(f"  成功: [green]{success_count}[/green]")
    console.print(f"  跳过: [dim]{skip_count}[/dim]")
    console.print(f"  失败: [yellow]{fail_count}[/yellow]")
    console.print(f"  加载记录数: [green]{total_records}[/green]")
    console.print(f"  当前向量库总数: {store.count()}")


@app.command()
def stats():
    """查看向量库统计"""

    reset_settings()
    settings = get_settings()

    console.print(Panel.fit("📊 向量库统计", style="bold blue"))
    _print_model_status(settings)

    store, _ = _build_components(settings)
    console.print(f"  总记录数: {store.count()}")
    console.print(f"  存储路径: {store.persist_dir}")
    console.print(f"  集合名称: {store.collection_name}")


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--confirm", "-y", help="确认清空"),
):
    """清空向量库"""

    if not confirm:
        console.print("[yellow]请使用 --confirm 确认清空操作[/yellow]")
        raise typer.Exit(1)

    reset_settings()
    settings = get_settings()

    store, _ = _build_components(settings)
    store.clear()
    console.print("[green]✅ 向量库已清空[/green]")


@app.command()
def config():
    """查看当前模型配置状态"""

    reset_settings()
    settings = get_settings()

    _print_model_status(settings)

    # 额外展示检索配置
    rt_table = Table(
        title="🔍 检索配置",
        show_header=True,
        header_style="bold cyan",
    )
    rt_table.add_column("参数", style="bold", width=30)
    rt_table.add_column("值", width=40)

    rt_table.add_row("语义 top_k", str(settings.retrieval.semantic.top_k))
    rt_table.add_row("语义阈值", str(settings.retrieval.semantic.score_threshold))
    rt_table.add_row("过滤 default_top_k", str(settings.retrieval.filter.default_top_k))
    rt_table.add_row("过滤字段", ", ".join(settings.retrieval.filter.filter_fields))
    rt_table.add_row("语义权重", str(settings.retrieval.hybrid.semantic_weight))
    rt_table.add_row("过滤权重", str(settings.retrieval.hybrid.filter_weight))
    rt_table.add_row("过滤扩倍率", str(settings.retrieval.hybrid.filter_expansion_ratio))
    rt_table.add_row("工具 search_top_k", str(settings.tools.search_top_k))
    rt_table.add_row("工具 filter_top_k", str(settings.tools.filter_top_k))

    console.print(rt_table)


def main():
    app()


if __name__ == "__main__":
    main()
