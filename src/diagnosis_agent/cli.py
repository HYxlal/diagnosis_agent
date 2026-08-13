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
import uuid
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
from .models.diagnostic_output import OutputCode, StandardOutput
from .models.input import InputIntent, InputType, ParsedInput, StandardEntities, StandardInput
from .models.incident import IncidentRecord
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


# ---------------------------------------------------------------------------
# 日志与通用工具
# ---------------------------------------------------------------------------

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
    """CLI 启动时统一展示所有模型的配置状态"""
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

    llm_key_set = bool(settings.llm.api_key)
    llm_status = "[green]✅ 已配置[/green]" if llm_key_set else "[red]❌ 未配置[/red]"
    llm_note = "诊断推理核心模型" if llm_key_set else "请设置 DASHSCOPE_API_KEY"
    table.add_row("LLM (推理)", settings.llm.model, llm_status, llm_note)

    emb_key_set = bool(settings.embedding.api_key)
    emb_status = "[green]✅ 已配置[/green]" if emb_key_set else "[red]❌ 未配置[/red]"
    emb_note = "向量编码模型" if emb_key_set else "回退到 ChromaDB 默认 embedding"
    table.add_row("Embedding", settings.embedding.model, emb_status, emb_note)

    if settings.input_router.enabled:
        router_status = "[green]✅ 已启用[/green]" if llm_key_set else "[yellow]⚠️ 回退模式[/yellow]"
        router_note = "轻量意图分类" if llm_key_set else "LLM 不可用，使用规则匹配"
        table.add_row("InputRouter", settings.input_router.model, router_status, router_note)
    else:
        table.add_row("InputRouter", settings.input_router.model, "[dim]⏸️ 已禁用[/dim]", "所有输入走默认诊断流程")

    table.add_row("向量存储", settings.vector_store.type, "[blue]📦 本地[/blue]", f"集合: {settings.vector_store.collection_name}")

    console.print(table)
    console.print()


def _build_components(settings: Settings):
    """构建存储"""
    from .utils.llm_factory import create_embedding
    store = ChromaVectorStore(
        persist_dir=settings.vector_store.persist_dir,
        collection_name=settings.vector_store.collection_name,
        embedding=create_embedding(),
    )
    return store, settings


# ---------------------------------------------------------------------------
# 输入收集
# ---------------------------------------------------------------------------

def _collect_files(
    file: Optional[List[str]],
    files: Optional[str],
) -> List[Path]:
    """收集所有输入文件（支持单文件、多文件、目录）"""
    all_files: List[Path] = []

    if file:
        all_files.extend(Path(f) for f in file)

    if files:
        dir_path = Path(files)
        if not dir_path.exists():
            console.print(f"[red]目录不存在: {files}[/red]")
            raise typer.Exit(1)
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(dir_path.glob(f"*{ext}"))

    return all_files


def _parse_traditional_input(
    text: Optional[str],
    file: Optional[List[str]],
    files: Optional[str],
) -> ParsedInput:
    """解析传统模式（文本/文件）输入

    三种分支：
    - 仅文本：直接 parse_input(text=...)
    - 单文件：parse_input(text, file_path=...)
    - 多文件：逐个解析后合并 bulk_records，text 只附加到第一个文件
    """
    all_files = _collect_files(file, files)

    if not all_files:
        try:
            return parse_input(text=text)
        except Exception as e:
            console.print(f"[red]输入解析失败: {e}[/red]")
            raise typer.Exit(1)

    if len(all_files) == 1:
        try:
            return parse_input(text=text, file_path=str(all_files[0]))
        except Exception as e:
            console.print(f"[red]输入解析失败: {e}[/red]")
            raise typer.Exit(1)

    # 多文件合并：text 只附加到首个文件，其余纯文件解析
    console.print(f"[yellow]检测到多个文件，将合并处理: {len(all_files)} 个[/yellow]")
    parsed: Optional[ParsedInput] = None
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
    return parsed


# ---------------------------------------------------------------------------
# 标准接口模式
# ---------------------------------------------------------------------------

def _load_standard_input(json_input: str) -> StandardInput:
    """加载并解析标准接口 JSON 输入"""
    try:
        with open(json_input, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        error_output = {"code": -1, "msg": f"JSON文件不存在: {json_input}"}
        console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        error_output = {"code": -1, "msg": f"JSON解析失败: {e}"}
        console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
        raise typer.Exit(1)

    try:
        return StandardInput(**raw_data)
    except Exception as e:
        error_output = {"code": -1, "msg": f"入参缺失关键信息无法诊断: {e}"}
        console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
        raise typer.Exit(1)


def _run_standard_diagnosis(
    standard_input: StandardInput,
    output_dir: str,
    generate_md: bool,
) -> dict:
    """执行标准接口诊断并输出/保存结果

    流程：构造 Agent → ContextManager 构建上下文 → diagnose_with_standard_input
          → 控制台输出 + 保存 JSON + 显示会话状态。
    错误状态（code != 0）只输出 code + msg，不输出 diagnosis_result。
    """
    from .agent.session_manager import SessionManager
    from .agent.context_manager import SimpleContextManager
    from .agent.context.summarizer import Summarizer
    from .agent.context.topic_detector import TopicDetector

    session_manager = SessionManager()
    session_id = standard_input.session_id or ""

    settings = get_settings()

    # 初始化上下文管理器（含摘要+话题检测）
    context_manager = SimpleContextManager(
        window_size=settings.context.window_size,
        max_tokens=settings.context.max_tokens,
        summarizer=Summarizer(
            strategy=settings.context.summary_strategy,
            max_tokens=settings.context.summary_max_tokens,
        ) if settings.context.summary_enabled else None,
        topic_detector=TopicDetector(
            strategy=settings.context.topic_detection_strategy,
            threshold_high=settings.context.topic_similarity_high,
            threshold_low=settings.context.topic_similarity_low,
            model=settings.context.topic_detection_model or None,
            switch_signal_words=settings.context.topic_signal_words.get("switch", []),
            continue_signal_words=settings.context.topic_signal_words.get("continue", []),
            entity_overlap_enabled=settings.context.entity_overlap_enabled,
            time_decay_short_sec=settings.context.topic_time_decay_short_sec if settings.context.topic_time_decay_enabled else 0,
            time_decay_short_max_len=settings.context.topic_time_decay_short_max_len,
            time_decay_long_sec=settings.context.topic_time_decay_long_sec,
            scope_detection_enabled=settings.context.scope_detection_enabled,
            scope_use_llm=settings.context.scope_use_llm,
            scope_out_keywords=settings.context.scope_out_keywords,
        ) if settings.context.topic_detection_enabled else None,
    )

    # 构建多轮上下文（三步降级 + 摘要注入）
    prepared_result = None
    ctx = session_manager.get_conversation_context(session_id)
    if ctx and session_id:
        prepared_result = context_manager.prepare_from_context(
            ctx, standard_input.raw_query,
        )
        if prepared_result.messages:
            console.print(
                f"  [dim]会话 {session_id}: 第 {session_manager.get_round_count(session_id)} 轮 "
                f"历史已注入 ({len(prepared_result.messages)} 条消息)"
            )

    # scope 检测判定不在范围内，直接返回 -3，不调用 Agent
    if prepared_result and not prepared_result.metadata.is_in_scope:
        from .models.converter import build_error_output
        standard_output = build_error_output(
            code=OutputCode.OUT_OF_SCOPE,
            msg="查询不在车辆电驱系统故障诊断范围内",
            standard_input=standard_input,
        )
    else:
        try:
            agent = LangChainDiagnosticAgent(settings=settings)
            standard_output = agent.diagnose_with_standard_input(
                standard_input,
                prepared_messages=prepared_result.messages if prepared_result else None,
            )
        except Exception as e:
            error_output = {"code": -2, "msg": f"Agent输出异常: {e}"}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            raise typer.Exit(1)

    # 存储本轮新增消息（成功时）
    if standard_output.code == 0 and session_id:
        from langchain_core.messages import messages_to_dict
        current_messages = messages_to_dict(getattr(agent, "_last_messages", []))
        session_manager.update(session_id, standard_input.raw_query, current_messages)

    if standard_output.code == 0:
        console.print(Panel.fit("✅ 诊断完成", style="bold green"))
        console.print(f"  状态码: [cyan]{standard_output.code}[/cyan]")
        console.print(f"  状态: {standard_output.msg}")

        if standard_output.diagnosis_result:
            result = standard_output.diagnosis_result
            console.print(f"  诊断置信度: {standard_output.diagnosis_confidence:.0%}")
            console.print(f"  根因数量: {len(result.fault_root_cause)}")
            console.print(f"  解决方案数量: {len(result.solution)}")

        # 会话状态（独立于 StandardOutput）
        if session_id:
            round_count = session_manager.get_round_count(session_id)
            console.print(f"  会话: {session_id} | 第 {round_count} 轮")

        console.print()
        console.print("  📄 标准输出JSON:")
        console.print(json.dumps(standard_output.model_dump(), ensure_ascii=False, indent=2))

        if generate_md:
            internal_output = agent._last_diagnostic_output
            if internal_output:
                md_path = generate_markdown_report(internal_output, output_dir=output_dir)
                console.print(f"  📄 Markdown 报告: {md_path}")
            else:
                console.print("[yellow]  ⚠️ 无法生成Markdown报告: 内部诊断输出不可用[/yellow]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_output_path = output_path / f"diagnosis_output_{standard_input.mcuid}.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(standard_output.model_dump(), ensure_ascii=False, indent=2))
        console.print(f"\n  💾 已保存到: {json_output_path}")
    else:
        console.print(Panel.fit("⚠️ 诊断失败", style="bold red"))
        error_output = {"code": standard_output.code, "msg": standard_output.msg}
        console.print(json.dumps(error_output, ensure_ascii=False, indent=2))

    return standard_output.model_dump()


# ---------------------------------------------------------------------------
# 传统模式
# ---------------------------------------------------------------------------

def _run_traditional_diagnosis(
    parsed: ParsedInput,
    output_dir: str,
    generate_md: bool,
    std_output: bool,
) -> None:
    """执行传统模式诊断并输出/保存结果

    流程：意图路由 → 诊断 → 按开关生成报告（MD / 标准JSON / 数据库条目）。
    --std-output 时构造临时 StandardInput 把内部输出转成对外格式。
    """
    settings = get_settings()
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

    try:
        agent = LangChainDiagnosticAgent(settings=settings)
        output = agent.diagnose(parsed)
    except Exception as e:
        console.print(f"[red]诊断失败: {e}[/red]")
        raise typer.Exit(1)

    if generate_md:
        md_path = generate_markdown_report(output, output_dir=output_dir)
        console.print(f"  📄 Markdown 报告: {md_path}")

    # 传统模式没有 StandardInput，需要构造临时的用于转换成标准输出格式
    if std_output:
        temp_standard_input = StandardInput(
            raw_query=parsed.description,
            mcuid="CLI",
            entities=StandardEntities(),
        )
        standard_output = diagnostic_output_to_standard(output, temp_standard_input)
        console.print(Panel.fit("📋 标准JSON输出", style="bold blue"))
        console.print(json.dumps(standard_output.model_dump(), ensure_ascii=False, indent=2))

    # 数据库条目始终生成（CSV + JSON）
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


# ---------------------------------------------------------------------------
# CLI 命令
# ---------------------------------------------------------------------------

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

    if json_input:
        console.print(f"[dim]使用标准接口模式，加载JSON输入: {json_input}[/dim]")
        standard_input = _load_standard_input(json_input)
        _run_standard_diagnosis(standard_input, output_dir=output_dir, generate_md=generate_md)
        return

    if not text and not file and not files:
        console.print("[red]请指定输入：--text/-f/--files 或 --json-input[/red]")
        raise typer.Exit(1)

    parsed = _parse_traditional_input(text=text, file=file, files=files)
    _run_traditional_diagnosis(
        parsed,
        output_dir=output_dir,
        generate_md=generate_md,
        std_output=std_output,
    )


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

    default_path = Path(settings.paths.samples_dir)

    all_files: List[Path] = []
    if file:
        all_files.extend(Path(f) for f in file)
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

    processed_dir = default_path / PROCESSED_DIR_NAME
    processed_dir.mkdir(exist_ok=True)

    total_files = len(all_files)
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_records = 0

    console.print(f"\n  找到 {total_files} 个文件待处理")
    console.print()

    store, _ = _build_components(settings)

    for idx, file_path in enumerate(all_files, 1):
        file_path = Path(file_path)

        processed_path = processed_dir / file_path.name
        if processed_path.exists():
            console.print(f"  [{idx}/{total_files}] [dim]跳过[/dim] {file_path.name}（已处理）")
            skip_count += 1
            continue

        console.print(f"  [{idx}/{total_files}] 处理 {file_path.name}...")

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

        records: List[IncidentRecord] = []
        for rec_dict in parsed.bulk_records:
            try:
                records.append(IncidentRecord.from_dict(rec_dict))
            except Exception as e:
                console.print(f"    [yellow]跳过记录: {e}[/yellow]")

        count = store.add_records(records)
        if count > 0:
            total_records += count
            success_count += 1
            console.print(f"    [green]成功加载 {count} 条记录[/green]")

            try:
                shutil.move(str(file_path), str(processed_path))
                console.print(f"    [dim]已移动到 {PROCESSED_DIR_NAME}/[/dim]")
            except Exception as e:
                console.print(f"    [yellow]移动文件失败: {e}[/yellow]")
        else:
            fail_count += 1
            console.print(f"    [yellow]未加载任何记录[/yellow]")

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


@app.command()
def chat(
    initial_question: Optional[str] = typer.Argument(None, help="初始问题（可选，不传则交互式输入）"),
    mcuid: str = typer.Option("CLI", "--mcuid", "-m", help="MCU 标识（可选，默认 CLI）"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="会话ID（可选，不填自动生成）"),
    output_dir: str = typer.Option("output", "--output", "-o", help="报告输出目录"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """交互式多轮诊断 — 进程常驻，持续接收问题，支持多轮追问

    为什么需要这个命令？
    - diagnose 命令是"一次性"的，进程退出后 SessionManager 内存状态就丢了。
    - chat 命令让进程常驻，同一 session_id 在内存中连续维护，多轮最自然。
    - 即使退出，SessionManager 也默认把历史持久化到 data/sessions/{session_id}.json，
      下次用 --session 指定同一 id 还能恢复。

    用法示例：
        python -m diagnosis_agent.cli chat                                           # 交互模式
        python -m diagnosis_agent.cli chat "发动机有异响"                            # 单次提问
        python -m diagnosis_agent.cli chat --mcuid MCU_001                           # 指定 MCU
        python -m diagnosis_agent.cli chat --mcuid MCU_001 --session my-session      # 指定会话

    输入 exit / quit / q 退出，Ctrl+C 也退出。
    """
    _setup_logging(verbose)
    reset_settings()
    settings = get_settings()
    from .agent.session_manager import SessionManager
    from .agent.langchain_agent import LangChainDiagnosticAgent
    from .agent.context_manager import SimpleContextManager
    from .agent.context.summarizer import Summarizer
    from .agent.context.topic_detector import TopicDetector

    # 初始化上下文管理器
    context_manager = SimpleContextManager(
        window_size=settings.context.window_size,
        max_tokens=settings.context.max_tokens,
        summarizer=Summarizer(
            strategy=settings.context.summary_strategy,
            max_tokens=settings.context.summary_max_tokens,
        ) if settings.context.summary_enabled else None,
        topic_detector=TopicDetector(
            strategy=settings.context.topic_detection_strategy,
            threshold_high=settings.context.topic_similarity_high,
            threshold_low=settings.context.topic_similarity_low,
            model=settings.context.topic_detection_model or None,
            switch_signal_words=settings.context.topic_signal_words.get("switch", []),
            continue_signal_words=settings.context.topic_signal_words.get("continue", []),
            entity_overlap_enabled=settings.context.entity_overlap_enabled,
            time_decay_short_sec=settings.context.topic_time_decay_short_sec if settings.context.topic_time_decay_enabled else 0,
            time_decay_short_max_len=settings.context.topic_time_decay_short_max_len,
            time_decay_long_sec=settings.context.topic_time_decay_long_sec,
            scope_detection_enabled=settings.context.scope_detection_enabled,
            scope_use_llm=settings.context.scope_use_llm,
            scope_out_keywords=settings.context.scope_out_keywords,
        ) if settings.context.topic_detection_enabled else None,
    )

    console.print(Panel.fit("🔍 故障诊断 Agent — 交互模式", style="bold blue"))
    _print_model_status(settings)

    sess_id = session_id or f"chat-{uuid.uuid4().hex[:8]}"
    agent = LangChainDiagnosticAgent(settings=settings)
    sm = SessionManager()

    console.print(f"  MCU: [cyan]{mcuid}[/cyan]")
    console.print(f"  会话: [dim]{sess_id}[/dim] (输入 exit/quit/q 退出)")
    console.print()

    round_num = sm.get_round_count(sess_id) + 1

    def _do_chat_round(query: str, sess_id: str, mcuid: str, round_num: int):
        """执行一轮对话"""
        nonlocal context_manager, agent, sm

        standard_input = StandardInput(
            raw_query=query,
            mcuid=mcuid,
            session_id=sess_id,
            entities=StandardEntities(),
        )

        # 构建上下文（三步降级 + 摘要注入）
        ctx = sm.get_conversation_context(sess_id)
        prepared = None
        if ctx:
            prepared = context_manager.prepare_from_context(ctx, query)

        # scope 检测判定不在范围内，直接返回 -3
        if prepared and not prepared.metadata.is_in_scope:
            from .models.converter import build_error_output
            standard_output = build_error_output(
                code=OutputCode.OUT_OF_SCOPE,
                msg="查询不在车辆电驱系统故障诊断范围内",
                standard_input=standard_input,
            )
        else:
            standard_output = agent.diagnose_with_standard_input(
                standard_input,
                prepared_messages=prepared.messages if prepared else None,
            )

        if standard_output.code == 0:
            from langchain_core.messages import messages_to_dict
            current_messages = messages_to_dict(getattr(agent, "_last_messages", []))
            sm.update(sess_id, query, current_messages)
            result = standard_output.diagnosis_result
            if result:
                console.print(f"  [cyan]分类:[/cyan] {result.classification}")
                console.print(f"  [cyan]根因:[/cyan] {result.fault_root_cause[0] if result.fault_root_cause else 'N/A'}")
                console.print(f"  [cyan]方案:[/cyan] {result.solution[0] if result.solution else 'N/A'}")
                console.print(f"  [cyan]置信度:[/cyan] {standard_output.diagnosis_confidence:.0%}")
                console.print()
        else:
            console.print(f"  [red]诊断失败: {standard_output.msg}[/red]")

    # 如果传了初始问题，先处理这一轮
    if initial_question:
        _do_chat_round(initial_question, sess_id, mcuid, round_num)
        round_num += 1

    while True:
        try:
            user_input = console.input(f"[bold green]第{round_num}轮[/bold green] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出交互模式[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print(f"[dim]已退出，会话 {sess_id} 共 {round_num - 1} 轮[/dim]")
            break

        _do_chat_round(user_input, sess_id, mcuid, round_num)
        round_num += 1


def main():
    app()


if __name__ == "__main__":
    main()
