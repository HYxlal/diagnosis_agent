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
from .models.input import InputIntent, InputType, ParsedInput, StandardInput
from .models.incident import IncidentRecord
from .parsers.unified import parse_input
from .reporting.entries import generate_both as generate_db_entries
from .reporting.markdown import generate_markdown_report
from .storage.chroma_store import ChromaVectorStore
from .knowledge.cli import knowledge_app

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
PROCESSED_DIR_NAME = "processed"

app = typer.Typer(
    name="diagnosis-agent",
    help="车辆故障诊断 Agent — 基于向量检索 + LLM 推理",
    no_args_is_help=True,
)

# 注册知识沉淀命令组
app.add_typer(knowledge_app, name="knowledge", help="知识沉淀管理")

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


def _auto_extract_knowledge(
    current_messages: list[dict],
    conversation_id: str,
    settings: "Settings",
) -> None:
    """自动提取本轮对话知识并提交审核

    从当前消息列表中提取 user/assistant 消息对，调用 knowledge 模块
    提取实体关系、提交审核。提取失败不阻塞主流程。
    """
    try:
        from langchain_openai import ChatOpenAI
        from .knowledge import ConversationKnowledgeExtractor
        from .knowledge.graph_writer import GraphWriter
        from .knowledge.edit_manager import ManualEditManager
        from .retrieval.neo4j_retriever import Neo4jFaultRetriever

        k_llm = ChatOpenAI(
            model=settings.knowledge.extraction_model,
            temperature=0.0,
            max_tokens=1024,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
        )
        k_neo4j = Neo4jFaultRetriever(settings=settings)
        k_graph_writer = None
        k_edit_manager = None
        try:
            from langchain_neo4j import Neo4jGraph
            if k_neo4j.available:
                k_graph = Neo4jGraph(
                    url=settings.neo4j.url,
                    username=settings.neo4j.user,
                    password=settings.neo4j.password,
                    database="neo4j",
                )
                k_edit_manager = ManualEditManager(graph=k_graph)
                k_graph_writer = GraphWriter(graph=k_graph, edit_manager=k_edit_manager)
        except Exception:
            pass
        k_extractor = ConversationKnowledgeExtractor(
            llm=k_llm,
            graph_writer=k_graph_writer,
            edit_manager=k_edit_manager,
            persistence_path=settings.knowledge.persistence_dir + "/" + settings.knowledge.persistence_file,
            tuple_delimiter=settings.knowledge.tuple_delimiter,
            record_delimiter=settings.knowledge.record_delimiter,
            completion_delimiter=settings.knowledge.completion_delimiter,
        )
        k_messages = [
            m for m in current_messages
            if m.get("type") in ("human", "ai")
            or m.get("role") in ("user", "assistant")
        ][-4:]
        if k_messages:
            kid = k_extractor.extract_and_submit(k_messages, conversation_id)
            if kid:
                console.print(f"  [dim]知识已提交审核: {kid[:12]}...[/dim]")
    except Exception:
        logger.warning("知识提取失败，不影响主流程", exc_info=True)


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
    from .agent.context.adaptive_window import AdaptiveWindowManager

    session_manager = SessionManager()
    conversation_id = standard_input.conversationId or ""

    settings = get_settings()

    # 初始化上下文管理器（含摘要+话题检测+自适应窗口）
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
        emergency_min_turns=settings.context.emergency_min_turns,
        adaptive_window_manager=AdaptiveWindowManager(
            initial_window=settings.context.window_size,
            min_window=settings.context.adaptive_window.min_window,
            max_window=settings.context.adaptive_window.max_window,
            target_utilization=settings.context.adaptive_window.target_utilization,
            adjustment_interval=settings.context.adaptive_window.adjustment_interval,
        ) if settings.context.adaptive_window.enabled else None,
    )

    # 构建多轮上下文（三步降级 + 摘要注入）
    prepared_result = None
    ctx = session_manager.get_conversation_context(conversation_id)
    if ctx and conversation_id:
        prepared_result = context_manager.prepare_from_context(
            ctx, standard_input.raw_query,
        )
        if prepared_result.messages:
            console.print(
                f"  [dim]会话 {conversation_id}: 第 {session_manager.get_round_count(conversation_id)} 轮 "
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
            agent = LangChainDiagnosticAgent(settings=settings, report_mode=generate_md)
            standard_output = agent.diagnose_with_standard_input(
                standard_input,
                prepared_messages=prepared_result.messages if prepared_result else None,
            )
        except Exception as e:
            error_output = {"code": -2, "msg": f"Agent输出异常: {e}"}
            console.print(json.dumps(error_output, ensure_ascii=False, indent=2))
            raise typer.Exit(1)

    # 存储本轮新增消息（成功时）
    if standard_output.code == 0 and conversation_id:
        from langchain_core.messages import messages_to_dict
        current_messages = messages_to_dict(getattr(agent, "_last_messages", []))
        session_manager.update(conversation_id, standard_input.raw_query, current_messages)

        # 自动提取本轮对话知识
        if settings.knowledge.enabled and current_messages:
            _auto_extract_knowledge(current_messages, conversation_id, settings)

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
        if conversation_id:
            round_count = session_manager.get_round_count(conversation_id)
            console.print(f"  会话: {conversation_id} | 第 {round_count} 轮")

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
        json_output_path = output_path / f"diagnosis_output_{standard_input.vehicleModel}.json"
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
        agent = LangChainDiagnosticAgent(settings=settings, report_mode=generate_md)
        output = agent.diagnose(parsed)
    except Exception as e:
        console.print(f"[red]诊断失败: {e}[/red]")
        raise typer.Exit(1)

    # 自动提取本轮对话知识
    if settings.knowledge.enabled:
        try:
            from langchain_core.messages import messages_to_dict
            current_messages = messages_to_dict(getattr(agent, "_last_messages", []))
            if current_messages:
                _auto_extract_knowledge(current_messages, "", settings)
        except Exception:
            logger.warning("知识提取失败，不影响主流程", exc_info=True)

    if generate_md:
        md_path = generate_markdown_report(output, output_dir=output_dir)
        console.print(f"  📄 Markdown 报告: {md_path}")

    # 传统模式没有 StandardInput，需要构造临时的用于转换成标准输出格式
    if std_output:
        temp_standard_input = StandardInput(
            raw_query=parsed.description,
            vehicleModel="CLI",
            VIN="",
            faultOccurTime="",
            mileage=0.0,
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
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="报告输出目录"),
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
    output_dir = output_dir or settings.report.output_dir

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

    default_path = Path("data/samples")

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
    rt_table.add_row("语义权重", str(settings.retrieval.hybrid.semantic_weight))
    rt_table.add_row("过滤权重", str(settings.retrieval.hybrid.filter_weight))
    rt_table.add_row("过滤扩倍率", str(settings.retrieval.hybrid.filter_expansion_ratio))
    rt_table.add_row("工具 search_top_k", str(settings.tools.search_top_k))

    console.print(rt_table)


@app.command()
def chat(
    initial_question: Optional[str] = typer.Argument(None, help="初始问题（可选，不传则交互式输入）"),
    mcuid: str = typer.Option("CLI", "--mcuid", "-m", help="MCU 标识（可选，默认 CLI）"),
    conversation_id: Optional[str] = typer.Option(None, "--session", "-s", help="会话ID（可选，不填自动生成）"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="报告输出目录"),
    generate_md: bool = typer.Option(False, "--generate-md", "-g", help="每轮诊断后生成 Markdown 报告和 CSV/JSON 数据库条目"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """交互式多轮诊断 — 进程常驻，持续接收问题，支持多轮追问

    为什么需要这个命令？
    - diagnose 命令是"一次性"的，进程退出后 SessionManager 内存状态就丢了。
    - chat 命令让进程常驻，同一 conversation_id 在内存中连续维护，多轮最自然。
    - 即使退出，SessionManager 会把历史持久化到 Redis（热层/温层）或磁盘，
      下次用 --session 指定同一 id 就能恢复之前的对话。

    用法示例：
        python -m diagnosis_agent.cli chat                                           # 交互模式
        python -m diagnosis_agent.cli chat "发动机有异响"                            # 单次提问
        python -m diagnosis_agent.cli chat --mcuid MCU_001                           # 指定 MCU
        python -m diagnosis_agent.cli chat --session bugfix-test                     # 恢复历史会话
        python -m diagnosis_agent.cli chat --generate-md                             # 每轮生成报告
        python -m diagnosis_agent.cli chat --mcuid MCU_001 --session my-session      # 指定会话

    输入 exit / quit / q 退出，Ctrl+C 也退出。
    """
    _setup_logging(verbose)
    reset_settings()
    settings = get_settings()
    output_dir = output_dir or settings.report.output_dir
    from .agent.session_manager import SessionManager
    from .agent.langchain_agent import LangChainDiagnosticAgent
    from .agent.context_manager import SimpleContextManager
    from .agent.context.summarizer import Summarizer
    from .agent.context.topic_detector import TopicDetector
    from .agent.context.adaptive_window import AdaptiveWindowManager

    # 初始化上下文管理器（含自适应窗口）
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
        emergency_min_turns=settings.context.emergency_min_turns,
        adaptive_window_manager=AdaptiveWindowManager(
            initial_window=settings.context.window_size,
            min_window=settings.context.adaptive_window.min_window,
            max_window=settings.context.adaptive_window.max_window,
            target_utilization=settings.context.adaptive_window.target_utilization,
            adjustment_interval=settings.context.adaptive_window.adjustment_interval,
        ) if settings.context.adaptive_window.enabled else None,
    )

    # 启动 Prometheus HTTP 端点（后台线程）
    from .agent.context.metrics import metrics
    http_port = 9090
    if metrics.start_http_server(port=http_port):
        console.print(f"  [dim]📊 指标端点: http://localhost:{http_port}/metrics[/dim]")
    else:
        console.print(f"  [dim]📊 指标端点: http://localhost:{http_port}/metrics (端口被占用，跳过)[/dim]")

    console.print(Panel.fit("🔍 故障诊断 Agent — 交互模式", style="bold blue"))
    _print_model_status(settings)

    sess_id = conversation_id or f"chat-{uuid.uuid4().hex[:8]}"
    agent = LangChainDiagnosticAgent(settings=settings, report_mode=generate_md)
    agent.show_tool_details = settings.tool_call.show_details
    sm = SessionManager()

    console.print(f"  MCU: [cyan]{mcuid}[/cyan]")
    console.print(f"  会话: [dim]{sess_id}[/dim] (输入 exit/quit/q 退出)")

    # 更新会话统计
    metrics.record_session()
    stats = sm.get_metrics()
    metrics.update_session_stats(stats["active_sessions"], stats["avg_turns"])

    # 恢复历史会话：/resume 风格完整展示所有历史对话 + Token 状态
    ctx = sm.get_conversation_context(sess_id)
    if ctx and ctx.total_turns > 0:
        from rich.table import Table
        console.print(f"\n  [bold cyan]📜 已恢复历史会话（共 {ctx.total_turns} 轮）[/bold cyan]")

        # 显示 Token 预算状态
        current_util = ctx.metadata.get("window_utilization", 0.0) if ctx.metadata else 0.0
        current_tokens = ctx.metadata.get("token_usage", 0) if ctx.metadata else 0
        max_toks = ctx.metadata.get("max_tokens", 0) if ctx.metadata else settings.context.max_tokens
        hot_count = len(ctx.hot_messages)
        summary_count = len(ctx.warm_summaries)
        console.print(f"  Token 预算: [cyan]{current_tokens}[/cyan]/[dim]{max_toks}[/dim] 用量: [cyan]{current_util:.0%}[/cyan]  | 热层: {hot_count} 条 | 温层: {summary_count} 个")

        # 分层展示：温层摘要 + 热层完整对话
        if ctx.warm_summaries:
            t = Table(title="温层摘要（已合并的历史话题）", show_header=True, header_style="bold dim")
            t.add_column("#", style="dim", width=3)
            t.add_column("话题", width=12)
            t.add_column("摘要内容", width=60)
            for i, s in enumerate(ctx.warm_summaries, 1):
                t.add_row(str(i), s.topic_label or "-", s.summary[:70] + ("..." if len(s.summary) > 70 else ""))
            console.print(t)

        # 完整展示所有热层历史消息（用户提问 + 助手回复）
        if ctx.hot_messages:
            console.print(f"\n  [bold]完整历史对话[/bold]")
            for msg in ctx.hot_messages:
                role = msg.get("role") or msg.get("type", "unknown")
                content = ""
                if isinstance(msg, dict):
                    if "content" in msg:
                        content = str(msg.get("content", ""))
                    elif "data" in msg and isinstance(msg["data"], dict):
                        content = str(msg["data"].get("content", ""))
                if not content:
                    continue
                if role in ("human", "user"):
                    console.print(f"\n  [green]👤 用户[/green]")
                    console.print(f"  {content}")
                elif role in ("ai", "assistant"):
                    console.print(f"\n  [blue]🤖 诊断助手[/blue]")
                    console.print(f"  {content}")
        else:
            console.print(f"  [dim]（历史消息已归档至冷层，摘要信息如上）[/dim]")
    console.print()

    round_num = sm.get_round_count(sess_id) + 1
    _current_sess_id = [sess_id]  # 闭包可变引用，话题切换时更新

    def _do_chat_round(query: str, mcuid: str, round_num: int):
        """执行一轮对话，话题切换时自动更新 _current_sess_id[0]"""
        nonlocal context_manager, agent, sm, output_dir, generate_md
        sess_id = _current_sess_id[0]

        standard_input = StandardInput(
            raw_query=query,
            vehicleModel=mcuid,
            conversationId=sess_id,
            VIN="CLI_DEFAULT",
            faultOccurTime="2026-01-01T00:00:00",
            mileage=0.0,
        )

        # 构建上下文（三步降级 + 摘要注入）
        ctx = sm.get_conversation_context(sess_id)
        prepared = None
        if ctx:
            prepared = context_manager.prepare_from_context(ctx, query)

        # 话题切换检测：归档旧会话，创建新会话，继续诊断
        if prepared and prepared.metadata.topic_changed:
            old_sess_id = sess_id
            new_sess_id = f"{old_sess_id}-topic-{sm.get_round_count(old_sess_id) + 1}"
            # 先归档完整会话（含热层消息），再清空热层为新话题准备
            sm.archive(old_sess_id)
            ctx.hot_messages = []
            _current_sess_id[0] = new_sess_id
            sess_id = new_sess_id
            console.print(f"  [dim]话题已切换，旧会话 {old_sess_id} 已归档，新会话 {new_sess_id}[/dim]")
            # 新会话首次获取上下文，无历史消息
            ctx = sm.get_conversation_context(sess_id)

        # scope 检测判定不在范围内，直接返回 -3
        if prepared and not prepared.metadata.is_in_scope:
            from .models.converter import build_error_output
            standard_output = build_error_output(
                code=OutputCode.OUT_OF_SCOPE,
                msg="查询不在车辆电驱系统故障诊断范围内",
                standard_input=standard_input,
            )
        else:
            # 启用流式打印 ReAct 步骤
            agent._enable_react_stream()
            standard_output = agent.diagnose_with_standard_input(
                standard_input,
                prepared_messages=prepared.messages if prepared else None,
            )
            agent._stream_callback = None

        if standard_output.code == 0:
            from langchain_core.messages import messages_to_dict
            current_messages = messages_to_dict(getattr(agent, "_last_messages", []))
            sm.update(sess_id, query, current_messages)

            # 获取更新后的上下文，显示当前 Token 用量
            ctx_after = sm.get_conversation_context(sess_id)
            meta = prepared.metadata if prepared else None
            used_toks = meta.token_usage if meta else 0
            max_toks_ctx = meta.max_tokens if meta else settings.context.max_tokens
            util_ctx = meta.window_utilization if meta else 0.0
            hot_count = len(ctx_after.hot_messages) if ctx_after else 0
            summary_count = len(ctx_after.warm_summaries) if ctx_after else 0

            # 自动提取本轮对话知识
            if settings.knowledge.enabled and current_messages:
                _auto_extract_knowledge(current_messages, sess_id, settings)

            result = standard_output.diagnosis_result
            if result:
                similar_count = 0
                last_output = getattr(agent, "_last_diagnostic_output", None)
                if last_output and last_output.report:
                    similar_count = len(last_output.report.similar_cases)

                console.print(f"  [cyan]分类:[/cyan] {result.classification}")
                console.print(f"  [cyan]相似工况:[/cyan] {similar_count} 条")
                console.print(f"  [cyan]根因:[/cyan] {result.fault_root_cause[0] if result.fault_root_cause else 'N/A'}")
                console.print(f"  [cyan]方案:[/cyan] {result.solution[0] if result.solution else 'N/A'}")
                console.print(f"  [cyan]置信度:[/cyan] {standard_output.diagnosis_confidence:.0%}")
                console.print(f"  [dim]Token 用量: {used_toks}/{max_toks_ctx} ({util_ctx:.0%}) | 热层消息: {hot_count} | 温层摘要: {summary_count}[/dim]")
                console.print()
                # 把格式化结果存入 metadata，恢复时可直接显示（不干扰 hot_messages 序列化）
                ctx = sm.get_conversation_context(sess_id)
                if ctx:
                    history = ctx.metadata.setdefault("diagnosis_history", [])
                    history.append({
                        "classification": result.classification,
                        "similar_count": similar_count,
                        "root_cause": result.fault_root_cause[0] if result.fault_root_cause else "",
                        "solution": result.solution[0] if result.solution else "",
                        "confidence": standard_output.diagnosis_confidence,
                    })
                    if sm._redis_store:
                        sm._redis_store.save(ctx, ttl=sm._idle_timeout)
                # 生成报告（--generate-md 参数启用）
                if generate_md:
                    internal_output = getattr(agent, "_last_diagnostic_output", None)
                    if internal_output:
                        md_path = generate_markdown_report(internal_output, output_dir=output_dir)
                        db_paths = generate_db_entries(internal_output, output_dir=output_dir)
                        console.print(f"  [dim]📄 Markdown: {md_path}[/dim]")
                        console.print(f"  [dim]📊 CSV: {db_paths.get('csv', '')}[/dim]")
        else:
            console.print(f"  [red]诊断失败: {standard_output.msg}[/red]")

    # 聊天空闲超时检测：后台线程 + PromptSession.app.exit() 优雅退出
    import threading
    from prompt_toolkit import PromptSession
    _chat_idle_timeout = settings.context.chat_idle_timeout
    _idle_stop = threading.Event()
    _idle_reset = threading.Event()

    _prompt_session = PromptSession()

    def _idle_timer():
        while not _idle_stop.is_set():
            timed_out = _idle_reset.wait(timeout=_chat_idle_timeout)
            if _idle_stop.is_set():
                return
            if not timed_out:
                _prompt_session.app.exit(exception=EOFError("聊天空闲超时"))
                return
            _idle_reset.clear()

    _idle_thread = threading.Thread(target=_idle_timer, daemon=True)
    _idle_thread.start()

    # 如果传了初始问题，先处理这一轮，然后重置计时器
    if initial_question:
        _do_chat_round(initial_question, mcuid, round_num)
        round_num += 1
        _idle_reset.set()

    while True:
        try:
            user_input = _prompt_session.prompt(f"第{round_num}轮 > ").strip()
        except (EOFError, KeyboardInterrupt):
            _idle_stop.set()
            _idle_reset.set()
            sm.archive(_current_sess_id[0])
            console.print("\n[dim]已退出交互模式（会话已归档）[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            _idle_stop.set()
            _idle_reset.set()
            sm.archive(_current_sess_id[0])
            console.print(f"[dim]已退出，会话 {_current_sess_id[0]} 共 {round_num - 1} 轮（已归档）[/dim]")
            break
        if user_input.lower() == "/tool":
            agent.show_tool_details = not agent.show_tool_details
            settings.tool_call.show_details = agent.show_tool_details
            status = "开启" if agent.show_tool_details else "关闭"
            console.print(f"[dim]工具调用详情已{status}[/dim]")
            continue

        _idle_reset.set()
        _do_chat_round(user_input, mcuid, round_num)

        # 显示指标面板
        metrics.record_turn()
        stats = sm.get_metrics()
        metrics.update_session_stats(stats["active_sessions"], stats["avg_turns"])
        panel_text = metrics.render_panel()
        if panel_text:
            console.print(Panel(panel_text, title="📊 运行指标", style="dim"))

        round_num += 1


# ---------------------------------------------------------------------------
# 会话管理命令
# ---------------------------------------------------------------------------


@app.command()
def session_list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """列出所有会话（活跃 + 归档）"""
    _setup_logging(verbose)
    from .agent.session_manager import SessionManager

    sm = SessionManager()
    active = sm.list_active()
    archived = sm.list_archived()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("状态", style="bold", width=10)
    table.add_column("会话 ID", width=40)
    table.add_column("轮次", justify="right", width=6)

    for sid in active:
        ctx = sm.get_conversation_context(sid)
        turns = ctx.total_turns if ctx else "?"
        table.add_row("[green]活跃[/green]", sid, str(turns))

    for sid in archived:
        # 从归档文件读取轮次
        import json
        archive_path = Path("data/sessions/archive") / f"{sid}.json"
        turns = "-"
        if archive_path.exists():
            try:
                with open(archive_path) as f:
                    data = json.load(f)
                turns = str(data.get("total_turns", "?"))
            except Exception:
                pass
        table.add_row("[dim]归档[/dim]", sid, turns)

    console.print(Panel.fit("📋 会话列表", style="bold blue"))
    if not active and not archived:
        console.print("  无会话")
    else:
        console.print(table)
        console.print(f"  活跃: {len(active)} | 归档: {len(archived)}")


@app.command()
def session_show(
    conversation_id: str = typer.Argument(..., help="会话 ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """查看会话详情"""
    _setup_logging(verbose)
    from .agent.session_manager import SessionManager

    sm = SessionManager()
    status = sm.get_session_status(conversation_id)

    console.print(Panel.fit(f"📄 会话: {conversation_id}", style="bold blue"))
    console.print(f"  状态: [cyan]{status}[/cyan]")

    if status == "archived":
        # 从冷层读取
        import json
        from pathlib import Path
        archive_path = Path("data/sessions/archive") / f"{conversation_id}.json"
        if archive_path.exists():
            with open(archive_path) as f:
                data = json.load(f)
            console.print(f"  总轮次: {data.get('total_turns', '?')}")
            # 兼容两种归档格式：ArchivedSession(topics) 和 ConversationContext(warm_summaries)
            topics = data.get("topics", data.get("warm_summaries", []))
            console.print(f"  话题数: {len(topics)}")
            for t in topics:
                summary = t.get("summary", "")[:100]
                label = t.get("topic_label", t.get("topic_id", "?"))
                console.print(f"    - [{label}] {summary}...")
    else:
        ctx = sm.get_conversation_context(conversation_id)
        if ctx:
            console.print(f"  总轮次: {ctx.total_turns}")
            console.print(f"  热层消息: {len(ctx.hot_messages)} 条")
            console.print(f"  温层摘要: {len(ctx.warm_summaries)} 个")
            console.print(f"  创建时间: {ctx.created_at[:19]}")
            console.print(f"  最后活动: {ctx.last_activity_at[:19]}")
            if ctx.current_topic:
                console.print(f"  当前话题: {ctx.current_topic.topic_label}")
        else:
            console.print("[yellow]  会话不在内存中[/yellow]")


@app.command()
def session_archive(
    conversation_id: str = typer.Argument(..., help="会话 ID"),
    confirm: bool = typer.Option(False, "--confirm", "-y", help="确认归档"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """手动归档会话"""
    _setup_logging(verbose)
    if not confirm:
        console.print("[yellow]请使用 --confirm 确认归档操作[/yellow]")
        raise typer.Exit(1)

    from .agent.session_manager import SessionManager

    sm = SessionManager()
    result = sm.archive(conversation_id)
    if result:
        console.print(f"[green]✅ 会话 {conversation_id} 已归档 ({result.total_turns} 轮)[/green]")
    else:
        console.print(f"[red]❌ 归档失败: 会话 {conversation_id} 不存在[/red]")


@app.command()
def session_audit(
    conversation_id: Optional[str] = typer.Argument(None, help="会话 ID（不指定则列出所有有审计日志的会话）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """查看审计日志"""
    _setup_logging(verbose)
    from .agent.context.audit import audit

    if conversation_id:
        events = audit.read_logs(conversation_id)
        console.print(Panel.fit(f"📋 审计日志: {conversation_id}", style="bold blue"))
        if not events:
            console.print("  无审计日志")
        else:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("时间", width=22)
            table.add_column("事件", style="bold", width=14)
            table.add_column("详情", width=60)
            for e in events:
                ts = e.get("_timestamp", "")[:19]
                event = e.get("event", "")
                detail = ", ".join(
                    f"{k}={v}" for k, v in e.items()
                    if k not in ("event", "conversation_id", "_timestamp")
                )
                table.add_row(ts, event, detail[:58])
            console.print(table)
            console.print(f"  共 {len(events)} 条记录")
    else:
        sessions = audit.list_sessions()
        console.print(Panel.fit("📋 审计日志会话列表", style="bold blue"))
        if not sessions:
            console.print("  无审计日志")
        else:
            for sid in sessions:
                events = audit.read_logs(sid)
                console.print(f"  [cyan]{sid}[/cyan] — {len(events)} 条记录")


@app.command()
def session_cleanup(
    confirm: bool = typer.Option(False, "--confirm", "-y", help="确认清理"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """手动清理过期归档"""
    _setup_logging(verbose)
    if not confirm:
        console.print("[yellow]请使用 --confirm 确认清理操作[/yellow]")
        raise typer.Exit(1)

    from .agent.retention import RetentionPolicy
    settings = get_settings()
    rt = settings.context.retention
    policy = RetentionPolicy(
        archive_dir="data/sessions/archive",
        retention_days=rt.retention_days,
        max_archive_size_mb=rt.max_archive_size_mb,
    )
    result = policy.cleanup()
    console.print(Panel.fit("🧹 归档清理完成", style="bold blue"))
    console.print(f"  删除文件: {result['deleted_files']}")
    console.print(f"  释放空间: {result['freed_bytes'] / 1024 / 1024:.1f} MB")
    console.print(f"  剩余文件: {result['remaining_files']}")


@app.command()
def adapter(
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """启动平台适配层服务（FastAPI）"""
    _setup_logging(verbose)
    console.print(Panel.fit("🔌 EV Drive 诊断 Agent 适配层", style="bold blue"))
    console.print(f"  监听: [cyan]{host}:{port}[/cyan]")
    console.print(f"  API:  [dim]http://{host}:{port}/api/v1/diagnoses/async[/dim]")
    console.print(f"  健康: [dim]http://{host}:{port}/health/live[/dim]")
    console.print(f"  Manifest: [dim]http://{host}:{port}/.well-known/diagnostic-agent-manifest[/dim]")
    import uvicorn
    from .adapter.server import app
    uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")


def main():
    app()


if __name__ == "__main__":
    main()
