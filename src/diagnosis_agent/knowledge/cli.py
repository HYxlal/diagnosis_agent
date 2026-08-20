"""知识沉淀 CLI 命令组"""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import get_settings, reset_settings
from .extractor import ConversationKnowledgeExtractor

logger = logging.getLogger(__name__)
console = Console()

# 知识沉淀命令组
knowledge_app = typer.Typer(name="knowledge", help="知识沉淀管理")

_extractor_cache: Optional[ConversationKnowledgeExtractor] = None


def _get_extractor() -> ConversationKnowledgeExtractor:
    """获取全局缓存的知识提取器实例"""
    global _extractor_cache
    if _extractor_cache is not None:
        return _extractor_cache

    settings = get_settings()
    from ..utils.llm_factory import create_llm

    # 尝试构建 GraphWriter + ManualEditManager（替代旧的 neo4j_driver/neo4j_graph）
    graph_writer = None
    edit_manager = None
    try:
        from ..retrieval.neo4j_retriever import Neo4jFaultRetriever
        from langchain_neo4j import Neo4jGraph
        from .graph_writer import GraphWriter
        from .edit_manager import ManualEditManager
        neo4j_retriever = Neo4jFaultRetriever(settings=settings)
        if neo4j_retriever.available:
            neo4j_graph = Neo4jGraph(
                url=settings.neo4j.url,
                username=settings.neo4j.user,
                password=settings.neo4j.password,
                database="neo4j",
            )
            edit_manager = ManualEditManager(graph=neo4j_graph)
            graph_writer = GraphWriter(graph=neo4j_graph, edit_manager=edit_manager)
    except Exception:
        pass

    _extractor_cache = ConversationKnowledgeExtractor(
        llm=create_llm(temperature=0.0, max_tokens=2048),
        graph_writer=graph_writer,
        edit_manager=edit_manager,
        persistence_path=settings.knowledge.persistence_dir + "/" + settings.knowledge.persistence_file,
        tuple_delimiter=settings.knowledge.tuple_delimiter,
        record_delimiter=settings.knowledge.record_delimiter,
        completion_delimiter=settings.knowledge.completion_delimiter,
    )
    return _extractor_cache


@knowledge_app.command("list-pending")
def knowledge_list_pending():
    """查看待审核列表"""
    reset_settings()
    extractor = _get_extractor()
    pending = extractor.get_pending_reviews()
    if not pending:
        console.print("[dim]暂无待审核的知识[/dim]")
        return

    table = Table(title="待审核知识列表", show_header=True, header_style="bold magenta")
    table.add_column("序号", style="dim", width=6)
    table.add_column("知识ID", style="dim", width=20)
    table.add_column("对话ID", width=12)
    table.add_column("实体", width=20)
    table.add_column("关系", width=20)
    table.add_column("摘要", width=20)
    table.add_column("创建时间", width=15)

    for i, k in enumerate(pending, 1):
        entities = ", ".join(e.entity_name for e in k.extracted_entities[:3])
        rels = ", ".join(
            f"{r.source_id}→{r.relation_type}→{r.target_id}"
            for r in k.extracted_relationships[:3]
        )
        summary = k.conversation_context[:100].replace("\n", " ")
        # 在表格中显示实体数/关系数
        entity_count = f"{len(k.extracted_entities)}个: {entities[:15]}"
        rel_count = f"{len(k.extracted_relationships)}个: {rels[:15]}"
        table.add_row(
            str(i),
            k.knowledge_id[:20],
            k.conversation_id[:12],
            entity_count,
            rel_count,
            summary[:20],
            k.created_at[:16],
        )

    console.print(table)


@knowledge_app.command("approve")
def knowledge_approve(
    knowledge_id: str,
    reviewer: str,
    comment: Optional[str] = None,
    write: bool = False,
):
    """审核通过

    Args:
        knowledge_id: 知识ID
        reviewer: 审核人
        comment: 审核备注
        write: 是否同时写入 Neo4j
    """
    reset_settings()
    extractor = _get_extractor()
    if extractor.review_knowledge(knowledge_id, True, reviewer, comment):
        console.print(f"[green]审核通过: {knowledge_id}[/green]")
        if write:
            if extractor.write_approved_knowledge(knowledge_id):
                console.print(f"[green]已写入 Neo4j: {knowledge_id}[/green]")
            else:
                console.print(f"[yellow]Neo4j 不可用，未写入: {knowledge_id}[/yellow]")
    else:
        console.print(f"[red]知识 ID 不存在: {knowledge_id}[/red]")


@knowledge_app.command("reject")
def knowledge_reject(
    knowledge_id: str,
    reviewer: str,
    comment: Optional[str] = None,
):
    """审核拒绝

    Args:
        knowledge_id: 知识ID
        reviewer: 审核人
        comment: 拒绝原因
    """
    reset_settings()
    extractor = _get_extractor()
    if extractor.review_knowledge(knowledge_id, False, reviewer, comment):
        console.print(f"[yellow]已拒绝: {knowledge_id}[/yellow]")
    else:
        console.print(f"[red]知识 ID 不存在: {knowledge_id}[/red]")


@knowledge_app.command("write")
def knowledge_write(knowledge_id: str):
    """写入 Neo4j（仅 approved 状态可写入）"""
    reset_settings()
    extractor = _get_extractor()
    knowledge = extractor.get_knowledge_by_id(knowledge_id)
    if not knowledge:
        console.print(f"[red]知识 ID 不存在: {knowledge_id}[/red]")
        return
    if knowledge.status != "approved":
        console.print(f"[red]知识状态为 {knowledge.status}，仅 approved 可写入[/red]")
        return
    if extractor.write_approved_knowledge(knowledge_id):
        console.print(f"[green]已写入 Neo4j: {knowledge_id}[/green]")
    else:
        console.print(f"[yellow]Neo4j 不可用，写入失败: {knowledge_id}[/yellow]")


@knowledge_app.command("stats")
def knowledge_stats():
    """查看统计信息"""
    reset_settings()
    extractor = _get_extractor()
    stats = extractor.get_knowledge_stats()

    table = Table(title="知识沉淀统计", show_header=True, header_style="bold cyan")
    table.add_column("指标", style="dim", width=25)
    table.add_column("数值", justify="right", width=10)
    table.add_row("总提取次数", str(stats.extractions))
    table.add_row("累计实体数", str(stats.entities_extracted))
    table.add_row("累计关系数", str(stats.relationships_extracted))
    table.add_row("提交审核次数", str(stats.submitted_for_review))
    table.add_row("审核通过", str(stats.approved_knowledge))
    table.add_row("审核拒绝", str(stats.rejected_knowledge))
    table.add_row("错误次数", str(stats.errors))
    table.add_row("合并实体数", str(stats.merged_entities))
    console.print(table)


@knowledge_app.command("serve")
def knowledge_serve(
    host: str = "0.0.0.0",
    port: int = 8090,
):
    """启动 Web 审核服务

    Args:
        host: 监听地址，默认 0.0.0.0
        port: 监听端口，默认 8090
    """
    from .web import run_web

    console.print(f"[bold cyan]知识沉淀审核服务[/bold cyan]")
    console.print(f"  待审核: [green]http://{host}:{port}[/green]")
    console.print(f"  知识库: [green]http://{host}:{port}/?tab=knowledge[/green]")
    run_web(host=host, port=port)