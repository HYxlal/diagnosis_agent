"""异步任务管理 — 执行诊断 + 回调平台 + 知识沉淀"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx

from ..config import get_settings
from .models import (
    DiagnosisTask,
    ExecutionStatus,
    PlatformDiagnosisRequest,
    to_callback_result,
    to_error_callback,
    to_standard_input,
)
from .session import build_history_messages

logger = logging.getLogger(__name__)


class TaskManager:
    """内存任务存储（生产环境可换 Redis）"""

    def __init__(self):
        self._tasks: dict[str, DiagnosisTask] = {}
        self._by_client_id: dict[str, str] = {}

    def create_task(self, req: PlatformDiagnosisRequest) -> DiagnosisTask:
        """创建诊断任务，相同 clientRequestId 返回已有任务（幂等）"""
        existing = self._by_client_id.get(req.clientRequestId)
        if existing:
            task = self._tasks.get(existing)
            if task:
                return task
        task = DiagnosisTask(req)
        self._tasks[task.requestId] = task
        self._by_client_id[req.clientRequestId] = task.requestId
        return task

    def get(self, request_id: str) -> DiagnosisTask | None:
        """按 requestId 查询任务"""
        return self._tasks.get(request_id)

    def get_by_client_id(self, client_id: str) -> DiagnosisTask | None:
        """按 clientRequestId 查询（幂等校验用）"""
        rid = self._by_client_id.get(client_id)
        if rid:
            return self._tasks.get(rid)
        return None


task_manager = TaskManager()


async def execute_diagnosis(task: DiagnosisTask, req: PlatformDiagnosisRequest):
    """后台执行诊断，完成后回调平台"""
    try:
        task.status = ExecutionStatus.RUNNING.value
        task.touch()

        logger.info(f"[{task.requestId}] 开始诊断: {req.raw_query[:60]}...")

        # 1. 构建 StandardInput
        std_input = to_standard_input(req)

        # 2. 获取历史消息
        history = build_history_messages(req.conversationId, req.context_history)

        # 3. 预处理 CAN 文件（如有）
        prepared = None
        if req.faultDataUrl:
            can_summary = await preprocess_can_file(req.faultDataUrl)
            if can_summary:
                from langchain_core.messages import SystemMessage
                prepared = [SystemMessage(content=can_summary)]
                logger.info(f"[{task.requestId}] CAN 报文预处理完成")

        # 4. 执行诊断
        from ..agent.langchain_agent import LangChainDiagnosticAgent
        settings = get_settings()
        agent = LangChainDiagnosticAgent(settings=settings)
        output = agent.diagnose_with_standard_input(
            std_input, history_messages=history, prepared_messages=prepared,
        )

        # 5. 构建回调结果
        if output.code == 0:
            callback_body = to_callback_result(task, output)
        else:
            callback_body = to_error_callback(task, output.msg)

        task.result = callback_body
        task.status = ExecutionStatus.SUCCESS.value
        task.touch()

        logger.info(f"[{task.requestId}] 诊断完成, 回调: {task.callbackUrl}")

        # 6. 回调平台
        await _callback_platform(task, callback_body)

        # 7. 异步知识提取（不阻塞回调）
        asyncio.create_task(
            _extract_knowledge_async(req, output, settings)
        )

    except Exception as e:
        logger.error(f"[{task.requestId}] 诊断失败: {e}", exc_info=True)
        task.status = ExecutionStatus.FAILED.value
        task.touch()
        callback_body = to_error_callback(task, str(e))
        await _callback_platform(task, callback_body)


async def _callback_platform(task: DiagnosisTask, body) -> bool:
    """回调平台，支持重试"""
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    task.callbackUrl,
                    json=body.model_dump() if hasattr(body, 'model_dump') else body,
                )
                if resp.is_success:
                    logger.info(f"[{task.requestId}] 回调成功")
                    return True
                logger.warning(f"[{task.requestId}] 回调返回 {resp.status_code}, 重试 {attempt+1}/3")
        except Exception as e:
            logger.warning(f"[{task.requestId}] 回调异常: {e}, 重试 {attempt+1}/3")
        await asyncio.sleep(1)
    logger.error(f"[{task.requestId}] 回调失败（3次重试）")
    return False


async def _extract_knowledge_async(
    req: PlatformDiagnosisRequest,
    output,
    settings,
):
    """异步知识提取：从平台上下文 + 本轮诊断中提取实体关系，写入 Neo4j 知识图谱

    不阻塞主流程，失败时静默忽略。
    """
    """异步知识提取"""
    try:
        if not settings.knowledge.enabled:
            return
        from ..knowledge.extractor import ConversationKnowledgeExtractor

        messages = []
        for turn in req.context_history or []:
            if turn.get("query"):
                messages.append({"role": "user", "content": turn["query"]})
            if turn.get("agent_answer"):
                messages.append({"role": "assistant", "content": turn["agent_answer"]})
        messages.append({"role": "user", "content": req.raw_query})
        if output and output.diagnosis_result:
            result_text = json.dumps(output.diagnosis_result.model_dump(), ensure_ascii=False)
            messages.append({"role": "assistant", "content": result_text})

        extractor = ConversationKnowledgeExtractor()
        knowledge = extractor.extract_from_conversation(messages)
        logger.info(f"[{req.conversationId}] 知识提取完成: {knowledge.knowledge_id}, "
                     f"实体={len(knowledge.extracted_entities)}, 关系={len(knowledge.extracted_relationships)}")
    except Exception as e:
        logger.warning(f"知识提取失败: {e}")


async def preprocess_can_file(fault_data_url: str) -> str | None:
    """下载 CAN 文件 → 解码 → 提取信号摘要 → 返回 Agent 可读文本

    fault_data_url 为逗号分隔的多文件链接，首个为 CAN 日志，第二个为 DBC 文件(可选)。
    """
    import pandas as pd

    urls = [u.strip() for u in fault_data_url.split(",") if u.strip()]
    if not urls:
        return None

    tmp_dir = Path("data/can_downloads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        can_url = urls[0]
        dbc_url = urls[1] if len(urls) > 1 else None
        can_path, dbc_path = None, None
        async with httpx.AsyncClient(timeout=300) as client:
            can_resp = await client.get(can_url)
            can_ext = Path(can_url).suffix or ".asc"
            can_path = str(tmp_dir / f"can_{uuid4().hex}{can_ext}")
            with open(can_path, "wb") as f:
                f.write(can_resp.content)
            if dbc_url:
                dbc_resp = await client.get(dbc_url)
                dbc_path = str(tmp_dir / f"dbc_{uuid4().hex}.dbc")
                with open(dbc_path, "wb") as f:
                    f.write(dbc_resp.content)

        if not dbc_path or not can_path:
            return None

        from ..tools.can_converter_tool import can_converter_impl
        result = json.loads(can_converter_impl(file_path=can_path, dbc_path=dbc_path, output_dir=str(tmp_dir)))
        if result.get("status") != "success":
            logger.warning(f"CAN 解码失败: {result.get('errors', 'unknown')}")
            return None

        capture = result.get("capture", {})
        output_files = result.get("output_files", [])
        csv_file = next((f for f in output_files if f.endswith(".csv")), None)
        signals = []
        summary_signals = ""
        if csv_file:
            df = pd.read_csv(csv_file)
            numeric_cols = df.select_dtypes(include="number").columns
            signals = list(numeric_cols)
            ranges = []
            for col in list(numeric_cols)[:15]:
                ranges.append(f"  {col}: {df[col].min():.1f} → {df[col].max():.1f} (均值 {df[col].mean():.1f})")
            summary_signals = "\n".join(ranges)

        lines = [
            "用户提供了 CAN 报文文件，已自动解码完毕。",
            f"采集信息: {capture.get('total_frames', '?')} 帧, "
            f"成功解码 {capture.get('decoded_frames', '?')} 帧, "
            f"时长 {capture.get('duration', '?')}s",
        ]
        if signals:
            lines.append(f"可用信号 ({len(signals)} 个): {', '.join(signals[:15])}")
            lines.append("关键信号变化区间:")
            lines.append(summary_signals)
        lines.append("请根据以上信号值结合用户故障描述进行故障诊断推理。")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"CAN 预处理失败: {e}")
        return None