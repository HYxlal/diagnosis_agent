"""异步任务管理 — 执行诊断 + 回调平台 + 知识沉淀

CAN 兜底解码逻辑已下沉到 Agent 主流程（见 tools/can_fallback.py），
本模块只负责请求接入、任务调度和回调。
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import httpx

from ..config import get_settings
from ..agent.langchain_agent import LangChainDiagnosticAgent
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
    """后台执行诊断，完成后回调平台

    职责：任务状态管理 + 委托 Agent 诊断 + 回调平台 + 知识提取。
    业务逻辑（预检索、CAN 兜底、主诊断）全部在 Agent 内部完成。
    """
    try:
        task.status = ExecutionStatus.RUNNING.value
        task.touch()

        logger.info(f"[{task.requestId}] 开始诊断: {req.raw_query[:60]}...")

        # 1. 构建 StandardInput + 获取历史消息
        std_input = to_standard_input(req)
        history = build_history_messages(req.conversationId, req.context_history)

        # 2. 委托 Agent 完成诊断（预检索 + CAN 兜底 + 主诊断全部在 Agent 内部）
        settings = get_settings()
        agent = LangChainDiagnosticAgent(settings=settings)
        output = await agent.diagnose_with_can_fallback(
            std_input, history_messages=history,
        )

        # 3. 构建回调结果
        if output.code == 0:
            callback_body = to_callback_result(task, output)
        else:
            callback_body = to_error_callback(task, output.msg)

        task.result = callback_body
        task.status = ExecutionStatus.SUCCESS.value
        task.touch()

        logger.info(f"[{task.requestId}] 诊断完成, 回调: {task.callbackUrl}")

        # 4. 回调平台
        await _callback_platform(task, callback_body)

        # 5. 异步知识提取（不阻塞回调）
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