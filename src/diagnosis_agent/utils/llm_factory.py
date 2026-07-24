"""LLM 工厂模块

统一创建 LLM 实例，支持多种模型配置。
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> ChatOpenAI:
    """创建 LLM 实例

    Args:
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成长度
        api_key: API Key
        api_base: API 基础 URL

    Returns:
        ChatOpenAI 实例
    """
    settings = get_settings()

    llm_model = model or settings.llm.model
    llm_temperature = temperature if temperature is not None else settings.llm.temperature
    llm_max_tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens
    llm_api_key = api_key or settings.llm.api_key
    llm_api_base = api_base or settings.llm.api_base

    if not llm_api_key:
        raise RuntimeError(
            "LLM API key 未配置。请设置 DASHSCOPE_API_KEY 环境变量。"
        )

    logger.info(
        f"创建 LLM: model={llm_model}, temperature={llm_temperature}, "
        f"max_tokens={llm_max_tokens}"
    )

    try:
        return ChatOpenAI(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            api_key=llm_api_key,
            base_url=llm_api_base,
        )
    except Exception as e:
        raise RuntimeError(f"LLM 初始化失败: {e}") from e


def create_embedding(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
):
    """创建 Embedding 实例"""
    settings = get_settings()

    emb_model = model or settings.embedding.model
    emb_api_key = api_key or settings.embedding.api_key or settings.llm.api_key
    emb_api_base = api_base or settings.embedding.api_base or settings.llm.api_base

    logger.info(f"创建 Embedding: model={emb_model}")

    # 使用自定义的 DashScopeEmbeddings 包装器，兼容阿里云 API
    from .embedding_wrapper import DashScopeEmbeddings
    return DashScopeEmbeddings(
        model=emb_model,
        api_key=emb_api_key,
        api_base=emb_api_base,
    )