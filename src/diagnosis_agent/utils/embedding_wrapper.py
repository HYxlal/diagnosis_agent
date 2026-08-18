"""Embedding 包装器

提供两种 embedding 实现，通过 `embedding.provider` 配置切换：

1. `openai`（默认）：用 `langchain_openai.OpenAIEmbeddings`，适用于所有
   OpenAI 兼容 API（阿里云百炼 /compatible-mode/v1/embeddings、DeepSeek、
   Moonshot、本地 vLLM 等）。阿里云 text-embedding-v2 已确认完全兼容
   OpenAI 响应格式（{"data": [{"embedding": [...], ...}]}）。
2. `dashscope`：保留旧的手写 requests 实现，作为非 OpenAI 兼容 API
   的回退模板（如 Cohere、Jina、本地 bge 服务），改造时照着改请求
   URL / body / 响应解析三处即可。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

from ..config import get_settings

logger = logging.getLogger(__name__)


class DashScopeEmbeddings(Embeddings):
    """非 OpenAI 兼容 API 的回退模板（原阿里云 DashScope 实现）

    保留这个类是为了：
    - 当 embedding.provider=dashscope 时走这个手写实现
    - 给非 OpenAI 兼容 API（Cohere/Jina/bge 等）提供改造模板：
      改请求 URL、body 格式、响应解析三处即可适配

    阿里云百炼现已原生兼容 OpenAI 格式，建议用 provider=openai。
    """

    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or get_settings().llm.api_key
        self.api_base = api_base or get_settings().llm.api_base
        if not self.api_base:
            self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档

        Raises:
            RuntimeError: 当 API 调用失败时抛出
        """
        if not texts:
            return []

        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本

        Raises:
            RuntimeError: 当 API 调用失败时抛出
        """
        result = self._embed_batch([text])
        if not result:
            raise RuntimeError("Embedding API 调用失败，无法获取查询向量")
        return result[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """调用 API 嵌入一批文本

        Raises:
            RuntimeError: 当 API 调用失败时抛出
        """
        url = f"{self.api_base}/embeddings"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": self.model,
            "input": texts,
        }

        try:
            response = requests.post(url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("data") and isinstance(data["data"], list):
                embeddings = []
                for item in data["data"]:
                    embedding = item.get("embedding", [])
                    if isinstance(embedding, list):
                        embeddings.append(embedding)
                    else:
                        import numpy as np
                        embeddings.append(np.array(embedding).tolist())
                return embeddings

            raise RuntimeError(f"Embedding API 响应格式异常: {data}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API 调用失败: {e}")
            raise RuntimeError(f"Embedding API 调用失败: {e}") from e

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入文档"""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询文本"""
        return self.embed_query(text)


def create_embedding_fn(
    model: str,
    api_key: str,
    api_base: str,
    provider: str = "openai",
) -> Embeddings:
    """工厂函数：按 provider 创建 Embedding 实例

    Args:
        model: 模型名（如 text-embedding-v2）
        api_key: API Key
        api_base: API 基础 URL
        provider: "openai"（默认，走 langchain_openai）或 "dashscope"（手写回退）

    Returns:
        LangChain Embeddings 实例
    """
    if provider == "dashscope":
        logger.info(f"使用 DashScope 手写 embedding: model={model}")
        return DashScopeEmbeddings(
            model=model,
            api_key=api_key,
            api_base=api_base,
        )

    # 默认走 OpenAI 兼容路径
    try:
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"使用 OpenAIEmbeddings: model={model}, base={api_base}")
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=api_base,
        )
    except Exception as e:
        logger.warning(
            f"OpenAIEmbeddings 初始化失败 ({e})，回退到 DashScope 手写实现"
        )
        return DashScopeEmbeddings(
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
