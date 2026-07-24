"""Embedding 包装器

解决阿里云 embedding API 与 langchain_openai.OpenAIEmbeddings 的兼容性问题。
阿里云 API 对输入格式有特殊要求，需要进行格式转换。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

from ..config import get_settings

logger = logging.getLogger(__name__)


class DashScopeEmbeddings:
    """兼容阿里云 DashScope embedding API 的包装器"""

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
        """批量嵌入文档"""
        if not texts:
            return []

        # 阿里云 API 限制每次最多 25 条
        batch_size = 25
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        result = self._embed_batch([text])
        return result[0] if result else []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """调用阿里云 API 嵌入一批文本"""
        url = f"{self.api_base}/embeddings"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # 阿里云 API 要求输入格式
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

            logger.error(f"Embedding API 响应格式异常: {data}")
            return []

        except Exception as e:
            logger.error(f"Embedding API 调用失败: {e}")
            return []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入文档"""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询文本"""
        return self.embed_query(text)