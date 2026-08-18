"""配置加载模块

从 .env 和 config.yaml 加载配置，API key / 模型 / 路径全外部化。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic 配置模型
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    model: str = "qwen-plus"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key: str = ""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-v4"
    api_key: str = ""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # dashscope（默认，手写 requests，兼容阿里云百炼 input 格式）
    # openai（走 langchain_openai，仅用于严格兼容 OpenAI tokenizer 的 API）
    provider: str = "dashscope"


class VectorStoreConfig(BaseModel):
    type: str = "chroma"
    persist_dir: str = "data/chroma"
    collection_name: str = "incidents"


class SemanticRetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.3


class HybridRetrievalConfig(BaseModel):
    """两段式检索权重融合（预留，待实现）"""
    semantic_weight: float = 0.7
    filter_weight: float = 0.3
    filter_expansion_ratio: int = 2


class RetrievalConfig(BaseModel):
    strategy: str = "neo4j_first"
    semantic: SemanticRetrievalConfig = Field(default_factory=SemanticRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)


class ToolsConfig(BaseModel):
    """Agent 工具默认参数"""
    search_top_k: int = 5


class ReportConfig(BaseModel):
    """报告输出配置"""
    output_dir: str = "output"


class Neo4jConfig(BaseModel):
    url: str = ""
    user: str = ""
    password: str = ""
    min_candidates: int = 3       # Neo4j 召回少于此数 → 触发 Chroma 兜底
    default_depth: int = 1       # 默认关系扩展深度
    fallback_to_chroma: bool = True  # Neo4j 不可用时是否降级到 Chroma


class ContextConfig(BaseModel):
    """上下文管理配置 — 分层记忆（热/温/冷）"""

    # ── Token 预算 ──
    max_tokens: int = 8000

    # ── 热层配置 ──
    window_size: int = 5

    # ── 温层配置 ──
    summary_enabled: bool = False
    summary_max_tokens: int = 500
    summary_strategy: str = "rule"

    # ── 话题检测配置 ──
    topic_detection_enabled: bool = False
    topic_detection_strategy: str = "rule"
    topic_detection_model: str = ""
    topic_similarity_high: float = 0.6
    topic_similarity_low: float = 0.27

    # ── 话题检测信号词 ──
    topic_signal_words: dict = Field(
        default_factory=lambda: {
            "switch": [
                "换一个问题",
                "换个话题",
                "新问题",
                "另外",
                "另一个问题",
                "不是这个",
                "换一个",
            ],
            "continue": [],
        }
    )

    # ── 实体重叠检测 ──
    entity_overlap_enabled: bool = False

    # ── 时间衰减检测 ──
    topic_time_decay_enabled: bool = False
    topic_time_decay_short_sec: int = 30
    topic_time_decay_short_max_len: int = 20
    topic_time_decay_long_sec: int = 1800

    # ── 领域范围检测（scope） ──
    scope_detection_enabled: bool = False
    scope_use_llm: bool = True
    scope_out_keywords: list[str] = Field(default_factory=lambda: [
        "电池包", "充电桩", "OBC", "音响", "空调", "外观",
        "娱乐", "导航", "座椅", "车窗", "天气", "股票",
        "外卖", "电影", "快递",
    ])

    # ── 摘要配置 ──
    summary_model: str = ""

    # ── 生命周期配置 ──
    session_idle_timeout: int = 3600
    session_max_lifetime: int = 86400
    chat_idle_timeout: int = 1800
    emergency_min_turns: int = 2

    # ── Redis 存储配置 ──
    redis: "RedisConfig" = Field(default_factory=lambda: RedisConfig())


class RedisConfig(BaseModel):
    """Redis 存储配置 — 热层/温层持久化"""
    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "session:"


class Settings(BaseModel):
    """全局配置根模型"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)


# ---------------------------------------------------------------------------
# 配置加载逻辑
# ---------------------------------------------------------------------------

def _resolve_env_placeholders(value: Any) -> Any:
    """递归解析字符串中的 ${ENV_VAR:default} 占位符"""

    if isinstance(value, str):
        # 匹配 ${VAR} 或 ${VAR:default}
        pattern = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        return pattern.sub(replacer, value)

    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]

    return value


def load_settings(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> Settings:
    """加载配置

    Args:
        config_path: config.yaml 路径，默认为项目根目录下的 config.yaml
        env_path: .env 文件路径，默认为项目根目录下的 .env

    Returns:
        Settings 实例
    """
    # 确定项目根目录（src 的上一级）
    project_root = Path(__file__).resolve().parents[2]

    # 加载 .env
    if env_path is None:
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)

    if env_path.exists():
        load_dotenv(env_path)

    # 加载 config.yaml
    if config_path is None:
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    raw_config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}

    # 解析 ${ENV_VAR:default} 占位符
    resolved_config = _resolve_env_placeholders(raw_config)

    return Settings(**resolved_config)


# 全局单例
_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """重置全局配置（主要用于测试）"""
    global _settings
    _settings = None
