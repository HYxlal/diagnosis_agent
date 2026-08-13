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


class FilterRetrievalConfig(BaseModel):
    default_top_k: int = 10
    filter_fields: list[str] = Field(
        default_factory=lambda: [
            "vehicle_type",
            "drive_code",
            "dtc_code",
            "dashboard_indicator",
            "fault_scenario",
        ]
    )


class HybridRetrievalConfig(BaseModel):
    semantic_weight: float = 0.7
    filter_weight: float = 0.3
    filter_expansion_ratio: int = 2


class RetrievalConfig(BaseModel):
    # 检索策略：chroma_only / neo4j_first / hybrid
    strategy: str = "neo4j_first"
    semantic: SemanticRetrievalConfig = Field(default_factory=SemanticRetrievalConfig)
    filter: FilterRetrievalConfig = Field(default_factory=FilterRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)


class ToolsConfig(BaseModel):
    """Agent 工具默认参数"""
    search_top_k: int = 5
    filter_top_k: int = 10


class InputRouterConfig(BaseModel):
    """InputRouter 意图路由配置"""
    enabled: bool = True
    model: str = "qwen-turbo"
    temperature: float = 0.1


class AgentConfig(BaseModel):
    max_iterations: int = 10
    verbose: bool = True
    similarity_threshold: float = 0.65


class ReportConfig(BaseModel):
    output_dir: str = "output"
    markdown_template: str = "default"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    samples_dir: str = "data/samples"
    output_dir: str = "output"


class AppConfig(BaseModel):
    name: str = "diagnosis_agent"
    version: str = "0.1.0"


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
    max_tokens: int = 8000        # 消息列表 token 预算上限

    # ── 热层配置 ──
    window_size: int = 5          # 热层窗口大小（轮次），保留最近 N 轮完整消息

    # ── 温层配置 ──
    summary_enabled: bool = False       # 是否启用摘要（Step 1 实现）
    summary_max_tokens: int = 500       # 摘要最大 token 数
    summary_strategy: str = "rule"      # 摘要策略: llm | rule | template

    # ── 话题检测配置 ──
    topic_detection_enabled: bool = False      # 是否启用话题检测
    topic_detection_strategy: str = "rule"     # 话题检测策略: embedding | llm | hybrid | rule
    topic_detection_model: str = ""            # 话题检测 LLM 模型，空则使用主 LLM
    topic_similarity_high: float = 0.6         # embedding 快筛高阈值（≥此值判 same）
    topic_similarity_low: float = 0.27         # embedding 快筛低阈值（≤此值判 diff）

    # ── 话题检测信号词（外部化配置） ──
    # switch: 检测到直接判 different（用户明确切换话题）
    # continue: 检测到增加延续倾向（辅助，默认空列表）
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
    # 用 ctx.current_topic.key_entities 与 query 做字符串匹配
    # 重叠率高（>50%）→ 辅助判 same
    entity_overlap_enabled: bool = False

    # ── 时间衰减检测 ──
    # 短间隔（<short_sec）+ 短消息 → 判延续 same
    # 长间隔（>long_sec）→ 判可能切换 different
    topic_time_decay_enabled: bool = False
    topic_time_decay_short_sec: int = 30       # 短间隔阈值（秒）
    topic_time_decay_short_max_len: int = 20   # 短消息最大字符数
    topic_time_decay_long_sec: int = 1800      # 长间隔阈值（秒）

    # ── 领域范围检测（scope） ──
    # 首轮 + 话题切换后，判断 query 是否在电驱诊断范围内
    # 只做 out 关键词快速排除，不做 in 关键词判断（太粗糙，如"电机相关的工程师"会被误判）
    scope_detection_enabled: bool = False
    scope_use_llm: bool = True                    # 关键词无法判断时，是否用 LLM 精判
    scope_out_keywords: list[str] = Field(default_factory=lambda: [
        "电池包", "充电桩", "OBC", "音响", "空调", "外观",
        "娱乐", "导航", "座椅", "车窗", "天气", "股票",
        "外卖", "电影", "快递",
    ])

    # ── 摘要配置 ──
    summary_model: str = ""                    # 摘要 LLM 模型，空则使用主 LLM

    # ── 生命周期配置 ──
    max_turns: int = 100                   # 最大保留轮次（超过后强制归档）
    session_idle_timeout: int = 3600       # 会话空闲超时（秒）
    session_max_lifetime: int = 86400      # 会话最大存活时间（秒）

    # ── 缓存配置 ──
    cache_context_window: int = 3          # 缓存上下文窗口大小

    # ── 降级配置 ──
    emergency_min_turns: int = 1           # 紧急截断时保留的最小轮次
    degradation_logging: bool = True       # 是否启用降级日志


class Settings(BaseModel):
    """全局配置根模型"""

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    input_router: InputRouterConfig = Field(default_factory=InputRouterConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
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
