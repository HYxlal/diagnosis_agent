"""prompts 模块 — 统一管理所有 LLM Prompt

所有 prompt 集中在这里，module-level 变量导入，方便修改和审查。
"""

from .summarizer import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_MERGE_PROMPT,
)
from .agent import (
    AGENT_SYSTEM_PROMPT,
)

__all__ = [
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_MERGE_PROMPT",
    "AGENT_SYSTEM_PROMPT",
]