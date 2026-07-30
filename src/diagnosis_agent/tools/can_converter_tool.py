"""CAN 报文转换工具

封装 graphrag_agent 复用的 CANDecoder/CANExporter，作为 LangChain StructuredTool
挂到 DiagnosticTools 上。

职责：
- 把 .asc/.blf/.mf4 等报文文件 + DBC 解码为结构化 CSV/Excel
- 返回 JSON 结果供 Agent 阅读（含采集统计 / DBC 信息 / 导出文件列表）

注意：本工具是同步执行，不做流式。Agent 调用后应基于返回的 JSON 文件列表
再决定是否读取 CSV 内容做进一步分析。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .can_analysis.decoder import CANDecoder
from .can_analysis.exporters import CANExporter
from .can_analysis.schemas import CanConverterOutput, ExportFormat

logger = logging.getLogger(__name__)


class CanConverterInputSchema(BaseModel):
    """can_converter 工具输入 schema

    list 参数保持为 list（StructuredTool 支持原生 list 序列化），
    避免 LLM 输出逗号分隔字符串导致解析歧义。
    """
    file_path: str = Field(
        description="CAN 报文文件路径，支持 asc/blf/mf4/mdf/trc/csv/log/sqlite 格式"
    )
    dbc_path: str = Field(description="DBC 文件路径，用于解码报文信号")
    output_dir: str = Field(
        default="./output",
        description="CSV/Excel 输出目录",
    )
    selected_signals: Optional[list[str]] = Field(
        default=None,
        description=(
            "要导出的信号名称列表，如 ['EngineSpeed', 'VehicleSpeed']。"
            "不指定则导出全部信号"
        ),
    )
    selected_ids: Optional[list[int]] = Field(
        default=None,
        description=(
            "要导出的消息 ID 列表（十进制整数），如 [161, 256]。"
            "不指定则导出全部消息"
        ),
    )
    export_format: str = Field(
        default="csv",
        description="导出格式: 'csv'、'xlsx' 或 'both'",
    )


def can_converter_impl(
    file_path: str,
    dbc_path: str,
    output_dir: str = "./output",
    selected_signals: Optional[list[str]] = None,
    selected_ids: Optional[list[int]] = None,
    export_format: str = "csv",
) -> str:
    """CAN 报文转换实现

    返回 JSON 字符串（CanConverterOutput 序列化），Agent 直接阅读。
    失败时返回 {"status": "error", "errors": [...]}，不抛异常。
    """
    # 校验导出格式
    try:
        fmt = ExportFormat(export_format.lower())
    except ValueError:
        return json.dumps(
            {
                "status": "error",
                "errors": [f"不支持的导出格式: {export_format}，可选: csv, xlsx, both"],
            },
            ensure_ascii=False,
        )

    # 加载 DBC
    try:
        decoder = CANDecoder(dbc_path)
    except FileNotFoundError as e:
        return json.dumps(
            {"status": "error", "errors": [str(e)]},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "errors": [f"DBC 加载失败: {str(e)}"]},
            ensure_ascii=False,
        )

    # 解析筛选条件
    filter_ids, warnings = decoder.resolve_ids(selected_ids, selected_signals)
    signal_filter = set(selected_signals) if selected_signals else None

    # 解码报文
    try:
        signal_data, capture = decoder.decode_file(
            file_path, filter_ids, signal_filter
        )
    except FileNotFoundError as e:
        return json.dumps(
            {"status": "error", "errors": [str(e)]},
            ensure_ascii=False,
        )
    except ValueError as e:
        return json.dumps(
            {"status": "error", "errors": [str(e)]},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "errors": [f"报文解码失败: {str(e)}"]},
            ensure_ascii=False,
        )

    # 无数据
    if not signal_data:
        result = CanConverterOutput(
            status="warning",
            capture=capture,
            warnings=warnings + ["未解码到任何有效数据，请检查 DBC 是否匹配报文"],
        )
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

    # 导出文件
    exporter = CANExporter(decoder)
    output_files = exporter.export(signal_data, output_dir, fmt)

    result = CanConverterOutput(
        status="success",
        capture=capture,
        dbc=decoder.get_dbc_info(),
        output_files=output_files,
        warnings=warnings,
    )

    logger.info(
        f"CAN 转换完成: file={file_path}, 帧数={capture.total_frames}, "
        f"成功={capture.decoded_frames}, 导出 {len(output_files)} 个文件"
    )
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


def build_can_converter_tool() -> StructuredTool:
    """构造 can_converter StructuredTool 实例

    供 DiagnosticTools.get_tool_list() 注册使用。
    """
    return StructuredTool.from_function(
        func=can_converter_impl,
        name="can_converter",
        description=(
            "将车辆 CAN/CAN FD 报文文件（.asc/.blf/.mf4 等）解码并转换为 CSV/Excel 文件。"
            "使用 DBC 文件解码报文中的物理信号值。"
            "适用场景：用户上传 CAN 报文文件后，先调用本工具将原始报文转为结构化 CSV，"
            "再进行信号分析和故障诊断。"
            "输入：file_path（报文文件路径）、dbc_path（DBC 路径）、output_dir（输出目录，默认 ./output）、"
            "selected_signals（信号名列表，可选）、selected_ids（消息 ID 列表，可选）、"
            "export_format（csv/xlsx/both，默认 csv）。"
            "返回：JSON 字符串，包含 status、capture（采集统计）、dbc（DBC 信息）、"
            "output_files（导出文件列表）、warnings、errors。"
        ),
        args_schema=CanConverterInputSchema,
    )
