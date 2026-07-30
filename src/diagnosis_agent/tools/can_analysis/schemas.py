"""CAN 报文解析工具的数据模型定义"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class CanLogFormat(str, Enum):
    """支持的 CAN 报文文件格式"""
    ASC = "asc"
    BLF = "blf"
    MF4 = "mf4"
    MDF = "mdf"
    TRC = "trc"
    CSV = "csv"
    LOG = "log"
    SQLITE = "sqlite"


class ExportFormat(str, Enum):
    """支持的导出格式"""
    CSV = "csv"
    XLSX = "xlsx"
    BOTH = "both"


class CanConverterInput(BaseModel):
    """CAN 转换工具的输入参数"""
    file_path: str = Field(..., description="CAN 报文文件路径")
    dbc_path: str = Field(..., description="DBC 文件路径")
    output_dir: str = Field(default="./output", description="CSV 输出目录")
    selected_ids: Optional[List[int]] = Field(
        default=None, description="筛选的消息 ID（十进制整数），不指定则导出全部"
    )
    selected_signals: Optional[List[str]] = Field(
        default=None, description="筛选的信号名称，不指定则导出全部"
    )
    export_format: ExportFormat = Field(
        default=ExportFormat.CSV, description="导出格式"
    )


class CaptureInfo(BaseModel):
    """数据采集统计信息"""
    duration_s: float = Field(default=0.0, description="数据持续时长（秒）")
    total_frames: int = Field(default=0, description="总报文数")
    decoded_frames: int = Field(default=0, description="成功解码数")
    skipped_frames: int = Field(default=0, description="跳过数")


class DBCInfo(BaseModel):
    """DBC 文件信息"""
    path: str = Field(default="", description="DBC 文件路径")
    message_count: int = Field(default=0, description="DBC 中定义的消息数")
    signal_count: int = Field(default=0, description="DBC 中定义的信号数")


class OutputFileInfo(BaseModel):
    """导出文件信息"""
    message_name: str = Field(default="", description="CAN 消息名称")
    message_id: str = Field(default="", description="消息 ID（十六进制字符串）")
    csv_path: str = Field(default="", description="导出的文件路径")
    row_count: int = Field(default=0, description="数据行数")
    signal_names: List[str] = Field(
        default_factory=list, description="包含的信号名称"
    )


class CanConverterOutput(BaseModel):
    """CAN 转换工具的输出结果"""
    status: str = Field(
        default="success", description="执行状态：success / error / warning"
    )
    capture: CaptureInfo = Field(default_factory=CaptureInfo)
    dbc: DBCInfo = Field(default_factory=DBCInfo)
    output_files: List[OutputFileInfo] = Field(
        default_factory=list, description="生成的输出文件列表"
    )
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    errors: List[str] = Field(default_factory=list, description="错误信息")
    summary: str = Field(default="", description="人类可读的处理摘要")

    def to_agent_context(self) -> str:
        """转换为 Agent（LLM）可直接使用的自然语言摘要"""
        lines = [
            "CAN 报文转换完成。",
            f"文件统计: 总 {self.capture.total_frames} 帧, "
            f"成功解码 {self.capture.decoded_frames} 帧, "
            f"跳过 {self.capture.skipped_frames} 帧。",
            f"DBC 信息: {self.dbc.path}, "
            f"包含 {self.dbc.message_count} 个消息定义, "
            f"{self.dbc.signal_count} 个信号定义。",
            f"输出文件: 共 {len(self.output_files)} 个。",
        ]
        for f in self.output_files:
            lines.append(
                f"  - 消息 {f.message_name} (ID=0x{f.message_id}): "
                f"{f.row_count} 行, 信号: {f.signal_names}"
            )
        if self.warnings:
            lines.append(f"警告: {'; '.join(self.warnings)}")
        return "\n".join(lines)
