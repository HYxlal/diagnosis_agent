"""CAN 解码数据导出模块

将解码后的信号时序数据导出为 CSV 或 Excel 文件。
"""

import os
import pandas as pd
from typing import Dict, List

from .schemas import ExportFormat, OutputFileInfo


class CANExporter:
    """CAN 数据导出器

    将 CANDecoder 解码后的 signal_data 导出为 CSV / Excel 文件。
    每个 CAN 消息生成一个独立文件，包含 timestamp 和所有信号列。

    使用示例:
        exporter = CANExporter(decoder)
        files = exporter.export(signal_data, './output', ExportFormat.CSV)
    """

    def __init__(self, decoder):
        """初始化导出器

        Args:
            decoder: CANDecoder 实例，用于获取消息名称等元信息
        """
        self.decoder = decoder

    def export(
        self,
        signal_data: dict,
        output_dir: str,
        export_format: ExportFormat = ExportFormat.CSV,
    ) -> List[OutputFileInfo]:
        """将解码数据导出为文件

        Args:
            signal_data: CANDecoder.decode_file() 返回的信号数据字典
            output_dir: 输出目录路径
            export_format: 导出格式（csv / xlsx / both）

        Returns:
            OutputFileInfo 列表，每项描述一个导出文件
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files: List[OutputFileInfo] = []

        for mid, data in signal_data.items():
            if not data or 'timestamp' not in data:
                continue

            df = pd.DataFrame(data).sort_values('timestamp').reset_index(drop=True)
            msg = self.decoder._id_to_msg.get(mid)
            msg_name = msg.name if msg else f"Unknown_{mid:03X}"
            base_name = f"0x{mid:03X}_{msg_name}"[:64]
            signal_names = [c for c in df.columns if c != 'timestamp']
            row_count = len(df)

            if export_format in (ExportFormat.CSV, ExportFormat.BOTH):
                csv_path = os.path.join(output_dir, f"{base_name}.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                output_files.append(OutputFileInfo(
                    message_name=msg_name,
                    message_id=f"{mid:03X}",
                    csv_path=csv_path,
                    row_count=row_count,
                    signal_names=signal_names,
                ))

            if export_format in (ExportFormat.XLSX, ExportFormat.BOTH):
                xlsx_path = os.path.join(output_dir, f"{base_name}.xlsx")
                sheet_name = f"0x{mid:03X}_{msg_name}"[:31]
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                output_files.append(OutputFileInfo(
                    message_name=msg_name,
                    message_id=f"{mid:03X}",
                    csv_path=xlsx_path,
                    row_count=row_count,
                    signal_names=signal_names,
                ))

        return output_files
