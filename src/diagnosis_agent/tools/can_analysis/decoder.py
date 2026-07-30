"""CAN 报文解码核心模块

负责加载 DBC 文件，解析信号-消息映射，并对 CAN 报文文件进行逐帧解码。
"""

import os
import cantools
import can
from collections import defaultdict
from typing import Optional, Set, Tuple, Dict

from .schemas import CaptureInfo, DBCInfo

SUPPORTED_FORMATS = {'asc', 'blf', 'mf4', 'mdf', 'trc', 'csv', 'log', 'sqlite'}


class CANDecoder:
    """CAN 报文解码器

    加载并缓存 DBC 数据库，提供信号名→消息ID的映射，支持对 CAN 报文文件
    进行逐帧解码，返回结构化的信号时序数据。

    使用示例:
        decoder = CANDecoder('/path/to/matrix.dbc')
        signal_data, capture = decoder.decode_file('/path/to/log.asc')
    """

    def __init__(self, dbc_path: str):
        """初始化解码器并加载 DBC 文件

        Args:
            dbc_path: DBC 文件路径

        Raises:
            FileNotFoundError: DBC 文件不存在
        """
        if not os.path.isfile(dbc_path):
            raise FileNotFoundError(f"DBC 文件不存在: {dbc_path}")
        self.dbc_path = dbc_path
        self.db = cantools.database.load_file(dbc_path)
        self._id_to_msg = {msg.frame_id: msg for msg in self.db.messages}
        self._signal_to_id: Dict[str, int] = {}
        for msg in self.db.messages:
            for sig in msg.signals:
                self._signal_to_id[sig.name] = msg.frame_id

    @property
    def message_ids(self) -> Set[int]:
        """返回 DBC 中所有消息 ID 集合"""
        return set(self._id_to_msg.keys())

    @property
    def signal_names(self) -> Set[str]:
        """返回 DBC 中所有信号名称集合"""
        return set(self._signal_to_id.keys())

    def resolve_ids(
        self,
        selected_ids: Optional[list] = None,
        selected_signals: Optional[list] = None,
    ) -> Tuple[Optional[Set[int]], list]:
        """根据用户指定的 ID 或信号名称，解析出最终要筛选的消息 ID 集合。

        Args:
            selected_ids: 用户指定的消息 ID 列表（十进制整数），空列表或 None 表示全部
            selected_signals: 用户指定的信号名称列表，空列表或 None 表示全部

        Returns:
            (filter_ids, warnings):
            - filter_ids: 要筛选的消息 ID 集合，None 表示不筛选
            - warnings: 解析过程中产生的警告信息列表
        """
        warnings: list = []
        filter_ids: Optional[Set[int]] = None

        if selected_signals:
            valid_sigs = []
            for sig_name in selected_signals:
                if sig_name in self._signal_to_id:
                    valid_sigs.append(sig_name)
                else:
                    warnings.append(f"信号 '{sig_name}' 在 DBC 中未找到，已忽略")
            if not valid_sigs:
                warnings.append("所有指定信号均无效，将导出全部消息")
            else:
                filter_ids = {self._signal_to_id[s] for s in valid_sigs}
                if selected_ids:
                    warnings.append(
                        "同时指定了 selected_ids 和 selected_signals，以信号名称为准"
                    )

        elif selected_ids:
            valid_ids = {mid for mid in selected_ids if mid in self._id_to_msg}
            invalid_ids = set(selected_ids) - valid_ids
            if invalid_ids:
                warnings.append(
                    f"消息 ID {[f'0x{mid:X}' for mid in invalid_ids]} "
                    f"在 DBC 中未找到，已忽略"
                )
            if not valid_ids:
                warnings.append("所有指定消息 ID 均无效，将导出全部消息")
            else:
                filter_ids = valid_ids

        return filter_ids, warnings

    def decode_file(
        self,
        file_path: str,
        filter_ids: Optional[Set[int]] = None,
        filter_signals: Optional[Set[str]] = None,
    ) -> Tuple[dict, CaptureInfo]:
        """解码 CAN 报文文件，返回信号时序数据和统计信息。

        Args:
            file_path: CAN 报文文件路径（支持 asc/blf/mf4/mdf/trc/csv/log/sqlite）
            filter_ids: 要保留的消息 ID 集合，None 表示全部
            filter_signals: 要保留的信号名称集合，None 表示全部

        Returns:
            (signal_data, capture_info):
            - signal_data: {message_id: {'timestamp': [...], 'signal_name': [...], ...}}
            - capture_info: 解码统计信息

        Raises:
            FileNotFoundError: CAN 报文文件不存在
            ValueError: 不支持的文件格式
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"CAN 报文文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: .{ext}，支持: {sorted(SUPPORTED_FORMATS)}"
            )

        signal_data: dict = defaultdict(lambda: defaultdict(list))
        total = ok = skip = 0
        first_ts = None
        last_ts = None

        for msg in can.io.LogReader(file_path):
            if not isinstance(msg, can.Message):
                continue
            total += 1
            mid = msg.arbitration_id

            if filter_ids is not None and mid not in filter_ids:
                skip += 1
                continue
            if mid not in self._id_to_msg:
                skip += 1
                continue

            try:
                decoded = self.db.decode_message(mid, msg.data)
                ts = msg.timestamp
                signal_data[mid]['timestamp'].append(ts)

                if first_ts is None:
                    first_ts = ts
                last_ts = ts

                for name, val in decoded.items():
                    if filter_signals and name not in filter_signals:
                        continue
                    if hasattr(val, 'value'):
                        val = val.value
                    signal_data[mid][name].append(val)
                ok += 1
            except Exception:
                skip += 1

        duration = (
            (last_ts - first_ts)
            if (first_ts is not None and last_ts is not None)
            else 0.0
        )

        capture = CaptureInfo(
            duration_s=round(duration, 3),
            total_frames=total,
            decoded_frames=ok,
            skipped_frames=skip,
        )
        return dict(signal_data), capture

    def get_dbc_info(self) -> DBCInfo:
        """获取当前加载的 DBC 文件信息"""
        return DBCInfo(
            path=self.dbc_path,
            message_count=len(self._id_to_msg),
            signal_count=len(self._signal_to_id),
        )
