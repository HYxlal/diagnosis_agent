"""CAN 报文兜底解码 — 预检索结果不足时，解码 CAN 文件生成信号摘要注入 Agent 上下文"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


async def preprocess_can_file(fault_data_url: str) -> str | None:
    """下载 CAN 文件 → 解码 → 提取信号摘要 → 返回 Agent 可读文本

    fault_data_url 为逗号分隔的多文件链接，首个为 CAN 日志，第二个为 DBC 文件(可选)。
    异步实现，不阻塞事件循环。
    """
    import pandas as pd

    urls = [u.strip() for u in fault_data_url.split(",") if u.strip()]
    if not urls:
        return None

    tmp_dir = Path("data/can_downloads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        can_url = urls[0]
        dbc_url = urls[1] if len(urls) > 1 else None
        can_path, dbc_path = None, None
        async with httpx.AsyncClient(timeout=300) as client:
            can_resp = await client.get(can_url)
            can_ext = Path(can_url).suffix or ".asc"
            can_path = str(tmp_dir / f"can_{uuid4().hex}{can_ext}")
            with open(can_path, "wb") as f:
                f.write(can_resp.content)
            if dbc_url:
                dbc_resp = await client.get(dbc_url)
                dbc_path = str(tmp_dir / f"dbc_{uuid4().hex}.dbc")
                with open(dbc_path, "wb") as f:
                    f.write(dbc_resp.content)

        if not dbc_path or not can_path:
            return None

        from .can_converter_tool import can_converter_impl
        result = json.loads(can_converter_impl(file_path=can_path, dbc_path=dbc_path, output_dir=str(tmp_dir)))
        if result.get("status") != "success":
            logger.warning(f"CAN 解码失败: {result.get('errors', 'unknown')}")
            return None

        capture = result.get("capture", {})
        output_files = result.get("output_files", [])

        # 遍历所有CSV文件，不再只取第一个
        all_signals = []
        all_ranges = []
        for of in output_files:
            csv_path = of.get("csv_path", "")
            msg_name = of.get("message_name", "未知消息")
            signal_names = of.get("signal_names", [])
            if not csv_path or not csv_path.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(csv_path)
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) == 0:
                    continue
                all_signals.extend(signal_names)
                all_ranges.append(f"  [{msg_name}]")
                for col in list(numeric_cols)[:15]:
                    all_ranges.append(
                        f"    {col}: {df[col].min():.1f} → {df[col].max():.1f} (均值 {df[col].mean():.1f})"
                    )
            except Exception as e:
                logger.warning(f"读取CSV失败 {csv_path}: {e}")
                continue

        summary_signals = "\n".join(all_ranges)

        lines = [
            "用户提供了 CAN 报文文件，已自动解码完毕。",
            f"采集信息: {capture.get('total_frames', '?')} 帧, "
            f"成功解码 {capture.get('decoded_frames', '?')} 帧, "
            f"时长 {capture.get('duration_s', '?')}s",
        ]
        if all_signals:
            unique_signals = list(dict.fromkeys(all_signals))
            lines.append(f"可用信号 ({len(unique_signals)} 个): {', '.join(unique_signals[:20])}")
            lines.append("各消息信号变化区间:")
            lines.append(summary_signals)
        lines.append("请根据以上信号值结合用户故障描述进行故障诊断推理。")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"CAN 预处理失败: {e}")
        return None
