# CAN 报文转换工具 (can_converter)

将车辆 CAN/CAN FD 报文文件（.asc / .blf / .mf4 等）结合 DBC 文件解码为结构化 CSV/Excel 数据。

## 功能概述

- **多格式支持**：asc、blf、mf4、mdf、trc、csv、log、sqlite
- **DBC 解码**：基于 cantools 将 CAN 报文原始字节解析为物理信号值
- **灵活筛选**：按消息 ID 或信号名称导出指定数据
- **双格式导出**：CSV 和 Excel（xlsx）均支持
- **Agent 集成**：封装为 LangChain BaseTool，可直接被 GraphRAG Agent 调用
- **结构化输出**：返回 JSON 格式结果，包含采集统计、DBC 信息和文件列表

## 模块结构

```
graphrag_agent/
├── can_analysis/                          # 纯领域逻辑（不含 Agent 依赖）
│   ├── __init__.py                        # 模块导出
│   ├── schemas.py                         # Pydantic 数据模型
│   ├── decoder.py                         # CAN/DBC 解码核心
│   └── exporters.py                       # CSV/Excel 导出
├── tools/                                 # LangChain Tool 集合
│   ├── __init__.py                        # 模块导出
│   └── can_converter_tool.py              # Agent 可调用的工具封装
└── asc_to_excel.py                        # 命令行入口（独立使用）
```

## 安装依赖

```bash
pip install cantools python-can pandas openpyxl pydantic langchain-core
```

## 使用方式

### 方式一：命令行独立使用

直接运行 `asc_to_excel.py`，修改配置区域的路径：

```python
INPUT_FILE = 'can_parser/log.asc'
DBC_FILE = 'can_parser/matrix.dbc'
OUTPUT_DIR = 'output'

SELECTED_SIGNALS = [
    'EngineSpeed',
    'VehicleSpeed',
]
```

```bash
python asc_to_excel.py
```

### 方式二：Python API 调用

```python
from graphrag_agent.tools.can_analysis.decoder import CANDecoder
from graphrag_agent.tools.can_analysis.exporters import CANExporter
from graphrag_agent.tools.can_analysis.schemas import ExportFormat

# 1. 解码
decoder = CANDecoder('/path/to/matrix.dbc')
signal_data, capture = decoder.decode_file('/path/to/log.asc')

print(f"总帧数: {capture.total_frames}, 解码成功: {capture.decoded_frames}")

# 2. 导出
exporter = CANExporter(decoder)
files = exporter.export(signal_data, './output', ExportFormat.CSV)

for f in files:
    print(f"{f.csv_path}: {f.row_count} 行, 信号: {f.signal_names}")
```

### 方式三：Agent 工具调用

```python
from graphrag_agent.tools import CanConverterTool

tool = CanConverterTool()
result = tool._run(
    file_path='/data/can_log.asc',
    dbc_path='/dbc/matrix.dbc',
    selected_signals=['EngineSpeed', 'MotorTorque'],
    export_format='csv',
)
print(result)  # JSON 字符串
```

Agent 中注册：

```python
from graphrag_agent.tools import CanConverterTool

class MyAgent(BaseAgent):
    def _setup_tools(self) -> list:
        return [
            self.search_tool.get_tool(),
            CanConverterTool(),  # 新增
        ]
```

## 参数说明

### CanConverterTool 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file_path` | string | 是 | - | CAN 报文文件路径 |
| `dbc_path` | string | 是 | - | DBC 文件路径 |
| `output_dir` | string | 否 | `./output` | 输出目录 |
| `selected_signals` | list | 否 | None | 信号名称列表，不指定则全部导出 |
| `selected_ids` | list | 否 | None | 消息 ID 列表（十进制），不指定则全部导出 |
| `export_format` | string | 否 | `csv` | 导出格式：`csv` / `xlsx` / `both` |

### 输出 JSON 结构

```json
{
  "status": "success",
  "capture": {
    "duration_s": 83.5,
    "total_frames": 428531,
    "decoded_frames": 420112,
    "skipped_frames": 8419
  },
  "dbc": {
    "path": "/dbc/matrix.dbc",
    "message_count": 85,
    "signal_count": 1204
  },
  "output_files": [
    {
      "message_name": "EMS_EngineData",
      "message_id": "0A1",
      "csv_path": "./output/0x0A1_EMS_EngineData.csv",
      "row_count": 28341,
      "signal_names": ["EngineSpeed", "EngineTorque", "CoolantTemp"]
    }
  ],
  "warnings": [],
  "errors": []
}
```

## 数据模型

| 类 | 用途 |
|----|------|
| `CANDecoder` | DBC 加载、信号←→ID 映射、文件解码 |
| `CANExporter` | 将解码数据导出为 CSV/Excel 文件 |
| `CanConverterInput` | 工具输入参数模型 |
| `CanConverterOutput` | 工具输出结果模型 |
| `CaptureInfo` | 数据采集统计信息 |
| `DBCInfo` | DBC 文件信息 |
| `OutputFileInfo` | 单个导出文件信息 |

## 常见问题

**Q: 解码后信号值为空或全是 0？**
A: 检查 DBC 是否匹配报文文件。DBC 中的消息 ID、信号起始位和长度需要与发送端一致。

**Q: 某些帧被跳过（skipped）？**
A: 可能原因：消息 ID 不在 DBC 中、报文长度不匹配、多路复用信号配置问题。检查控制台输出的警告信息。

**Q: 支持 CAN FD 吗？**
A: 支持。`cantools` 和 `python-can` 均支持 CAN FD 报文解码。

**Q: 大文件（>1GB）性能如何？**
A: 当前实现为流式读取，内存占用可控。但导出为 Excel 时可能因行数过多导致性能下降，建议大文件使用 CSV 格式。

**Q: Agent 调用时如何处理错误？**
A: 工具返回的 JSON 中 `status` 字段为 `"error"` 时，`errors` 字段包含具体错误信息。Agent 可根据此信息决定后续行为。
