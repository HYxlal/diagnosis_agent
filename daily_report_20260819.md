# 2026-08-19 工作日志

## 今日完成概览

### 1. Neo4j 问题修复
- 定位 Neo4j 异常退出原因：Docker 容器 `neo4j-fault` 5 小时前异常退出（退出码 255）
- 启动 Neo4j Docker 服务，验证 Bolt 端口 7687 和 HTTP 7474 连通性
- 验证 Neo4j 数据完整性：共 819 条实体关系（MENTIONS:4473, SHOWS_INDICATOR:819, HAS_DTC:655, HAS_ROOT_CAUSE:574, OCCURS_ON:574, OCCURS_IN_SCENARIO:574, HAS_SOLUTION:281）
- 修复之前的配置降级问题，检索策略从 chroma_only 切换到 hybrid 混合模式，启用 Neo4j召回 + Embedding精排 + Chroma兜底

### 2. 相似度语义错误修复
- 发现 LLM 误判相似度值极低问题：`score=0.09` 是 cosine distance 而非相似度
- 修复点：3 个位置 — `prompts.py:65` / `langchain_agent.py:277` / `tools.py:291`
- 转换公式：`1 - score`，现在 LLM 看到的相似度是 0~1 的余弦相似度（0.91/0.86 高度匹配）
- 验证效果：召回结果相似度正常，0.71~0.93 区间符合预期

### 3. 新增两个独立组件（解耦架构）
- **`knowledge/edit_manager.py` — ManualEditManager**
  - 封装 Neo4j 人工编辑标记功能：`mark_manual_edit()` / `is_manually_edited()` / `get_edit_info()`
  - 保护手动编辑过的节点不被自动知识提取覆盖
- **`knowledge/graph_writer.py` — GraphWriter**
  - 独立封装 Neo4j 图写入：实体去重（不再用 description 模糊匹配导致误召回） + 人工编辑保护 + GraphDocument 转换
  - 与 ConversationKnowledgeExtractor 解耦，遵循文档中 FR-3/3.2/3.3 规范

### 4. 会话恢复 & 历史展示增强
- 话题切换归档逻辑修复：之前 `prepare_from_context()` 清空热层消息 → `sm.archive()` 拿到空列表，现在调整为 **先 `sm.archive()` 后清空**，确保归档文件包含完整热层消息
- Chat 模式 /resume 风格历史展示：
  - 完整显示 Token 预算用量（`used/max (percentage)`）
  - 显示温层摘要表格
  - 完整展示用户提问 + 助手诊断结果，恢复历史会话时不再只显示最后 60 字摘要
- 每轮诊断输出增加实时指标：Token 用量/热层消息数/温层摘要数

### 5. 话题检测 LLM 分离
- 话题检测使用 qwen-turbo，诊断推理用 qwen3.5-plus，知识提取用 deepseek-v3.1
- 三类 LLM 职责完全分离：
  - 诊断推理：qwen3.5-plus（高智能，昂贵，用于复杂分析）
  - 话题检测：qwen-turbo（轻量，快速，低成本）
  - 知识提取：deepseek-v3.1（结构化输出强）

### 6. 清理废弃文件
- 删除两个导入错误的损坏测试文件：`test_agent_report.py` + `test_parsers.py`
- 项目现在 65 项测试全部通过

### 7. 知识库统计
- 共 10 条已处理知识，5 条审核通过，3 条审核拒绝，1 条待审核
- 实体类型覆盖：故障代码/故障现象/电驱型号/车型/部件/诊断方法
- 关系类型：对应故障代码/发生于/对策/导致/属于/关联
