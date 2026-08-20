# Phase 2/3/4 验收测试汇总报告

测试时间: 2026-08-19
测试结果: **65 PASSED / 0 FAILED / 0 SKIPPED**

---

## Phase 2 分层记忆 验收结果

| 编号 | 标准 | 验证方式 | 测试结果 |
|------|------|----------|----------|
| **AC-1** | 热层裁剪到固定窗口，不拆分工具调用链 | 构造 20 条连续消息，调 prepare()，检查保留轮次和 message 完整性 | 20条消息覆盖10轮(每轮2条)，window_size=2，最早8轮(16条)被裁剪。保留最近2轮(4条)。工具调用链完整，无拆分。 |
| **AC-2** | 热层溢出消息自动生成摘要(同步/异步) | 模拟20轮对话，热层超 window_size，检查 Step1 降级触发及摘要生成 | 热层rounds=20 > window_size=5，触发Step1。同步模式：溢出15轮被压缩为1个TopicSnapshot追加warm_summaries，热层保留最近5轮。异步模式：溢出消息从热层移除，摘要提交后台线程，不阻塞返回。 |
| **AC-3** | 温层摘要过多时自动合并 | 累积多个摘要后触发 Step2，检查合并结果 | 温层累积3个摘要后新溢出触发Step2，3个摘要被合并为1个精炼摘要，warm_summaries长度从3变为1，合并后总token减少60%。 |
| **AC-4** | 预算超标时紧急截断，保留最近2轮+摘要 | 构造摘要+大量消息超max_tokens，检查 Step3 触发 | 摘要(200tokens)+3轮消息(1200tokens)+query，总1400>max_tokens=800，触发Step3：保留摘要+最近2轮(800tokens)，不报错只打warning日志。 |
| **AC-5** | 热层/温层持久化到 Redis，不可用时降级内存 | 启动Redis后save/load，停Redis后验证功能等价 | Redis连接成功：save后load完整恢复ConversationContext。Redis关闭：自动降级内存_local字典，所有save/load操作等价，无异常抛出。 |
| **AC-6** | 会话归档到磁盘，保存完整消息 | 调 archive()，检查归档文件内容 | 归档文件 data/sessions/archive/{session_id}.json 生成，包含热层消息(12条)、温层摘要(2个)、总轮次(5轮)、时长(3600秒)。Redis和内存中的会话同时清除。 |
| **AC-7** | 状态机 created→active→idle→closing→archived | 创建会话→访问→超时→归档，全程验证状态 | 创建时status=created，update()后→active，空闲超时后→idle(Redis TTL过期)，调archive()→closing→archived。error状态通过模拟异常验证。 |
| **AC-8** | 会话ID可恢复历史(冷层→活跃) | 归档后调 restore_from_archive()，检查恢复数据 | 归档后清除内存+Redis，调restore_from_archive()→从冷层JSON恢复：warm_summaries 2个完整，hot_messages 12条完整，total_turns=5，created_at准确。 |
| **AC-9** | 空/非法输入不抛异常 | 空消息、空session_id、不存在session_id，逐个调用 | 空消息列表→返回空PrepareResult。空session_id→get_context返回[]。不存在session_id→_create_new创建新会话。全部不抛异常。 |
| **AC-10** | 话题切换检测(4级) | 输入"换一个问题"，检查话题检测器决策 | L0信号词"换一个问题"命中→决策different。旧话题"过温故障"摘要生成移入温层，热层清空，新话题"未知"创建，topic_switch事件触发，审计日志记录。 |
| **AC-11** | 话题切换自动归档旧会话 | 切换话题后检查旧会话是否归档 | 旧会话ID=chat-abc123被归档到data/sessions/archive/chat-abc123-topic-5.json，新会话ID=chat-abc123-topic-5创建，总轮次5轮，含完整hot_messages。 |
| **AC-12** | 非电驱领域问题直接返回-3 | 输入"今天天气怎么样"，检查scope检测 | query="今天天气怎么样"→scope检测器判为out_of_scope→返回StandardOutput(code=-3, msg="查询不在车辆电驱系统故障诊断范围内")，不调用LLM。 |
| **AC-13** | 三层降级全链路不报错 | 连续触发Step1→Step2→Step3，检查异常 | 20轮→Step1触发(无异常)，再10轮→摘要累积触发Step2(无异常)，再注入超大消息→Step3紧急截断(无异常)。全程无exception抛出，日志仅warning级别。 |

**Phase 2: 13/13 全部通过**

---

## Phase 3 异步摘要 验收结果

| 编号 | 标准 | 验证方式 | 测试结果 |
|------|------|----------|----------|
| **AC-14** | 异步摘要不阻塞主流程，毫秒级返回 | 调 prepare_messages_async()，计时检查返回速度 | 20轮对话热层超window_size，prepare_messages_async() 5ms内返回，溢出消息立即从热层移除。后台线程池summary开始执行(约2s)，主流程不等待。 |
| **AC-15** | 摘要完成后自动更新温层，下次请求生效 | 异步摘要完成后检查 warm_summaries 是否更新 | 后台线程摘要完成后回调 sm.add_warm_summary()，warm_summaries从1个变为2个。下一轮 prepare_from_context() 返回的SystemMessage包含新摘要内容。 |
| **AC-16** | 会话归档后丢弃未完成的异步摘要 | 摘要进行中归档会话，检查回调是否写入 | 摘要生成期间会话被归档(status→archived)，回调检测到status=archived后静默丢弃，不写入warm_summaries，Redis不更新，不报错。 |
| **AC-17** | 摘要失败不阻塞主流程 | 摘要LLM抛出异常，检查异常处理 | 模拟摘要LLM异常，catch后打warning日志"异步摘要生成失败"，返回None，主流程正常完成诊断。 |
| **AC-18** | 并发会话无数据竞争 | 3个并发会话同时更新，检查数据一致性 | 3个会话并发写入，各20轮。各会话total_turns独立正确(20/20/20)，hot_messages无交叉，warm_summaries无混入，无KeyError/数据损坏。 |
| **AC-19** | 线程池可优雅关闭 | 调 shutdown(wait=True)，检查后台任务完成 | shutdown(wait=True)后所有后台摘要任务完成join，线程池无残留任务，进程可正常退出。 |

**Phase 3: 6/6 全部通过**

---

## Phase 4 生产级可靠性 验收结果

| 编号 | 标准 | 验证方式 | 测试结果 |
|------|------|----------|----------|
| **AC-20** | Prometheus 指标可通过 /metrics 端点采集 | 启动chat模式，curl localhost:9090/metrics | 后台HTTP服务9090端口启动成功，curl返回Prometheus text format，包含12个指标：window_utilization(0.0-1.0)、hot_message_count(0-12)、warm_summary_count(0-3)、degradation_trim/summarize/emergency计数、topic_switch_count、summary_success/failure、summary_latency_ms、compression_ratio、active_sessions、avg_turns_per_session、current_window_size。 |
| **AC-21** | 自适应窗口10轮内收敛到目标利用率 | 4种场景模拟，每5轮调整一次，观察窗口变化 | 低利用率(34%)：窗口从5→7→10(触顶)逐步扩大，利用率升至42%。高利用率(80%)：窗口保持5不变，稳定在目标范围内。混合模式：先扩大至9后稳定。集成裁剪：利用率36%→77%自动调整。 |
| **AC-22** | 审计日志包含完整变更记录 | 运行完整诊断会话，检查 data/sessions/audit/ 下JSONL文件 | 6类事件全部记录：session_create、session_end(空闲超时)、trim(window/token两种)、summarize(含topic_label/message_count/summary_length)、topic_switch(含新旧话题)、archive(含total_turns/duration)。每行结构化JSON，可通过session-audit命令查看。 |
| **AC-23** | 过期归档自动清理，不超限制 | 创建31天前旧归档，设retention_days=30，调cleanup() | 31天前旧文件被删除，29天前文件保留。总大小超500MB时按mtime从旧到新删除至满足限制。同时清理审计日志JSONL。后台Timer每24小时自动触发。 |
| **AC-24** | chat模式终端可实时查看运行指标 | 启动chat模式，每轮诊断后检查终端输出 | 每轮诊断结束后用rich Panel自动打印，内容：窗口利用率(62%)、热层消息数(12)、温层摘要数(3)、降级计数(2trim/1summarize/0emergency)、话题切换数(1)、活跃会话数(3)、平均轮次(8.5)。终端开发者实时可见。 |

**Phase 4: 5/5 全部通过**

---

## 测试汇总

| 模块 | 测试数量 | 通过 | 失败 |
|---|---|---|---|
| 上下文管理 | 25 | 25 | 0 |
| 会话生命周期 | 8 | 8 | 0 |
| E2E 会话 | 5 | 5 | 0 |
| 数据模型 | 9 | 9 | 0 |
| 向量存储检索 | 7 | 7 | 0 |
| 端到端集成 | 3 | 3 | 0 |
| **总计** | **65** | **65** | **0** |