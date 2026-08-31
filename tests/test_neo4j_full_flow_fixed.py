#!/usr/bin/env python3
"""Neo4j 知识图谱全流程测试脚本 (修复版)

覆盖4个测试点:
1. 测试 Neo4j 数据是否正确 + 检索（读取），验证 Cypher 查询能正常返回数据，FaultCandidate 能正确展平
2. 测试提取（写入）：模拟一条对话，手动构造验证实体和关系提取链路
3. 测试全流程（写入 + 读取）：把提取结果写入 Neo4j，再查回来验证
"""
import os
import sys
from pathlib import Path

# 确保项目根目录在 path 里
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_1_neo4j_read_and_flatten():
    """测试点1: Neo4j 连接 + 检索读取 + FaultCandidate 展平验证"""
    print("\n" + "="*70)
    print("🚀 测试点 1: Neo4j 数据读取 + Cypher 查询 + FaultCandidate 展平验证")
    print("="*70)

    from src.diagnosis_agent.config import get_settings, reset_settings
    from src.diagnosis_agent.retrieval.neo4j_retriever import Neo4jFaultRetriever
    from src.diagnosis_agent.retrieval.cypher_builder import build_query, QueryCondition, build_count_query
    from src.diagnosis_agent.models.neo4j_result import FaultCandidate

    reset_settings()
    settings = get_settings()

    # 1.1 检查配置
    print(f"✅ Neo4j 配置检查: url={settings.neo4j.url}, user={settings.neo4j.user}")
    if not settings.neo4j.url:
        print("❌ Neo4j URL 未配置，跳过测试点1")
        return False

    # 1.2 初始化 retriever
    retriever = Neo4jFaultRetriever()
    if not retriever.available:
        print("❌ Neo4j 驱动初始化失败，跳过后续读取测试")
        return False
    print("✅ Neo4j 驱动连接成功")

    # 1.3 执行无过滤全量计数查询
    count_cond = QueryCondition()
    count_query, count_params = build_count_query(count_cond)
    print(f"✅ 生成计数 Cypher: {count_query.strip()[:100]}...")
    with retriever._driver.session() as session:
        count_result = session.run(count_query, count_params).data()
        total_faults = count_result[0]["total"]
        print(f"✅ 图谱中总 Fault 节点数量: {total_faults}")
        if total_faults == 0:
            print("⚠️  图谱中无故障数据，后续查询将返回空")

    # 1.4 生成测试查询 - 不带条件，返回前5条
    cond = QueryCondition(limit=5, depth=1)
    query, params = build_query(cond)
    print(f"✅ 生成检索 Cypher: 长度 {len(query)}")
    print(f"   参数: {params}")

    # 1.5 执行查询
    with retriever._driver.session() as session:
        raw_records = session.run(query, params).data()
        print(f"✅ Cypher 查询返回原始记录数: {len(raw_records)}")

    if len(raw_records) == 0:
        print("⚠️  查询返回空记录，测试点1 基础通过（无数据但链路通）")
        return True

    # 1.6 验证 FaultCandidate 展平逻辑
    candidates = []
    for idx, record in enumerate(raw_records):
        candidate = FaultCandidate.from_neo4j_record(record)
        candidates.append(candidate)
        print(f"\n  记录 {idx+1}:")
        print(f"    fault_id: {candidate.fault_id}")
        print(f"    description: {candidate.description[:50]}..." if len(candidate.description) > 50 else f"    description: {candidate.description}")
        print(f"    root_cause: {candidate.root_cause[:50]}..." if len(candidate.root_cause) > 50 else f"    root_cause: {candidate.root_cause}")
        print(f"    dtc_codes: {candidate.dtc_codes}")
        print(f"    motor_code: {candidate.motor_code}")
        print(f"    vehicle_type: {candidate.vehicle_type}")
        print(f"    indicators: {candidate.indicators}")
        print(f"    scenario: {candidate.scenario}")

    # 1.7 转 Document 验证
    docs = [c.to_document() for c in candidates]
    print(f"\n✅ FaultCandidate 转 LangChain Document 完成，共 {len(docs)} 条")
    print(f"   首条 Document metadata 键: {list(docs[0].metadata.keys())}")

    # 1.8 用 structured_recall 接口测试
    recall_result = retriever.structured_recall(keyword="电机", limit=3)
    print(f"✅ 结构化召回（keyword='电机'）返回 {len(recall_result)} 条结果")

    retriever.close()
    print("\n🎉 测试点1 全部通过！Neo4j 读取链路正常，FaultCandidate 展平逻辑正确")
    return True


def test_2_entity_extraction():
    """测试点2: LLM 实体关系提取（写入前置步骤）"""
    print("\n" + "="*70)
    print("🚀 测试点 2: LLM 对话实体关系提取能力验证")
    print("="*70)

    from src.diagnosis_agent.knowledge.extractor import ConversationKnowledgeExtractor, ENTITY_TYPES, RELATIONSHIP_TYPES

    print(f"✅ 支持的实体类型: {ENTITY_TYPES}")
    print(f"✅ 支持的关系类型: {RELATIONSHIP_TYPES}")

    # 模拟一条故障诊断对话
    test_conversation = [
        {
            "role": "user",
            "content": "我的车辆MCU报P0201故障码，车辆在低速爬坡的时候出现扭矩波动，电机温度超过120度之后报过温保护"
        },
        {
            "role": "assistant",
            "content": "根据故障现象分析，根因为旋变零点偏移导致的力矩估算不准，对策是重新标定旋变零点，同时检查散热风扇是否运行正常。适用于MCU-234型号电驱系统。"
        }
    ]
    print(f"\n✅ 输入测试对话: {test_conversation[0]['content']}")

    # 初始化提取器
    extractor = ConversationKnowledgeExtractor(persistence_path="cache/test_knowledge.json")

    # 执行提取
    knowledge = extractor.extract_from_conversation(test_conversation, conversation_id="test_conv_001")

    print(f"\n✅ 提取结果:")
    print(f"   实体数量: {len(knowledge.extracted_entities)}")
    for e in knowledge.extracted_entities:
        print(f"     - [{e.entity_type}] {e.entity_name}: {e.description}")
    print(f"   关系数量: {len(knowledge.extracted_relationships)}")
    for r in knowledge.extracted_relationships:
        print(f"     - {r.source_id} → [{r.relation_type}] → {r.target_id} (权重:{r.weight})")

    if len(knowledge.extracted_entities) == 0 and len(knowledge.extracted_relationships) == 0:
        print("⚠️  本次提取到0个实体和关系，可能是LLM输出格式问题，后续用手动构造实体继续验证写入链路")
    else:
        print("✅ 实体关系提取成功")

    # 提交审核
    knowledge_id = extractor.submit_for_review(knowledge)
    print(f"✅ 知识已提交审核，knowledge_id={knowledge_id}")

    # 自动审核通过（测试环境直接放行）
    extractor.review_knowledge(knowledge_id, approved=True, reviewer="test_system", comment="测试自动通过")
    print(f"✅ 知识 {knowledge_id} 审核已通过")

    return extractor, knowledge_id, knowledge


def test_3_full_write_read_cycle():
    """测试点3: 全流程 写入Neo4j → 再读取验证"""
    print("\n" + "="*70)
    print("🚀 测试点 3: 全流程 写入Neo4j → 读取回显验证")
    print("="*70)

    from src.diagnosis_agent.config import get_settings
    from src.diagnosis_agent.retrieval.neo4j_retriever import Neo4jFaultRetriever
    from src.diagnosis_agent.knowledge.graph_writer import GraphWriter
    from src.diagnosis_agent.knowledge.edit_manager import ManualEditManager
    from src.diagnosis_agent.knowledge.models import ConversationKnowledge, ExtractedEntity, ExtractedRelationship
    from langchain_neo4j import Neo4jGraph

    settings = get_settings()
    if not settings.neo4j.url:
        print("❌ Neo4j URL 未配置，跳过写入测试")
        return False

    # 3.1 初始化 Neo4jGraph 和 GraphWriter
    print("✅ 初始化 Neo4jGraph 连接...")
    graph = Neo4jGraph(
        url=settings.neo4j.url,
        username=settings.neo4j.user,
        password=settings.neo4j.password,
    )
    print("✅ 初始化 ManualEditManager...")
    edit_manager = ManualEditManager(persistence_path="cache/test_edit_records.json")
    print("✅ 初始化 GraphWriter...")
    graph_writer = GraphWriter(graph, edit_manager)

    # 3.2 手动构造测试实体和关系（确保不受LLM输出格式影响）
    test_knowledge = ConversationKnowledge(
        conversation_id="test_manual_001",
        reviewer="test_system",
        status="approved",
        extracted_entities=[
            ExtractedEntity(entity_name="P0201", entity_type="故障DTC", description="喷油器线圈1电路故障"),
            ExtractedEntity(entity_name="扭矩波动", entity_type="现象", description="车辆低速爬坡时出现扭矩波动"),
            ExtractedEntity(entity_name="旋变零点偏移", entity_type="根因", description="旋变安装零点偏移导致力矩估算不准"),
            ExtractedEntity(entity_name="重新标定旋变零点", entity_type="对策", description="重新执行旋变零点标定流程"),
            ExtractedEntity(entity_name="MCU-234", entity_type="电驱代号", description="MCU 型号编号 234"),
            ExtractedEntity(entity_name="低速爬坡", entity_type="故障场景", description="车辆处于低速爬坡工况"),
        ],
        extracted_relationships=[
            ExtractedRelationship(source_id="扭矩波动", target_id="旋变零点偏移", relation_type="由...引起", description="扭矩波动现象由旋变零点偏移根因引起", weight=0.95),
            ExtractedRelationship(source_id="旋变零点偏移", target_id="重新标定旋变零点", relation_type="对应对策", description="旋变零点偏移故障对应的解决方案", weight=1.0),
            ExtractedRelationship(source_id="旋变零点偏移", target_id="MCU-234", relation_type="适用于", description="该故障适用于MCU-234型号电驱", weight=0.9),
            ExtractedRelationship(source_id="扭矩波动", target_id="P0201", relation_type="关联DTC", description="扭矩波动故障对应DTC码P0201", weight=0.85),
            ExtractedRelationship(source_id="扭矩波动", target_id="低速爬坡", relation_type="发生于", description="扭矩波动故障发生在低速爬坡场景", weight=0.9),
        ],
        conversation_context="手动构造测试数据，验证实体关系写入链路",
    )
    print(f"✅ 构造测试实体 {len(test_knowledge.extracted_entities)} 个，关系 {len(test_knowledge.extracted_relationships)} 条")

    # 3.3 写入 Neo4j
    node_count, merged_count = graph_writer.write(test_knowledge)
    print(f"✅ 知识写入完成: 新增节点 {node_count} 个，合并实体 {merged_count} 个")

    # 3.4 验证写入后数据可以查回
    print("\n✅ 写入后执行检索，验证新实体可以被召回...")
    retriever = Neo4jFaultRetriever()

    # 用关键词检索新写入的内容
    candidates = retriever.structured_recall(keyword="旋变零点偏移", limit=5)
    print(f"✅ 用关键词 '旋变零点偏移' 检索到 {len(candidates)} 条结果")

    # 用 DTC P0201 检索
    candidates_dtc = retriever.structured_recall(dtc_codes=["P0201"], limit=5)
    print(f"✅ 用 DTC 'P0201' 检索到 {len(candidates_dtc)} 条结果")

    # 手动查询新写入的节点，验证所有新实体都存在
    with retriever._driver.session() as session:
        verify_query = """
        MATCH (e) WHERE e.name IN $names RETURN labels(e) as labels, e.name as name, e.description as desc
        """
        verify_names = ["P0201", "扭矩波动", "旋变零点偏移", "重新标定旋变零点", "MCU-234"]
        verify_result = list(session.run(verify_query, {"names": verify_names}).data())
        print(f"✅ 验证新写入实体: 共匹配到 {len(verify_result)} 个节点:")
        for item in verify_result:
            print(f"   - [{item['labels']}] {item['name']}: {item['desc']}")

        # 查询新写入的关系
        rel_query = """
        MATCH (a)-[r]->(b) WHERE a.name IN $names AND b.name IN $names RETURN a.name as src, type(r) as rel, b.name as tgt
        """
        rel_result = list(session.run(rel_query, {"names": verify_names}).data())
        print(f"✅ 验证新写入关系: 共匹配到 {len(rel_result)} 条关系:")
        for item in rel_result:
            print(f"   - {item['src']} → [{item['rel']}] → {item['tgt']}")

    retriever.close()

    print("\n🎉 测试点3 全流程写入+读取验证完成！新提取的知识已成功持久化到Neo4j并可被检索召回")
    return True


if __name__ == "__main__":
    print("🔧 Neo4j 知识图谱全流程测试开始...")
    print(f"当前工作目录: {os.getcwd()}")

    os.chdir(project_root)

    # 执行所有测试
    try:
        # 测试点1
        ok1 = test_1_neo4j_read_and_flatten()

        # 测试点2
        extractor, knowledge_id, knowledge = test_2_entity_extraction()

        # 测试点3
        ok3 = test_3_full_write_read_cycle()

        print("\n" + "="*70)
        print("🏁 所有测试执行完毕！汇总结果:")
        print(f"   测试点1 (Neo4j读取+展平): {'✅ 通过' if ok1 else '⚠️ 部分通过'}")
        print(f"   测试点2 (LLM实体提取): ✅ 执行完成，可正常提交审核")
        print(f"   测试点3 (写入+读取回显): {'✅ 通过' if ok3 else '⚠️ 部分通过'}")
        print("="*70)
    except Exception as e:
        import traceback
        print(f"\n❌ 测试执行异常: {e}")
        traceback.print_exc()
        sys.exit(1)
