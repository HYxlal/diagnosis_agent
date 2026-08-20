"""Neo4j 知识图谱 Schema 迁移脚本

迁移目标：
- 节点标签：从 `__Entity__` + 二级标签 → 仅保留二级标签 + entity_type 属性
- 实体类型：旧 8 种 → 新 8 种（现象/根因/对策/故障DTC/电驱代号/车辆类型/仪表指示灯/故障场景）
- 关系类型：旧英文/中文 → 新中文语义关系

用法：
  python scripts/migrate_graph_schema.py          # 执行迁移
  python scripts/migrate_graph_schema.py --dry-run # 只预览不执行
"""

import argparse
import logging
import os
import sys

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate")

# 实体类型 → Neo4j 标签
ENTITY_TYPE_TO_LABEL = {
    "现象": "Fault",
    "根因": "RootCause",
    "对策": "Solution",
    "故障DTC": "DTC",
    "电驱代号": "MotorType",
    "车辆类型": "VehicleType",
    "仪表指示灯": "Indicator",
    "故障场景": "Scenario",
}

# 旧实体类型 → 新实体类型
OLD_TO_NEW_ENTITY_TYPE = {
    "故障现象": "现象",
    "故障代码": "故障DTC",
    "电驱型号": "电驱代号",
    "车型": "车辆类型",
    "部件": "根因",
    "诊断方法": "对策",
}

# 旧关系 → 新关系映射
RELATION_RENAME = {
    "HAS_ROOT_CAUSE": "由...引起",
    "HAS_DTC": "关联DTC",
    "SHOWS_INDICATOR": "亮起",
    "OCCURS_ON": "出现于",
    "OCCURS_IN_SCENARIO": "发生于",
}

# 所有目标标签
_ALL_LABELS = list(ENTITY_TYPE_TO_LABEL.values())


def get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "sDK2aesu")
    return GraphDatabase.driver(uri, auth=(user, password))


def run_query(driver, query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return result.data()


def run_query_no_return(driver, query, params=None):
    with driver.session() as session:
        session.run(query, params or {})


def step1_clean_dirty_data(driver, dry_run):
    logger.info("Step 1: 清理脏数据")

    # 1a. 删除 __Entity__.id 唯一约束（后续要移除 __Entity__ 标签，约束不再需要）
    if dry_run:
        with driver.session() as session:
            result = session.run(
                "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
                "WHERE labelsOrTypes = ['__Entity__'] AND properties = ['id'] "
                "RETURN count(*) AS c"
            ).data()
            if result[0]["c"] > 0:
                logger.info("  __Entity__.id 约束待删除: 1")
            else:
                logger.info("  __Entity__.id 约束: 无")
    else:
        constraints = run_query(
            driver,
            "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
            "WHERE labelsOrTypes = ['__Entity__'] AND properties = ['id'] "
            "RETURN name",
        )
        for c in constraints:
            run_query_no_return(driver, f"DROP CONSTRAINT {c['name']}")
            logger.info(f"  __Entity__.id 约束已删除: {c['name']}")

    # 1b. 修复 DTC 节点中存成列表字符串的脏数据
    q1 = """
    MATCH (d:DTC)
    WHERE d.id STARTS WITH '['
    SET d.id = replace(replace(replace(d.id, "['", ''), "']", ''), "', '", ', ')
    RETURN count(d) AS fixed
    """
    if dry_run:
        with driver.session() as session:
            result = session.run(
                "MATCH (d:DTC) WHERE d.id STARTS WITH '[' RETURN count(d) AS c"
            ).data()
            logger.info(f"  DTC 脏数据待修复: {result[0]['c']}")
    else:
        result = run_query(driver, q1)
        logger.info(f"  DTC 脏数据已修复: {result[0]['fixed']}")


def step2_delete_obsolete_relations(driver, dry_run):
    logger.info("Step 2: 删除废弃关系")

    # 2a. 删除 MENTIONS
    q1 = "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c"
    q1d = "MATCH ()-[r:MENTIONS]->() DELETE r"
    if dry_run:
        result = run_query(driver, q1)
        logger.info(f"  MENTIONS 待删除: {result[0]['c']}")
    else:
        result = run_query(driver, q1)
        run_query_no_return(driver, q1d)
        logger.info(f"  MENTIONS 已删除: {result[0]['c']}")

    # 2b. 删除 属于、参考
    for rel_type in ["属于", "参考"]:
        q = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c"
        qd = f"MATCH ()-[r:{rel_type}]->() DELETE r"
        if dry_run:
            result = run_query(driver, q)
            logger.info(f"  {rel_type} 待删除: {result[0]['c']}")
        else:
            result = run_query(driver, q)
            run_query_no_return(driver, qd)
            logger.info(f"  {rel_type} 已删除: {result[0]['c']}")


def step3_rename_structured_relations(driver, dry_run):
    logger.info("Step 3: 结构化关系重命名")

    for old_type, new_type in RELATION_RENAME.items():
        q = f"MATCH ()-[r:{old_type}]->() RETURN count(r) AS c"
        qd = f"""
        MATCH (s)-[r:{old_type}]->(t)
        CREATE (s)-[:`{new_type}`]->(t)
        DELETE r
        """
        if dry_run:
            result = run_query(driver, q)
            logger.info(f"  {old_type} → {new_type}: {result[0]['c']} 条")
        else:
            result = run_query(driver, q)
            run_query_no_return(driver, qd)
            logger.info(f"  {old_type} → {new_type}: {result[0]['c']} 条已迁移")

    # 3f. HAS_SOLUTION 拆成 对应对策 + 适用于
    q_sol = "MATCH ()-[r:HAS_SOLUTION]->() RETURN count(r) AS c"
    q_sol_migrate = """
    MATCH (f:Fault)-[r:HAS_SOLUTION]->(s:Solution)
    OPTIONAL MATCH (f)-[:`由...引起`]->(rc:RootCause)
    FOREACH (_ IN CASE WHEN rc IS NOT NULL THEN [1] ELSE [] END |
      CREATE (rc)-[:`对应对策`]->(s)
    )
    CREATE (s)-[:`适用于`]->(f)
    DELETE r
    """
    if dry_run:
        result = run_query(driver, q_sol)
        logger.info(f"  HAS_SOLUTION → 对应对策 + 适用于: {result[0]['c']} 条")
    else:
        result = run_query(driver, q_sol)
        run_query_no_return(driver, q_sol_migrate)
        logger.info(f"  HAS_SOLUTION → 对应对策 + 适用于: {result[0]['c']} 条已迁移")

    # 3g. OCCURS_ON_VEHICLE → 配备（通过 Fault 桥接）
    q_ov = "MATCH ()-[r:OCCURS_ON_VEHICLE]->() RETURN count(r) AS c"
    q_ov_migrate = """
    MATCH (f:Fault)-[r:OCCURS_ON_VEHICLE]->(v:VehicleType)
    OPTIONAL MATCH (f)-[:`出现于`]->(m:MotorType)
    FOREACH (_ IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
      CREATE (v)-[:`配备`]->(m)
    )
    DELETE r
    """
    if dry_run:
        result = run_query(driver, q_ov)
        logger.info(f"  OCCURS_ON_VEHICLE → 配备: {result[0]['c']} 条")
    else:
        result = run_query(driver, q_ov)
        run_query_no_return(driver, q_ov_migrate)
        logger.info(f"  OCCURS_ON_VEHICLE → 配备: {result[0]['c']} 条已迁移")


def step4_transform_extraction_relations(driver, dry_run):
    logger.info("Step 4: 提取关系重建")

    # 4a. 导致 → 由...引起 / 导致（按方向）
    q_cause = "MATCH ()-[r:导致]->() RETURN count(r) AS c"
    q_cause_migrate = """
    // 现象→根因 归 由...引起
    MATCH (s:Fault)-[r:导致]->(t:RootCause)
    CREATE (s)-[:`由...引起`]->(t)
    DELETE r
    WITH 1 AS _
    // 根因→现象 归 导致
    MATCH (s:RootCause)-[r:导致]->(t:Fault)
    CREATE (s)-[:导致]->(t)
    DELETE r
    """
    if dry_run:
        result = run_query(driver, q_cause)
        logger.info(f"  导致 待拆分: {result[0]['c']} 条")
    else:
        result = run_query(driver, q_cause)
        run_query_no_return(driver, q_cause_migrate)
        logger.info(f"  导致 已拆分: {result[0]['c']} 条")

    # 4b. 对应故障代码 → 关联DTC
    q_dtc = "MATCH ()-[r:对应故障代码]->() RETURN count(r) AS c"
    q_dtc_migrate = """
    MATCH (s)-[r:对应故障代码]->(t)
    CREATE (s)-[:`关联DTC`]->(t)
    DELETE r
    """
    if dry_run:
        result = run_query(driver, q_dtc)
        logger.info(f"  对应故障代码 → 关联DTC: {result[0]['c']} 条")
    else:
        result = run_query(driver, q_dtc)
        run_query_no_return(driver, q_dtc_migrate)
        logger.info(f"  对应故障代码 → 关联DTC: {result[0]['c']} 条已迁移")

    # 4c. 对策 → 对应对策 + 适用于（方向反转）
    q_ce = "MATCH ()-[r:对策]->() RETURN count(r) AS c"
    q_ce_migrate = """
    MATCH (s:Solution)-[r:对策]->(t:Fault)
    OPTIONAL MATCH (t)-[:`由...引起`]->(rc:RootCause)
    FOREACH (_ IN CASE WHEN rc IS NOT NULL THEN [1] ELSE [] END |
      CREATE (rc)-[:`对应对策`]->(s)
    )
    CREATE (s)-[:`适用于`]->(t)
    DELETE r
    """
    if dry_run:
        result = run_query(driver, q_ce)
        logger.info(f"  对策 待反转: {result[0]['c']} 条")
    else:
        result = run_query(driver, q_ce)
        run_query_no_return(driver, q_ce_migrate)
        logger.info(f"  对策 已反转: {result[0]['c']} 条")


def step5_migrate_node_labels(driver, dry_run):
    logger.info("Step 5: 节点标签迁移")

    # 5a. 给结构化节点补 entity_type 属性
    label_entity_type_map = {
        "Fault": "现象",
        "RootCause": "根因",
        "Solution": "对策",
        "DTC": "故障DTC",
        "MotorType": "电驱代号",
        "VehicleType": "车辆类型",
        "Indicator": "仪表指示灯",
        "Scenario": "故障场景",
    }
    for label, entity_type in label_entity_type_map.items():
        q = f"""
        MATCH (n:{label})
        WHERE n.entity_type IS NULL OR n.entity_type = ''
        SET n.entity_type = '{entity_type}'
        RETURN count(n) AS updated
        """
        if dry_run:
            with driver.session() as session:
                result = session.run(
                    f"MATCH (n:{label}) WHERE n.entity_type IS NULL OR n.entity_type = '' RETURN count(n) AS c"
                ).data()
                if result[0]["c"] > 0:
                    logger.info(f"  {label} 补 entity_type: {result[0]['c']} 个")
        else:
            result = run_query(driver, q)
            if result[0]["updated"] > 0:
                logger.info(f"  {label} 补 entity_type: {result[0]['updated']} 个")

    # 5b. 纯 __Entity__ 节点按旧 entity_type 映射到新标签
    for old_type, new_type in OLD_TO_NEW_ENTITY_TYPE.items():
        new_label = ENTITY_TYPE_TO_LABEL[new_type]
        q = f"""
        MATCH (n:__Entity__)
        WHERE n.entity_type = '{old_type}'
          AND NOT (n:Fault OR n:RootCause OR n:Solution OR n:DTC
                   OR n:MotorType OR n:VehicleType OR n:Indicator OR n:Scenario)
        SET n.entity_type = '{new_type}'
        SET n:{new_label}
        REMOVE n:__Entity__
        RETURN count(n) AS migrated
        """
        if dry_run:
            with driver.session() as session:
                result = session.run(
                    f"MATCH (n:__Entity__) WHERE n.entity_type = '{old_type}' "
                    f"AND NOT (n:Fault OR n:RootCause OR n:Solution OR n:DTC "
                    f"OR n:MotorType OR n:VehicleType OR n:Indicator OR n:Scenario) "
                    f"RETURN count(n) AS c"
                ).data()
                if result[0]["c"] > 0:
                    logger.info(f"  __Entity__ ({old_type}→{new_type}→{new_label}): {result[0]['c']} 个")
        else:
            result = run_query(driver, q)
            if result[0]["migrated"] > 0:
                logger.info(f"  __Entity__ ({old_type}→{new_type}→{new_label}): {result[0]['migrated']} 个已迁移")

    # 5c. 删除剩余的 __Entity__ 标签（已不再需要）
    q_remove = """
    MATCH (n:__Entity__)
    WHERE n:Fault OR n:RootCause OR n:Solution OR n:DTC
       OR n:MotorType OR n:VehicleType OR n:Indicator OR n:Scenario
    REMOVE n:__Entity__
    RETURN count(n) AS removed
    """
    if dry_run:
        with driver.session() as session:
            result = session.run(
                "MATCH (n:__Entity__) WHERE n:Fault OR n:RootCause OR n:Solution OR n:DTC "
                "OR n:MotorType OR n:VehicleType OR n:Indicator OR n:Scenario "
                "RETURN count(n) AS c"
            ).data()
            logger.info(f"  __Entity__ 基标签待移除: {result[0]['c']} 个")
    else:
        result = run_query(driver, q_remove)
        logger.info(f"  __Entity__ 基标签已移除: {result[0]['removed']} 个")

    # 5d. 删除无 entity_type 的纯 __Entity__ 残留节点
    q_clean = """
    MATCH (n:__Entity__)
    WHERE n.entity_type IS NULL OR n.entity_type = ''
    DETACH DELETE n
    RETURN count(n) AS deleted
    """
    if dry_run:
        with driver.session() as session:
            result = session.run(
                "MATCH (n:__Entity__) WHERE n.entity_type IS NULL OR n.entity_type = '' RETURN count(n) AS c"
            ).data()
            if result[0]["c"] > 0:
                logger.info(f"  无类型残留节点待删除: {result[0]['c']} 个")
            else:
                logger.info("  无类型残留节点: 0")
    else:
        result = run_query(driver, q_clean)
        if result[0]["deleted"] > 0:
            logger.info(f"  无类型残留节点已删除: {result[0]['deleted']} 个")


def step6_summary(driver):
    """迁移完成后打印统计"""
    logger.info("=" * 50)
    logger.info("迁移完成统计:")

    for label in _ALL_LABELS:
        result = run_query(driver, f"MATCH (n:{label}) RETURN count(n) AS c")
        logger.info(f"  {label}: {result[0]['c']} 个节点")

    logger.info("")
    new_rel_types = [
        "由...引起", "导致", "对应对策", "适用于",
        "发生于", "关联DTC", "亮起", "配备", "出现于", "排除",
        "关联", "互斥", "并存",
    ]
    for rel_type in new_rel_types:
        result = run_query(driver, f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS c")
        logger.info(f"  {rel_type}: {result[0]['c']} 条关系")


def main():
    parser = argparse.ArgumentParser(description="Neo4j 知识图谱 Schema 迁移")
    parser.add_argument("--dry-run", action="store_true", help="只预览不执行")
    args = parser.parse_args()

    driver = get_driver()

    try:
        # 验证连接
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("Neo4j 连接正常")
    except Exception as e:
        logger.error(f"Neo4j 连接失败: {e}")
        sys.exit(1)

    if args.dry_run:
        logger.info("=" * 50)
        logger.info("DRY RUN 模式 — 仅预览，不修改数据")
        logger.info("=" * 50)

    step1_clean_dirty_data(driver, args.dry_run)
    step2_delete_obsolete_relations(driver, args.dry_run)
    step3_rename_structured_relations(driver, args.dry_run)
    step4_transform_extraction_relations(driver, args.dry_run)
    step5_migrate_node_labels(driver, args.dry_run)

    if not args.dry_run:
        step6_summary(driver)

    driver.close()
    logger.info("迁移完成")


if __name__ == "__main__":
    main()