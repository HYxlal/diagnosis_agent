"""知识沉淀 Web 审核页面

提供待审核列表、审核操作、知识库浏览功能。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from ..config import get_settings, reset_settings

logger = logging.getLogger(__name__)

security = HTTPBasic(auto_error=False)


def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """如果配置了 web_username/web_password，则验证 Basic Auth"""
    settings = get_settings()
    if not settings.knowledge.web_password:
        return
    expected_user = settings.knowledge.web_username or "admin"
    if credentials is None:
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})
    if credentials.username != expected_user or credentials.password != settings.knowledge.web_password:
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})


def _get_extractor():
    """延迟导入避免循环依赖"""
    from .cli import _get_extractor as _get

    return _get()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    reviewer: str
    comment: Optional[str] = None
    write_to_neo4j: bool = True


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="知识沉淀审核", version="1.0.0")


def _get_neo4j_graph():
    """获取 Neo4jGraph 实例"""
    extractor = _get_extractor()
    return extractor._graph_writer._graph if extractor._graph_writer else None


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(_=Depends(require_auth)):
    """审核页面"""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/pending")
async def list_pending(_=Depends(require_auth)):
    """待审核列表"""
    reset_settings()
    extractor = _get_extractor()
    pending = extractor.get_pending_reviews()
    return [
        {
            "knowledge_id": k.knowledge_id,
            "conversation_id": k.conversation_id,
            "status": k.status,
            "created_at": k.created_at,
            "review_comment": k.review_comment,
            "reviewer": k.reviewer,
            "entities": [
                {
                    "name": e.entity_name,
                    "type": e.entity_type,
                    "description": e.description,
                }
                for e in k.extracted_entities
            ],
            "relationships": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type,
                    "description": r.description,
                    "weight": r.weight,
                }
                for r in k.extracted_relationships
            ],
            "conversation_preview": k.conversation_context[:200] if k.conversation_context else "",
        }
        for k in pending
    ]


@app.get("/api/history")
async def list_history(_=Depends(require_auth)):
    """已审核列表"""
    reset_settings()
    extractor = _get_extractor()
    all_knowledge = extractor._processed_knowledge
    reviewed = [k for k in all_knowledge.values() if k.status in ("approved", "rejected")]
    reviewed.sort(key=lambda k: k.review_time or "", reverse=True)
    return [
        {
            "knowledge_id": k.knowledge_id,
            "status": k.status,
            "created_at": k.created_at,
            "review_comment": k.review_comment,
            "reviewer": k.reviewer,
            "review_time": k.review_time,
            "entities": [
                {"name": e.entity_name, "type": e.entity_type}
                for e in k.extracted_entities
            ],
            "relationships": [
                {"source": r.source_id, "type": r.relation_type, "target": r.target_id}
                for r in k.extracted_relationships
            ],
            "conversation_preview": k.conversation_context[:200] if k.conversation_context else "",
        }
        for k in reviewed
    ]


@app.post("/api/approve/{knowledge_id}")
async def approve(knowledge_id: str, body: ReviewRequest, _=Depends(require_auth)):
    """审核通过"""
    reset_settings()
    extractor = _get_extractor()
    ok = extractor.review_knowledge(
        knowledge_id, True, body.reviewer, body.comment
    )
    if not ok:
        raise HTTPException(404, "知识 ID 不存在")
    if body.write_to_neo4j:
        extractor.write_approved_knowledge(knowledge_id)
    return {"status": "approved", "knowledge_id": knowledge_id}


@app.post("/api/reject/{knowledge_id}")
async def reject(knowledge_id: str, body: ReviewRequest, _=Depends(require_auth)):
    """审核拒绝"""
    reset_settings()
    extractor = _get_extractor()
    ok = extractor.review_knowledge(
        knowledge_id, False, body.reviewer, body.comment
    )
    if not ok:
        raise HTTPException(404, "知识 ID 不存在")
    return {"status": "rejected", "knowledge_id": knowledge_id}


@app.get("/api/knowledge")
async def get_knowledge(_=Depends(require_auth)):
    """已积累的知识图谱（从 Neo4j 查询）"""
    reset_settings()
    graph = _get_neo4j_graph()
    if graph is None:
        return {"entities": [], "relationships": [], "error": "Neo4j 不可用"}

    try:
        entities = graph.query(
            "MATCH (e) WHERE (e:Fault OR e:RootCause OR e:Solution "
            "OR e:DTC OR e:MotorType OR e:VehicleType OR e:Indicator OR e:Scenario) "
            "RETURN e.id AS id, e.name AS name, "
            "e.entity_type AS type, e.description AS description, "
            "e.source AS source, e.knowledge_id AS knowledge_id "
            "ORDER BY e.name LIMIT 200"
        )
        relationships = graph.query(
            "MATCH (a)-[r]->(b) "
            "WHERE (a:Fault OR a:RootCause OR a:Solution "
            "OR a:DTC OR a:MotorType OR a:VehicleType OR a:Indicator OR a:Scenario) "
            "AND (b:Fault OR b:RootCause OR b:Solution "
            "OR b:DTC OR b:MotorType OR b:VehicleType OR b:Indicator OR b:Scenario) "
            "RETURN a.name AS source, type(r) AS rel_type, "
            "b.name AS target, r.description AS description, "
            "r.weight AS weight, r.knowledge_id AS knowledge_id "
            "LIMIT 200"
        )
        return {"entities": entities, "relationships": relationships}
    except Exception as e:
        logger.error(f"查询知识图谱失败: {e}")
        return {"entities": [], "relationships": [], "error": str(e)}


@app.get("/api/stats")
async def get_stats(_=Depends(require_auth)):
    """知识沉淀统计"""
    reset_settings()
    extractor = _get_extractor()
    stats = extractor.get_knowledge_stats()
    return stats.model_dump()


# ---------------------------------------------------------------------------
# HTML 页面
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识沉淀审核</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; color: #333; }
.header { background: #1a1a2e; color: #fff; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 20px; }
.tabs { display: flex; gap: 4px; background: #1a1a2e; padding: 0 24px; }
.tab { padding: 10px 20px; cursor: pointer; border: none; background: transparent; color: #8892b0; font-size: 14px; border-radius: 6px 6px 0 0; transition: .2s; }
.tab:hover { color: #ccd6f6; }
.tab.active { background: #f5f5f5; color: #1a1a2e; font-weight: 600; }
.content { padding: 24px; max-width: 1400px; margin: 0 auto; }
.card { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.card-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px; }
.card-title { font-size: 16px; font-weight: 600; color: #1a1a2e; }
.card-meta { font-size: 12px; color: #8892b0; }
.entity-tag { display: inline-block; background: #e8f4fd; color: #0a66b9; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
.rel-tag { display: inline-block; background: #e8fae8; color: #1a7a1a; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
.actions { display: flex; gap: 8px; margin-top: 12px; }
.btn { padding: 6px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: .2s; }
.btn-approve { background: #10b981; color: #fff; }
.btn-approve:hover { background: #059669; }
.btn-reject { background: #ef4444; color: #fff; }
.btn-reject:hover { background: #dc2626; }
.btn-secondary { background: #e5e7eb; color: #374151; }
.btn-secondary:hover { background: #d1d5db; }
.comment-input { flex: 1; padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 13px; }
.reviewer-input { width: 120px; padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 13px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 13px; }
th { background: #f9fafb; font-weight: 600; color: #6b7280; }
tr:hover { background: #f9fafb; }
.empty { text-align: center; color: #9ca3af; padding: 40px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
.stat-card { background: #fff; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.stat-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
.stat-label { font-size: 12px; color: #8892b0; margin-top: 4px; }
.toast { position: fixed; top: 16px; right: 16px; padding: 12px 20px; border-radius: 8px; color: #fff; font-size: 14px; z-index: 1000; animation: slideIn .3s; }
.toast-success { background: #10b981; }
.toast-error { background: #ef4444; }
@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
.section { display: none; }
.section.active { display: block; }
</style>
</head>
<body>
<div class="header">
  <h1>知识沉淀审核</h1>
  <span id="stats-summary" style="font-size:13px;color:#8892b0;">加载中...</span>
</div>
<div class="tabs">
  <button class="tab active" onclick="switchTab('review')">待审核</button>
  <button class="tab" onclick="switchTab('history')">历史审核</button>
  <button class="tab" onclick="switchTab('knowledge')">知识库</button>
</div>
<div class="content">

  <!-- 待审核 Tab -->
  <div id="section-review" class="section active">
    <div class="stats-grid" id="review-stats"></div>
    <div id="pending-list"></div>
  </div>

  <!-- 历史审核 Tab -->
  <div id="section-history" class="section">
    <div id="history-list"></div>
  </div>

  <!-- 知识库 Tab -->
  <div id="section-knowledge" class="section">
    <div class="card">
      <div class="card-title" style="margin-bottom:12px">实体列表</div>
      <div style="overflow-x:auto">
        <table id="entity-table">
          <thead><tr><th>名称</th><th>类型</th><th>描述</th><th>来源</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-title" style="margin-bottom:12px">关系列表</div>
      <div style="overflow-x:auto">
        <table id="rel-table">
          <thead><tr><th>源实体</th><th>关系</th><th>目标实体</th><th>描述</th><th>权重</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<div id="toast-container"></div>

<script>
// ========================================================================
// 状态管理
// ========================================================================
let currentTab = 'review';

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById('section-' + tab).classList.add('active');
  if (tab === 'review') loadPending();
  if (tab === 'history') loadHistory();
  if (tab === 'knowledge') loadKnowledge();
}

// ========================================================================
// Toast 通知
// ========================================================================
function toast(msg, type) {
  const el = document.createElement('div');
  el.className = 'toast toast-' + type;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// ========================================================================
// 待审核列表
// ========================================================================
async function loadPending() {
  try {
    const [pending, stats] = await Promise.all([
      fetch('/api/pending').then(r => r.json()),
      fetch('/api/stats').then(r => r.json()),
    ]);
    renderStats(stats);
    renderPending(pending);
  } catch (e) {
    document.getElementById('pending-list').innerHTML = '<div class="empty">加载失败: ' + e.message + '</div>';
  }
}

function renderStats(stats) {
  document.getElementById('review-stats').innerHTML = [
    ['总提取', stats.extractions],
    ['提取实体', stats.entities_extracted],
    ['提取关系', stats.relationships_extracted],
    ['提交审核', stats.submitted_for_review],
    ['已通过', stats.approved_knowledge],
    ['已拒绝', stats.rejected_knowledge],
    ['合并实体', stats.merged_entities],
  ].map(([label, val]) => '<div class="stat-card"><div class="stat-value">' + val + '</div><div class="stat-label">' + label + '</div></div>').join('');
  document.getElementById('stats-summary').textContent = '已提取 ' + stats.entities_extracted + ' 实体 / ' + stats.relationships_extracted + ' 关系';
}

function renderPending(items) {
  if (!items.length) {
    document.getElementById('pending-list').innerHTML = '<div class="empty">暂无待审核的知识</div>';
    return;
  }
  document.getElementById('pending-list').innerHTML = items.map((k, i) => `
    <div class="card" id="card-${k.knowledge_id}">
      <div class="card-header">
        <div>
          <div class="card-title">#${i + 1} ${k.knowledge_id.substring(0, 12)}...</div>
          <div class="card-meta">${k.created_at.substring(0, 16)} | 对话 ${(k.conversation_id || '无').substring(0, 12)}</div>
        </div>
      </div>
      <div style="margin-bottom:8px">
        ${k.entities.map(e => `<span class="entity-tag">${e.name} (${e.type})</span>`).join('')}
      </div>
      <div style="margin-bottom:8px">
        ${k.relationships.map(r => `<span class="rel-tag">${r.source} → ${r.type} → ${r.target}</span>`).join('')}
      </div>
      <div style="font-size:12px;color:#8892b0;margin-bottom:12px;max-height:60px;overflow:hidden;">
        ${k.conversation_preview}
      </div>
      ${k.review_comment ? '<div style="font-size:12px;color:#e6c384;margin-bottom:8px;border-left:3px solid #e6c384;padding-left:8px;">备注: ' + k.review_comment + '</div>' : ''}
      <div class="actions">
        <input class="reviewer-input" placeholder="审核人" id="reviewer-${k.knowledge_id}" value="admin">
        <input class="comment-input" placeholder="审核备注（可选）" id="comment-${k.knowledge_id}">
        <button class="btn btn-approve" onclick="doApprove('${k.knowledge_id}')">通过</button>
        <button class="btn btn-reject" onclick="doReject('${k.knowledge_id}')">拒绝</button>
      </div>
    </div>
  `).join('');
}

async function doApprove(id) {
  const reviewer = document.getElementById('reviewer-' + id).value || 'admin';
  const comment = document.getElementById('comment-' + id).value;
  try {
    const res = await fetch('/api/approve/' + id, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer, comment, write_to_neo4j: true}),
    });
    if (res.ok) {
      toast('审核通过: ' + id.substring(0, 12) + '...', 'success');
      document.getElementById('card-' + id).remove();
      loadPending();
    } else {
      const err = await res.json();
      toast('操作失败: ' + err.detail, 'error');
    }
  } catch (e) {
    toast('请求失败: ' + e.message, 'error');
  }
}

async function doReject(id) {
  const reviewer = document.getElementById('reviewer-' + id).value || 'admin';
  const comment = document.getElementById('comment-' + id).value;
  try {
    const res = await fetch('/api/reject/' + id, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reviewer, comment, write_to_neo4j: false}),
    });
    if (res.ok) {
      toast('已拒绝: ' + id.substring(0, 12) + '...', 'success');
      document.getElementById('card-' + id).remove();
      loadPending();
    } else {
      const err = await res.json();
      toast('操作失败: ' + err.detail, 'error');
    }
  } catch (e) {
    toast('请求失败: ' + e.message, 'error');
  }
}

// ========================================================================
// 历史审核
// ========================================================================
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const items = await res.json();
    renderHistory(items);
  } catch (e) {
    document.getElementById('history-list').innerHTML = '<div class="empty">加载失败: ' + e.message + '</div>';
  }
}

function renderHistory(items) {
  if (!items.length) {
    document.getElementById('history-list').innerHTML = '<div class="empty">暂无已审核的知识</div>';
    return;
  }
  document.getElementById('history-list').innerHTML = items.map((k, i) => {
    const statusIcon = k.status === 'approved' ? '✅' : '❌';
    const statusColor = k.status === 'approved' ? '#10b981' : '#ef4444';
    return `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">${statusIcon} #${i + 1} ${k.knowledge_id.substring(0, 12)}...</div>
          <div class="card-meta">${k.created_at.substring(0, 16)} | 审核: ${k.reviewer || '未知'} ${k.review_time ? k.review_time.substring(0, 16) : ''}</div>
        </div>
        <span style="color:${statusColor};font-weight:600;font-size:13px;">${k.status === 'approved' ? '已通过' : '已拒绝'}</span>
      </div>
      ${k.review_comment ? '<div style="font-size:12px;color:#e6c384;margin-bottom:8px;border-left:3px solid #e6c384;padding-left:8px;">备注: ' + k.review_comment + '</div>' : ''}
      <div style="margin-bottom:8px">
        ${k.entities.map(e => '<span class="entity-tag">' + e.name + ' (' + e.type + ')</span>').join('')}
      </div>
      <div style="margin-bottom:8px">
        ${k.relationships.map(r => '<span class="rel-tag">' + r.source + ' → ' + r.type + ' → ' + r.target + '</span>').join('')}
      </div>
      <div style="font-size:12px;color:#8892b0;max-height:60px;overflow:hidden;">
        ${k.conversation_preview}
      </div>
    </div>
  `}).join('');
}

// ========================================================================
// 知识库浏览
// ========================================================================
async function loadKnowledge() {
  try {
    const data = await fetch('/api/knowledge').then(r => r.json());
    if (data.error) {
      document.getElementById('entity-table').querySelector('tbody').innerHTML =
        '<tr><td colspan="4" class="empty">' + data.error + '</td></tr>';
      return;
    }
    renderEntities(data.entities);
    renderRelationships(data.relationships);
  } catch (e) {
    document.getElementById('entity-table').querySelector('tbody').innerHTML =
      '<tr><td colspan="4" class="empty">加载失败: ' + e.message + '</td></tr>';
  }
}

function renderEntities(entities) {
  const tbody = document.getElementById('entity-table').querySelector('tbody');
  if (!entities.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">暂无知识积累</td></tr>';
    return;
  }
  tbody.innerHTML = entities.map(e => `
    <tr>
      <td><strong>${e.name || e.id}</strong></td>
      <td><span class="entity-tag">${e.type || '-'}</span></td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.description || '-'}</td>
      <td style="font-size:12px;color:#8892b0">${e.source || '-'}</td>
    </tr>
  `).join('');
}

function renderRelationships(rels) {
  const tbody = document.getElementById('rel-table').querySelector('tbody');
  if (!rels.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">暂无关系</td></tr>';
    return;
  }
  tbody.innerHTML = rels.map(r => `
    <tr>
      <td>${r.source}</td>
      <td><span class="rel-tag">${r.rel_type}</span></td>
      <td>${r.target}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.description || '-'}</td>
      <td>${r.weight || '-'}</td>
    </tr>
  `).join('');
}

// ========================================================================
// 初始化
// ========================================================================
loadPending();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 启动函数
# ---------------------------------------------------------------------------

def run_web(host: str = "0.0.0.0", port: int = 8090) -> None:
    """启动 Web 审核服务"""
    import uvicorn

    reset_settings()
    logger.info(f"知识沉淀审核服务启动: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")