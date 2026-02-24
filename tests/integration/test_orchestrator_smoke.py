from src.orchestrator.main import PoseidonOrchestrator
from src.state.schemas import AgentResult, AgentTask, OrchestratorRequest


def test_orchestrator_smoke(monkeypatch):
    orchestrator = PoseidonOrchestrator()

    monkeypatch.setattr(
        orchestrator.query_agent,
        "execute",
        lambda task, context: AgentResult(
            task_id=task.task_id,
            agent="query_understanding",
            status="success",
            confidence=0.95,
            data={"intent": {"variable": "temp", "location": "Mumbai", "lat": 19.0, "lon": 72.8}},
        ),
    )
    monkeypatch.setattr(
        orchestrator.retrieval_agent,
        "execute",
        lambda task, context: AgentResult(
            task_id=task.task_id,
            agent="data_retrieval",
            status="success",
            confidence=0.9,
            data={"row_count": 2, "raw_data_path": "data/fake.csv", "columns": ["temp"], "cache_key": "k"},
        ),
    )
    monkeypatch.setattr(
        orchestrator.analysis_agent,
        "execute",
        lambda task, context: AgentResult(
            task_id=task.task_id,
            agent="analysis",
            status="success",
            confidence=0.9,
            data={
                "summary": "ok",
                "row_count": 2,
                "columns": ["temp"],
                "data": [{"temp": 1.0}, {"temp": 2.0}],
                "analysis": {"trend": "computed", "comparative": "available"},
            },
        ),
    )
    monkeypatch.setattr(
        orchestrator.validation_agent,
        "execute",
        lambda task, context: AgentResult(
            task_id=task.task_id,
            agent="validation",
            status="success",
            confidence=0.88,
            data={
                "quality_report": {
                    "passed": True,
                    "confidence": 0.88,
                    "issues": [],
                    "metrics": {"row_count": 2.0},
                }
            },
        ),
    )

    req = OrchestratorRequest(query="temp near mumbai")
    resp = orchestrator.execute(req)
    assert resp.status == "success"
    assert resp.result["row_count"] == 2
    assert resp.result["validation"]["passed"] is True
    assert resp.confidence == 0.88
    assert resp.trace_id
