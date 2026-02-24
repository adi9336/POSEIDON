from src.agents.validation.agent import ValidationAgent
from src.state.schemas import AgentTask


def test_validation_agent_passes_plausible_data():
    agent = ValidationAgent()
    task = AgentTask(
        agent="validation",
        task_type="validate_quality",
        payload={
            "intent": {"depth": 500},
            "analysis": {
                "row_count": 2,
                "data": [
                    {"temp": 12.5, "psal": 35.2, "latitude": 19.0, "longitude": 72.8},
                    {"temp": 13.1, "psal": 35.0, "latitude": 19.1, "longitude": 72.9},
                ],
            },
            "retrieval": {"row_count": 2},
        },
    )
    result = agent.execute(task, {})
    report = result.data["quality_report"]
    assert report["passed"] is True
    assert result.confidence > 0.7


def test_validation_agent_flags_invalid_depth_and_no_data():
    agent = ValidationAgent()
    task = AgentTask(
        agent="validation",
        task_type="validate_quality",
        payload={
            "intent": {"depth": 9000},
            "analysis": {"row_count": 0, "data": []},
            "retrieval": {"row_count": 0},
        },
    )
    result = agent.execute(task, {})
    report = result.data["quality_report"]
    assert report["passed"] is False
    assert any("No records" in issue for issue in report["issues"])
    assert any("Argo operating range" in issue for issue in report["issues"])
    assert result.confidence < 0.6


def test_validation_agent_flags_time_window_inconsistency():
    agent = ValidationAgent()
    task = AgentTask(
        agent="validation",
        task_type="validate_quality",
        payload={
            "intent": {"time_range": ("2025-01-01", "2025-01-31")},
            "analysis": {
                "row_count": 4,
                "data": [
                    {"time": "2025-01-05T00:00:00", "temp": 12.0, "psal": 35.0, "pres": 100},
                    {"time": "2025-01-11T00:00:00", "temp": 12.2, "psal": 35.1, "pres": 110},
                    {"time": "2025-02-03T00:00:00", "temp": 12.1, "psal": 35.0, "pres": 120},
                    {"time": "2025-02-04T00:00:00", "temp": 11.9, "psal": 34.9, "pres": 130},
                ],
            },
            "retrieval": {"row_count": 4},
        },
    )
    result = agent.execute(task, {})
    report = result.data["quality_report"]
    assert report["passed"] is False
    assert any("outside requested time window" in issue.lower() for issue in report["issues"])
    assert report["metrics"]["time_outside_rate"] > 0.05


def test_validation_agent_flags_high_outlier_rate():
    agent = ValidationAgent()
    temps = [10.0] * 16 + [25.0] * 4  # 20% outlier pattern
    data = [
        {"temp": t, "psal": 35.0, "pres": 200, "latitude": 19.0, "longitude": 72.8}
        for t in temps
    ]
    task = AgentTask(
        agent="validation",
        task_type="validate_quality",
        payload={
            "intent": {"depth": 200},
            "analysis": {"row_count": len(data), "data": data},
            "retrieval": {"row_count": len(data)},
        },
    )
    result = agent.execute(task, {})
    report = result.data["quality_report"]
    assert report["passed"] is False
    assert any("outlier rate" in issue.lower() for issue in report["issues"])
    assert report["metrics"]["temp_outlier_rate"] > 0.10
