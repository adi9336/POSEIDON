from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.state.schemas import AgentResult, AgentTask


class VisualizationAgent(BaseAgent):
    name = "visualization"

    def plan(self, context: Dict[str, Any]) -> List[AgentTask]:
        return [AgentTask(agent=self.name, task_type="visualize", payload={})]

    def execute(self, task: AgentTask, context: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent=self.name,
            status="partial",
            confidence=0.0,
            data={"note": "Visualization agent is scaffolded for phase 2."},
        )

