from __future__ import annotations

import time
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.state.schemas import AgentResult, AgentTask
from src.tools.intent_extractor import extract_intent_with_llm


class QueryUnderstandingAgent(BaseAgent):
    name = "query_understanding"

    def plan(self, context: Dict[str, Any]) -> List[AgentTask]:
        return [
            AgentTask(
                agent=self.name,
                task_type="parse_query",
                payload={"query": context.get("query", "")},
            )
        ]

    def execute(self, task: AgentTask, context: Dict[str, Any]) -> AgentResult:
        started = time.perf_counter()
        query = task.payload.get("query", "")
        try:
            intent = extract_intent_with_llm(query)
            q = (query or "").lower()
            requested_variables = []
            if any(x in q for x in ["temp", "temperature"]):
                requested_variables.append("temp")
            if any(x in q for x in ["salinity", "psal"]):
                requested_variables.append("psal")
            if "nitrate" in q:
                requested_variables.append("nitrate")
            if not requested_variables and intent.variable:
                requested_variables = [intent.variable]

            confidence = 0.9
            clarification_needed = False
            clarification = None

            if not intent.variable:
                confidence = 0.55
                clarification_needed = True
                clarification = (
                    "Which parameter do you need (temperature, salinity, nitrate)?"
                )

            data = {
                "intent": intent.model_dump(),
                "requested_variables": sorted(set(requested_variables)),
                "clarification_needed": clarification_needed,
                "clarification_question": clarification,
            }
            return AgentResult(
                task_id=task.task_id,
                agent=self.name,
                status="success",
                confidence=confidence,
                data=data,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent=self.name,
                status="error",
                confidence=0.0,
                error=str(exc),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
