from __future__ import annotations

import time
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.state.models import FloatChatState, ScientificIntent
from src.state.schemas import AgentResult, AgentTask
from src.tools.fetcher import fetch_argo_data


class DataRetrievalAgent(BaseAgent):
    name = "data_retrieval"

    def plan(self, context: Dict[str, Any]) -> List[AgentTask]:
        return [
            AgentTask(
                agent=self.name,
                task_type="fetch_data",
                payload={
                    "query": context.get("query", ""),
                    "source": "argo",
                    "intent": context.get("intent", {}),
                },
            )
        ]

    def execute(self, task: AgentTask, context: Dict[str, Any]) -> AgentResult:
        started = time.perf_counter()
        try:
            intent_payload = task.payload.get("intent", {})
            intent = (
                intent_payload
                if isinstance(intent_payload, ScientificIntent)
                else ScientificIntent(**intent_payload)
            )

            state = FloatChatState(user_query=context.get("query", ""), intent=intent)
            df = fetch_argo_data(intent, state=state)

            result_payload = {
                "source": task.payload.get("source", "argo"),
                "query_plan": {
                    "source_priority": ["argo", "noaa", "nasa"],
                    "selected": "argo",
                },
                "row_count": int(len(df)) if df is not None else 0,
                "raw_data_path": state.raw_data,
                "columns": list(df.columns) if df is not None and not df.empty else [],
                "cache_key": f"{intent.location or 'global'}:{intent.variable or 'all'}",
            }

            confidence = 0.85 if result_payload["row_count"] > 0 else 0.5
            return AgentResult(
                task_id=task.task_id,
                agent=self.name,
                status="success",
                confidence=confidence,
                data=result_payload,
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

