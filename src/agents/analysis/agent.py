from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.state.models import FloatChatState, ScientificIntent
from src.state.schemas import AgentResult, AgentTask
from src.tools.processor import ArgoDataProcessor


class AnalysisAgent(BaseAgent):
    name = "analysis"

    def plan(self, context: Dict[str, Any]) -> List[AgentTask]:
        return [
            AgentTask(
                agent=self.name,
                task_type="analyze_data",
                payload={
                    "query": context.get("query", ""),
                    "intent": context.get("intent", {}),
                    "raw_data_path": context.get("raw_data_path", ""),
                    "requested_variables": context.get("requested_variables", []),
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

            state = FloatChatState(
                user_query=task.payload.get("query", ""),
                intent=intent,
                raw_data=task.payload.get("raw_data_path", ""),
            )
            processor = ArgoDataProcessor(state=state)
            processed = processor.process_query(state.user_query)
            if processed.get("status") == "error":
                raise RuntimeError(processed.get("message", "analysis failed"))

            variable_insights = self._build_variable_insights_parallel(
                processed.get("data", []),
                task.payload.get("requested_variables", []),
            )

            data = {
                "summary": processed.get("summary", ""),
                "row_count": processed.get("row_count", 0),
                "columns": processed.get("columns", []),
                "data": processed.get("data", []),
                "variable_insights": variable_insights,
                "analysis": {
                    "trend": "computed" if processed.get("row_count", 0) > 0 else "none",
                    "comparative": "available" if processed.get("row_count", 0) > 1 else "limited",
                    "parallel_branches": len(variable_insights),
                },
            }
            return AgentResult(
                task_id=task.task_id,
                agent=self.name,
                status="success",
                confidence=0.9 if processed.get("row_count", 0) > 0 else 0.6,
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

    def _build_variable_insights_parallel(
        self, rows: List[Dict[str, Any]], requested_variables: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        if not rows:
            return {}

        available_vars = []
        for candidate in ["temp", "psal", "nitrate"]:
            if any(isinstance(r, dict) and r.get(candidate) is not None for r in rows):
                available_vars.append(candidate)

        target_vars = requested_variables or available_vars
        target_vars = [v for v in target_vars if v in available_vars]
        if not target_vars:
            return {}

        def summarize(var_name: str) -> Dict[str, Any]:
            values = [
                float(r[var_name])
                for r in rows
                if isinstance(r, dict) and r.get(var_name) is not None
            ]
            if not values:
                return {"count": 0}
            values_sorted = sorted(values)
            n = len(values_sorted)
            return {
                "count": n,
                "min": round(values_sorted[0], 3),
                "max": round(values_sorted[-1], 3),
                "mean": round(sum(values_sorted) / n, 3),
                "median": round(
                    values_sorted[n // 2]
                    if n % 2 == 1
                    else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2.0,
                    3,
                ),
            }

        insights: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(3, len(target_vars))) as pool:
            results = list(pool.map(lambda name: (name, summarize(name)), target_vars))
        for name, summary in results:
            insights[name] = summary
        return insights
