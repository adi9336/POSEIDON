from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from src.agent.Retrieving_Agent import run_argo_workflow
from src.agents.analysis.agent import AnalysisAgent
from src.agents.data_retrieval.agent import DataRetrievalAgent
from src.agents.query_understanding.agent import QueryUnderstandingAgent
from src.agents.validation.agent import ValidationAgent
from src.orchestrator.router import PolicyRouter
from src.orchestrator.state_manager import StateManager
from src.state.schemas import (
    AgentResult,
    EventType,
    ExecutionPlan,
    OrchestratorRequest,
    OrchestratorResponse,
    StreamEvent,
)


class WorkflowState(TypedDict, total=False):
    query: str
    trace_id: str
    conversation_id: str
    context: Dict[str, Any]
    result: Dict[str, Any]
    error: str
    clarification_needed: bool
    clarification_question: str


class PoseidonOrchestrator:
    def __init__(
        self,
        state_manager: StateManager | None = None,
        router: PolicyRouter | None = None,
    ) -> None:
        self.state_manager = state_manager or StateManager(os.getenv("REDIS_URL"))
        self.router = router or PolicyRouter()
        self.query_agent = QueryUnderstandingAgent()
        self.retrieval_agent = DataRetrievalAgent()
        self.analysis_agent = AnalysisAgent()
        self.validation_agent = ValidationAgent()
        self.max_task_retries = 2
        self.max_query_retries = 3
        self._events: Dict[str, List[StreamEvent]] = defaultdict(list)
        self._graph = self._build_graph()

    def _emit(self, event: StreamEvent) -> None:
        self._events[event.conversation_id].append(event)

    def pop_events(self, conversation_id: str) -> List[StreamEvent]:
        events = self._events.get(conversation_id, [])
        self._events[conversation_id] = []
        return events

    def _run_with_retry(
        self,
        fn: Callable[[], AgentResult],
        conversation_id: str,
        trace_id: str,
        agent_name: str,
    ) -> AgentResult:
        last_error = None
        for attempt in range(self.max_task_retries + 1):
            try:
                result = fn()
                if result.status != "error":
                    return result
                last_error = result.error
            except Exception as exc:
                last_error = str(exc)
            if attempt < self.max_task_retries:
                self._emit(
                    StreamEvent(
                        event_type=EventType.AGENT_PROGRESS,
                        trace_id=trace_id,
                        conversation_id=conversation_id,
                        message=f"Retrying {agent_name} (attempt {attempt + 2})",
                    )
                )
                time.sleep(2**attempt)
        return AgentResult(
            task_id=str(uuid4()),
            agent=agent_name,
            status="error",
            confidence=0.0,
            error=last_error or "unknown error",
        )

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("understanding", self._understanding_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("analysis", self._analysis_node)
        graph.add_node("validation", self._validation_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_edge("understanding", "retrieval")
        graph.add_edge("retrieval", "analysis")
        graph.add_edge("analysis", "validation")
        graph.add_edge("validation", "finalize")
        graph.add_edge("finalize", END)
        graph.set_entry_point("understanding")
        return graph.compile()

    def _understanding_node(self, state: WorkflowState) -> WorkflowState:
        context = state.get("context", {})
        task = self.query_agent.plan({"query": state["query"]})[0]
        result = self._run_with_retry(
            lambda: self.query_agent.execute(task, context),
            state["conversation_id"],
            state["trace_id"],
            self.query_agent.name,
        )
        if result.status == "error":
            return {"error": result.error or "understanding failed"}

        intent = result.data.get("intent", {})
        context["intent"] = intent
        context["requested_variables"] = result.data.get("requested_variables", [])
        context["understanding_confidence"] = result.confidence
        self._emit(
            StreamEvent(
                event_type=EventType.AGENT_PROGRESS,
                trace_id=state["trace_id"],
                conversation_id=state["conversation_id"],
                message="Query understanding completed",
                data={"intent": intent, "confidence": result.confidence},
            )
        )
        return {
            "context": context,
            "clarification_needed": result.data.get("clarification_needed", False),
            "clarification_question": result.data.get("clarification_question", ""),
        }

    def _retrieval_node(self, state: WorkflowState) -> WorkflowState:
        if state.get("clarification_needed"):
            return {}
        context = state.get("context", {})
        task = self.retrieval_agent.plan(
            {"query": state["query"], "intent": context.get("intent", {})}
        )[0]
        result = self._run_with_retry(
            lambda: self.retrieval_agent.execute(task, context),
            state["conversation_id"],
            state["trace_id"],
            self.retrieval_agent.name,
        )
        if result.status == "error":
            return {"error": result.error or "retrieval failed"}

        context["raw_data_path"] = result.data.get("raw_data_path", "")
        context["retrieval"] = result.data
        context["retrieval_confidence"] = result.confidence
        self._emit(
            StreamEvent(
                event_type=EventType.PARTIAL_RESULT,
                trace_id=state["trace_id"],
                conversation_id=state["conversation_id"],
                message="Data retrieval completed",
                data={"row_count": result.data.get("row_count", 0)},
            )
        )
        return {"context": context}

    def _analysis_node(self, state: WorkflowState) -> WorkflowState:
        if state.get("clarification_needed"):
            return {}
        context = state.get("context", {})
        task = self.analysis_agent.plan(
            {
                "query": state["query"],
                "intent": context.get("intent", {}),
                "raw_data_path": context.get("raw_data_path", ""),
                "requested_variables": context.get("requested_variables", []),
            }
        )[0]
        result = self._run_with_retry(
            lambda: self.analysis_agent.execute(task, context),
            state["conversation_id"],
            state["trace_id"],
            self.analysis_agent.name,
        )
        if result.status == "error":
            return {"error": result.error or "analysis failed"}
        context["analysis"] = result.data
        context["analysis_confidence"] = result.confidence
        return {"context": context}

    def _validation_node(self, state: WorkflowState) -> WorkflowState:
        if state.get("clarification_needed"):
            return {}
        context = state.get("context", {})
        task = self.validation_agent.plan(
            {
                "intent": context.get("intent", {}),
                "analysis": context.get("analysis", {}),
                "retrieval": context.get("retrieval", {}),
            }
        )[0]
        result = self._run_with_retry(
            lambda: self.validation_agent.execute(task, context),
            state["conversation_id"],
            state["trace_id"],
            self.validation_agent.name,
        )
        if result.status == "error":
            return {"error": result.error or "validation failed"}
        context["validation"] = result.data.get("quality_report", {})
        context["validation_confidence"] = result.confidence
        self._emit(
            StreamEvent(
                event_type=EventType.AGENT_PROGRESS,
                trace_id=state["trace_id"],
                conversation_id=state["conversation_id"],
                message="Validation completed",
                data={"quality_report": context["validation"]},
            )
        )
        return {"context": context}

    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        context = state.get("context", {})
        if state.get("clarification_needed"):
            result = {
                "status": "clarification_needed",
                "message": state.get("clarification_question") or "Need more detail",
            }
            return {"result": result}
        if state.get("error"):
            return {"result": {"status": "error", "message": state["error"]}}
        quality_report = context.get("validation", {})
        overall_confidence = quality_report.get(
            "confidence",
            (
                context.get("understanding_confidence", 0.0) * 0.2
                + context.get("retrieval_confidence", 0.0) * 0.3
                + context.get("analysis_confidence", 0.0) * 0.5
            ),
        )
        return {
            "result": {
                "status": "success",
                "summary": context.get("analysis", {}).get("summary", ""),
                "data": context.get("analysis", {}).get("data", []),
                "row_count": context.get("analysis", {}).get("row_count", 0),
                "columns": context.get("analysis", {}).get("columns", []),
                "intent": context.get("intent", {}),
                "retrieval": context.get("retrieval", {}),
                "analysis": context.get("analysis", {}).get("analysis", {}),
                "variable_insights": context.get("analysis", {}).get("variable_insights", {}),
                "validation": quality_report,
                "confidence": round(float(overall_confidence), 3),
            }
        }

    def _run_legacy(self, request: OrchestratorRequest) -> OrchestratorResponse:
        result = run_argo_workflow(request.query)
        processed = result.get("processed", {})
        response_result = {
            "summary": processed.get("summary", result.get("final_answer", "")),
            "data": processed.get("data", []),
            "row_count": processed.get("row_count", 0),
            "columns": processed.get("columns", []),
            "intent": result.get("intent"),
            "final_answer": result.get("final_answer"),
        }
        return OrchestratorResponse(
            status="success",
            trace_id=str(uuid4()),
            conversation_id=request.conversation_id,
            result=response_result,
            confidence=0.7,
        )

    def execute(self, request: OrchestratorRequest) -> OrchestratorResponse:
        mode = os.getenv("POSEIDON_ORCHESTRATOR_MODE", request.mode).lower()
        if mode == "legacy":
            return self._run_legacy(request)

        trace_id = str(uuid4())
        self._emit(
            StreamEvent(
                event_type=EventType.STARTED,
                trace_id=trace_id,
                conversation_id=request.conversation_id,
                message="Orchestrator execution started",
                data={"mode": mode},
            )
        )

        state: WorkflowState = {
            "query": request.query,
            "trace_id": trace_id,
            "conversation_id": request.conversation_id,
            "context": {
                "latency_budget_ms": request.latency_budget_ms,
                "cost_budget": request.cost_budget,
                "provider_health": {"openai": "healthy", "anthropic": "unknown"},
            },
        }

        final_state = self._graph.invoke(state)
        result = final_state.get("result", {})
        status = result.get("status", "success")
        confidence = (
            float(result.get("confidence", 0.85))
            if status == "success"
            else 0.5
        )

        response = OrchestratorResponse(
            status="clarification_needed" if status == "clarification_needed" else ("error" if status == "error" else "success"),
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            result=result,
            confidence=confidence,
            plan=ExecutionPlan(conversation_id=request.conversation_id),
            error=result.get("message") if status == "error" else None,
        )

        self.state_manager.set(
            request.conversation_id,
            {
                "query": request.query,
                "trace_id": trace_id,
                "status": response.status,
                "result": response.result,
            },
        )

        self._emit(
            StreamEvent(
                event_type=EventType.COMPLETED if response.status != "error" else EventType.FAILED,
                trace_id=trace_id,
                conversation_id=request.conversation_id,
                message="Orchestrator execution completed",
                data={"status": response.status, "confidence": response.confidence},
            )
        )
        return response
