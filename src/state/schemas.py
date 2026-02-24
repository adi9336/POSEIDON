from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    agent: str
    task_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 2
    timeout_ms: int = 60000


class AgentResult(BaseModel):
    task_id: str
    agent: str
    status: Literal["success", "error", "partial"] = "success"
    confidence: float = 0.0
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class QualityReport(BaseModel):
    passed: bool = True
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    conversation_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    tasks: List[AgentTask] = Field(default_factory=list)


class OrchestratorRequest(BaseModel):
    query: str
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    mode: Literal["legacy", "hybrid", "multi"] = "hybrid"
    latency_budget_ms: int = 120000
    cost_budget: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorResponse(BaseModel):
    status: Literal["success", "error", "clarification_needed"] = "success"
    trace_id: str
    conversation_id: str
    result: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    plan: Optional[ExecutionPlan] = None
    error: Optional[str] = None


class EventType(str, Enum):
    STARTED = "started"
    AGENT_PROGRESS = "agent_progress"
    PARTIAL_RESULT = "partial_result"
    COMPLETED = "completed"
    FAILED = "failed"


class StreamEvent(BaseModel):
    event_type: EventType
    trace_id: str
    conversation_id: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

