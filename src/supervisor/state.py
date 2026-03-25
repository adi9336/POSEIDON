"""
Supervisor conversation state model.
Tracks multi-turn interactions between user and Supervisor Agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SupervisorPhase(str, Enum):
    """Phases of the Supervisor conversation lifecycle."""
    GREETING = "greeting"
    CLARIFYING = "clarifying"
    CONFIRMING_LOCATION = "confirming_location"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


class ClarificationQuestion(BaseModel):
    """A single clarification question for the user."""
    id: str
    text: str
    options: List[str] = Field(default_factory=list)
    default: Optional[str] = None
    required: bool = True


class WorkflowPlan(BaseModel):
    """An execution plan awaiting user approval."""
    name: str = "ARGO Data Analysis"
    steps: List[str] = Field(default_factory=list)
    estimated_time: str = "30-60 seconds"
    estimated_cost: str = "$0.05-0.15"
    data_volume: str = "Unknown"
    argo_float_count: int = 0


class SupervisorResponse(BaseModel):
    """Response from the Supervisor Agent to the UI layer."""
    response_type: str  # clarification_needed | map_confirmation_needed | approval_needed | ready_to_execute | message | error
    message: str = ""
    clarification_questions: List[ClarificationQuestion] = Field(default_factory=list)
    map_data: Optional[Dict[str, Any]] = None  # lat, lon, radius_km, float_count
    workflow_plan: Optional[WorkflowPlan] = None
    suggestions: List[str] = Field(default_factory=list)


class SupervisorConversationState(BaseModel):
    """
    Full conversation state for a Supervisor session.
    Tracks the progression: greeting → clarifying → confirming_location → approval → executing → completed
    Persists session_memory across multiple queries within the same session.
    """
    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    phase: SupervisorPhase = SupervisorPhase.GREETING
    user_query: str = ""
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # Collected parameters
    clarifications_collected: Dict[str, Any] = Field(default_factory=dict)
    confirmed_coordinates: Optional[Dict[str, float]] = None  # {lat, lon, radius_km}
    confirmed_intent: Optional[Dict[str, Any]] = None

    # Plans & results
    workflow_plan: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None

    # ── Session memory (persists across queries) ──
    session_memory: List[Dict[str, Any]] = Field(default_factory=list)
    # Each entry: {"query": str, "intent": dict, "result_summary": str, "timestamp": str}
    previous_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Tracking
    clarification_turn_count: int = 0
    max_clarification_turns: int = 3
    execution_counter: int = 0  # For unique chart keys

    def add_message(self, role: str, content: str) -> None:
        """Add a message to chat history."""
        self.chat_history.append({"role": role, "content": content})

    def advance_phase(self, new_phase: SupervisorPhase) -> None:
        """Move to a new phase."""
        self.phase = new_phase

    def is_ready_to_execute(self) -> bool:
        """Check if we have enough info to execute the workflow."""
        return (
            self.phase == SupervisorPhase.AWAITING_APPROVAL
            or self.phase == SupervisorPhase.EXECUTING
        )

    def merge_clarifications(self, answers: Dict[str, Any]) -> None:
        """Merge user-provided clarification answers into collected state."""
        self.clarifications_collected.update(answers)
        self.clarification_turn_count += 1

    def save_to_memory(self, result_summary: str = "") -> None:
        """Save the current query+result into session memory before resetting."""
        from datetime import datetime
        entry = {
            "query": self.user_query,
            "intent": self.confirmed_intent or {},
            "coordinates": self.confirmed_coordinates,
            "result_summary": result_summary,
            "timestamp": datetime.now().isoformat(),
        }
        self.session_memory.append(entry)

    def reset_for_new_query(self) -> None:
        """
        Reset phase/intent/clarifications for a new query while
        preserving session_memory and chat_history (conversation context).
        """
        self.phase = SupervisorPhase.GREETING
        self.user_query = ""
        self.clarifications_collected = {}
        self.confirmed_coordinates = None
        self.confirmed_intent = None
        self.workflow_plan = None
        self.execution_result = None
        self.clarification_turn_count = 0
        self.execution_counter += 1
        # session_memory and chat_history are preserved

    def get_memory_summary(self) -> str:
        """Get a compact summary of previous queries in this session."""
        if not self.session_memory:
            return ""
        lines = ["Previous queries in this session:"]
        for i, entry in enumerate(self.session_memory[-5:], 1):  # Last 5
            lines.append(
                f"  {i}. \"{entry['query']}\" → {entry.get('result_summary', 'completed')}"
            )
        return "\n".join(lines)
