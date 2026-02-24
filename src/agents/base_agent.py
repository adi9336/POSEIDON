from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.state.schemas import AgentResult, AgentTask, QualityReport


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def plan(self, context: Dict[str, Any]) -> List[AgentTask]:
        """Generate one or more tasks for this agent."""

    @abstractmethod
    def execute(self, task: AgentTask, context: Dict[str, Any]) -> AgentResult:
        """Execute a task and return a structured result."""

    def validate(
        self, result: AgentResult, context: Dict[str, Any]
    ) -> Optional[QualityReport]:
        """Optional result validation hook."""
        return None

