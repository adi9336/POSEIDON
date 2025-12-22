"""
State models for the Argo Float Chat system.
These models define the data structures used throughout the workflow.
"""

from pydantic import BaseModel, field_validator, Field
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime
from enum import Enum


class ScientificIntent(BaseModel):
    variable: Optional[str] = None  # temp, psal, nitrate, etc.
    operation: Optional[str] = None  # trend, difference, anomaly, etc.
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    depth: Optional[float] = None
    depth_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None
    context_needed: Optional[str] = None  # marine life impact, climate reasoning

    @field_validator("time_range", mode="before")
    @classmethod
    def validate_time_range(cls, v):
        """Ensure time_range is properly formatted"""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            # Convert list to tuple and handle None values
            return tuple(v) if len(v) == 2 else None
        return v

    @field_validator("depth_range", mode="before")
    @classmethod
    def validate_depth_range(cls, v):
        """Ensure depth_range is properly formatted"""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return tuple(v) if len(v) == 2 else None
        return v


class FloatChatState(BaseModel):
    user_query: str
    intent: Optional[ScientificIntent] = None
    dataset: Optional[str] = None  # ArgoFloats or ArgoProfile
    erddap_url: Optional[str] = None
    raw_data: str = ""  # Path to CSV file or raw data
    processed: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    status: Optional[str] = (
        None  # Tracks processing status: None, "processing", "processed", "error"
    )
    error: Optional[str] = None  # Stores error message if status is "error"

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrame
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class AgentRole(str, Enum):
    """Roles for messages in the agent's conversation."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A message in the conversation."""
    role: AgentRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """State for the React agent, extending FloatChatState with conversation management."""
    conversation_id: str  # Unique identifier for the conversation
    messages: List[Message] = Field(default_factory=list)  # Conversation history
    current_state: Optional[FloatChatState] = None  # Current processing state
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, role: Union[AgentRole, str], content: str, **metadata) -> None:
        """Add a message to the conversation."""
        if isinstance(role, str):
            role = AgentRole(role.lower())
        
        self.messages.append(
            Message(
                role=role,
                content=content,
                metadata=metadata or {}
            )
        )
        self.updated_at = datetime.utcnow()

    def get_conversation_history(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get the conversation history in a format suitable for the LLM."""
        return [
            {"role": msg.role.value, "content": msg.content, **msg.metadata}
            for msg in self.messages[-max_messages:]
        ]

    def update_state(self, state: FloatChatState) -> None:
        """Update the current processing state."""
        self.current_state = state
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ],
            "current_state": self.current_state.dict() if self.current_state else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create an AgentState from a dictionary."""
        messages = [
            Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data.get("metadata", {})
            )
            for msg_data in data.get("messages", [])
        ]
        
        current_state_data = data.get("current_state")
        current_state = FloatChatState(**current_state_data) if current_state_data else None
        
        return cls(
            conversation_id=data["conversation_id"],
            messages=messages,
            current_state=current_state,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
