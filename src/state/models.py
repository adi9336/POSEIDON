"""
State models for the Argo Float Chat system.
These models define the data structures used throughout the workflow.
"""
from pydantic import BaseModel, field_validator
from typing import Optional, Tuple, Dict, Any, List

class ScientificIntent(BaseModel):
    variable: Optional[str] = None        # temp, psal, nitrate, etc.
    operation: Optional[str] = None       # trend, difference, anomaly, etc.
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    depth: Optional[float] = None
    depth_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None
    context_needed: Optional[str] = None  # marine life impact, climate reasoning

    @field_validator('time_range', mode='before')
    @classmethod
    def validate_time_range(cls, v):
        """Ensure time_range is properly formatted"""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            # Convert list to tuple and handle None values
            return tuple(v) if len(v) == 2 else None
        return v
    
    @field_validator('depth_range', mode='before')
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
    dataset: Optional[str] = None          # ArgoFloats or ArgoProfile
    erddap_url: Optional[str] = None
    raw_data: Optional[Any] = None         # Pandas DataFrame
    processed: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrame