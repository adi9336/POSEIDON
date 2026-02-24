from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError


@dataclass
class RegisteredTool:
    name: str
    tool: Callable[[Any], Any]
    input_model: Optional[Type[BaseModel]] = None
    output_model: Optional[Type[BaseModel]] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        tool: Callable[[Any], Any],
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        self._tools[name] = RegisteredTool(
            name=name, tool=tool, input_model=input_model, output_model=output_model
        )

    def invoke(self, name: str, payload: Any, trace_ctx: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        item = self._tools[name]

        validated_payload = payload
        if item.input_model is not None:
            try:
                validated_payload = item.input_model.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"Invalid payload for tool '{name}': {exc}") from exc

        result = item.tool(validated_payload)

        if item.output_model is not None:
            try:
                return item.output_model.model_validate(result)
            except ValidationError as exc:
                raise ValueError(f"Invalid output from tool '{name}': {exc}") from exc
        return result

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())


TOOL_REGISTRY = ToolRegistry()

