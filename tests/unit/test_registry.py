import pytest
from pydantic import BaseModel

from src.tools.registry import ToolRegistry


class InputModel(BaseModel):
    value: int


class OutputModel(BaseModel):
    doubled: int


def _double(payload: InputModel):
    return {"doubled": payload.value * 2}


def test_registry_register_and_invoke():
    registry = ToolRegistry()
    registry.register("double", _double, input_model=InputModel, output_model=OutputModel)
    result = registry.invoke("double", {"value": 4})
    assert result.doubled == 8


def test_registry_validation_failure():
    registry = ToolRegistry()
    registry.register("double", _double, input_model=InputModel, output_model=OutputModel)
    with pytest.raises(ValueError):
        registry.invoke("double", {"value": "bad"})

