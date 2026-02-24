from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class VectorStoreAdapter(ABC):
    @abstractmethod
    def add(self, id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
        ...

    @abstractmethod
    def search(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        ...


class ChromaAdapter(VectorStoreAdapter):
    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}

    def add(self, id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
        self._items[id] = {"text": text, "metadata": metadata or {}}

    def search(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        matches = [
            {"id": item_id, **value}
            for item_id, value in self._items.items()
            if text.lower() in value["text"].lower()
        ]
        return matches[:limit]


class PineconeAdapter(VectorStoreAdapter):
    """
    Scaffold adapter for phase-2 integration.
    """

    def add(self, id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
        raise NotImplementedError("Pinecone adapter will be implemented in phase 2")

    def search(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError("Pinecone adapter will be implemented in phase 2")

