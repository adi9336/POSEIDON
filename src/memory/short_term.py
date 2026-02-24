from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict


class ShortTermMemory:
    def __init__(self, max_items: int = 20) -> None:
        self.max_items = max_items
        self._store: Dict[str, Deque[Dict[str, Any]]] = {}

    def append(self, conversation_id: str, item: Dict[str, Any]) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = deque(maxlen=self.max_items)
        self._store[conversation_id].append(item)

    def get(self, conversation_id: str) -> list[Dict[str, Any]]:
        return list(self._store.get(conversation_id, deque()))

