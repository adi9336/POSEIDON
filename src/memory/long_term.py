from __future__ import annotations

from typing import Any, Dict, List


class LongTermMemory:
    """
    Phase-1 abstraction.
    Backends:
    - chroa/chromadb in phase 2
    - pinecone adapter in phase 2
    """

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self._items.append(item)

    def search(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Placeholder lexical search; vector store adapters are phase 2.
        results = [x for x in self._items if text.lower() in str(x).lower()]
        return results[:limit]

