from __future__ import annotations

import json
from typing import Any, Dict, Optional


class StateManager:
    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis = None
        self._local: Dict[str, Dict[str, Any]] = {}
        if redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.from_url(redis_url, decode_responses=True)
            except Exception:
                self._redis = None

    def set(self, conversation_id: str, data: Dict[str, Any]) -> None:
        if self._redis is not None:
            self._redis.set(f"poseidon:session:{conversation_id}", json.dumps(data))
        else:
            self._local[conversation_id] = data

    def get(self, conversation_id: str) -> Dict[str, Any]:
        if self._redis is not None:
            raw = self._redis.get(f"poseidon:session:{conversation_id}")
            if not raw:
                return {}
            return json.loads(raw)
        return self._local.get(conversation_id, {})

