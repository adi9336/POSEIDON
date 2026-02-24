from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RoutingDecision:
    provider: str
    model: str
    fallback_chain: list[str]


class PolicyRouter:
    """
    Policy-based model router.
    Defaults to OpenAI and allows Anthropic override via config/env in phase 1.
    """

    def __init__(self, policy: Dict[str, Any] | None = None) -> None:
        self.policy = policy or {}
        self.default_provider = self.policy.get("default_provider", "openai")
        self.default_models = self.policy.get(
            "default_models",
            {
                "orchestrator": "gpt-4o-mini",
                "query_understanding": "gpt-4o-mini",
                "data_retrieval": "gpt-4o-mini",
                "analysis": "gpt-4o-mini",
            },
        )

    def choose(self, task_type: str, context: Dict[str, Any]) -> RoutingDecision:
        provider_health = context.get("provider_health", {"openai": "healthy"})
        use_anthropic = os.getenv("POSEIDON_ENABLE_ANTHROPIC", "false").lower() == "true"
        preferred_provider = context.get("preferred_provider", self.default_provider)

        if preferred_provider == "anthropic" and use_anthropic and provider_health.get("anthropic") == "healthy":
            provider = "anthropic"
        else:
            provider = "openai"

        model = self.default_models.get(task_type, self.default_models.get("orchestrator", "gpt-4o-mini"))
        fallback_chain = ["openai:gpt-4o-mini", "openai:gpt-4.1-mini"]
        if use_anthropic:
            fallback_chain.insert(0, "anthropic:claude-3-5-sonnet-latest")

        return RoutingDecision(provider=provider, model=model, fallback_chain=fallback_chain)

