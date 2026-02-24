from src.orchestrator.router import PolicyRouter


def test_router_defaults_to_openai():
    router = PolicyRouter()
    decision = router.choose("analysis", {"provider_health": {"openai": "healthy"}})
    assert decision.provider == "openai"
    assert decision.model
    assert len(decision.fallback_chain) >= 1

