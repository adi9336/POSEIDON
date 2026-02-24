# POSEIDON Multi-Agent Architecture

## Runtime modes

- `legacy`: Use existing `run_argo_workflow` pipeline.
- `hybrid`: Default. Use orchestrator for `/v1/query`, keep `/query` compatibility.
- `multi`: Force orchestrator path.

Set with `POSEIDON_ORCHESTRATOR_MODE=legacy|hybrid|multi`.

## MVP agents

- QueryUnderstandingAgent
- DataRetrievalAgent
- AnalysisAgent

## APIs

- `POST /v1/query`
- `WS /v1/stream/{conversation_id}`
- Legacy compatibility: `POST /query`

## Storage

- Stage 1: SQLite for data.
- Stage 1: optional Redis for session memory.
- Stage 2: Postgres + SQLAlchemy migration path.

