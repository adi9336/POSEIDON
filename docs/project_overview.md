# POSEIDON Project Overview

## What This Project Is

POSEIDON is an oceanographic analysis application built around Argo float data. It accepts natural-language questions, resolves them into a scientific intent, fetches matching observations from ERDDAP, runs analysis and validation, and presents the result through:

- a Streamlit chat UI in `app.py`
- a FastAPI service in `src/api/server.py`
- scheduled autonomous campaigns in `src/campaigns/`

The codebase currently contains three layers that coexist:

1. `src/agent/`: the original legacy LangGraph workflow (`extract_intent -> fetch_data -> process_data`)
2. `src/orchestrator/` + `src/agents/`: the newer multi-agent orchestration layer
3. `src/supervisor/`: a conversational control layer for clarification, map confirmation, and approval

That layered history explains why some modules overlap in responsibility.

## End-to-End Runtime Flow

### Streamlit path

`app.py` is the main UI entrypoint. It initializes:

- `PoseidonOrchestrator` for execution
- `SupervisorAgent` for conversation management
- `CampaignManager` in the background when campaign support is available

The user message first goes to the Supervisor. The Supervisor decides whether the system should:

- ask clarification questions
- ask the user to confirm the target area on a map
- show an approval card with time/cost/data estimates
- execute immediately

Once approved, the app constructs `OrchestratorRequest` from the supervisor-confirmed intent and calls `PoseidonOrchestrator.execute()`. During execution, the app polls orchestrator events and renders progress, diagnostics, and charts.

### API path

`src/api/server.py` exposes:

- `POST /query`: legacy compatibility path using `run_argo_workflow`
- `POST /v1/query`: orchestrator path
- `WS /v1/stream/{conversation_id}`: execution event stream
- `GET /health` and `GET /stats`: SQLite-backed operational checks

### Campaign path

`src/campaigns/manager.py` creates one orchestrator instance and schedules recurring jobs from `config/campaigns.yaml`. Current campaign types are:

- anomaly watch
- weekly report
- float watcher

Campaign results also feed the memory layer.

## Core Architecture

### 1. Supervisor layer

Files:

- `src/supervisor/agent.py`
- `src/supervisor/state.py`
- `src/supervisor/clarification.py`
- `src/supervisor/map_interface.py`
- `src/supervisor/planner.py`

Responsibilities:

- manage multi-turn conversation state
- apply higher-level reasoning about ambiguity and geographic scope
- ask for missing parameters
- confirm location with float coverage checks
- produce a human-readable execution plan before running expensive work

Important detail: the Supervisor mixes rule-based logic and LLM reasoning. Rule-based clarification comes from `ClarificationEngine`; higher-level geographic reasoning comes from OpenAI with a strict JSON response contract.

### 2. Orchestrator layer

Files:

- `src/orchestrator/main.py`
- `src/orchestrator/router.py`
- `src/orchestrator/state_manager.py`
- `src/state/schemas.py`

Responsibilities:

- build the agent graph with LangGraph
- execute agent stages in order
- retry failed stages
- emit progress events for UI/API streaming
- persist per-conversation state through Redis or local memory fallback

The orchestrator graph currently runs:

1. query understanding
2. data retrieval
3. analysis
4. validation
5. finalize

It also enriches execution context before the first stage by reading historical insights from the memory subsystem.

### 3. Agent layer

Files:

- `src/agents/query_understanding/agent.py`
- `src/agents/data_retrieval/agent.py`
- `src/agents/analysis/agent.py`
- `src/agents/validation/agent.py`
- `src/agents/visualization/agent.py`
- `src/agents/base_agent.py`

Responsibilities:

- `QueryUnderstandingAgent`: parse user query into structured intent and requested variables
- `DataRetrievalAgent`: call the fetcher and capture the generated CSV path
- `AnalysisAgent`: load data through `ArgoDataProcessor`, compute summaries, and build variable insights
- `ValidationAgent`: apply scientific plausibility and outlier checks, then produce confidence scores
- `VisualizationAgent`: auxiliary, not central to the main execution path

### 4. Legacy workflow layer

Files:

- `src/agent/Retrieving_Agent.py`
- `src/agent/react_agent.py`
- `src/agent/graph.py`

This is the older direct workflow. It still matters because:

- the API compatibility route depends on it
- the orchestrator still retains a `legacy` execution mode

The legacy path is simpler, but it duplicates logic now handled by the orchestrator and supervisor stack.

### 5. Tooling and data-processing layer

Files:

- `src/tools/fetcher.py`
- `src/tools/processor.py`
- `src/tools/intent_extractor.py`
- `src/tools/intent_extractor_model.py`
- `src/tools/geosolver.py`
- `src/tools/water_mass_classifier.py`
- `src/tools/registry.py`
- `src/tools/visualizer.py`

Responsibilities:

- fetch remote Argo observations from ERDDAP
- persist fetched snapshots into `data/`
- load data into SQLite
- generate SQL-backed summaries
- classify locations and marine regions
- perform fast local intent extraction with spaCy when available
- support anomaly and water-mass interpretation

`src/tools/processor.py` is the main scientific-processing bridge between raw fetched CSV data and queryable SQLite-backed analysis.

### 6. Memory layer

Files:

- `src/memory/context_builder.py`
- `src/memory/insight_store.py`
- `src/memory/insight_retriever.py`
- `src/memory/short_term.py`
- `src/memory/long_term.py`
- `src/memory/vector_store.py`

Responsibilities:

- write completed analyses into structured SQLite memory plus ChromaDB embeddings
- retrieve historical baselines and semantic matches before new runs
- enrich current analysis with trend context and prior observations

This is what allows the orchestrator to compare current results against previous runs and campaigns.

## Data and State Model

Two model families are used:

- `src/state/models.py`: domain state like `ScientificIntent` and legacy chat state
- `src/state/schemas.py`: orchestrator transport models like `AgentTask`, `AgentResult`, `OrchestratorRequest`, and streaming events

In practice:

- `ScientificIntent` represents the parsed scientific request
- `FloatChatState` supports the legacy graph flow
- orchestrator schemas support the newer staged multi-agent execution

## Configuration

Files under `config/` are small but important:

- `agents.yaml`: enablement, retries, and timeouts
- `tools.yaml`: tool availability flags
- `policies.yaml`: provider defaults and routing policy
- `llms.yaml`: model assignments
- `campaigns.yaml`: schedule and region/float configuration for campaigns

Most operational behavior is still partially hardcoded in Python, so these configs act more like policy hints than a complete control plane.

## Frontend and UX

`app.py` provides:

- conversational chat history
- clarification widgets
- folium-based map confirmation
- workflow approval UI
- live progress events
- diagnostics cards
- Plotly charts for trends, depth profiles, maps, correlations, and variable insights

This file is large because it combines app bootstrapping, execution control, and visualization helpers in one module.

## Runtime Directories

The cleaned structure separates source code from generated state:

- `data/`: fetched CSV files and the primary SQLite database
- `db/`: insight memory SQLite and ChromaDB state
- `models/`: locally trained intent/anomaly models
- `reports/`: generated campaign outputs
- `logs/`: runtime logs
- `notebooks/`: exploratory notebooks

`src/core/paths.py` is now the single place defining those paths.

## Tests

The current test suite covers:

- orchestrator smoke execution
- router behavior
- registry behavior
- validation agent behavior
- processor and tool-level logic
- geosolver behavior

Tests are useful for refactoring safety, but coverage is still strongest around isolated modules rather than full UI or full external-data flows.

## Practical Strengths

- clear separation between supervisor, orchestrator, and tool execution responsibilities
- typed request/response models for the orchestrator
- multiple interfaces over the same engine: UI, API, campaigns
- memory subsystem adds a historical dimension
- validation adds scientific sanity checks instead of returning raw data blindly

## Current Technical Debt

- legacy and new execution paths coexist, so responsibilities are duplicated
- `app.py` is doing too much UI and orchestration work in one module
- packaging metadata is inconsistent (`pyproject.toml` vs `setup.py`)
- config files do not yet fully drive runtime behavior
- naming conventions still reflect multiple development phases

## Folder-by-Folder Reference

- `app.py`: Streamlit interface and execution UI
- `run_campaigns.py`: standalone campaign runner
- `config/`: YAML policy and schedule files
- `docs/`: architecture and overview documentation
- `notebooks/`: ad hoc analysis notebooks
- `scripts/`: setup and local model training helpers
- `src/agent/`: legacy workflow implementation
- `src/agents/`: orchestrator-stage agents
- `src/api/`: FastAPI app
- `src/campaigns/`: scheduled autonomous analyses
- `src/core/`: shared path and runtime utilities
- `src/memory/`: retrieval-augmented memory and insight persistence
- `src/orchestrator/`: main multi-agent execution graph
- `src/skills/`: prompt/skill definitions
- `src/state/`: shared models and schemas
- `src/supervisor/`: conversational control layer
- `src/tools/`: fetch, process, geospatial, and scientific utilities
- `tests/`: regression tests

## Summary

POSEIDON is best understood as a multi-interface ocean-analysis platform with a legacy core and a newer agentic architecture layered on top. The most important runtime path today is:

`Streamlit/FastAPI -> Supervisor -> Orchestrator -> Agents -> Tools -> SQLite/Memory -> UI/API response`

That path is the one to preserve if the project is further refactored.
