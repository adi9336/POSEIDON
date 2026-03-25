# Full Project Description and File-by-File Analysis

## Executive Summary
POSEIDON is a multi-agent oceanographic analysis platform that combines a Supervisor conversation layer, an orchestrator graph, task-specific agents, scientific tooling, and optional autonomous campaigns. It supports Streamlit UI usage and FastAPI API usage.

Primary runtime path:
`User Query -> Supervisor -> Orchestrator -> (Query Understanding -> Data Retrieval -> Analysis -> Validation) -> Final Response + Memory Write-back`

## Runtime Components
- UI entrypoint: `streamlit_app.py` -> `app.py`
- API entrypoint: `src/api/server.py`
- Orchestrator graph: `src/orchestrator/main.py`
- Supervisor flow: `src/supervisor/*`
- Agent implementations: `src/agents/*`
- Legacy fallback pipeline: `src/agent/*`
- Data and science tools: `src/tools/*`
- Context and memory: `src/memory/*`
- Config-driven controls: `config/*.yaml`

## File Inventory Metrics
- Total analyzed files: 91
- Python modules: 69

## File-by-File Analysis

### Agent
- **src\agents\__init__.py**
  - Lines: 14, Size: 0.5 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agents\analysis\__init__.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agents\analysis\agent.py**
  - Lines: 149, Size: 6.7 KB
  - Role: Classes: AnalysisAgent
- **src\agents\base_agent.py**
  - Lines: 17, Size: 0.7 KB
  - Role: Classes: BaseAgent
- **src\agents\data_retrieval\__init__.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agents\data_retrieval\agent.py**
  - Lines: 64, Size: 2.59 KB
  - Role: Classes: DataRetrievalAgent
- **src\agents\query_understanding\__init__.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agents\query_understanding\agent.py**
  - Lines: 66, Size: 2.62 KB
  - Role: Classes: QueryUnderstandingAgent
- **src\agents\validation\__init__.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agents\validation\agent.py**
  - Lines: 180, Size: 8.38 KB
  - Role: Classes: ValidationAgent
- **src\agents\visualization\__init__.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agents\visualization\agent.py**
  - Lines: 16, Size: 0.68 KB
  - Role: Classes: VisualizationAgent

### API
- **src\api\server.py**
  - Lines: 196, Size: 6.99 KB
  - Role: Classes: QueryRequest, QueryResponse || Functions: _intent_to_dict, _run_legacy_query

### Campaigns
- **src\campaigns\__init__.py**
  - Lines: 1, Size: 0.06 KB
  - Role: Module-level helpers/constants or imports only.
- **src\campaigns\anomaly_watch.py**
  - Lines: 47, Size: 2.4 KB
  - Role: Classes: AnomalyWatchCampaign
- **src\campaigns\base_campaign.py**
  - Lines: 60, Size: 2.53 KB
  - Role: Classes: BaseCampaign
- **src\campaigns\float_watcher.py**
  - Lines: 68, Size: 3.13 KB
  - Role: Classes: FloatWatcherCampaign || Functions: _point_to_region
- **src\campaigns\manager.py**
  - Lines: 59, Size: 2.73 KB
  - Role: Classes: CampaignManager
- **src\campaigns\weekly_report.py**
  - Lines: 35, Size: 1.63 KB
  - Role: Classes: WeeklyReportCampaign

### Configuration
- **config\agents.yaml**
  - Lines: 17, Size: 0.29 KB
  - Role: Header: agents: ¦   query_understanding: ¦     enabled: true ¦     retries: 2
- **config\campaigns.yaml**
  - Lines: 24, Size: 0.44 KB
  - Role: Header: campaigns: ¦   anomaly_watch: ¦     enabled: true ¦     hour: 6
- **config\llms.yaml**
  - Lines: 16, Size: 0.33 KB
  - Role: Header: llms: ¦   orchestrator: ¦     provider: openai ¦     model: gpt-4o-mini
- **config\policies.yaml**
  - Lines: 14, Size: 0.36 KB
  - Role: Header: policies: ¦   default_provider: openai ¦   max_task_retries: 2 ¦   max_query_retries: 3
- **config\tools.yaml**
  - Lines: 13, Size: 0.2 KB
  - Role: Header: tools: ¦   sql_generator: ¦     enabled: true ¦   data_fetcher:

### Core
- **src\core\__init__.py**
  - Lines: 1, Size: 0.06 KB
  - Role: Module-level helpers/constants or imports only.
- **src\core\paths.py**
  - Lines: 22, Size: 0.94 KB
  - Role: Functions: ensure_runtime_dirs

### Documentation
- **docs\architecture.md**
  - Lines: 18, Size: 0.59 KB
  - Role: Header: # POSEIDON Multi-Agent Architecture ¦ ## Runtime modes ¦ - `legacy`: Use existing `run_argo_workflow` pipeline. ¦ - `hybrid`: Default. Use orchestrator for `/v1/query`, keep `/query` compatibility.
- **docs\full_project_description.md**
  - Lines: 295, Size: 14.88 KB
  - Role: Header: # Full Project Description and File-by-File Analysis ¦ ## Executive Summary ¦ POSEIDON is a multi-agent oceanographic analysis platform that combines a Supervisor conversation layer, an orchestrator graph, task-specific agents, scientific tooling, and optional autonomous campaigns. It supports Streamlit UI usage and FastAPI API usage. ¦ Primary runtime path:
- **docs\implementation_plan.md**
  - Lines: 121, Size: 8.63 KB
  - Role: Header: # POSEIDON Supervisor Agent Architecture Redesign ¦ A production-grade Supervisor Agent that sits between the user and the existing 5-agent orchestrator, providing intelligent clarification, geographic confirmation, workflow approval, and progress communication. ¦ ## User Review Required ¦ > [!IMPORTANT]
- **docs\project_overview.md**
  - Lines: 203, Size: 9.78 KB
  - Role: Header: # POSEIDON Project Overview ¦ ## What This Project Is ¦ POSEIDON is an oceanographic analysis application built around Argo float data. It accepts natural-language questions, resolves them into a scientific intent, fetches matching observations from ERDDAP, runs analysis and validation, and presents the result through: ¦ - a Streamlit chat UI in `app.py`
- **docs\project_workflow.html**
  - Lines: 247, Size: 7.82 KB
  - Role: 

### Legacy Agent
- **src\agent\__init__.py**
  - Lines: 1, Size: 0.05 KB
  - Role: Module-level helpers/constants or imports only.
- **src\agent\graph.py**
  - Lines: 61, Size: 2.83 KB
  - Role: Classes: ArgoWorkflow || Functions: create_argo_workflow, run_argo_workflow
- **src\agent\react_agent.py**
  - Lines: 185, Size: 7.75 KB
  - Role: Classes: ReactAgent || Functions: main
- **src\agent\Retrieving_Agent.py**
  - Lines: 199, Size: 7.84 KB
  - Role: Functions: extract_intent, fetch_data, create_argo_workflow, run_argo_workflow

### Memory
- **src\memory\__init__.py**
  - Lines: 16, Size: 0.53 KB
  - Role: Module-level helpers/constants or imports only.
- **src\memory\context_builder.py**
  - Lines: 45, Size: 1.93 KB
  - Role: Classes: ContextBuilder
- **src\memory\insight_retriever.py**
  - Lines: 140, Size: 6.22 KB
  - Role: Classes: InsightRetriever
- **src\memory\insight_store.py**
  - Lines: 124, Size: 5.67 KB
  - Role: Classes: InsightStore
- **src\memory\long_term.py**
  - Lines: 17, Size: 0.62 KB
  - Role: Classes: LongTermMemory
- **src\memory\short_term.py**
  - Lines: 13, Size: 0.63 KB
  - Role: Classes: ShortTermMemory
- **src\memory\vector_store.py**
  - Lines: 30, Size: 1.34 KB
  - Role: Classes: VectorStoreAdapter, ChromaAdapter, PineconeAdapter

### Notebooks
- **notebooks\test.ipynb**
  - Lines: 481, Size: 233.61 KB
  - Role: Jupyter notebook for exploratory workflows.
- **notebooks\visualisati.ipynb**
  - Lines: 309, Size: 59.46 KB
  - Role: Jupyter notebook for exploratory workflows.

### Orchestration
- **src\orchestrator\__init__.py**
  - Lines: 4, Size: 0.22 KB
  - Role: Module-level helpers/constants or imports only.
- **src\orchestrator\main.py**
  - Lines: 386, Size: 16.49 KB
  - Role: Classes: WorkflowState, PoseidonOrchestrator
- **src\orchestrator\router.py**
  - Lines: 39, Size: 1.72 KB
  - Role: Classes: RoutingDecision, PolicyRouter
- **src\orchestrator\state_manager.py**
  - Lines: 25, Size: 1.02 KB
  - Role: Classes: StateManager

### Root
- **app.py**
  - Lines: 600, Size: 29.22 KB
  - Role: Functions: initialize, render_diagnostics, render_visualizations, render_clarification_dialog, render_map_confirmation, render_approval_card, _handle_supervisor_response, _progress_from_events, execute_with_live_progress, main, _run_execution
- **pyproject.toml**
  - Lines: 37, Size: 0.91 KB
  - Role: Header: [build-system] ¦ requires = ["setuptools>=42", "wheel"] ¦ build-backend = "setuptools.build_meta" ¦ [project]
- **README.md**
  - Lines: 105, Size: 3.46 KB
  - Role: Header: # POSEIDON: Multi-Agent Oceanographic Analysis ¦ POSEIDON is a multi-agent oceanographic analysis system for Argo float data. ¦ It supports natural language querying, retrieval, analysis, validation, and interactive visualization. ¦ ## Features
- **requirements.txt**
  - Lines: 24, Size: 0.42 KB
  - Role: Header: langgraph==0.1.19 ¦ langchain-core==0.2.43 ¦ langchain-community==0.2.4 ¦ langchain-openai==0.1.25
- **requirements-dev.txt**
  - Lines: 14, Size: 0.24 KB
  - Role: Header: -r requirements.txt ¦ pytest>=8.4.2 ¦ black==26.1a1 ¦ isort==7.0.0
- **run_campaigns.py**
  - Lines: 25, Size: 0.7 KB
  - Role: Module-level helpers/constants or imports only.
- **runtime.txt**
  - Lines: 1, Size: 0.01 KB
  - Role: Header: python-3.11
- **setup.py**
  - Lines: 2, Size: 0.04 KB
  - Role: Module-level helpers/constants or imports only.
- **src\__init__.py**
  - Lines: 1, Size: 0.05 KB
  - Role: Module-level helpers/constants or imports only.
- **streamlit_app.py**
  - Lines: 3, Size: 0.06 KB
  - Role: Module-level helpers/constants or imports only.
- **uv.lock**
  - Lines: 1260, Size: 261.46 KB
  - Role: 

### Scripts
- **scripts\setup.sh**
  - Lines: 5, Size: 0.2 KB
  - Role: Header: #!/bin/bash ¦ # Run once to set up all POSEIDON dependencies and model downloads ¦ pip install -r requirements.txt ¦ python -m spacy download en_core_web_sm
- **scripts\train_models.py**
  - Lines: 84, Size: 3.26 KB
  - Role: Functions: train_anomaly_detector, train_intent_ner

### Skills
- **src\skills\data_visualization\SKILL.md**
  - Lines: 11, Size: 0.58 KB
  - Role: Header: --- ¦ name: data-visualization ¦ description: Produce ocean-data visual outputs and reporting directives. Use when users request plots, trends, maps, or export-ready charts from retrieved and analyzed datasets. ¦ ---
- **src\skills\oceanographic_analysis\SKILL.md**
  - Lines: 10, Size: 0.62 KB
  - Role: Header: --- ¦ name: oceanographic-analysis ¦ description: Analyze oceanographic query outputs with domain checks and trend logic. Use when requests require interpretation of temperature, salinity, depth, anomalies, or basin-aware insights. ¦ ---
- **src\skills\sql_generation\SKILL.md**
  - Lines: 10, Size: 0.56 KB
  - Role: Header: --- ¦ name: sql-generation ¦ description: Generate safe parameterized SQL plans for oceanographic retrieval tasks. Use when natural language requests must be translated into query filters, joins, and time/depth constraints. ¦ ---

### State Models
- **src\state\__init__.py**
  - Lines: 27, Size: 0.6 KB
  - Role: Module-level helpers/constants or imports only.
- **src\state\models.py**
  - Lines: 138, Size: 5.68 KB
  - Role: Classes: ScientificIntent, FloatChatState, AgentRole, Message, AgentState
- **src\state\schemas.py**
  - Lines: 61, Size: 2.26 KB
  - Role: Classes: AgentTask, AgentResult, QualityReport, ExecutionPlan, OrchestratorRequest, OrchestratorResponse, EventType, StreamEvent

### Supervisor
- **src\supervisor\__init__.py**
  - Lines: 1, Size: 0.08 KB
  - Role: Module-level helpers/constants or imports only.
- **src\supervisor\agent.py**
  - Lines: 474, Size: 22.31 KB
  - Role: Classes: SupervisorAgent
- **src\supervisor\clarification.py**
  - Lines: 202, Size: 7.19 KB
  - Role: Classes: ClarificationEngine
- **src\supervisor\map_interface.py**
  - Lines: 117, Size: 3.9 KB
  - Role: Classes: MapInterface
- **src\supervisor\planner.py**
  - Lines: 99, Size: 3.96 KB
  - Role: Classes: WorkflowPlanner
- **src\supervisor\state.py**
  - Lines: 118, Size: 5.27 KB
  - Role: Classes: SupervisorPhase, ClarificationQuestion, WorkflowPlan, SupervisorResponse, SupervisorConversationState

### Tests
- **tests\integration\test_orchestrator_smoke.py**
  - Lines: 68, Size: 2.25 KB
  - Role: Functions: test_orchestrator_smoke
- **tests\test_geosolver.py**
  - Lines: 157, Size: 7.38 KB
  - Role: Classes: TestCentroidAccuracy, TestFuzzyMatching, TestCityResolution, TestPointClassification, TestEdgeCases, TestBackwardCompat, TestRegionListing || Functions: classifier
- **tests\test_processor.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **tests\test_tools.py**
  - Lines: 0, Size: 0 KB
  - Role: Module-level helpers/constants or imports only.
- **tests\unit\test_registry.py**
  - Lines: 19, Size: 0.75 KB
  - Role: Classes: InputModel, OutputModel || Functions: _double, test_registry_register_and_invoke, test_registry_validation_failure
- **tests\unit\test_router.py**
  - Lines: 7, Size: 0.31 KB
  - Role: Functions: test_router_defaults_to_openai
- **tests\unit\test_validation_agent.py**
  - Lines: 85, Size: 3.29 KB
  - Role: Functions: test_validation_agent_passes_plausible_data, test_validation_agent_flags_invalid_depth_and_no_data, test_validation_agent_flags_time_window_inconsistency, test_validation_agent_flags_high_outlier_rate

### Tooling
- **src\tools\__init__.py**
  - Lines: 8, Size: 0.38 KB
  - Role: Module-level helpers/constants or imports only.
- **src\tools\fetch.py**
  - Lines: 271, Size: 10.2 KB
  - Role: Functions: get_default_time_range, fetch_argo_data, _fetch_with_params, print_test_results
- **src\tools\fetcher.py**
  - Lines: 256, Size: 10.89 KB
  - Role: Functions: get_default_time_range, fetch_argo_data, _fetch_with_params
- **src\tools\geosolver.py**
  - Lines: 414, Size: 16.78 KB
  - Role: Classes: MarinePolygonClassifier, GeoSolver || Functions: _build_polygons, resolve_location_fast
- **src\tools\intent_extractor.py**
  - Lines: 300, Size: 12.19 KB
  - Role: Functions: parse_relative_time, extract_intent_with_llm, fallback_intent_extraction
- **src\tools\intent_extractor_model.py**
  - Lines: 66, Size: 2.41 KB
  - Role: Functions: extract_intent_fast, _spacy_extract
- **src\tools\OCEAN_profile_tool.py**
  - Lines: 464, Size: 16.64 KB
  - Role: Functions: pressure_to_depth, quality_control, bin_vertical_profile, smooth_profile, compute_physics, compute_depth_context, detect_thermocline, classify_layers, extract_insights, analyze_single_profile, analyze_ocean_profile, ocean_profile_analysis_tool, main
- **src\tools\processor.py**
  - Lines: 421, Size: 18.94 KB
  - Role: Classes: ArgoDataProcessor || Functions: process_data
- **src\tools\registry.py**
  - Lines: 45, Size: 1.84 KB
  - Role: Classes: RegisteredTool, ToolRegistry
- **src\tools\visualizer.py**
  - Lines: 180, Size: 6.47 KB
  - Role: Classes: CSVVisualizer || Functions: main
- **src\tools\water_mass_classifier.py**
  - Lines: 121, Size: 5.23 KB
  - Role: Functions: _load_anomaly_model, classify_water_mass_rule_based, classify_water_masses, get_dominant_water_mass, detect_anomalies

## Deployment Readiness Notes
- Streamlit deployment files exist: `.streamlit/config.toml`, `streamlit_app.py`, `runtime.txt`.
- Production dependencies are separated from dev dependencies (`requirements.txt` vs `requirements-dev.txt`).
- Campaigns and semantic memory are toggled by environment flags to keep cloud runtime lightweight.

## Operational Risks to Track
- Some modules still use deprecated `datetime.utcnow()` and should be migrated to timezone-aware datetime.
- Legacy (`src/agent`) and modern (`src/agents`) paths coexist, increasing maintenance complexity.
- Optional heavy NLP/vector components require explicit enablement and resources in hosted environments.
