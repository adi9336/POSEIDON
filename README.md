# POSEIDON: Multi-Agent Oceanographic Analysis

POSEIDON is a multi-agent oceanographic analysis system for Argo float data.
It supports natural language querying, retrieval, analysis, validation, and interactive visualization.

## Features

- Multi-agent orchestration (`query_understanding -> data_retrieval -> analysis -> validation`)
- Natural language query parsing for location, depth, time range, and variables
- Argo ERDDAP data retrieval plus SQLite-backed analytics
- Validation with confidence scoring, time-window checks, and outlier-rate checks
- Streamlit app with live execution progress, diagnostics, and Plotly visualizations
- FastAPI endpoints including `/v1/query` and `/v1/stream/{conversation_id}`
- Supervisor-guided workflow with clarification, map confirmation, and approval
- Autonomous campaign scheduling for recurring monitoring workflows

## Requirements

- Python 3.11+
- OpenAI API key

## Setup

```powershell
git clone https://github.com/adi9336/POSEIDON.git
cd POSEIDON
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\pip3.exe install -r requirements.txt
```

For local development tooling (tests, lint, training extras):

```powershell
.\.venv\Scripts\pip3.exe install -r requirements-dev.txt
```

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

## Run Streamlit App

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

App URL:

- `http://localhost:8501`

## Run FastAPI Server

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.api.server:app --reload
```

API URLs:

- Docs: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`
- Health: `http://127.0.0.1:8000/health`

## API Endpoints

- `POST /query`: legacy compatibility endpoint
- `POST /v1/query`: multi-agent orchestrator response
- `WS /v1/stream/{conversation_id}`: execution event streaming

## Example Query

```json
{
  "query": "Analyze temperature, salinity, and nitrate near Arabian Sea from 2024-05-01 to 2024-06-30 and give key insights.",
  "mode": "multi"
}
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Project Structure

```text
POSEIDON/
|-- app.py
|-- streamlit_app.py
|-- .streamlit/
|-- config/
|-- data/
|-- docs/
|-- notebooks/
|-- scripts/
|-- src/
|   |-- agent/        # legacy LangGraph pipeline
|   |-- agents/       # orchestrator agents
|   |-- api/
|   |-- campaigns/
|   |-- core/         # shared runtime paths
|   |-- memory/
|   |-- orchestrator/
|   |-- skills/
|   |-- state/
|   |-- supervisor/
|   `-- tools/
|-- tests/
|-- runtime.txt
|-- requirements.txt
|-- requirements-dev.txt
`-- pyproject.toml
```

See `docs/project_overview.md` for the full architecture walkthrough.

## Notes

- Runtime database now lives under `data/argo_data.db`.
- Campaign scheduler is disabled by default in Streamlit deployment (`POSEIDON_ENABLE_CAMPAIGNS=false`).
- Semantic memory embedding is disabled by default for lightweight cloud deploys (`POSEIDON_ENABLE_SEMANTIC_MEMORY=false`).
- If you see OpenAI `401 invalid_api_key`, rotate or update your key in `.env`.

## Deploy To Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app and set the main file to `streamlit_app.py`.
3. Add required secrets (at minimum `OPENAI_API_KEY`) in Streamlit App Settings.
4. Keep default lightweight runtime flags:
   - `POSEIDON_ENABLE_CAMPAIGNS=false`
   - `POSEIDON_ENABLE_SEMANTIC_MEMORY=false`
