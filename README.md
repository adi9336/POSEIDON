# POSEIDON: Multi-Agent Oceanographic Analysis

POSEIDON is a multi-agent oceanographic analysis system for Argo float data.  
It supports natural language querying, retrieval, analysis, validation, and rich interactive visualization.

## Features

- Multi-agent orchestration (`query_understanding -> data_retrieval -> analysis -> validation`)
- Natural language query parsing (location, depth, time range, variables)
- Argo ERDDAP data retrieval + SQLite-backed analytics
- Validation with confidence scoring, time-window checks, and outlier-rate checks
- Streamlit app with:
  - live execution progress events
  - diagnostics cards
  - Plotly interactive visualizations (trends, depth profile, geo view, correlations)
- FastAPI endpoints including `/v1/query` and `/v1/stream/{conversation_id}`

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

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

## Run Streamlit App

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
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

- `POST /query`  
  Legacy compatibility endpoint.

- `POST /v1/query`  
  Multi-agent orchestrator response (`status`, `result`, `confidence`, `trace_id`).

- `WS /v1/stream/{conversation_id}`  
  Execution event streaming.

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
├── app.py
├── config/
├── docs/
├── src/
│   ├── agent/
│   ├── agents/
│   ├── api/
│   ├── memory/
│   ├── orchestrator/
│   ├── skills/
│   ├── state/
│   └── tools/
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Notes

- `argo_data.db` is runtime state and should generally not be committed.
- If you see OpenAI `401 invalid_api_key`, rotate/update your key in `.env`.

