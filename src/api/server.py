from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.Retrieving_Agent import run_argo_workflow
from src.orchestrator.main import PoseidonOrchestrator
from src.state.schemas import (
    OrchestratorRequest,
    OrchestratorResponse,
)

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="POSEIDON API",
    description="Multi-agent orchestrator and legacy-compatible Argo query API",
    version="2.0.0",
)
orchestrator = PoseidonOrchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about Argo data")


class QueryResponse(BaseModel):
    status: str
    summary: str
    data: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    intent: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


def _intent_to_dict(intent_obj: Any) -> Dict[str, Any]:
    if intent_obj is None:
        return {}
    if isinstance(intent_obj, dict):
        return intent_obj
    if hasattr(intent_obj, "model_dump"):
        return intent_obj.model_dump()
    if hasattr(intent_obj, "dict"):
        return intent_obj.dict()
    return {}


def _run_legacy_query(query: str) -> QueryResponse:
    result = run_argo_workflow(query)
    processed = result.get("processed", {})
    intent_obj = _intent_to_dict(result.get("intent"))
    if not processed:
        raise HTTPException(
            status_code=404,
            detail="No data was processed. Please check your query and try again.",
        )
    return QueryResponse(
        status="success",
        summary=processed.get("summary", "No summary available"),
        data=processed.get("data", []),
        row_count=len(processed.get("data", [])),
        columns=processed.get("columns", []),
        final_answer=result.get("final_answer"),
        intent={
            "lat": intent_obj.get("lat"),
            "lon": intent_obj.get("lon"),
            "depth": intent_obj.get("depth"),
            "time_range": intent_obj.get("time_range"),
            "location": intent_obj.get("location"),
        },
    )


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "POSEIDON API is running",
        "version": "2.0.0",
        "endpoints": {
            "legacy_query": "/query",
            "v1_query": "/v1/query",
            "stream": "/v1/stream/{conversation_id}",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    try:
        with sqlite3.connect("argo_data.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM argo_data").fetchone()[0]
        return {"status": "healthy", "database": "connected", "records": count}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def process_query_legacy(request: QueryRequest):
    """
    Compatibility endpoint for one release cycle.
    """
    try:
        return _run_legacy_query(request.query)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/query", response_model=OrchestratorResponse)
async def process_query_v1(request: OrchestratorRequest):
    try:
        logger.info(f"v1 query received: {request.query}")
        response = orchestrator.execute(request)
        return response
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/stream/{conversation_id}")
async def stream_events(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    try:
        while True:
            events = orchestrator.pop_events(conversation_id)
            if events:
                for event in events:
                    await websocket.send_json(event.model_dump(mode="json"))
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation_id={conversation_id}")
    except Exception as e:
        logger.exception(e)
        await websocket.close(code=1011)


@app.get("/examples")
async def get_example_queries():
    return {
        "examples": [
            "What's the salinity at 500m depth near Mumbai in January 2024?",
            "Show temperature data at 200m depth near latitude 19, longitude 72",
            "Get ocean data between 100m and 300m depth in the Arabian Sea",
            "What's the temperature at 150m depth near Chennai?",
            "Show me salinity readings at 400m near Goa in March 2024",
        ]
    }


@app.get("/stats")
async def get_database_stats():
    try:
        with sqlite3.connect("argo_data.db") as conn:
            total = conn.execute("SELECT COUNT(*) FROM argo_data").fetchone()[0]
            depth_stats = conn.execute(
                "SELECT MIN(pres) as min_depth, MAX(pres) as max_depth FROM argo_data"
            ).fetchone()
            time_stats = conn.execute(
                "SELECT MIN(time) as earliest, MAX(time) as latest FROM argo_data"
            ).fetchone()
            platforms = conn.execute(
                "SELECT COUNT(DISTINCT platform_number) FROM argo_data"
            ).fetchone()[0]
            location_stats = conn.execute(
                """
                SELECT
                    MIN(latitude) as min_lat, MAX(latitude) as max_lat,
                    MIN(longitude) as min_lon, MAX(longitude) as max_lon
                FROM argo_data
                """
            ).fetchone()

        return {
            "total_records": total,
            "depth_range": {"min": depth_stats[0], "max": depth_stats[1]},
            "time_range": {"earliest": time_stats[0], "latest": time_stats[1]},
            "platforms": platforms,
            "location_bounds": {
                "latitude": {"min": location_stats[0], "max": location_stats[1]},
                "longitude": {"min": location_stats[2], "max": location_stats[3]},
            },
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
