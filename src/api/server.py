from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing modules
from src.state.models import FloatChatState
from src.agent.Retrieving_Agent import run_argo_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Argo Float Data API",
    description="API for querying Argo oceanographic float data",
    version="1.0.0",
)

# Configure CORS to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about Argo data")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What's the salinity at 500m depth near Mumbai in January 2024?"
            }
        }


class QueryResponse(BaseModel):
    status: str
    summary: str
    data: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    intent: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = Field(
        None, description="The final processed answer from the workflow, if available"
    )


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    details: Optional[str] = None


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "online",
        "message": "Argo Float Data API is running",
        "version": "1.0.0",
        "endpoints": {"query": "/query", "health": "/health", "docs": "/docs"},
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        import sqlite3

        with sqlite3.connect("argo_data.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM argo_data").fetchone()[0]

        return {"status": "healthy", "database": "connected", "records": count}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query about Argo float data by forwarding it to the Retrieving_Agent.

    Args:
        request: QueryRequest containing the user's natural language query

    Returns:
        QueryResponse with processed data and summary

    Example:
        POST /query
        {
            "query": "What's the temperature at 200m depth near latitude 19, longitude 72?"
        }
    """
    try:
        logger.info("=" * 60)
        logger.info(f"🌊 NEW QUERY RECEIVED: {request.query}")
        logger.info("=" * 60)

        # Run the Argo workflow
        logger.info("🔄 Starting Argo workflow...")
        result = run_argo_workflow(request.query)

        # Extract the processed data from the workflow result
        processed = result.get("processed", {})

        if not processed:
            error_msg = "No data was processed. Please check your query and try again."
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)

        logger.info("=" * 60)
        logger.info(
            f"✅ QUERY COMPLETED: {len(processed.get('data', []))} rows returned"
        )
        logger.info("=" * 60)

        # Return the response
        return QueryResponse(
            status="success",
            summary=processed.get("summary", "No summary available"),
            data=processed.get("data", []),
            row_count=len(processed.get("data", [])),
            columns=processed.get("columns", []),
            final_answer=result.get("final_answer"),
            intent={
                "lat": result.get("intent", {}).get("lat"),
                "lon": result.get("intent", {}).get("lon"),
                "depth": result.get("intent", {}).get("depth"),
                "time_range": result.get("intent", {}).get("time_range"),
                "location": result.get("intent", {}).get("location"),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error processing query: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.exception(e)  # Log full stack trace
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/examples")
async def get_example_queries():
    """Get example queries that users can try"""
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
    """Get statistics about the database"""
    try:
        import sqlite3
        import pandas as pd

        with sqlite3.connect("argo_data.db") as conn:
            # Get total count
            total = conn.execute("SELECT COUNT(*) FROM argo_data").fetchone()[0]

            # Get depth range
            depth_stats = conn.execute(
                "SELECT MIN(pres) as min_depth, MAX(pres) as max_depth FROM argo_data"
            ).fetchone()

            # Get time range
            time_stats = conn.execute(
                "SELECT MIN(time) as earliest, MAX(time) as latest FROM argo_data"
            ).fetchone()

            # Get unique platforms
            platforms = conn.execute(
                "SELECT COUNT(DISTINCT platform_number) FROM argo_data"
            ).fetchone()[0]

            # Get location bounds
            location_stats = conn.execute("""SELECT 
                    MIN(latitude) as min_lat, MAX(latitude) as max_lat,
                    MIN(longitude) as min_lon, MAX(longitude) as max_lon
                FROM argo_data""").fetchone()

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
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"status": "error", "message": exc.detail, "status_code": exc.status_code}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled Exception: {str(exc)}")
    logger.exception(exc)
    return {
        "status": "error",
        "message": "An unexpected error occurred",
        "details": str(exc),
    }


# Run the app
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Argo Float Data API Server...")
    logger.info("📍 Server will be available at: http://localhost:8000")
    logger.info("📚 API Documentation at: http://localhost:8000/docs")
    logger.info("🔍 Alternative docs at: http://localhost:8000/redoc")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
