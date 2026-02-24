import os
import pandas as pd
import sqlite3
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.messages import HumanMessage

# Local imports
from src.state.models import FloatChatState, ScientificIntent

# Configure logging with colored output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "argo_data.db"
DEFAULT_LAT_LON_RANGE = 3.0  # ±3 degrees
DEFAULT_DEPTH_TOLERANCE = 10.0  # meters
DEFAULT_QUERY_LIMIT = 100


class ArgoDataProcessor:
    """Handles all data processing for Argo float data."""

    def __init__(self, state: Optional[FloatChatState] = None):
        """Initialize with optional state."""
        logger.info("🔧 Initializing ArgoDataProcessor...")
        self.state = state
        self.llm = None
        self.sql_tool = None
        self._init_components()
        logger.info("✅ ArgoDataProcessor initialized successfully")

    def _init_components(self):
        """Initialize database and LLM components."""
        logger.info("🔄 Initializing components (database and LLM)...")
        self._init_db()
        self._init_llm()
        logger.info("✅ All components initialized successfully")

    def _init_db(self):
        """Initialize SQLite database and import data if needed."""
        try:
            logger.info(f"🗄️  Connecting to database: {DB_PATH}")
            with sqlite3.connect(DB_PATH) as conn:
                logger.info("✅ Database connection established")

                self._create_tables(conn)
                logger.info("✅ Database tables verified/created")

                if self._should_import_data(conn):
                    logger.info("📥 Importing initial data...")
                    self._import_initial_data(conn)
                else:
                    logger.info("✅ Database already contains data, skipping import")

        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            raise

    def _create_tables(self, conn: sqlite3.Connection):
        """Create necessary database tables if they don't exist."""
        logger.info("Creating/verifying database schema...")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS argo_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform_number TEXT,
                latitude REAL,
                longitude REAL,
                time TEXT,
                temp REAL,
                psal REAL,
                pres REAL,
                nitrate REAL
            )
        """)
        logger.info("argo_data table verified/created")

        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_lat_lon_time ON argo_data(latitude, longitude, time, pres)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pres ON argo_data(pres)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON argo_data(time)")
        logger.info("Database indexes created successfully")

    def _should_import_data(self, conn: sqlite3.Connection) -> bool:
        """Check if we need to import data into the database."""
        count = conn.execute("SELECT COUNT(*) FROM argo_data;").fetchone()[0]
        logger.info(f"Current database row count: {count}")
        if not self.state or not hasattr(self.state, "raw_data"):
            return False
        raw_data = (self.state.raw_data or "").strip()
        if not raw_data:
            return False
        return Path(raw_data).exists()

    def _import_initial_data(self, conn: sqlite3.Connection):
        """Import initial data from CSV if available."""
        try:
            logger.info(f"Reading CSV file: {self.state.raw_data}")
            df = pd.read_csv(self.state.raw_data)
            logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

            if "nitrate" not in df.columns:
                df["nitrate"] = None

            required_cols = [
                "platform_number",
                "latitude",
                "longitude",
                "time",
                "temp",
                "psal",
                "pres",
                "nitrate",
            ]
            df = df[required_cols].copy()

            logger.info("Importing data to database with deduplication...")
            insert_sql = """
                INSERT OR IGNORE INTO argo_data
                (platform_number, latitude, longitude, time, temp, psal, pres, nitrate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            rows = list(df.itertuples(index=False, name=None))
            before = conn.execute("SELECT COUNT(*) FROM argo_data;").fetchone()[0]
            conn.executemany(insert_sql, rows)
            conn.commit()
            after = conn.execute("SELECT COUNT(*) FROM argo_data;").fetchone()[0]
            inserted = after - before
            logger.info(
                f"Imported {inserted} new rows ({len(rows)} processed) from {self.state.raw_data}"
            )

        except Exception as e:
            logger.error(f"Error importing data: {e}")
            raise

    def _init_llm(self):
        """Initialize the language model and SQL tools."""
        try:
            logger.info("🤖 Initializing OpenAI LLM...")
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info("✅ OpenAI LLM initialized successfully")

            logger.info("🔗 Setting up SQL database connection...")
            db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
            self.sql_tool = QuerySQLDataBaseTool(db=db)
            logger.info("✅ SQL tools initialized successfully")

        except Exception as e:
            logger.error(f"❌ LLM initialization error: {e}")
            raise

    def _get_table_columns(self) -> List[str]:
        """Get list of columns in the argo_data table."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(argo_data);")
            columns = [col[1] for col in cursor.fetchall()]
            logger.info(f"✅ Retrieved {len(columns)} columns from database")
            return columns

    def _get_sample_data(self, limit: int = 3) -> List[Dict]:
        """Get sample data from the database."""
        with sqlite3.connect(DB_PATH) as conn:
            sample = pd.read_sql_query(
                "SELECT * FROM argo_data LIMIT ?", conn, params=(limit,)
            ).to_dict("records")
            logger.info(f"✅ Retrieved {len(sample)} sample records")
            return sample

    def _build_sql_conditions(
        self, intent: ScientificIntent, depth_tolerance: float = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Build SQL WHERE conditions based on intent."""
        logger.info("🔍 Building SQL query conditions...")
        conditions = []
        params = {}

        # Handle location - both lat and lon must be provided
        if intent.lat is not None and intent.lon is not None:
            conditions.extend(
                [
                    "latitude BETWEEN :lat_min AND :lat_max",
                    "longitude BETWEEN :lon_min AND :lon_max",
                ]
            )
            params.update(
                {
                    "lat_min": float(intent.lat) - DEFAULT_LAT_LON_RANGE,
                    "lat_max": float(intent.lat) + DEFAULT_LAT_LON_RANGE,
                    "lon_min": float(intent.lon) - DEFAULT_LAT_LON_RANGE,
                    "lon_max": float(intent.lon) + DEFAULT_LAT_LON_RANGE,
                }
            )
            logger.info(
                f"✅ Location filter: Lat {intent.lat}±{DEFAULT_LAT_LON_RANGE}, Lon {intent.lon}±{DEFAULT_LAT_LON_RANGE}"
            )

        # Handle depth with dynamic tolerance
        if intent.depth is not None:
            target_depth = float(intent.depth)
            tolerance = (
                depth_tolerance
                if depth_tolerance is not None
                else DEFAULT_DEPTH_TOLERANCE
            )
            conditions.append("pres BETWEEN :depth_min AND :depth_max")
            params.update(
                {
                    "depth_min": target_depth - tolerance,
                    "depth_max": target_depth + tolerance,
                }
            )
            logger.info(f"✅ Depth filter: {target_depth}±{tolerance}m")

        # Handle time range
        if intent.time_range:
            start_date, end_date = intent.time_range
            conditions.append("time BETWEEN :start_date AND :end_date")
            params.update({"start_date": start_date, "end_date": end_date})
            logger.info(f"✅ Time range filter: {start_date} to {end_date}")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        logger.info(
            f"✅ SQL conditions built successfully ({len(conditions)} conditions)"
        )
        return where_clause, params

    def execute_sql_query(
        self, query: str, params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Execute a parameterized SQL query and return results as DataFrame."""
        try:
            logger.info("🔎 Executing SQL query...")
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"✅ Query executed successfully: {len(df)} rows returned")
                return df
        except Exception as e:
            logger.error(f"❌ SQL query error: {e}")
            raise

    def generate_nlp_summary(self, df: pd.DataFrame, query: str) -> str:
        """Generate a natural language summary of the query results using OpenAI."""
        logger.info("📝 Generating natural language summary...")

        if df.empty:
            logger.warning("⚠️  No data found matching query criteria")
            return "No data found matching the query criteria."

        try:
            # Prepare data statistics
            logger.info("📊 Calculating data statistics...")
            stats = {}
            for col in ["temp", "psal", "pres"]:
                if col in df.columns:
                    col_stats = df[col].describe()
                    stats[col] = {
                        "min": round(col_stats.get("min", 0), 2),
                        "max": round(col_stats.get("max", 0), 2),
                        "mean": round(col_stats.get("mean", 0), 2),
                        "count": int(col_stats.get("count", 0)),
                    }
            logger.info(f"✅ Statistics calculated for {len(stats)} parameters")

            # Get unique locations and platform counts
            location_info = ""
            if "latitude" in df.columns and "longitude" in df.columns:
                unique_locations = df[["latitude", "longitude"]].drop_duplicates()
                location_info = (
                    f"Data collected from {len(unique_locations)} unique locations. "
                )
                logger.info(f"✅ Found {len(unique_locations)} unique locations")

            # Get platform info
            platform_info = ""
            if "platform_number" in df.columns:
                unique_platforms = df["platform_number"].nunique()
                platform_info = (
                    f"Data comes from {unique_platforms} different Argo floats. "
                )
                logger.info(f"✅ Data from {unique_platforms} Argo floats")

            # Generate prompt for OpenAI
            logger.info("🤖 Calling OpenAI API for summary generation...")
            prompt = f"""
            You are an oceanographic data assistant. Analyze the following data and provide a concise, 
            insightful summary in natural language. Focus on key patterns, anomalies, and notable observations.
            
            Query: {query}
            
            Data Overview:
            - Number of data points: {len(df)}
            {location_info}{platform_info}
            
            Statistics:
            {chr(10).join([f"- {k}: {v}" for k, v in stats.items()])}
            
            Provide a 3-5 sentence summary that would be helpful for an oceanographer.
            Include any notable patterns, ranges, or anomalies in the data.
            If the data is limited, suggest what additional information might be needed.
            """

            # Call OpenAI API
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info("✅ OpenAI summary generated successfully")

            return response.content.strip()

        except Exception as e:
            logger.error(f"❌ Error generating NLP summary with OpenAI: {e}")
            logger.info("🔄 Falling back to basic summary...")

            # Fallback to basic summary if OpenAI fails
            try:
                basic_summary = (
                    f"Found {len(df)} data points. "
                    f"Temperature: {df['temp'].min():.1f} to {df['temp'].max():.1f}°C. "
                    f"Salinity: {df['psal'].min():.1f} to {df['psal'].max():.1f} PSU. "
                    f"Depth: {df['pres'].min():.1f} to {df['pres'].max():.1f} m."
                )
                logger.info("✅ Basic fallback summary generated")
                return basic_summary
            except:
                logger.error("❌ Summary generation completely failed")
                return "Summary generation failed. Please check the data manually."

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return structured results."""
        try:
            logger.info(f"🚀 Processing query: '{query}'")

            # Get intent from state or use default
            intent = getattr(self.state, "intent", None)
            if not intent:
                raise ValueError("No intent available for processing")
            logger.info("✅ Intent retrieved from state")

            # Log database info for debugging
            try:
                columns = self._get_table_columns()
                logger.info(f"📋 Available columns: {', '.join(columns)}")

                samples = self._get_sample_data()
                logger.info(f"📊 Sample data retrieved ({len(samples)} records)")
            except Exception as e:
                logger.warning(f"⚠️  Could not fetch database info: {e}")

            # First try with default depth tolerance
            logger.info("🔍 Building initial query with default tolerance...")
            conditions, params = self._build_sql_conditions(
                intent, DEFAULT_DEPTH_TOLERANCE
            )

            # Build the SQL query
            sql = f"""
                SELECT temp, psal, latitude, longitude, time, pres, platform_number
                FROM argo_data
                WHERE {conditions}
                ORDER BY ABS(pres - {intent.depth if intent.depth else 0}), time DESC
                LIMIT {DEFAULT_QUERY_LIMIT}
            """

            logger.info(f"📝 SQL Query: {sql[:100]}...")
            logger.info(f"📝 Parameters: {params}")

            df = self.execute_sql_query(sql, params)

            # If no results, try with wider depth range
            if df.empty and intent.depth is not None:
                logger.warning("⚠️  No data found with initial depth range")
                logger.info("🔄 Retrying with 5x wider depth tolerance...")

                conditions, params = self._build_sql_conditions(
                    intent, DEFAULT_DEPTH_TOLERANCE * 5
                )
                sql = f"""
                    SELECT temp, psal, latitude, longitude, time, pres, platform_number
                    FROM argo_data
                    WHERE {conditions}
                    ORDER BY ABS(pres - {intent.depth}), time DESC
                    LIMIT {DEFAULT_QUERY_LIMIT}
                """
                df = self.execute_sql_query(sql, params)

                if not df.empty:
                    logger.info("✅ Data found with wider tolerance")

            # Generate summary with more context
            summary = self.generate_nlp_summary(df, query)

            logger.info(
                f"✅ Query processing completed successfully: {len(df)} results"
            )

            return {
                "status": "success",
                "data": df.to_dict(orient="records"),
                "summary": summary,
                "row_count": len(df),
                "columns": list(df.columns),
            }

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "message": error_msg}


def process_data(state: FloatChatState) -> FloatChatState:
    """
    LangGraph node that processes data based on the current state.
    """
    logger.info("=" * 60)
    logger.info("🌊 STARTING DATA PROCESSING NODE")
    logger.info("=" * 60)

    try:
        # Input validation
        logger.info("🔍 Validating input state...")
        if not hasattr(state, "user_query") or not state.user_query:
            raise ValueError("No query provided for processing")
        logger.info(f"✅ Query validated: '{state.user_query}'")

        # Initialize and process
        processor = ArgoDataProcessor(state=state)

        logger.info("🎯 Executing query processing...")
        result = processor.process_query(state.user_query)

        # Handle results
        if result["status"] == "error":
            raise Exception(result.get("message", "Unknown error in data processing"))

        logger.info("✅ Query executed successfully")

        # Update state
        logger.info("💾 Updating state with results...")
        state.processed = {
            "data": result["data"],
            "summary": result["summary"],
            "row_count": result["row_count"],
            "columns": result["columns"],
        }
        state.status = "processed"

        # Save the summary to final_answer
        state.final_answer = result["summary"]

        logger.info("=" * 60)
        logger.info(
            f"✅ DATA PROCESSING COMPLETED: {result['row_count']} rows processed"
        )
        logger.info(f"📊 Summary: {state.final_answer}")
        logger.info("=" * 60)

    except Exception as e:
        error_msg = f"Error in process_data: {str(e)}"
        logger.error("=" * 60)
        logger.error(f"❌ DATA PROCESSING FAILED")
        logger.error(f"❌ {error_msg}")
        logger.error("=" * 60)
        state.status = "error"
        state.error = error_msg

    return state

