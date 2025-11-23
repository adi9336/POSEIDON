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

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        self.state = state
        self.llm = None
        self.sql_tool = None
        self._init_components()
        
    def _init_components(self):
        """Initialize database and LLM components."""
        self._init_db()
        self._init_llm()
    
    def _init_db(self):
        """Initialize SQLite database and import data if needed."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                self._create_tables(conn)
                if self._should_import_data(conn):
                    self._import_initial_data(conn)
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def _create_tables(self, conn: sqlite3.Connection):
        """Create necessary database tables if they don't exist."""
        # First, drop the table if it exists to ensure clean schema
        conn.execute("DROP TABLE IF EXISTS argo_data")
        
        # Create the table with the correct schema
        conn.execute("""
            CREATE TABLE argo_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform_number TEXT,
                latitude REAL,
                longitude REAL,
                time TEXT,
                temp REAL,
                psal REAL,
                pres REAL,
                nitrate REAL,
                -- Add indexes for better query performance
                CONSTRAINT idx_lat_lon_time UNIQUE (latitude, longitude, time, pres)
            )
        """)
        
        # Create additional indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pres ON argo_data(pres)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON argo_data(time)")

    def _should_import_data(self, conn: sqlite3.Connection) -> bool:
        """Check if we need to import data into the database."""
        count = conn.execute("SELECT COUNT(*) FROM argo_data;").fetchone()[0]
        return count == 0 and self.state and hasattr(self.state, 'raw_data') and self.state.raw_data

    def _import_initial_data(self, conn: sqlite3.Connection):
        """Import initial data from CSV if available."""
        try:
            df = pd.read_csv(self.state.raw_data)
            df.to_sql("argo_data", conn, if_exists="append", index=False)
            logger.info(f"Imported {len(df)} rows from {self.state.raw_data}")
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            raise

    def _init_llm(self):
        """Initialize the language model and SQL tools."""
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
            self.sql_tool = QuerySQLDataBaseTool(db=db)
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            raise

    def _get_table_columns(self) -> List[str]:
        """Get list of columns in the argo_data table."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(argos_data);")
            return [col[1] for col in cursor.fetchall()]

    def _get_sample_data(self, limit: int = 3) -> List[Dict]:
        """Get sample data from the database."""
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                "SELECT * FROM argo_data LIMIT ?", 
                conn, 
                params=(limit,)
            ).to_dict('records')

    def _build_sql_conditions(self, intent: ScientificIntent, depth_tolerance: float = None) -> Tuple[str, Dict[str, Any]]:
        """Build SQL WHERE conditions based on intent."""
        conditions = []
        params = {}
        
        # Handle location - both lat and lon must be provided
        if intent.lat is not None and intent.lon is not None:
            conditions.extend([
                "latitude BETWEEN :lat_min AND :lat_max",
                "longitude BETWEEN :lon_min AND :lon_max"
            ])
            params.update({
                'lat_min': float(intent.lat) - DEFAULT_LAT_LON_RANGE,
                'lat_max': float(intent.lat) + DEFAULT_LAT_LON_RANGE,
                'lon_min': float(intent.lon) - DEFAULT_LAT_LON_RANGE,
                'lon_max': float(intent.lon) + DEFAULT_LAT_LON_RANGE
            })
        
        # Handle depth with dynamic tolerance
        if intent.depth is not None:
            target_depth = float(intent.depth)
            tolerance = depth_tolerance if depth_tolerance is not None else DEFAULT_DEPTH_TOLERANCE
            conditions.append("pres BETWEEN :depth_min AND :depth_max")
            params.update({
                'depth_min': target_depth - tolerance,
                'depth_max': target_depth + tolerance
            })
        
        # Handle time range
        if intent.time_range:
            start_date, end_date = intent.time_range
            conditions.append("time BETWEEN :start_date AND :end_date")
            params.update({
                'start_date': start_date,
                'end_date': end_date
            })
        
        return " AND ".join(conditions) if conditions else "1=1", params

    def execute_sql_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a parameterized SQL query and return results as DataFrame."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                return pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            raise

    def generate_nlp_summary(self, df: pd.DataFrame, query: str) -> str:
        """Generate a natural language summary of the query results using OpenAI."""
        if df.empty:
            return "No data found matching the query criteria."
        
        try:
            # Prepare data statistics
            stats = {}
            for col in ['temp', 'psal', 'pres']:
                if col in df.columns:
                    col_stats = df[col].describe()
                    stats[col] = {
                        'min': round(col_stats.get('min', 0), 2),
                        'max': round(col_stats.get('max', 0), 2),
                        'mean': round(col_stats.get('mean', 0), 2),
                        'count': int(col_stats.get('count', 0))
                    }
            
            # Get unique locations and platform counts
            location_info = ""
            if 'latitude' in df.columns and 'longitude' in df.columns:
                unique_locations = df[['latitude', 'longitude']].drop_duplicates()
                location_info = f"Data collected from {len(unique_locations)} unique locations. "
            
            # Get platform info
            platform_info = ""
            if 'platform_number' in df.columns:
                unique_platforms = df['platform_number'].nunique()
                platform_info = f"Data comes from {unique_platforms} different Argo floats. "
            
            # Prepare context for OpenAI
            context = {
                'query': query,
                'stats': stats,
                'row_count': len(df),
                'location_info': location_info,
                'platform_info': platform_info,
                'columns': list(df.columns)
            }
            
            # Generate prompt for OpenAI
            prompt = f"""
            You are an oceanographic data assistant. Analyze the following data and provide a concise, 
            insightful summary in natural language. Focus on key patterns, anomalies, and notable observations.
            
            Query: {query}
            
            Data Overview:
            - Number of data points: {row_count}
            {location_info}{platform_info}
            
            Statistics:
            {stats}
            
            Provide a 3-5 sentence summary that would be helpful for an oceanographer.
            Include any notable patterns, ranges, or anomalies in the data.
            If the data is limited, suggest what additional information might be needed.
            """.format(
                query=context['query'],
                row_count=context['row_count'],
                location_info=context['location_info'],
                platform_info=context['platform_info'],
                stats='\n'.join([f"- {k}: {v}" for k, v in context['stats'].items()])
            )
            
            # Call OpenAI API
            response = self.llm.invoke([
                {"role": "system", "content": "You are a helpful oceanographic data analyst."},
                {"role": "user", "content": prompt}
            ])
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating NLP summary with OpenAI: {e}")
            # Fallback to basic summary if OpenAI fails
            try:
                basic_summary = (
                    f"Found {len(df)} data points. "
                    f"Temperature: {df['temp'].min():.1f} to {df['temp'].max():.1f}°C. "
                    f"Salinity: {df['psal'].min():.1f} to {df['psal'].max():.1f} PSU. "
                    f"Depth: {df['pres'].min():.1f} to {df['pres'].max():.1f} m."
                )
                return basic_summary
            except:
                return "Summary generation failed. Please check the data manually."

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return structured results."""
        try:
            logger.info(f"Processing query: {query}")
            
            # Get intent from state or use default
            intent = getattr(self.state, 'intent', None)
            if not intent:
                raise ValueError("No intent available for processing")
            
            # Log database info for debugging
            try:
                logger.info(f"Available columns in database: {self._get_table_columns()}")
                logger.info(f"Sample data: {self._get_sample_data()}")
            except Exception as e:
                logger.warning(f"Could not fetch database info: {e}")
            
            # First try with default depth tolerance
            conditions, params = self._build_sql_conditions(intent, DEFAULT_DEPTH_TOLERANCE)
            
            # Build the SQL query
            sql = f"""
                SELECT temp, psal, latitude, longitude, time, pres, platform_number
                FROM argo_data
                WHERE {conditions}
                ORDER BY ABS(pres - {intent.depth if intent.depth else 0}), time DESC
                LIMIT {DEFAULT_QUERY_LIMIT}
            """
            
            logger.info(f"Executing SQL: {sql} with params: {params}")
            df = self.execute_sql_query(sql, params)
            
            # If no results, try with wider depth range
            if df.empty and intent.depth is not None:
                logger.warning("No data found with initial depth range, trying wider range...")
                conditions, params = self._build_sql_conditions(intent, DEFAULT_DEPTH_TOLERANCE * 5)  # 5x tolerance
                sql = f"""
                    SELECT temp, psal, latitude, longitude, time, pres, platform_number
                    FROM argo_data
                    WHERE {conditions}
                    ORDER BY ABS(pres - {intent.depth}), time DESC
                    LIMIT {DEFAULT_QUERY_LIMIT}
                """
                df = self.execute_sql_query(sql, params)
            
            # Generate summary with more context
            summary = self.generate_nlp_summary(df, query)
            
            return {
                "status": "success",
                "data": df.to_dict(orient='records'),
                "summary": summary,
                "row_count": len(df),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }

def process_data(state: FloatChatState) -> FloatChatState:
    """
    LangGraph node that processes data based on the current state.
    """
    logger.info("Starting data processing")
    
    try:
        # Input validation
        if not hasattr(state, 'user_query') or not state.user_query:
            raise ValueError("No query provided for processing")
            
        # Initialize and process
        processor = ArgoDataProcessor(state=state)
        result = processor.process_query(state.user_query)
        
        # Handle results
        if result["status"] == "error":
            raise Exception(result.get("message", "Unknown error in data processing"))
            
        # Update state
        state.processed = {
            "data": result["data"],
            "summary": result["summary"],
            "row_count": result["row_count"],
            "columns": result["columns"]
        }
        state.status = "processed"
        logger.info(f"Successfully processed {result['row_count']} rows")
        
    except Exception as e:
        error_msg = f"Error in process_data: {str(e)}"
        logger.error(error_msg)
        state.status = "error"
        state.error = error_msg
        
    return state