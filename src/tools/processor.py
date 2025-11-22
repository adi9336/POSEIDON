import os
import pandas as pd
import sqlite3
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.messages import HumanMessage

from src.state.models import FloatChatState, ScientificIntent

# Database configuration
DB_PATH = "argo_data.db"

class DataProcessor:
    def __init__(self, state: Optional[FloatChatState] = None):
        self.llm = None
        self.sql_tool = None
        self.state = state
        self._init_db()
        self._init_llm()

    def _init_db(self):
        """Initialize the SQLite database with Argo data if it doesn't exist."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()

            # Create table if it doesn't exist
            cur.execute("""
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
                );
            """)

            # Check if we need to import data
            cur.execute("SELECT COUNT(*) FROM argo_data;")
            count = cur.fetchone()[0]

            if count == 0 and hasattr(self.state, 'raw_data') and self.state.raw_data:
                try:
                    df = pd.read_csv(self.state.raw_data)
                    df.to_sql("argo_data", conn, if_exists="append", index=False)
                    print(f"✔ Imported {len(df)} rows from {self.state.raw_data}")
                except Exception as e:
                    print(f"❌ Error importing data: {str(e)}")
                    raise

            conn.commit()
        except Exception as e:
            print(f"❌ Database initialization error: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def _init_llm(self):
        """Initialize the language model and SQL tool."""
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # Using a more reliable model
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
            self.sql_tool = QuerySQLDataBaseTool(db=db)
        except Exception as e:
            print(f"❌ Error initializing LLM: {str(e)}")
            raise

    def _sql_to_df(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            return pd.read_sql_query(sql_query, conn)
        except Exception as e:
            print(f"❌ SQL query error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def _generate_sql(self, query: str, intent: Optional[ScientificIntent] = None) -> str:
        """Generate SQL query from natural language and intent."""
        try:
            # Default values
            lat_condition = ""
            lon_condition = ""
            depth_condition = ""
            time_condition = ""
            
            # Add conditions based on intent if available
            if intent:
                # Handle location
                if hasattr(intent, 'lat') and intent.lat is not None:
                    lat = float(intent.lat)
                    lat_range = 3.0  # ±3 degrees
                    lat_condition = f"AND latitude BETWEEN {lat - lat_range} AND {lat + lat_range}"
                
                if hasattr(intent, 'lon') and intent.lon is not None:
                    lon = float(intent.lon)
                    lon_range = 3.0  # ±3 degrees
                    lon_condition = f"AND longitude BETWEEN {lon - lon_range} AND {lon + lon_range}"

                # Handle depth
                if hasattr(intent, 'depth') and intent.depth is not None:
                    depth = float(intent.depth)
                    depth_condition = f"AND ABS(pres - {depth}) <= 10"  # Within 10m of target depth

                # Handle time range
                if hasattr(intent, 'time_range') and intent.time_range:
                    start_date, end_date = intent.time_range
                    time_condition = f"AND time >= '{start_date}' AND time <= '{end_date}'"

            # Build the SQL query
            sql = f"""
            SELECT psal, latitude, longitude, time, pres, platform_number
            FROM argo_data
            WHERE 1=1
            {lat_condition}
            {lon_condition}
            {depth_condition}
            {time_condition}
            ORDER BY time DESC
            LIMIT 100
            """
            
            return sql.strip()
            
        except Exception as e:
            print(f"❌ Error generating SQL: {str(e)}")
            # Fallback to a simple query if there's an error
            return "SELECT psal, latitude, longitude, time, pres, platform_number FROM argo_data LIMIT 100"

    def _generate_nlp_summary(self, df: pd.DataFrame, query: str) -> str:
        """Generate a natural language summary of the data."""
        try:
            if df.empty:
                return "No data found matching your query."

            prompt = f"""
            User asked: {query}

            Data statistics:
            {df.describe().to_string()}

            Please provide a clear, concise summary of this data in simple terms.
            Focus on key patterns, trends, and any interesting observations.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"❌ Error generating NLP summary: {str(e)}")
            return "I couldn't generate a summary due to an error."

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return results."""
        try:
            print(f"\n🔍 Processing query: {query}")
            
            # Get intent from state if available
            intent = getattr(self.state, 'intent', None)
            
            # Generate and execute SQL
            sql_query = self._generate_sql(query, intent)
            print(f"📝 Generated SQL: {sql_query}")
            
            df = self._sql_to_df(sql_query)
            print(f"📊 Retrieved {len(df)} rows")
            
            # Generate summary
            summary = self._generate_nlp_summary(df, query)
            
            return {
                "status": "success",
                "data": df.to_dict(orient='records'),
                "summary": summary,
                "row_count": len(df),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error in process_query: {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }

def process_data(state: FloatChatState) -> FloatChatState:
    """
    LangGraph node that processes data based on the current state.
    """
    print("\n📊 Processing data...")
    
    if not hasattr(state, 'user_query') or not state.user_query:
        error_msg = "No query provided for processing"
        print(f"❌ {error_msg}")
        state.status = "error"
        state.error = error_msg
        return state
    
    try:
        # Initialize processor with state
        processor = DataProcessor(state=state)
        
        # Process the query
        result = processor.process_query(state.user_query)
        
        if result["status"] == "error":
            raise Exception(result.get("message", "Unknown error in data processing"))
        
        # Update state with results
        state.processed = {
            "data": result["data"],
            "summary": result["summary"],
            "row_count": result["row_count"],
            "columns": result["columns"]
        }
        state.status = "processed"
        
        print(f"✅ Successfully processed {result['row_count']} rows")
        return state
        
    except Exception as e:
        error_msg = f"Error processing data: {str(e)}"
        print(f"❌ {error_msg}")
        state.status = "error"
        state.error = error_msg
        return state