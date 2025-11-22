import os
import json
import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.state.models import ScientificIntent


# ---------------------------------------------------------
# 1. Initialize LLM
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ No OPENAI_API_KEY detected. Using fallback heuristics.")

llm = None
if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )


# ---------------------------------------------------------
# 2. Intent Extraction Function
# ---------------------------------------------------------
def extract_intent_with_llm(query: str) -> ScientificIntent:
    """
    Extract scientific intent directly into ScientificIntent schema.
    """
    try:
        if llm is None:
            raise RuntimeError("LLM unavailable")

        # Create the prompt
        prompt = f"""Extract scientific intent from this query:
Query: {query}

Return the result as a JSON object with these exact field names:
{{
  "variable": "temp" | "psal" | "nitrate" | null,
  "operation": "trend" | "anomaly" | "average" | null,
  "location": string | null,  // Extract location name like "Mumbai", "Arabian Sea", etc.
  "lat": number | null,       // Only if explicitly provided (e.g., "lat=19.0860")
  "lon": number | null,       // Only if explicitly provided (e.g., "lon=72.8777")
  "depth": number | null,
  "depth_range": [min, max] | null,
  "time_range": ["start_date", "end_date"] | null,
  "context_needed": string | null
}}

IMPORTANT RULES:
- Extract location name from phrases like "near Mumbai", "at Mumbai", "in Arabian Sea"
- Only set lat/lon if EXPLICITLY provided as numbers in the query
- If location is mentioned but no coordinates given, set "location" and leave lat/lon as null
- Convert depth units to meters (e.g., 1.5km → 1500)
- Time ranges: use ISO 8601 format (e.g., "2024-01-01")
- For month names like "January 2024", convert to ["2024-01-01", "2024-01-31"]

Examples:

Query: "What's the salinity at 500m depth near Mumbai in January 2024?"
Response:
{{
  "variable": "psal",
  "operation": null,
  "location": "Mumbai",
  "lat": null,
  "lon": null,
  "depth": 500,
  "depth_range": null,
  "time_range": ["2024-01-01", "2024-01-31"],
  "context_needed": null
}}

Query: "Show temperature at lat=19.0860, lon=72.8777"
Response:
{{
  "variable": "temp",
  "operation": null,
  "location": null,
  "lat": 19.0860,
  "lon": 72.8777,
  "depth": null,
  "depth_range": null,
  "time_range": null,
  "context_needed": null
}}"""

        # Call the LLM
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # Extract JSON from the response
            json_str = content.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].strip()
                
            obj = json.loads(json_str)
            
            # Convert arrays to tuples for Pydantic
            if 'time_range' in obj:
                if obj['time_range'] is None or obj['time_range'] == [None, None]:
                    obj['time_range'] = None
                elif isinstance(obj['time_range'], list):
                    obj['time_range'] = tuple(obj['time_range'])
                
            if 'depth_range' in obj:
                if obj['depth_range'] is None or obj['depth_range'] == [None, None]:
                    obj['depth_range'] = None
                elif isinstance(obj['depth_range'], list):
                    obj['depth_range'] = tuple(obj['depth_range'])
            
            print(f"✓ Extracted intent: {obj}")
            return ScientificIntent(**obj)
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {content}")
            raise

    except Exception as e:
        print(f"⚠️ Fallback mode triggered: {str(e)}")
        return fallback_intent_extraction(query)


# ---------------------------------------------------------
# 3. Fallback Extraction (no LLM)
# ---------------------------------------------------------
def fallback_intent_extraction(query: str) -> ScientificIntent:
    """Simple keyword-based extraction when LLM is unavailable"""
    q = query.lower()
    obj = {}

    # Variable detection
    if "temp" in q:
        obj["variable"] = "temp"
    elif "salin" in q or "psal" in q:
        obj["variable"] = "psal"
    elif "nitrate" in q:
        obj["variable"] = "nitrate"

    # Operation detection
    if "trend" in q:
        obj["operation"] = "trend"
    elif "diff" in q or "difference" in q:
        obj["operation"] = "difference"
    elif "anom" in q:
        obj["operation"] = "anomaly"

    # Time range detection
    if "6 month" in q or "six month" in q:
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=180)
        obj["time_range"] = (
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d")
        )
    elif "last month" in q:
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=30)
        obj["time_range"] = (
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d")
        )
    else:
        obj["time_range"] = None

    # Set defaults for other fields
    obj.setdefault("location", None)
    obj.setdefault("lat", None)
    obj.setdefault("lon", None)
    obj.setdefault("depth", None)
    obj.setdefault("depth_range", None)
    obj.setdefault("context_needed", "Using fallback extraction - LLM unavailable")

    return ScientificIntent(**obj)


if __name__ == "__main__":
    print("\n🔍 Testing Scientific Intent Extraction\n")

    test_queries = [
        "Show me the salinity trend at lat:19.0760 and lon:72.877 at 1000m depth in jan 2024",
        "Temperature data from last month near Mumbai",
        "What's the nitrate level at 500m depth?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        intent = extract_intent_with_llm(query)
        print(f"➡️ Extracted Intent:")
        for field, value in intent.model_dump().items():
            if value is not None:
                print(f"   • {field}: {value}")
        print("-" * 60)