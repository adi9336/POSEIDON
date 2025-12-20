import os
import json
import datetime
import re
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
# 2. Helper Function: Parse Relative Time (moved to top level)
# ---------------------------------------------------------
def parse_relative_time(time_str: str) -> tuple:
    """Parse relative time expressions into start and end dates.

    Args:
        time_str: String containing time expression (e.g., 'last 30 days', 'last month')

    Returns:
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    """
    today = datetime.datetime.utcnow()
    time_str = time_str.lower().strip()

    try:
        # Handle 'last month' specifically
        if time_str == "last month":
            # Get first day of current month
            first_of_current = today.replace(day=1)
            # Get last day of previous month
            last_day_prev_month = first_of_current - datetime.timedelta(days=1)
            # Get first day of previous month
            first_day_prev_month = last_day_prev_month.replace(day=1)

            start_date = first_day_prev_month.strftime("%Y-%m-%d")
            end_date = last_day_prev_month.strftime("%Y-%m-%d")
            print(f"   ⏰ Time range for 'last month': {start_date} to {end_date}")
            return start_date, end_date

        # Handle other time expressions
        if "last" in time_str:
            num = 1  # Default to 1 if no number is specified

            # Extract number if present (e.g., 'last 3 days' -> 3)
            match = re.search(r"last\s+(\d+)", time_str)
            if match:
                num = int(match.group(1))

            if "day" in time_str:
                end_date = today
                start_date = end_date - datetime.timedelta(days=num)
            elif "week" in time_str:
                end_date = today
                start_date = end_date - datetime.timedelta(weeks=num)
            elif "month" in time_str:
                # For 'last X months' (not just 'last month' which is handled above)
                end_date = today.replace(day=1) - datetime.timedelta(days=1)
                start_date = end_date.replace(day=1)
                if num > 1:
                    # Approximate month calculation
                    start_date = start_date.replace(
                        month=max(1, start_date.month - num + 1)
                    )
            elif "year" in time_str:
                end_date = today
                start_date = end_date.replace(year=end_date.year - num, month=1, day=1)
                end_date = end_date.replace(month=12, day=31)
            else:
                return None, None

            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            print(
                f"   ⏰ Time range for '{time_str}': {start_date_str} to {end_date_str}"
            )
            return start_date_str, end_date_str

    except Exception as e:
        print(f"⚠️ Error parsing time string '{time_str}': {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None

    return None, None


# ---------------------------------------------------------
# 3. Intent Extraction Function
# ---------------------------------------------------------
def extract_intent_with_llm(query: str) -> ScientificIntent:
    """
    Extract scientific intent directly into ScientificIntent schema.
    """
    try:
        if llm is None:
            raise RuntimeError("LLM unavailable")

        # Get current date for relative time expressions
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        # Create the prompt
        prompt = f"""Extract scientific intent from this query:
Query: {query}
Current Date: {today}

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
- For relative time like "last month", "last 30 days", calculate from current date ({today})
- For month names like "January 2024", convert to ["2024-01-01", "2024-01-31"]
- "last month" means the complete previous month (e.g., if today is Dec 19, 2024, last month is Nov 1-30, 2024)
- "last 30 days" means 30 days back from today
- If no time is specified, leave time_range as null

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

Query: "Temperature data from last month"
Response (assuming today is 2024-12-19):
{{
  "variable": "temp",
  "operation": null,
  "location": null,
  "lat": null,
  "lon": null,
  "depth": null,
  "depth_range": null,
  "time_range": ["2024-11-01", "2024-11-30"],
  "context_needed": null
}}"""

        # Call the LLM
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            # Extract JSON from the response
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()

            obj = json.loads(json_str)

            # Convert arrays to tuples for Pydantic
            if "time_range" in obj:
                if obj["time_range"] is None or obj["time_range"] == [None, None]:
                    obj["time_range"] = None
                elif isinstance(obj["time_range"], list):
                    obj["time_range"] = tuple(obj["time_range"])

            if "depth_range" in obj:
                if obj["depth_range"] is None or obj["depth_range"] == [None, None]:
                    obj["depth_range"] = None
                elif isinstance(obj["depth_range"], list):
                    obj["depth_range"] = tuple(obj["depth_range"])

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
# 4. Fallback Extraction (no LLM)
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
    else:
        obj["variable"] = None

    # Operation detection
    if "trend" in q:
        obj["operation"] = "trend"
    elif "diff" in q or "difference" in q:
        obj["operation"] = "difference"
    elif "anom" in q:
        obj["operation"] = "anomaly"
    else:
        obj["operation"] = None

    # Time range detection
    time_range = None

    # Check for specific time range patterns
    if "last month" in q:
        time_range = parse_relative_time("last month")
    elif "last week" in q:
        time_range = parse_relative_time("last week")
    elif "last year" in q:
        time_range = parse_relative_time("last year")
    elif "last" in q and ("day" in q or "days" in q):
        # Extract number of days if specified (e.g., 'last 30 days')
        match = re.search(r"last\s+(\d+)\s+day", q)
        if match:
            num_days = int(match.group(1))
            time_range = parse_relative_time(f"last {num_days} days")
        else:
            time_range = parse_relative_time("last 1 days")

    obj["time_range"] = (
        time_range if time_range and time_range[0] and time_range[1] else None
    )

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
        "What's the nitrate level at 500m depth?",
        "Show me temperature for last 30 days",
        "Salinity trend in last week",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        intent = extract_intent_with_llm(query)
        print(f"➡️ Extracted Intent:")
        for field, value in intent.model_dump().items():
            if value is not None:
                print(f"   • {field}: {value}")
        print("-" * 60)
