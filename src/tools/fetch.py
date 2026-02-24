"""
Argo Data Fetcher
Fetches oceanographic data from Argo floats using direct ERDDAP CSV API.
"""

import datetime
import os
from typing import Optional
import pandas as pd
import requests
from dotenv import load_dotenv
from io import StringIO

try:
    from src.state.models import FloatChatState
except ImportError:
    # Create a mock FloatChatState class if the module is not available
    class FloatChatState:
        def __init__(self):
            self.raw_data = None

# Import geosolver for location resolution
try:
    from .geosolver import resolve_location_fast
    GEOSOLVER_AVAILABLE = True
except ImportError:
    GEOSOLVER_AVAILABLE = False
    def resolve_location_fast(location: str) -> tuple[float, float] | None:
        print(f"⚠️  Geosolver not available. Cannot resolve location: {location}")
        return None

load_dotenv()


def get_default_time_range(days_back: int = 30) -> tuple[str, str]:
    """Get default time range based on days back from current date."""
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def fetch_argo_data(
    intent, retry_with_broader_search: bool = True, state: Optional[FloatChatState] = None
) -> pd.DataFrame:
    """
    Fetch Argo float data for a specific region and time range.

    Args:
        intent: An object with attributes: lat, lon, depth, depth_range,
                time_range, variable, operation, location, context_needed
        retry_with_broader_search: If True, retry with expanded parameters if no data found
        state: Optional FloatChatState to store the output path in raw_data

    Returns:
        pd.DataFrame: DataFrame containing the Argo float data, or empty DataFrame if no time range is specified
    """
    # Check if time range is provided
    if not hasattr(intent, "time_range") or not intent.time_range:
        print("\n⚠️  No time range specified. Using default of last 30 days.")
        print(
            "   You can specify a time range like: 'last month', 'last 7 days', or 'from 2024-01-01 to 2024-01-31'\n"
        )
        # Set a default time range
        intent.time_range = get_default_time_range()

    # Try with original parameters first
    df = _fetch_with_params(intent, radius_degrees=3, days_back=30, state=state)

    # If no data and retry enabled, try broader search
    if df.empty and retry_with_broader_search:
        print(f"   ℹ️  No data found. Trying broader search (5° radius)...")
        df = _fetch_with_params(intent, radius_degrees=5, days_back=60, state=state)

        if df.empty:
            print(f"   ℹ️  Still no data. Trying larger area (10° radius, 180 days)...")
            df = _fetch_with_params(
                intent, radius_degrees=10, days_back=180, state=state
            )

    return df


def _fetch_with_params(
    intent,
    radius_degrees: float = 3,
    days_back: int = 30,
    state: Optional[FloatChatState] = None,
    persist: bool = False,
) -> pd.DataFrame:
    """
    Core stateless ERDDAP fetcher.
    
    Args:
        intent: Object containing query parameters
        radius_degrees: Search radius in degrees
        days_back: Number of days to look back if no time range is provided
        state: Optional FloatChatState for storing results
        persist: Whether to save results to disk
        
    Returns:
        pd.DataFrame: Fetched data or empty DataFrame if no data found
    """

    try:
        lat, lon = None, None

        if hasattr(intent, "lat") and intent.lat is not None:
            lat, lon = intent.lat, intent.lon
        elif hasattr(intent, "location") and intent.location:
            coords = resolve_location_fast(intent.location)
            if coords:
                lat, lon = coords
            else:
                return pd.DataFrame()

        if lat is None or lon is None:
            return pd.DataFrame()

        lat_min = max(lat - radius_degrees, -90)
        lat_max = min(lat + radius_degrees, 90)
        lon_min = max(lon - radius_degrees, -180)
        lon_max = min(lon + radius_degrees, 180)

        if not intent.time_range:
            start_date, end_date = get_default_time_range(days_back)
        else:
            start, end = intent.time_range
            start_date = str(start)
            end_date = str(end)

        start_time = f"{start_date}T00:00:00Z"
        end_time = f"{end_date}T23:59:59Z"

        variables = [
            "platform_number",
            "latitude",
            "longitude",
            "time",
            "pres",
            "temp",
            "psal",
        ]

        base_url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
        variables_str = ",".join(variables)

        url = (
            f"{base_url}?"
            f"{variables_str}"
            f"&latitude>={lat_min}&latitude<={lat_max}"
            f"&longitude>={lon_min}&longitude<={lon_max}"
            f"&time>={start_time}&time<={end_time}"
        )

        if hasattr(intent, "depth") and intent.depth:
            url += f"&pres<={intent.depth}"

        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            return pd.DataFrame()

        df = pd.read_csv(StringIO(response.text), skiprows=[1])
        if df.empty:
            return pd.DataFrame()

        df.columns = df.columns.str.lower()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        # ---------------------------
        # OPTIONAL PERSISTENCE BLOCK
        # ---------------------------
        if persist:
            output_dir = "data"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"argo_data_{timestamp}.csv")
            df.to_csv(output_path, index=False)

            if state is not None:
                state.raw_data = output_path

        return df

    except Exception as e:
        return pd.DataFrame()


def print_test_results(df: pd.DataFrame, test_name: str) -> None:
    """Print test results in a formatted way."""
    print(f"\n📋 {test_name}")
    if df is None or df.empty:
        print("❌ No data found")
        return
    
    print(f"✅ Data found: {len(df)} rows")
    print(f"📅 Time range: {df['time'].min()} to {df['time'].max()}" if 'time' in df.columns else "⏰ No time data")
    print(f"📍 Location: {df['latitude'].min():.2f}°N to {df['latitude'].max():.2f}°N, "
          f"{df['longitude'].min():.2f}°E to {df['longitude'].max():.2f}°E" 
          if 'latitude' in df.columns and 'longitude' in df.columns else "🌍 No location data")
    print(f"📊 Data columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    """Test the fetcher with various scenarios"""
    print("\n🧪 Testing Argo Data Fetcher\n")
    print("=" * 80)
    
    # Check for required packages
    try:
        import pandas as pd
        import requests
    except ImportError as e:
        print(f"❌ Error: Required package not found: {e}")
        print("Please install required packages with: pip install pandas requests")
        exit(1)
    
    class TestIntent:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Test 1: Basic test with coordinates
    print("\n📍 TEST 1: Basic test with coordinates")
    print("-" * 80)
    try:
        test1 = TestIntent(
            lat=20.0,
            lon=70.0,
            depth=1000,
            time_range=("2024-12-01", "2024-12-15"),
            location="Arabian Sea"
        )
        df1 = fetch_argo_data(test1)
        print_test_results(df1, "Basic coordinates test")
    except Exception as e:
        print(f"❌ Test 1 failed: {str(e)}")

    # Test 2: Location name resolution
    print("\n\n📍 TEST 2: Location name resolution")
    print("-" * 80)
    try:
        test2 = TestIntent(
            location="Bay of Bengal",
            depth=500,
            time_range=("2024-11-01", "2024-12-15")
        )
        df2 = fetch_argo_data(test2)
        print_test_results(df2, "Location name resolution")
    except Exception as e:
        print(f"❌ Test 2 failed: {str(e)}")
        df2 = pd.DataFrame()

    # Test 3: No time range (should use default)
    print("\n\n📍 TEST 3: No time range (using default)")
    print("-" * 80)
    try:
        test3 = TestIntent(
            lat=35.0,
            lon=-120.0,
            depth=2000,
            location="California Current"
        )
        df3 = fetch_argo_data(test3)
        print_test_results(df3, "Default time range")
    except Exception as e:
        print(f"❌ Test 3 failed: {str(e)}")
        df3 = pd.DataFrame()

    # Test 4: Shallow depth test
    print("\n\n📍 TEST 4: Shallow depth test")
    print("-" * 80)
    try:
        test4 = TestIntent(
            lat=-30.0,
            lon=150.0,
            depth=200,
            time_range=("2024-11-15", "2024-12-15"),
            location="Tasman Sea"
        )
        df4 = fetch_argo_data(test4)
        print_test_results(df4, "Shallow depth test")
    except Exception as e:
        print(f"❌ Test 4 failed: {str(e)}")
        df4 = pd.DataFrame()

    # Test 5: Large time range
    print("\n\n📍 TEST 5: Large time range")
    print("-" * 80)
    try:
        test5 = TestIntent(
            lat=0.0,
            lon=-170.0,
            depth=1000,
            time_range=("2024-01-01", "2024-12-31"),
            location="Equatorial Pacific"
        )
        df5 = fetch_argo_data(test5)
        print_test_results(df5, "Large time range")
    except Exception as e:
        print(f"❌ Test 5 failed: {str(e)}")
        df5 = pd.DataFrame()

    print("\n" + "=" * 80)
    print("\n📊 Test Summary")
    print("-" * 80)
    print(f"✅ Tests completed: 5")
    print(f"📊 Data found in: {sum(1 for df in [df1, df2, df3, df4, df5] if not df.empty)}/5 tests")
    print("\n💡 Tips:")
    print("  • Check the output above for any warnings or errors")
    print("  • Some regions may have sparse Argo float coverage")
    print("  • Try adjusting time ranges or locations if no data is found")
    print("  • For more details, visit: https://www.ocean-ops.org/board")
    print("=" * 80)