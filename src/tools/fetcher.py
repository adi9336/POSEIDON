"""
Argo Data Fetcher
Fetches oceanographic data from Argo floats using direct ERDDAP CSV API.
"""
import datetime
import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Try to import geosolver, provide fallback if not available
try:
    from geosolver import resolve_location_fast
    GEOSOLVER_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: geosolver module not found. Location name resolution disabled.")
    GEOSOLVER_AVAILABLE = False
    
    # Fallback: basic location cache
    KNOWN_LOCATIONS = {
        'mumbai': (19.0760, 72.8777),
        'arabian sea': (15.0, 65.0),
        'bay of bengal': (15.0, 88.0),
        'chennai': (13.0827, 80.2707),
        'goa': (15.2993, 74.1240),
        'new york': (40.7128, -74.0060),
    }
    
    def resolve_location_fast(location: str):
        """Fallback location resolver"""
        if not location:
            return None
        loc_lower = location.lower().strip()
        coords = KNOWN_LOCATIONS.get(loc_lower)
        if coords:
            print(f"   🗺️  '{location}' → lat={coords[0]:.4f}, lon={coords[1]:.4f} (cached)")
        return coords

load_dotenv()


def fetch_argo_data(intent, retry_with_broader_search=True) -> pd.DataFrame:
    """
    Fetch Argo float data for a specific region and time range.
    
    Args:
        intent: An object with attributes: lat, lon, depth, depth_range, 
                time_range, variable, operation, location, context_needed
        retry_with_broader_search: If True, retry with expanded parameters if no data found
        
    Returns:
        pd.DataFrame: DataFrame containing the Argo float data
    """
    
    # Try with original parameters first
    df = _fetch_with_params(intent, radius_degrees=3, days_back=30)
    
    # If no data and retry enabled, try broader search
    if df.empty and retry_with_broader_search:
        print(f"   ℹ️  No data found. Trying broader search (5° radius)...")
        df = _fetch_with_params(intent, radius_degrees=5, days_back=60)
        
        if df.empty:
            print(f"   ℹ️  Still no data. Trying larger area (10° radius, 180 days)...")
            df = _fetch_with_params(intent, radius_degrees=10, days_back=180)
    
    return df


def _fetch_with_params(intent, radius_degrees=3, days_back=30) -> pd.DataFrame:
    """
    Internal function to fetch data with specific parameters using ERDDAP CSV API.
    """
    try:
        # Extract latitude/longitude
        lat = None
        lon = None
        
        # First, check if lat/lon are directly provided
        if hasattr(intent, 'lat') and intent.lat is not None and \
           hasattr(intent, 'lon') and intent.lon is not None:
            lat = intent.lat
            lon = intent.lon
            print(f"   📍 Using provided coordinates: ({lat}, {lon})")
        
        # If not, try to resolve from location name
        elif hasattr(intent, 'location') and intent.location:
            print(f"   🗺️  No coordinates provided. Resolving location: '{intent.location}'")
            coords = resolve_location_fast(intent.location)
            if coords:
                lat, lon = coords
                print(f"   ✅ Resolved to: lat={lat:.4f}, lon={lon:.4f}")
            else:
                print(f"   ❌ Could not resolve location: '{intent.location}'")
                return pd.DataFrame()
        
        # If still no coordinates, cannot proceed
        if lat is None or lon is None:
            print(f"   ⚠️  No location specified (no lat/lon or location name)")
            return pd.DataFrame()
        
        # Calculate search area
        lat_min = max(lat - radius_degrees, -90)
        lat_max = min(lat + radius_degrees, 90)
        lon_min = max(lon - radius_degrees, -180)
        lon_max = min(lon + radius_degrees, 180)
        print(f"   📏 Search radius: ±{radius_degrees}°")
        print(f"   📐 Range: lat [{lat_min:.1f}, {lat_max:.1f}], lon [{lon_min:.1f}, {lon_max:.1f}]")

        # Extract time range
        if hasattr(intent, 'time_range') and intent.time_range is not None and \
           intent.time_range[0] is not None and intent.time_range[1] is not None:
            start, end = intent.time_range
            start_date = start if isinstance(start, str) else start.strftime("%Y-%m-%d")
            end_date = end if isinstance(end, str) else end.strftime("%Y-%m-%d")
        else:
            end_date_dt = datetime.datetime.utcnow()
            start_date_dt = end_date_dt - datetime.timedelta(days=days_back)
            start_date = start_date_dt.strftime("%Y-%m-%d")
            end_date = end_date_dt.strftime("%Y-%m-%d")
        
        start_time = f"{start_date}T00:00:00Z"
        end_time = f"{end_date}T23:59:59Z"
        print(f"   📅 Time: {start_date} to {end_date}")
        
        # Determine which variables to request
        variables = ["platform_number", "latitude", "longitude", "time", "pres", "temp", "psal"]
        
        # Add variable-specific fields if specified
        if hasattr(intent, 'variable') and intent.variable:
            var = intent.variable.lower()
            if var == 'nitrate' and 'nitrate' not in variables:
                variables.append("nitrate")
            elif var == 'oxygen' and 'doxy' not in variables:
                variables.append("doxy")
        
        # Build ERDDAP URL
        base_url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
        variables_str = ",".join(variables)
        
        url = (
            f"{base_url}?"
            f"{variables_str}"
            f"&latitude>={lat_min}&latitude<={lat_max}"
            f"&longitude>={lon_min}&longitude<={lon_max}"
            f"&time>={start_time}&time<={end_time}"
        )
        
        # Add depth constraint if specified
        if hasattr(intent, 'depth') and intent.depth is not None:
            url += f"&pres<={intent.depth}"
            print(f"   🌊 Depth: 0 to {intent.depth}m")
        elif hasattr(intent, 'depth_range') and intent.depth_range is not None:
            depth_min, depth_max = intent.depth_range
            url += f"&pres>={depth_min}&pres<={depth_max}"
            print(f"   🌊 Depth: {depth_min}m to {depth_max}m")
        
        print(f"   🔍 Querying ERDDAP...")
        print(f"   🔗 URL: {url[:100]}..." if len(url) > 100 else f"   🔗 URL: {url}")
        
        # Fetch data
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), skiprows=[1])  # Skip units row
            
            if df.empty or len(df) == 0:
                print(f"   ⚠️  No data found for these parameters")
                return pd.DataFrame()
            
            # Rename columns to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Convert time to datetime
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            
            print(f"   ✅ Retrieved {len(df)} data points")
            if 'platform_number' in df.columns:
                print(f"   🎈 From {df['platform_number'].nunique()} unique floats")
            
            # Save to CSV
            output_dir = "data"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"argo_data_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_file)
            
            df.to_csv(output_path, index=False)
            print(f"   💾 Saved to: {output_path}")
            
            return df
            
        elif response.status_code == 404:
            print(f"   ⚠️  No data found (404)")
            return pd.DataFrame()
        else:
            print(f"   ❌ Error: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return pd.DataFrame()
        
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


if __name__ == "__main__":
    """Test the fetcher independently"""
    print("\n🧪 Testing Argo Data Fetcher\n")
    print("="*60)
    
    # Test case 1: Arabian Sea (sparse coverage)
    class TestIntent1:
        lat = 20.0
        lon = 70.0
        depth = 1000
        depth_range = None
        time_range = None
        variable = "nitrate"
        operation = None
        location = "Arabian Sea"
        context_needed = None
    
    print("\n📍 TEST 1: Arabian Sea")
    print("-" * 60)
    test1 = TestIntent1()
    df1 = fetch_argo_data(test1)
    
    if not df1.empty:
        print(f"\n✅ Success! Found {len(df1)} records")
        print(f"\nColumns: {', '.join(df1.columns)}")
        print(f"\nFirst 3 rows:")
        print(df1.head(3))
    
    # Test case 2: North Pacific (better coverage)
    print("\n" + "="*60)
    print("\n📍 TEST 2: North Pacific (known good coverage)")
    print("-" * 60)
    
    class TestIntent2:
        lat = 30.0
        lon = -140.0
        depth = 500
        depth_range = None
        time_range = ("2024-10-01", "2024-10-31")
        variable = "temp"
        operation = None
        location = "North Pacific"
        context_needed = None
    
    test2 = TestIntent2()
    df2 = fetch_argo_data(test2, retry_with_broader_search=False)
    
    if not df2.empty:
        print(f"\n✅ Success! Found {len(df2)} records")
        if 'temp' in df2.columns:
            print(f"Temperature range: {df2['temp'].min():.2f}°C to {df2['temp'].max():.2f}°C")
    else:
        print("\n❌ No data found")
    
    print("\n" + "="*60)
    print("\n💡 Tips:")
    print("  • Argo floats are sparse in some regions (like Arabian Sea)")
    print("  • Pacific and Atlantic oceans have better coverage")
    print("  • Try broader time ranges if no data found")
    print("  • Check https://www.ocean-ops.org/board for float locations")