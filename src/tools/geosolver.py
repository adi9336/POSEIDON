"""
GeoSolver - Converts location names to latitude/longitude coordinates
Uses OpenStreetMap's Nominatim API (free, no API key required)
"""
import requests
import time
from typing import Optional, Tuple, Dict
from functools import lru_cache


class GeoSolver:
    """Resolves location names to geographic coordinates"""
    
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {
            'User-Agent': 'ArgoFloatApp/1.0'  # Required by Nominatim
        }
        
    @lru_cache(maxsize=100)
    def resolve_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Convert a location name to (latitude, longitude).
        
        Args:
            location: Location name (e.g., "Mumbai", "Arabian Sea", "New York")
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        if not location:
            return None
            
        try:
            print(f"🗺️  Resolving location: '{location}'")
            
            params = {
                'q': location,
                'format': 'json',
                'limit': 1
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    display_name = data[0].get('display_name', location)
                    print(f"   ✓ Found: {display_name}")
                    print(f"   ✓ Coordinates: lat={lat:.4f}, lon={lon:.4f}")
                    return (lat, lon)
                else:
                    print(f"   ⚠ Location not found")
                    return None
            else:
                print(f"   ✗ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ✗ Error resolving location: {str(e)}")
            return None
    
    def get_marine_location_info(self, location: str) -> Dict:
        """
        Get detailed information about a marine location.
        Returns lat, lon, and additional context.
        """
        coords = self.resolve_location(location)
        
        if coords:
            lat, lon = coords
            return {
                'location': location,
                'latitude': lat,
                'longitude': lon,
                'found': True,
                'marine_region': self._get_marine_region(lat, lon)
            }
        else:
            return {
                'location': location,
                'latitude': None,
                'longitude': None,
                'found': False,
                'marine_region': None
            }
    
    def _get_marine_region(self, lat: float, lon: float) -> str:
        """Identify the general marine region based on coordinates"""
        # Simple region classification
        if 0 <= lat <= 30 and 50 <= lon <= 80:
            return "Arabian Sea"
        elif 0 <= lat <= 30 and 80 <= lon <= 100:
            return "Bay of Bengal"
        elif -40 <= lat <= 10 and 20 <= lon <= 120:
            return "Indian Ocean"
        elif 20 <= lat <= 60 and -100 <= lon <= -50:
            return "North Atlantic Ocean"
        elif -60 <= lat <= 20 and -100 <= lon <= 20:
            return "South Atlantic Ocean"
        elif 0 <= lat <= 60 and 100 <= lon <= 180:
            return "North Pacific Ocean"
        elif -60 <= lat <= 0 and 100 <= lon <= 180:
            return "South Pacific Ocean"
        else:
            return "Unknown marine region"


# Common marine locations cache (for faster lookups)
KNOWN_MARINE_LOCATIONS = {
    'mumbai': (19.0760, 72.8777),
    'arabian sea': (15.0, 65.0),
    'bay of bengal': (15.0, 88.0),
    'indian ocean': (-20.0, 80.0),
    'chennai': (13.0827, 80.2707),
    'kolkata': (22.5726, 88.3639),
    'goa': (15.2993, 74.1240),
    'kochi': (9.9312, 76.2673),
    'new york': (40.7128, -74.0060),
    'gulf of mexico': (25.0, -90.0),
    'north atlantic': (40.0, -30.0),
}


def resolve_location_fast(location: str) -> Optional[Tuple[float, float]]:
    """
    Quick location lookup with cached common locations.
    Falls back to API if not in cache.
    """
    if not location:
        return None
    
    location_lower = location.lower().strip()
    
    # Check cache first
    if location_lower in KNOWN_MARINE_LOCATIONS:
        lat, lon = KNOWN_MARINE_LOCATIONS[location_lower]
        print(f"🗺️  Location '{location}' → lat={lat:.4f}, lon={lon:.4f} (cached)")
        return (lat, lon)
    
    # Use GeoSolver for unknown locations
    solver = GeoSolver()
    return solver.resolve_location(location)


if __name__ == "__main__":
    """Test the GeoSolver"""
    print("\n" + "="*60)
    print("Testing GeoSolver")
    print("="*60)
    
    solver = GeoSolver()
    
    test_locations = [
        "Mumbai",
        "Arabian Sea",
        "New York",
        "Bay of Bengal",
        "Chennai",
        "Gulf of Mexico",
        "Unknown Location XYZ"
    ]
    
    for location in test_locations:
        print(f"\n📍 Testing: {location}")
        result = solver.get_marine_location_info(location)
        
        if result['found']:
            print(f"   ✓ Lat: {result['latitude']:.4f}")
            print(f"   ✓ Lon: {result['longitude']:.4f}")
            print(f"   ✓ Region: {result['marine_region']}")
        else:
            print(f"   ✗ Not found")
        
        time.sleep(1)  # Be nice to the API
    
    print("\n" + "="*60)
    print("Testing fast lookup with cache:")
    print("="*60)
    
    cached_location = "Mumbai"
    coords = resolve_location_fast(cached_location)
    print(f"Result: {coords}")