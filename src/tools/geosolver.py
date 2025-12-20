"""
GeoSolver - Converts location names to latitude/longitude coordinates
Uses Google Geocoding API (requires API key)
"""

import os
import requests
import time
from typing import Optional, Tuple, Dict
from functools import lru_cache


class GeoSolver:
    """Resolves location names to geographic coordinates"""

    def __init__(self, api_key: str = None):
        """Initialize GeoSolver with Google Geocoding API.

        Args:
            api_key: Google Cloud API key with Geocoding API enabled.
                    If not provided, will try to get from GOOGLE_MAPS_API_KEY environment variable.
        """
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google Maps API key is required. Either pass it to the constructor or set GOOGLE_MAPS_API_KEY environment variable."
            )
        self.headers = {"Accept": "application/json"}

    @lru_cache(maxsize=100)
    def resolve_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Convert a location name to (latitude, longitude) using Google Geocoding API.

        Args:
            location: Location name (e.g., "Mumbai", "Arabian Sea", "New York")

        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        if not location:
            return None

        try:
            print(f"🗺️  Resolving location: '{location}'")

            params = {"address": location, "key": self.api_key}

            response = requests.get(
                self.base_url, params=params, headers=self.headers, timeout=5
            )

            data = response.json()

            if response.status_code == 200 and data["status"] == "OK":
                result = data["results"][0]
                location = result["geometry"]["location"]
                lat = location["lat"]
                lng = location["lng"]
                formatted_address = result.get("formatted_address", location)
                print(f"   ✓ Found: {formatted_address}")
                print(f"   ✓ Coordinates: lat={lat:.4f}, lng={lng:.4f}")
                return (lat, lng)
            else:
                error_status = data.get("status", "UNKNOWN_ERROR")
                error_message = data.get("error_message", "No error details provided")
                print(
                    f"   ✗ Google Geocoding API error: {error_status} - {error_message}"
                )
                return None

        except Exception as e:
            print(f"   ✗ Error resolving location: {str(e)}")
            return None

    def get_marine_location_info(self, location: str) -> Dict:
        """
        Get detailed information about a marine location.
        Returns lat, lng, and additional context.
        """
        coords = self.resolve_location(location)

        if coords:
            lat, lng = coords
            return {
                "location": location,
                "latitude": lat,
                "longitude": lng,
                "found": True,
                "marine_region": self._get_marine_region(lat, lng),
            }
        else:
            return {
                "location": location,
                "latitude": None,
                "longitude": None,
                "found": False,
                "marine_region": None,
            }

    def _get_marine_region(self, lat: float, lng: float) -> str:
        """Identify the general marine region based on coordinates"""
        # Simple region classification
        if 0 <= lat <= 30 and 50 <= lng <= 80:
            return "Arabian Sea"
        elif 0 <= lat <= 30 and 80 <= lng <= 100:
            return "Bay of Bengal"
        elif -40 <= lat <= 10 and 20 <= lng <= 120:
            return "Indian Ocean"
        elif 20 <= lat <= 60 and -100 <= lng <= -50:
            return "North Atlantic Ocean"
        elif -60 <= lat <= 20 and -100 <= lng <= 20:
            return "South Atlantic Ocean"
        elif 0 <= lat <= 60 and 100 <= lng <= 180:
            return "North Pacific Ocean"
        elif -60 <= lat <= 0 and 100 <= lng <= 180:
            return "South Pacific Ocean"
        else:
            return "Unknown marine region"


# Common marine locations cache (for faster lookups)
KNOWN_MARINE_LOCATIONS = {
    "mumbai": (19.0760, 72.8777),
    "arabian sea": (15.0, 65.0),
    "bay of bengal": (15.0, 88.0),
    "indian ocean": (-20.0, 80.0),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "goa": (15.2993, 74.1240),
    "kochi": (9.9312, 76.2673),
    "new york": (40.7128, -74.0060),
    "gulf of mexico": (25.0, -90.0),
    "north atlantic": (40.0, -30.0),
}


def resolve_location_fast(
    location: str, api_key: str = None
) -> Optional[Tuple[float, float]]:
    """
    Quick location lookup with cached common locations.
    Falls back to Google Geocoding API if not in cache.

    Args:
        location: Location name to look up
        api_key: Optional Google Maps API key. If not provided, will try to get from environment.
    """
    if not location:
        return None

    location_lower = location.lower().strip()

    # Check cache first
    if location_lower in KNOWN_MARINE_LOCATIONS:
        lat, lng = KNOWN_MARINE_LOCATIONS[location_lower]
        print(f"🗺️  Location '{location}' → lat={lat:.4f}, lng={lng:.4f} (cached)")
        return (lat, lng)

    # Use GeoSolver for unknown locations
    solver = GeoSolver(api_key=api_key)
    return solver.resolve_location(location)


if __name__ == "__main__":
    """Test the GeoSolver"""
    import os

    print("\n" + "=" * 60)
    print("Testing Google Geocoding API Integration")
    print("=" * 60)

    # Get API key from environment or prompt
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        api_key = input("Enter your Google Maps API key: ").strip()

    if not api_key:
        print("Error: Google Maps API key is required for testing.")
        print(
            "Please set the GOOGLE_MAPS_API_KEY environment variable or enter it when prompted."
        )
        exit(1)

    solver = GeoSolver(api_key=api_key)

    test_locations = [
        "Mumbai",
        "Arabian Sea",
        "New York",
        "Bay of Bengal",
        "Chennai",
        "Gulf of Mexico",
        "Unknown Location XYZ",
    ]

    for location in test_locations:
        print(f"\n📍 Testing: {location}")
        result = solver.get_marine_location_info(location)

        if result["found"]:
            print(f"   ✓ Lat: {result['latitude']:.4f}")
            print(f"   ✓ Lng: {result['longitude']:.4f}")
            print(f"   ✓ Region: {result['marine_region']}")
        else:
            print(f"   ✗ Not found")

        time.sleep(1)  # Be nice to the API

    print("\n" + "=" * 60)
    print("Testing fast lookup with cache:")
    print("=" * 60)

    cached_location = "Mumbai"
    coords = resolve_location_fast(cached_location, api_key=api_key)
    print(f"Result: {coords}")

    print("\nNote: To use this in your application, you can either:")
    print("1. Set the GOOGLE_MAPS_API_KEY environment variable, or")
    print("2. Pass your API key directly to GeoSolver(api_key='your_api_key')")
