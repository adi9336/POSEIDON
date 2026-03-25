"""
GeoSolver - Marine Polygon Classifier for Geo Resolution

Self-contained geographic resolver for oceanographic locations.
Uses shapely polygons to define ocean basin boundaries and fuzzy string
matching to resolve location names — no external API required.
"""

import logging
from typing import Dict, List, Optional, Tuple

from functools import lru_cache
from shapely.geometry import Point, Polygon
from thefuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marine region polygons — approximate boundaries as (lat, lon) rings.
# Each polygon is defined counter-clockwise.  Coordinates are
# (latitude, longitude) which shapely treats as (y, x).
# ---------------------------------------------------------------------------

MARINE_REGION_COORDS: Dict[str, List[Tuple[float, float]]] = {
    # ── Indian Ocean basins ──
    "Arabian Sea": [
        (5, 44), (25, 44), (25, 65), (25, 77), (20, 77),
        (15, 77), (8, 77), (5, 72), (5, 44),
    ],
    "Bay of Bengal": [
        (5, 77), (8, 77), (15, 77), (20, 77), (22, 88),
        (22, 95), (16, 97), (10, 93), (5, 85), (5, 77),
    ],
    "Andaman Sea": [
        (5, 93), (10, 93), (16, 97), (16, 100), (5, 100), (5, 93),
    ],
    "Indian Ocean": [
        (-40, 20), (-40, 120), (-5, 120), (-5, 100), (5, 100),
        (5, 44), (0, 44), (-5, 40), (-10, 30), (-40, 20),
    ],
    # ── Atlantic Ocean ──
    "North Atlantic Ocean": [
        (0, -80), (0, -15), (30, -15), (50, -10),
        (60, -10), (65, -40), (50, -60), (30, -80), (0, -80),
    ],
    "South Atlantic Ocean": [
        (-60, -70), (-60, 20), (0, 20), (0, -15), (0, -80), (-60, -70),
    ],
    "Gulf of Mexico": [
        (18, -98), (18, -83), (21, -83), (25, -83), (30, -90),
        (30, -98), (26, -98), (18, -98),
    ],
    "Caribbean Sea": [
        (9, -84), (9, -60), (18, -60), (18, -75), (18, -84), (9, -84),
    ],
    "Mediterranean Sea": [
        (30, -6), (30, 36), (38, 36), (42, 30), (46, 16),
        (44, 5), (42, -6), (36, -6), (30, -6),
    ],
    "North Sea": [
        (51, -4), (51, 9), (58, 9), (62, 5), (62, -4), (51, -4),
    ],
    "Baltic Sea": [
        (53, 9), (53, 30), (60, 30), (66, 26), (66, 18),
        (60, 14), (55, 12), (53, 9),
    ],
    "Norwegian Sea": [
        (62, -10), (62, 15), (72, 25), (78, 15), (78, -10), (62, -10),
    ],
    "Black Sea": [
        (41, 27), (41, 42), (44, 42), (47, 40), (47, 30), (44, 27), (41, 27),
    ],
    # ── Pacific Ocean ──
    # NOTE: Regions crossing the antimeridian (±180°) are split into
    # east and west halves so shapely can handle them.  Their centroids
    # are overridden in _CENTROID_OVERRIDES below.
    "North Pacific Ocean (W)": [
        (0, 100), (0, 180), (60, 180), (60, 170), (50, 140), (30, 120), (0, 100),
    ],
    "North Pacific Ocean (E)": [
        (0, -180), (0, -100), (30, -100), (50, -120), (60, -150), (60, -180), (0, -180),
    ],
    "South Pacific Ocean (W)": [
        (-60, 120), (-60, 180), (0, 180), (0, 120), (-60, 120),
    ],
    "South Pacific Ocean (E)": [
        (-60, -180), (-60, -70), (0, -100), (0, -180), (-60, -180),
    ],
    "South China Sea": [
        (0, 100), (0, 120), (10, 120), (18, 120),
        (22, 118), (22, 110), (10, 105), (0, 100),
    ],
    "East China Sea": [
        (22, 118), (22, 130), (30, 130), (34, 124), (28, 118), (22, 118),
    ],
    "Sea of Japan": [
        (34, 127), (34, 140), (44, 140), (52, 142),
        (52, 133), (44, 130), (34, 127),
    ],
    "Bering Sea (W)": [
        (52, 162), (52, 180), (66, 180), (66, 162), (52, 162),
    ],
    "Bering Sea (E)": [
        (52, -180), (52, -160), (66, -160), (66, -180), (52, -180),
    ],
    "Coral Sea": [
        (-25, 143), (-25, 170), (-10, 170), (-10, 143), (-25, 143),
    ],
    "Tasman Sea": [
        (-47, 147), (-47, 170), (-30, 170), (-30, 150), (-34, 147), (-47, 147),
    ],
    # ── Polar ──
    "Southern Ocean": [
        (-90, -180), (-90, 180), (-60, 180), (-60, -180), (-90, -180),
    ],
    "Arctic Ocean": [
        (78, -180), (78, 180), (90, 180), (90, -180), (78, -180),
    ],
    # ── Marginal seas ──
    "Red Sea": [
        (12, 41), (12, 44), (20, 38), (28, 33), (30, 33), (30, 35),
        (28, 35), (20, 40), (12, 44), (12, 41),
    ],
    "Persian Gulf": [
        (24, 48), (24, 56), (27, 57), (30, 50), (30, 48), (24, 48),
    ],
}

# ---------------------------------------------------------------------------
# Coastal / port city coordinates  →  (lat, lon)
# These are used when a user queries by city name; we resolve the city to
# its coordinates, then do a point-in-polygon check to find the nearby
# marine region.
# ---------------------------------------------------------------------------

CITY_COORDS: Dict[str, Tuple[float, float]] = {
    # ── India ──
    "mumbai": (19.076, 72.878),
    "chennai": (13.083, 80.271),
    "kolkata": (22.573, 88.364),
    "kochi": (9.931, 76.267),
    "goa": (15.299, 74.124),
    "visakhapatnam": (17.687, 83.218),
    "mangalore": (12.914, 74.856),
    "thiruvananthapuram": (8.524, 76.936),
    # ── Americas ──
    "new york": (40.713, -74.006),
    "miami": (25.761, -80.192),
    "los angeles": (33.941, -118.408),
    "san francisco": (37.775, -122.419),
    "rio de janeiro": (-22.907, -43.173),
    "buenos aires": (-34.604, -58.382),
    "havana": (23.114, -82.367),
    # ── Europe ──
    "london": (51.507, -0.128),
    "lisbon": (38.722, -9.139),
    "barcelona": (41.386, 2.170),
    "marseille": (43.297, 5.370),
    "istanbul": (41.009, 28.978),
    "oslo": (59.914, 10.752),
    "copenhagen": (55.676, 12.568),
    # ── East Asia ──
    "tokyo": (35.682, 139.692),
    "shanghai": (31.230, 121.474),
    "hong kong": (22.320, 114.169),
    "singapore": (1.352, 103.820),
    "manila": (14.600, 120.984),
    # ── Oceania ──
    "sydney": (-33.869, 151.209),
    "auckland": (-36.849, 174.764),
    # ── Middle East / Africa ──
    "dubai": (25.205, 55.270),
    "jeddah": (21.486, 39.183),
    "cape town": (-33.925, 18.424),
    "dar es salaam": (-6.792, 39.208),
    "mombasa": (-4.043, 39.666),
}

# ---------------------------------------------------------------------------
# Centroid overrides for regions split across the antimeridian.
# The canonical name maps to a manually chosen ocean-centre point.
# ---------------------------------------------------------------------------

_CENTROID_OVERRIDES: Dict[str, Tuple[float, float]] = {
    "North Pacific Ocean": (30.0, -160.0),
    "South Pacific Ocean": (-30.0, -150.0),
    "Bering Sea": (59.0, 180.0),
}

# Maps a canonical display name → list of internal polygon part names.
_SPLIT_REGION_PARTS: Dict[str, List[str]] = {
    "North Pacific Ocean": ["North Pacific Ocean (W)", "North Pacific Ocean (E)"],
    "South Pacific Ocean": ["South Pacific Ocean (W)", "South Pacific Ocean (E)"],
    "Bering Sea": ["Bering Sea (W)", "Bering Sea (E)"],
}

# ---------------------------------------------------------------------------
# Build shapely Polygon objects (cached at module level)
# ---------------------------------------------------------------------------

_REGION_POLYGONS: Dict[str, Polygon] = {}
_REGION_CENTROIDS: Dict[str, Tuple[float, float]] = {}
_CANONICAL_NAMES: List[str] = []  # user-facing region names (no "(W)/(E)" suffixes)


def _build_polygons() -> None:
    """Build shapely polygons from coordinate lists (called once at import)."""
    seen_canonical: set = set()

    for name, coords in MARINE_REGION_COORDS.items():
        # shapely uses (x, y) = (lon, lat)
        ring = [(lon, lat) for lat, lon in coords]
        poly = Polygon(ring)
        _REGION_POLYGONS[name] = poly

        # Determine the canonical (display) name
        canonical = name
        for canon, parts in _SPLIT_REGION_PARTS.items():
            if name in parts:
                canonical = canon
                break

        if canonical not in seen_canonical:
            seen_canonical.add(canonical)
            _CANONICAL_NAMES.append(canonical)

        # Set centroid (use override for split regions, else compute)
        if canonical in _CENTROID_OVERRIDES:
            _REGION_CENTROIDS[canonical] = _CENTROID_OVERRIDES[canonical]
        elif canonical not in _REGION_CENTROIDS:
            centroid = poly.centroid
            _REGION_CENTROIDS[canonical] = (centroid.y, centroid.x)


_build_polygons()


# ---------------------------------------------------------------------------
# Marine Polygon Classifier
# ---------------------------------------------------------------------------

class MarinePolygonClassifier:
    """
    Self-contained marine geographic resolver.

    • Fuzzy-matches location names to ocean basin polygons.
    • Resolves city names via point-in-polygon classification.
    • Returns oceanographically correct centroids — no external API.
    """

    FUZZY_THRESHOLD = 65  # minimum fuzz score to accept a match

    def __init__(self) -> None:
        self._region_names = list(_CANONICAL_NAMES)

    # ── Public API ──

    @lru_cache(maxsize=256)
    def resolve_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Convert a location name to (latitude, longitude).

        Resolution order:
        1. Fuzzy match against marine region names → return centroid.
        2. Exact match against known city names → return city coords.
        3. Return None.
        """
        if not location:
            return None

        location_clean = location.strip()
        location_lower = location_clean.lower()

        # 1) Try fuzzy match against marine regions
        best_region, best_score = self._fuzzy_match_region(location_lower)
        if best_region and best_score >= self.FUZZY_THRESHOLD:
            centroid = _REGION_CENTROIDS[best_region]
            print(
                f"🗺️  Location '{location_clean}' → {best_region} "
                f"(centroid {centroid[0]:.2f}°N, {centroid[1]:.2f}°E) "
                f"[fuzzy score {best_score}]"
            )
            return centroid

        # 2) Try city lookup
        if location_lower in CITY_COORDS:
            lat, lon = CITY_COORDS[location_lower]
            region = self.classify_point(lat, lon)
            print(
                f"🗺️  City '{location_clean}' → lat={lat:.4f}, lon={lon:.4f} "
                f"(region: {region})"
            )
            return (lat, lon)

        # 3) Not found
        print(f"🗺️  ✗ Could not resolve location: '{location_clean}'")
        return None

    def classify_point(self, lat: float, lon: float) -> str:
        """Return the marine region name that contains the given point."""
        pt = Point(lon, lat)  # shapely uses (x, y) = (lon, lat)
        for name, poly in _REGION_POLYGONS.items():
            if poly.contains(pt):
                # Map internal part names back to canonical names
                for canon, parts in _SPLIT_REGION_PARTS.items():
                    if name in parts:
                        return canon
                return name
        return "Unknown marine region"

    def get_marine_location_info(self, location: str) -> Dict:
        """Get detailed information about a marine location."""
        coords = self.resolve_location(location)

        if coords:
            lat, lon = coords
            region = self.classify_point(lat, lon)
            # Get bounding box from the matched polygon
            bbox = None
            if region in _REGION_POLYGONS:
                bounds = _REGION_POLYGONS[region].bounds  # (minx, miny, maxx, maxy)
                bbox = {
                    "lat_min": bounds[1],
                    "lon_min": bounds[0],
                    "lat_max": bounds[3],
                    "lon_max": bounds[2],
                }
            return {
                "location": location,
                "latitude": lat,
                "longitude": lon,
                "found": True,
                "marine_region": region,
                "bounding_box": bbox,
            }
        return {
            "location": location,
            "latitude": None,
            "longitude": None,
            "found": False,
            "marine_region": None,
            "bounding_box": None,
        }

    def get_region_centroid(self, region_name: str) -> Optional[Tuple[float, float]]:
        """Return the centroid of a named marine region (exact match)."""
        return _REGION_CENTROIDS.get(region_name)

    def list_regions(self) -> List[str]:
        """Return a sorted list of all known marine region names."""
        return sorted(self._region_names)

    # ── Internal helpers ──

    def _fuzzy_match_region(self, query: str) -> Tuple[Optional[str], int]:
        """Fuzzy match a query against all region names. Returns (name, score)."""
        best_name: Optional[str] = None
        best_score = 0

        for region_name in self._region_names:
            score = fuzz.token_sort_ratio(query, region_name.lower())
            if score > best_score:
                best_score = score
                best_name = region_name

        return best_name, best_score


# ---------------------------------------------------------------------------
# Backward-compatible wrapper class
# ---------------------------------------------------------------------------

class GeoSolver:
    """
    Backward-compatible wrapper around MarinePolygonClassifier.

    Accepts an optional ``api_key`` parameter for signature compatibility,
    but it is ignored — no external API is used.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._classifier = MarinePolygonClassifier()
        if api_key:
            logger.debug("GeoSolver: api_key provided but ignored (marine polygon classifier is self-contained)")

    @lru_cache(maxsize=100)
    def resolve_location(self, location: str) -> Optional[Tuple[float, float]]:
        return self._classifier.resolve_location(location)

    def get_marine_location_info(self, location: str) -> Dict:
        return self._classifier.get_marine_location_info(location)

    def _get_marine_region(self, lat: float, lng: float) -> str:
        return self._classifier.classify_point(lat, lng)


# ---------------------------------------------------------------------------
# Module-level convenience function (primary public API)
# ---------------------------------------------------------------------------

# Singleton instance
_classifier = MarinePolygonClassifier()


def resolve_location_fast(
    location: str, api_key: Optional[str] = None
) -> Optional[Tuple[float, float]]:
    """
    Quick location lookup using the marine polygon classifier.

    Args:
        location: Location name to look up (e.g. "Arabian Sea", "Mumbai")
        api_key:  Accepted for backward compatibility but ignored.

    Returns:
        Tuple of (latitude, longitude) or None if not found.
    """
    return _classifier.resolve_location(location)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Marine Polygon Classifier — Self Test")
    print("=" * 60)

    classifier = MarinePolygonClassifier()

    print(f"\n📋 Known regions ({len(classifier.list_regions())}):")
    for r in classifier.list_regions():
        centroid = _REGION_CENTROIDS[r]
        print(f"   • {r:30s}  centroid = ({centroid[0]:7.2f}°N, {centroid[1]:7.2f}°E)")

    test_queries = [
        "Arabian Sea",
        "arabian sea",
        "Bay of Bengal",
        "bay of bengal",
        "Indian Ocean",
        "Mumbai",
        "Chennai",
        "Gulf of Mexico",
        "Mediterranean Sea",
        "New York",
        "Tokyo",
        "North Pacific",
        "Southern Ocean",
        "Arctic Ocean",
        "Unknown Location XYZ",
    ]

    print(f"\n📍 Testing {len(test_queries)} queries:")
    print("-" * 60)

    for query in test_queries:
        result = classifier.resolve_location(query)
        if result:
            lat, lon = result
            region = classifier.classify_point(lat, lon)
            print(f"   ✓ '{query}' → ({lat:.2f}, {lon:.2f}) [{region}]")
        else:
            print(f"   ✗ '{query}' → NOT FOUND")

    print("\n" + "=" * 60)
    print("Point-in-polygon classification tests:")
    print("-" * 60)

    test_points = [
        (15.0, 65.0, "Should be Arabian Sea"),
        (15.0, 88.0, "Should be Bay of Bengal"),
        (40.0, -30.0, "Should be North Atlantic"),
        (25.0, -90.0, "Should be Gulf of Mexico"),
        (-80.0, 0.0, "Should be Southern Ocean"),
    ]

    for lat, lon, expected in test_points:
        region = classifier.classify_point(lat, lon)
        print(f"   ({lat:6.1f}, {lon:6.1f}) → {region:30s}  ({expected})")

    print("\n✅ No API key required. Fully self-contained.")
    print("=" * 60)
