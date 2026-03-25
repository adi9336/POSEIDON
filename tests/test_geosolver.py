"""
Tests for the Marine Polygon Classifier (geosolver.py).

Verifies that the self-contained marine polygon classifier correctly
resolves ocean basin names, city names, and performs point-in-polygon
classification — all without an external API.
"""

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    from src.tools.geosolver import MarinePolygonClassifier
    return MarinePolygonClassifier()


# ---------------------------------------------------------------------------
# Centroid accuracy
# ---------------------------------------------------------------------------

class TestCentroidAccuracy:
    """Verify that centroids for major basins are oceanographically sensible."""

    @pytest.mark.parametrize("region, lat_range, lon_range", [
        ("Arabian Sea",         (10, 20),   (55, 70)),
        ("Bay of Bengal",       (10, 20),   (80, 95)),
        ("Indian Ocean",        (-30, -10), (60, 90)),
        ("Gulf of Mexico",      (20, 28),   (-95, -85)),
        ("North Atlantic Ocean",(20, 40),   (-55, -30)),
        ("Mediterranean Sea",   (32, 42),   (8, 22)),
        ("North Pacific Ocean", (25, 35),   (-170, -150)),
        ("South Pacific Ocean", (-35, -25), (-160, -140)),
        ("Southern Ocean",      (-80, -70), (-10, 10)),
        ("Arctic Ocean",        (80, 90),   (-10, 10)),
    ])
    def test_centroid_in_expected_range(self, classifier, region, lat_range, lon_range):
        centroid = classifier.get_region_centroid(region)
        assert centroid is not None, f"No centroid found for '{region}'"
        lat, lon = centroid
        assert lat_range[0] <= lat <= lat_range[1], (
            f"{region} centroid lat {lat} not in {lat_range}"
        )
        assert lon_range[0] <= lon <= lon_range[1], (
            f"{region} centroid lon {lon} not in {lon_range}"
        )


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

class TestFuzzyMatching:
    """Verify case-insensitive and slightly misspelled names resolve correctly."""

    @pytest.mark.parametrize("query", [
        "Arabian Sea",
        "arabian sea",
        "ARABIAN SEA",
        "Arabian  sea",
    ])
    def test_arabian_sea_variants(self, classifier, query):
        result = classifier.resolve_location(query)
        assert result is not None

    @pytest.mark.parametrize("query", [
        "Bay of Bengal",
        "bay of bengal",
        "BAY OF BENGAL",
    ])
    def test_bay_of_bengal_variants(self, classifier, query):
        result = classifier.resolve_location(query)
        assert result is not None

    def test_partial_name_north_pacific(self, classifier):
        """'North Pacific' (without 'Ocean') should still match."""
        result = classifier.resolve_location("North Pacific")
        assert result is not None


# ---------------------------------------------------------------------------
# City → region resolution
# ---------------------------------------------------------------------------

class TestCityResolution:
    """Verify that city names resolve to coordinates in the correct ocean region."""

    @pytest.mark.parametrize("city, expected_region", [
        ("Mumbai",   "Arabian Sea"),
        ("Chennai",  "Bay of Bengal"),
        ("Kochi",    "Arabian Sea"),
        ("Goa",      "Arabian Sea"),
    ])
    def test_indian_cities(self, classifier, city, expected_region):
        result = classifier.resolve_location(city)
        assert result is not None, f"Could not resolve '{city}'"
        lat, lon = result
        region = classifier.classify_point(lat, lon)
        assert region == expected_region, (
            f"'{city}' classified as '{region}', expected '{expected_region}'"
        )


# ---------------------------------------------------------------------------
# Point-in-polygon classification
# ---------------------------------------------------------------------------

class TestPointClassification:

    @pytest.mark.parametrize("lat, lon, expected", [
        (15.0,  65.0,  "Arabian Sea"),
        (15.0,  88.0,  "Bay of Bengal"),
        (40.0, -30.0,  "North Atlantic Ocean"),
        (25.0, -90.0,  "Gulf of Mexico"),
        (-80.0,  0.0,  "Southern Ocean"),
        (38.0,  15.0,  "Mediterranean Sea"),
    ])
    def test_point_in_polygon(self, classifier, lat, lon, expected):
        region = classifier.classify_point(lat, lon)
        assert region == expected, f"({lat}, {lon}) classified as '{region}', expected '{expected}'"


# ---------------------------------------------------------------------------
# Negative / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_unknown_location_returns_none(self, classifier):
        assert classifier.resolve_location("Narnia") is None

    def test_empty_string_returns_none(self, classifier):
        assert classifier.resolve_location("") is None

    def test_none_returns_none(self, classifier):
        assert classifier.resolve_location(None) is None


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_geosolver_class_no_api_key(self):
        """GeoSolver() should work without an API key."""
        from src.tools.geosolver import GeoSolver
        solver = GeoSolver()
        result = solver.resolve_location("Arabian Sea")
        assert result is not None

    def test_geosolver_class_with_api_key(self):
        """GeoSolver(api_key='fake') should silently ignore the key."""
        from src.tools.geosolver import GeoSolver
        solver = GeoSolver(api_key="fake-key-ignored")
        result = solver.resolve_location("Bay of Bengal")
        assert result is not None

    def test_resolve_location_fast(self):
        """Module-level resolve_location_fast should work."""
        from src.tools.geosolver import resolve_location_fast
        result = resolve_location_fast("Gulf of Mexico")
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_resolve_location_fast_with_api_key(self):
        """api_key param accepted but ignored."""
        from src.tools.geosolver import resolve_location_fast
        result = resolve_location_fast("Mediterranean Sea", api_key="ignored")
        assert result is not None


# ---------------------------------------------------------------------------
# Region listing
# ---------------------------------------------------------------------------

class TestRegionListing:

    def test_list_regions_returns_nonempty(self, classifier):
        regions = classifier.list_regions()
        assert len(regions) >= 20

    def test_list_regions_sorted(self, classifier):
        regions = classifier.list_regions()
        assert regions == sorted(regions)

    def test_key_regions_present(self, classifier):
        regions = classifier.list_regions()
        for expected in [
            "Arabian Sea", "Bay of Bengal", "Indian Ocean",
            "North Pacific Ocean", "South Pacific Ocean",
            "Gulf of Mexico", "Mediterranean Sea",
        ]:
            assert expected in regions, f"'{expected}' missing from region list"
