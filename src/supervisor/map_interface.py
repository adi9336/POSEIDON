"""
Map interface for the Supervisor Agent.
Generates Folium maps for geographic confirmation in Streamlit.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Optional

from src.core.paths import ARGO_DB_PATH

logger = logging.getLogger(__name__)


class MapInterface:
    """
    Provides map-related utilities for the Supervisor:
    - Generate confirmation map data (lat, lon, radius)
    - Check ARGO float coverage in a given area
    """

    def generate_map_data(
        self,
        lat: float,
        lon: float,
        radius_km: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Create map confirmation data for the UI layer.
        The UI (Streamlit) will render this using Folium.
        """
        float_count = self.check_argo_coverage(lat, lon, radius_km)
        radius_deg = radius_km / 111.0  # approximate conversion

        return {
            "center_lat": lat,
            "center_lon": lon,
            "radius_km": radius_km,
            "radius_deg": radius_deg,
            "argo_float_count": float_count,
            "bounds": {
                "min_lat": lat - radius_deg,
                "max_lat": lat + radius_deg,
                "min_lon": lon - radius_deg,
                "max_lon": lon + radius_deg,
            },
        }

    def check_argo_coverage(
        self,
        lat: float,
        lon: float,
        radius_km: float = 100.0,
    ) -> int:
        """
        Query the SQLite database for ARGO float observations
        within <radius_km> of the given coordinates.
        Returns the count of distinct platform_numbers.
        """
        radius_deg = radius_km / 111.0
        try:
            with sqlite3.connect(ARGO_DB_PATH) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT platform_number)
                    FROM argo_data
                    WHERE latitude BETWEEN ? AND ?
                      AND longitude BETWEEN ? AND ?
                    """,
                    (
                        lat - radius_deg,
                        lat + radius_deg,
                        lon - radius_deg,
                        lon + radius_deg,
                    ),
                ).fetchone()
                return row[0] if row else 0
        except Exception as exc:
            logger.warning(f"Could not check ARGO coverage: {exc}")
            return 0

    def get_nearby_floats(
        self,
        lat: float,
        lon: float,
        radius_km: float = 100.0,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        """
        Get float positions near the given coordinates for map markers.
        """
        radius_deg = radius_km / 111.0
        try:
            with sqlite3.connect(ARGO_DB_PATH) as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT platform_number, latitude, longitude,
                           MAX(time) as last_seen
                    FROM argo_data
                    WHERE latitude BETWEEN ? AND ?
                      AND longitude BETWEEN ? AND ?
                    GROUP BY platform_number
                    ORDER BY last_seen DESC
                    LIMIT ?
                    """,
                    (
                        lat - radius_deg,
                        lat + radius_deg,
                        lon - radius_deg,
                        lon + radius_deg,
                        limit,
                    ),
                ).fetchall()
                return [
                    {
                        "platform_number": r[0],
                        "latitude": r[1],
                        "longitude": r[2],
                        "last_seen": r[3],
                    }
                    for r in rows
                ]
        except Exception as exc:
            logger.warning(f"Could not fetch nearby floats: {exc}")
            return []
