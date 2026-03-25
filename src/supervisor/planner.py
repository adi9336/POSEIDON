"""
Workflow Planner for the Supervisor Agent.
Produces approval-ready execution plans with estimated time, cost, and steps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.supervisor.map_interface import MapInterface
from src.supervisor.state import WorkflowPlan

logger = logging.getLogger(__name__)


class WorkflowPlanner:
    """
    Generates a human-readable execution plan from a complete set of parameters.
    The plan is shown to the user for approval before execution begins.
    """

    def __init__(self) -> None:
        self.map_interface = MapInterface()

    def build_plan(
        self,
        intent: Dict[str, Any],
        coordinates: Optional[Dict[str, float]] = None,
    ) -> WorkflowPlan:
        """
        Build a WorkflowPlan from a resolved intent + confirmed coordinates.
        """
        variable = intent.get("variable", "temperature")
        location = intent.get("location", "specified region")
        depth = intent.get("depth")
        operation = intent.get("operation", "summary")
        time_days = intent.get("time_range_days", 30)

        # Get float count if we have coordinates
        float_count = 0
        radius_km = 100.0
        if coordinates:
            lat = coordinates.get("lat", 0.0)
            lon = coordinates.get("lon", 0.0)
            radius_km = coordinates.get("radius_km", 100.0)
            float_count = self.map_interface.check_argo_coverage(lat, lon, radius_km)

        # Build descriptive steps
        steps = self._build_steps(
            variable=variable,
            location=location,
            depth=depth,
            operation=operation,
            float_count=float_count,
            radius_km=radius_km,
        )

        # Estimate time and cost
        est_time = self._estimate_time(float_count, operation)
        est_cost = self._estimate_cost(float_count, operation)
        data_vol = self._estimate_data_volume(float_count, time_days)

        return WorkflowPlan(
            name=f"{variable.upper()} Analysis — {location}",
            steps=steps,
            estimated_time=est_time,
            estimated_cost=est_cost,
            data_volume=data_vol,
            argo_float_count=float_count,
        )

    def _build_steps(
        self,
        variable: str,
        location: str,
        depth: Optional[float],
        operation: str,
        float_count: int,
        radius_km: float,
    ) -> list[str]:
        depth_str = f" at {depth}m depth" if depth else ""
        steps = [
            f"1. Search for ARGO floats near {location} ({radius_km}km radius)",
            f"2. Retrieve {variable} data{depth_str} from {float_count or '~'} floats",
        ]

        if operation in ("summary", "trend", "comparison"):
            steps.append("3. Compute statistical analysis (mean, std, trends)")
        if operation in ("trend", "comparison"):
            steps.append("4. Generate trend visualizations")
        steps.append(f"{len(steps) + 1}. Validate data quality and confidence scores")
        steps.append(f"{len(steps) + 1}. Return results with interactive charts")

        return steps

    def _estimate_time(self, float_count: int, operation: str) -> str:
        base = 15  # seconds
        if float_count > 20:
            base += 20
        if operation in ("trend", "comparison"):
            base += 15
        return f"{base}-{base + 20} seconds"

    def _estimate_cost(self, float_count: int, operation: str) -> str:
        base = 0.03  # base API cost
        if operation in ("trend", "comparison"):
            base += 0.05
        if float_count > 10:
            base += 0.02
        return f"${base:.2f} (API calls)"

    def _estimate_data_volume(self, float_count: int, time_days: int) -> str:
        rows_est = max(float_count * time_days * 2, 100)
        if rows_est < 1000:
            return f"~{rows_est} rows (~{rows_est * 50 // 1024} KB)"
        return f"~{rows_est} rows (~{rows_est * 50 // (1024 * 1024):.1f} MB)"
