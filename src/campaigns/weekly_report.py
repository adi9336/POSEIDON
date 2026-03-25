"""Weekly regional ocean health report across 5 ocean basins."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict
from src.campaigns.base_campaign import BaseCampaign

DEFAULT_REGIONS = [
    "Arabian Sea", "Bay of Bengal",
    "Indian Ocean", "North Pacific Ocean", "North Atlantic Ocean"
]


class WeeklyReportCampaign(BaseCampaign):
    name = "weekly_report"

    def __init__(self, orchestrator, regions=None):
        super().__init__(orchestrator)
        self.regions = regions or DEFAULT_REGIONS

    def execute(self) -> Dict[str, Any]:
        week   = datetime.utcnow().strftime("%Y-W%W")
        report = {"campaign": self.name, "week": week,
                  "generated_at": datetime.utcnow().isoformat(), "regions": {}}

        for region in self.regions:
            result = self.query_orchestrator(
                query=f"Comprehensive analysis of temperature, salinity, "
                      f"and anomalies in {region} over the last 7 days. "
                      f"Compare to recent historical baseline.",
                region=region
            )
            report["regions"][region] = {
                "summary":           result.get("summary"),
                "confidence":        result.get("confidence"),
                "variable_insights": result.get("variable_insights"),
                "validation":        result.get("validation"),
                "trend_context":     result.get("trend_context"),
                "baseline":          result.get("historical_baseline"),
                "n_past_obs":        result.get("n_past_observations", 0),
            }

        return report
