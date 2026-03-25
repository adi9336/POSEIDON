"""Daily anomaly watch — scans key regions, alerts on high anomaly rate."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List
from src.campaigns.base_campaign import BaseCampaign

DEFAULT_REGIONS    = ["Arabian Sea", "Bay of Bengal", "Indian Ocean"]
ALERT_THRESHOLD    = 0.10


class AnomalyWatchCampaign(BaseCampaign):
    name = "anomaly_watch"

    def __init__(self, orchestrator, regions=None, threshold=ALERT_THRESHOLD):
        super().__init__(orchestrator)
        self.regions   = regions or DEFAULT_REGIONS
        self.threshold = threshold

    def execute(self) -> Dict[str, Any]:
        report = {"campaign": self.name, "generated_at": datetime.utcnow().isoformat(), "regions": {}}

        for region in self.regions:
            self.log.info(f"Scanning {region}...")
            result = self.query_orchestrator(
                query=f"Check for anomalous temperature or salinity profiles "
                      f"in {region} over the last 10 days. Flag any unusual readings.",
                region=region
            )

            anomaly_rate = result.get("validation", {}).get("metrics", {}).get("temp_outlier_rate", 0.0) or 0.0
            last_known   = result.get("last_anomaly")

            # Deduplicate: only alert if this is a NEW anomaly
            is_new_anomaly = anomaly_rate > self.threshold
            if is_new_anomaly and last_known:
                try:
                    last_dt = datetime.fromisoformat(last_known["recorded_at"])
                    if (datetime.utcnow() - last_dt).total_seconds() < 86400:
                        is_new_anomaly = False  # same anomaly from today, skip
                except Exception:
                    pass

            if is_new_anomaly:
                self._log_alert(
                    f"Anomaly detected in {region}: rate={anomaly_rate:.1%}",
                    {"region": region, "anomaly_rate": anomaly_rate, "summary": result.get("summary")}
                )

            report["regions"][region] = {
                "anomaly_rate":   anomaly_rate,
                "alert_fired":    is_new_anomaly,
                "summary":        result.get("summary"),
                "confidence":     result.get("confidence"),
                "history_used":   result.get("history_available", False),
                "n_past_obs":     result.get("n_past_observations", 0),
            }

        return report
