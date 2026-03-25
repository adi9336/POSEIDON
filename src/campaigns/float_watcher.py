"""
Event-driven float watcher — detects new ARGO float data hourly.
Triggers deep analysis only when genuinely new data arrives.
"""
from __future__ import annotations
import hashlib
import requests
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.campaigns.base_campaign import BaseCampaign
from src.tools.geosolver import MarinePolygonClassifier

log = logging.getLogger(__name__)
DEFAULT_FLOATS = ["6900000", "6900001", "3902123"]

_classifier = MarinePolygonClassifier()


def _point_to_region(lat: float, lon: float) -> Optional[str]:
    """Given coordinates, return which ocean basin they fall in."""
    region = _classifier.classify_point(lat, lon)
    return region if region != "Unknown marine region" else None


class FloatWatcherCampaign(BaseCampaign):
    name = "float_watcher"

    def __init__(self, orchestrator, float_ids: Optional[List[str]] = None):
        super().__init__(orchestrator)
        self.float_ids   = float_ids or DEFAULT_FLOATS
        self._seen: Dict[str, str] = {}   # float_id → last content hash

    def _get_latest_hash(self, float_id: str) -> Optional[str]:
        try:
            url = (
                f"https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
                f"platform_number,time,latitude,longitude"
                f"&platform_number=%22{float_id}%22"
                f"&orderByMax(%22time%22)"
            )
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return hashlib.md5(resp.text.encode()).hexdigest()
        except Exception as exc:
            log.warning(f"Hash fetch failed for float {float_id}: {exc}")
        return None

    def execute(self) -> Dict[str, Any]:
        report = {"campaign": self.name, "generated_at": datetime.utcnow().isoformat(), "floats": {}}

        for float_id in self.float_ids:
            current_hash = self._get_latest_hash(float_id)
            if current_hash is None:
                continue

            previous_hash = self._seen.get(float_id)
            self._seen[float_id] = current_hash

            if previous_hash is None:
                report["floats"][float_id] = {"status": "initialized"}
                continue

            if current_hash == previous_hash:
                report["floats"][float_id] = {"status": "no_new_data"}
                continue

            # New data detected — trigger full analysis
            self.log.info(f"New data from float {float_id} — triggering analysis")
            result = self.query_orchestrator(
                query=f"Analyze the latest profile from ARGO float {float_id}. "
                      f"Detect anomalies, classify water mass, compute thermocline depth. "
                      f"Compare to any historical data for this float's region.",
                region="global"
            )
            report["floats"][float_id] = {
                "status":     "analyzed",
                "summary":    result.get("summary"),
                "confidence": result.get("confidence"),
                "history_used": result.get("history_available", False),
            }

        return report
