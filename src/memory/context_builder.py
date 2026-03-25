"""
ContextBuilder — assembles enriched context dict from InsightRetriever.
Injected into orchestrator state before any agent runs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict
from src.memory.insight_retriever import InsightRetriever

log = logging.getLogger(__name__)


class ContextBuilder:

    def __init__(self) -> None:
        self.retriever = InsightRetriever()

    def build(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        region   = intent.get("location") or "unknown"
        variable = intent.get("variable")

        try:
            history  = self.retriever.retrieve(region, variable)
        except Exception as exc:
            log.warning(f"InsightRetriever failed: {exc}")
            return {
                "history_available": False,
                "n_past_observations": 0,
                "region_resolved": region,
            }

        baseline = history.get("baseline", {})
        ts       = history.get("time_series", [])
        semantic = history.get("semantic_matches", [])
        anomaly  = history.get("last_anomaly")
        trend    = history.get("trend_context", {})

        return {
            # For ValidationAgent — compare current vs rolling average
            "historical_baseline":     baseline,

            # For AnalysisAgent — trend direction
            "trend_context":           trend,

            # For LLM summary — inject as text
            "recent_summaries":        [m["summary"] for m in semantic[:2]],

            # For campaign alert dedup — don't re-fire same event
            "last_anomaly":            anomaly,

            # For DataRetrievalAgent — skip re-fetch if fresh cache exists
            "historical_time_series":  ts[:5],

            # Meta flags
            "history_available":       len(ts) > 0,
            "n_past_observations":     baseline.get("n_observations", 0),
            "region_resolved":         region,
        }
