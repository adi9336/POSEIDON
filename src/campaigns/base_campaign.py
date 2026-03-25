"""
BaseCampaign — shared behaviour for all automated campaigns.
All campaigns inherit from this. Handles logging, error recovery,
and automatic insight store write-back.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.orchestrator.main import PoseidonOrchestrator
from src.state.schemas import OrchestratorRequest

log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
LOGS_DIR    = Path("logs")
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class BaseCampaign(ABC):

    name: str = "base"

    def __init__(self, orchestrator: PoseidonOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.log = logging.getLogger(f"campaign.{self.name}")

    def run(self) -> None:
        """Entry point called by scheduler. Wraps execute() with error handling."""
        self.log.info(f"Campaign {self.name} started")
        try:
            result = self.execute()
            self._save_report(result)
            self.log.info(f"Campaign {self.name} completed successfully")
        except Exception as exc:
            self.log.error(f"Campaign {self.name} failed: {exc}", exc_info=True)

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Implement campaign logic here. Must return a result dict."""

    def query_orchestrator(self, query: str, region: str) -> Dict[str, Any]:
        """Run one orchestrator query, tagged with this campaign as source."""
        req = OrchestratorRequest(query=query, mode="multi")
        response = self.orchestrator.execute(req)
        result = response.result or {}
        result["_campaign_source"] = self.name
        return result

    def _save_report(self, result: Dict[str, Any]) -> None:
        ts       = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = REPORTS_DIR / f"{self.name}_{ts}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.log.info(f"Report saved: {filename}")

    def _log_alert(self, message: str, data: Dict[str, Any]) -> None:
        entry = {
            "campaign":   self.name,
            "timestamp":  datetime.utcnow().isoformat(),
            "message":    message,
            "data":       data,
        }
        alert_log = LOGS_DIR / f"{self.name}_alerts.jsonl"
        with open(alert_log, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self.log.warning(f"ALERT: {message}")
