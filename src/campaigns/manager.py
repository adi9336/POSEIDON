"""
CampaignManager — central scheduler.
One shared orchestrator instance for both chatbot and all campaigns.
"""
from __future__ import annotations

import logging
import yaml
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler

from src.core.paths import CAMPAIGNS_CONFIG_PATH
from src.orchestrator.main import PoseidonOrchestrator

log = logging.getLogger(__name__)


class CampaignManager:

    def __init__(self, config_path: str = str(CAMPAIGNS_CONFIG_PATH)) -> None:
        self.orchestrator = PoseidonOrchestrator()
        self.scheduler    = BackgroundScheduler(timezone="UTC")
        self.config       = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
        log.warning(f"Campaign config not found at {path}, using defaults")
        return {"campaigns": {}}

    def start(self) -> None:
        cfg = self.config.get("campaigns", {})

        if cfg.get("anomaly_watch", {}).get("enabled", True):
            from src.campaigns.anomaly_watch import AnomalyWatchCampaign
            c = AnomalyWatchCampaign(
                self.orchestrator,
                regions=cfg["anomaly_watch"].get("regions"),
                threshold=cfg["anomaly_watch"].get("alert_threshold", 0.10)
            )
            self.scheduler.add_job(c.run, "cron",
                hour=cfg["anomaly_watch"].get("hour", 6), id="anomaly_watch")
            log.info("Anomaly watch registered — daily at 06:00 UTC")

        if cfg.get("weekly_report", {}).get("enabled", True):
            from src.campaigns.weekly_report import WeeklyReportCampaign
            c = WeeklyReportCampaign(
                self.orchestrator,
                regions=cfg["weekly_report"].get("regions")
            )
            self.scheduler.add_job(c.run, "cron",
                day_of_week="mon", hour=cfg["weekly_report"].get("hour", 7),
                id="weekly_report")
            log.info("Weekly report registered — every Monday 07:00 UTC")

        if cfg.get("float_watcher", {}).get("enabled", True):
            from src.campaigns.float_watcher import FloatWatcherCampaign
            c = FloatWatcherCampaign(
                self.orchestrator,
                float_ids=cfg["float_watcher"].get("float_ids")
            )
            self.scheduler.add_job(c.run, "interval", hours=1, id="float_watcher")
            log.info("Float watcher registered — every hour")

        self.scheduler.start()
        log.info("CampaignManager started — all campaigns running in background")

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        log.info("CampaignManager stopped")
