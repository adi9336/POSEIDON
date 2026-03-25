"""
InsightRetriever — read side of memory.
Called BEFORE every orchestrator run to enrich context with history.
"""
from __future__ import annotations

import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.core.paths import CHROMA_DIR, INSIGHTS_DB_PATH

log = logging.getLogger(__name__)
SQLITE_PATH = str(INSIGHTS_DB_PATH)
CHROMA_PATH = str(CHROMA_DIR)


class InsightRetriever:

    def __init__(self) -> None:
        self._chroma = None
        self._collection = None
        self._embedder = None

    def _get_chroma(self):
        if self._chroma is None:
            import chromadb
            self._chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            self._collection = self._chroma.get_or_create_collection("poseidon_insights")
        return self._collection

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def retrieve(
        self,
        region: str,
        variable: Optional[str] = None,
        n_semantic: int = 3,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """Single call that returns everything needed to enrich context."""
        enable_semantic = (
            os.getenv("POSEIDON_ENABLE_SEMANTIC_MEMORY", "false").lower() == "true"
        )
        return {
            "semantic_matches":      self._semantic_search(region, variable, n_semantic) if enable_semantic else [],
            "time_series":           self._time_series(region, variable, days_back),
            "baseline":              self._compute_baseline(region, variable),
            "last_anomaly":          self._last_anomaly(region),
            "trend_context":         self._compute_trend(region),
        }

    def _semantic_search(self, region: str, variable: Optional[str], n: int) -> List[Dict]:
        try:
            query = f"{region} {variable or ''} oceanographic analysis"
            embed = self._get_embedder().encode(query).tolist()
            results = self._get_chroma().query(
                query_embeddings=[embed],
                n_results=min(n, 10),
                where={"region": region} if region != "unknown" else None
            )
            if not results["documents"] or not results["documents"][0]:
                return []
            return [
                {"summary": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        except Exception as exc:
            log.warning(f"Semantic search failed: {exc}")
            return []

    def _time_series(self, region: str, variable: Optional[str], days_back: int) -> List[Dict]:
        since = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                rows = conn.execute("""
                    SELECT recorded_at, thermocline_depth, water_mass,
                           anomaly_rate, surface_temp, salinity_mean, confidence, source
                    FROM insights
                    WHERE region = ? AND recorded_at > ?
                    ORDER BY recorded_at DESC LIMIT 30
                """, (region, since)).fetchall()
            cols = ["recorded_at","thermocline_depth","water_mass",
                    "anomaly_rate","surface_temp","salinity_mean","confidence","source"]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def _compute_baseline(self, region: str, variable: Optional[str]) -> Dict[str, Any]:
        since = (datetime.utcnow() - timedelta(days=30)).isoformat()
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                row = conn.execute("""
                    SELECT AVG(thermocline_depth), AVG(surface_temp),
                           AVG(salinity_mean), AVG(anomaly_rate), COUNT(*)
                    FROM insights
                    WHERE region = ? AND recorded_at > ?
                """, (region, since)).fetchone()
            if row and row[4] and row[4] > 0:
                return {
                    "avg_thermocline_m": round(row[0], 1) if row[0] else None,
                    "avg_surface_temp":  round(row[1], 2) if row[1] else None,
                    "avg_salinity":      round(row[2], 2) if row[2] else None,
                    "avg_anomaly_rate":  round(row[3], 4) if row[3] else None,
                    "n_observations":    row[4],
                }
        except Exception:
            pass
        return {"n_observations": 0}

    def _compute_trend(self, region: str) -> Dict[str, Any]:
        ts = self._time_series(region, None, 30)
        if len(ts) < 2:
            return {}
        depths = [r["thermocline_depth"] for r in ts if r.get("thermocline_depth")]
        temps  = [r["surface_temp"]      for r in ts if r.get("surface_temp")]
        trend = {}
        if len(depths) >= 2:
            delta = depths[0] - depths[-1]
            trend["thermocline_trend"] = (
                f"shoaling {abs(delta):.1f}m over {len(depths)} observations" if delta < -2
                else f"deepening {abs(delta):.1f}m over {len(depths)} observations" if delta > 2
                else "stable"
            )
        if len(temps) >= 2:
            delta = temps[0] - temps[-1]
            trend["temperature_trend"] = (
                f"warming +{delta:.2f}°C" if delta > 0.1
                else f"cooling {delta:.2f}°C" if delta < -0.1
                else "stable"
            )
        return trend

    def _last_anomaly(self, region: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                row = conn.execute("""
                    SELECT recorded_at, anomaly_rate, summary, source
                    FROM insights
                    WHERE region = ? AND anomaly_rate > 0.10
                    ORDER BY recorded_at DESC LIMIT 1
                """, (region,)).fetchone()
            if row:
                return {"recorded_at": row[0], "anomaly_rate": row[1],
                        "summary": row[2], "source": row[3]}
        except Exception:
            pass
        return None
