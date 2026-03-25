"""
InsightStore — dual write: ChromaDB (semantic) + SQLite (structured).
Called after EVERY orchestrator execution, chatbot or campaign.
"""
from __future__ import annotations

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.paths import CHROMA_DIR, INSIGHTS_DB_PATH, ensure_runtime_dirs

log = logging.getLogger(__name__)

ensure_runtime_dirs()
SQLITE_PATH = str(INSIGHTS_DB_PATH)
CHROMA_PATH = str(CHROMA_DIR)


class InsightStore:

    def __init__(self) -> None:
        self._init_sqlite()
        self._chroma = None
        self._collection = None
        self._embedder = None

    # ── lazy-load heavy deps ──────────────────────────────────────────
    def _get_chroma(self):
        if self._chroma is None:
            import chromadb
            self._chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            self._collection = self._chroma.get_or_create_collection(
                "poseidon_insights",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ── SQLite schema ─────────────────────────────────────────────────
    def _init_sqlite(self) -> None:
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id                TEXT PRIMARY KEY,
                    region            TEXT NOT NULL,
                    variable          TEXT,
                    source            TEXT NOT NULL,
                    recorded_at       TEXT NOT NULL,
                    thermocline_depth REAL,
                    water_mass        TEXT,
                    anomaly_rate      REAL,
                    surface_temp      REAL,
                    salinity_mean     REAL,
                    summary           TEXT,
                    confidence        REAL,
                    raw_json          TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_region_time ON insights(region, recorded_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly ON insights(region, anomaly_rate)")

    # ── public write ──────────────────────────────────────────────────
    def write(
        self,
        result: Dict[str, Any],
        region: str,
        source: str = "chatbot"
    ) -> str:
        """Write one analysis result. Returns the insight_id."""
        insight_id = f"{region.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        summary    = result.get("summary", "")
        vi         = result.get("variable_insights", {})
        physics    = result.get("physics", {})
        intent     = result.get("intent", {})
        validation = result.get("validation", {})

        # 1. SQLite structured row
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO insights
                    (id, region, variable, source, recorded_at,
                     thermocline_depth, water_mass, anomaly_rate,
                     surface_temp, salinity_mean, summary, confidence, raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    insight_id,
                    region,
                    intent.get("variable"),
                    source,
                    datetime.utcnow().isoformat(),
                    physics.get("thermocline_depth_m"),
                    result.get("water_mass_prediction"),
                    validation.get("metrics", {}).get("temp_outlier_rate"),
                    physics.get("surface_temp"),
                    vi.get("psal", {}).get("mean"),
                    summary,
                    result.get("confidence", 0.0),
                    json.dumps(result, default=str)[:4000]
                ))
        except Exception as exc:
            log.error(f"InsightStore SQLite write failed: {exc}")

        # 2. ChromaDB vector embedding
        enable_semantic = (
            os.getenv("POSEIDON_ENABLE_SEMANTIC_MEMORY", "false").lower() == "true"
        )
        if summary and enable_semantic:
            try:
                text  = f"{region} {intent.get('variable','')} {summary}"
                embed = self._get_embedder().encode(text).tolist()
                self._get_chroma().add(
                    ids=[insight_id],
                    embeddings=[embed],
                    documents=[summary],
                    metadatas=[{
                        "region":            region,
                        "source":            source,
                        "recorded_at":       datetime.utcnow().isoformat(),
                        "thermocline_depth": float(physics.get("thermocline_depth_m") or -1.0),
                        "water_mass":        result.get("water_mass_prediction", "unknown"),
                        "confidence":        float(result.get("confidence", 0.0)),
                    }]
                )
            except Exception as exc:
                log.error(f"InsightStore ChromaDB write failed: {exc}")

        log.info(f"Insight written: {insight_id} ({source}, {region})")
        return insight_id
