from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"

ARGO_DB_PATH = DATA_DIR / "argo_data.db"
CAMPAIGNS_CONFIG_PATH = CONFIG_DIR / "campaigns.yaml"
CHROMA_DIR = DB_DIR / "chroma"
INSIGHTS_DB_PATH = DB_DIR / "insights.db"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.pkl"
INTENT_NER_MODEL_PATH = MODELS_DIR / "argo_intent_ner"
TRAINING_DATA_PATH = DATA_DIR / "argo_training.csv"


def ensure_runtime_dirs() -> None:
    """Create runtime directories used by the application if missing."""
    for path in (DATA_DIR, DB_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
