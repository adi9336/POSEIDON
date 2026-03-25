"""
Water mass classifier using scikit-learn IsolationForest + rule-based T-S classification.
Train once with: python scripts/train_models.py
Then load and call classify_water_masses() at inference.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from src.core.paths import ANOMALY_MODEL_PATH

log = logging.getLogger(__name__)

_anomaly_model = None  # module-level singleton


def _load_anomaly_model():
    global _anomaly_model
    if _anomaly_model is None:
        if not ANOMALY_MODEL_PATH.exists():
            log.info("Anomaly detector not trained yet. Run scripts/train_models.py")
            return None
        import joblib
        _anomaly_model = joblib.load(str(ANOMALY_MODEL_PATH))
        log.info("Anomaly detector loaded")
    return _anomaly_model


# ── Rule-based T-S water mass classification ────────────────────────
# Based on standard T-S diagrams for major water masses

_WATER_MASS_RULES = [
    # (name, temp_min, temp_max, psal_min, psal_max, depth_min, depth_max)
    ("Antarctic Bottom Water (AABW)",     -2.0,  2.0,  34.6,  34.8,  3000, 6000),
    ("North Atlantic Deep Water (NADW)",   1.5,  4.0,  34.8,  35.0,  1500, 4000),
    ("Antarctic Intermediate Water (AAIW)", 2.0,  6.0,  33.8,  34.5,   500, 1500),
    ("North Indian High Salinity (NIHSW)", 20.0, 30.0, 35.5,  36.8,     0,  200),
    ("Arabian Sea High Salinity (ASHSW)",  24.0, 30.0, 36.0,  37.0,     0,  150),
    ("Bay of Bengal Low Salinity (BoBLS)",  25.0, 30.0, 30.0,  34.0,     0,  100),
    ("Red Sea Water (RSW)",                5.0,  15.0, 35.0,  35.5,   500, 1500),
    ("Persian Gulf Water (PGW)",          15.0,  22.0, 35.5,  37.5,   200,  400),
    ("Subtropical Mode Water (STMW)",     16.0,  20.0, 35.2,  36.0,   100,  400),
    ("Tropical Surface Water (TSW)",      25.0,  32.0, 34.0,  35.5,     0,  100),
    ("Subantarctic Mode Water (SAMW)",      4.0, 10.0, 34.2,  34.8,   200,  600),
    ("Mediterranean Overflow Water (MOW)",  6.0, 13.0, 35.5,  36.5,   800, 1500),
]


def classify_water_mass_rule_based(
    temp: float, psal: float, depth: float = 0.0
) -> str:
    """Classify a single T-S-depth point using rule-based ranges."""
    for name, t_min, t_max, s_min, s_max, d_min, d_max in _WATER_MASS_RULES:
        if (t_min <= temp <= t_max and
            s_min <= psal <= s_max and
            d_min <= depth <= d_max):
            return name
    return "unclassified"


def classify_water_masses(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Classify a list of ARGO profile rows into water mass categories.
    Uses rule-based T-S classification.
    Returns list of labels: e.g. ['NIHSW', 'AAIW', 'AABW', ...]
    """
    if not rows:
        return []

    results = []
    for r in rows:
        temp = r.get("temp")
        psal = r.get("psal")
        pres = r.get("pres", 0.0)
        if temp is not None and psal is not None:
            results.append(classify_water_mass_rule_based(
                float(temp), float(psal), float(pres or 0.0)
            ))
        else:
            results.append("unknown")
    return results


def get_dominant_water_mass(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Returns the most common water mass label from a profile batch."""
    labels = classify_water_masses(rows)
    if not labels or all(l in ("unknown", "unclassified") for l in labels):
        return None
    from collections import Counter
    valid = [l for l in labels if l not in ("unknown", "unclassified")]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def detect_anomalies(rows: List[Dict[str, Any]]) -> List[bool]:
    """
    Detect anomalies using trained IsolationForest model.
    Returns list of booleans (True = anomalous).
    Falls back to simple z-score if model not available.
    """
    if not rows:
        return []

    model = _load_anomaly_model()

    if model is not None:
        import pandas as pd
        features = ["temp", "psal", "pres", "latitude", "longitude"]
        records = [{f: r.get(f) for f in features} for r in rows]
        df = pd.DataFrame(records).dropna()
        if df.empty:
            return [False] * len(rows)
        try:
            preds = model.predict(df)  # -1 = anomaly, 1 = normal
            # Map back to original rows
            result = [False] * len(rows)
            valid_idx = [i for i, r in enumerate(rows)
                         if all(r.get(f) is not None for f in features)]
            for i, pred in zip(valid_idx, preds):
                result[i] = (pred == -1)
            return result
        except Exception as exc:
            log.warning(f"IsolationForest prediction failed: {exc}")

    # Fallback: simple z-score on temperature
    temps = [float(r["temp"]) for r in rows if r.get("temp") is not None]
    if len(temps) < 5:
        return [False] * len(rows)

    mean_t = sum(temps) / len(temps)
    std_t = (sum((t - mean_t) ** 2 for t in temps) / len(temps)) ** 0.5
    if std_t < 0.01:
        return [False] * len(rows)

    results = []
    for r in rows:
        if r.get("temp") is not None:
            z = abs(float(r["temp"]) - mean_t) / std_t
            results.append(z > 3.0)
        else:
            results.append(False)
    return results
