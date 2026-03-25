"""
Fast local intent extraction using spaCy NER.
Replaces GPT-4o-mini call for basic entity parsing.
Falls back to existing GPT extractor for complex queries.
Train with: python scripts/train_models.py
"""
from __future__ import annotations

import re
import logging
from typing import Optional
from src.core.paths import INTENT_NER_MODEL_PATH
from src.state.models import ScientificIntent

log = logging.getLogger(__name__)
MODEL_PATH = INTENT_NER_MODEL_PATH


def extract_intent_fast(query: str) -> ScientificIntent:
    """
    Primary: spaCy NER (5ms, local, free).
    Fallback: regex heuristics from existing fallback_intent_extraction.
    Final fallback: original GPT extractor.
    """
    # Try spaCy first
    if MODEL_PATH.exists():
        try:
            return _spacy_extract(query)
        except Exception as exc:
            log.warning(f"spaCy extraction failed, falling back: {exc}")

    # Try regex heuristics
    try:
        from src.tools.intent_extractor import fallback_intent_extraction
        return fallback_intent_extraction(query)
    except Exception:
        pass

    # Final fallback to GPT
    from src.tools.intent_extractor import extract_intent_with_llm
    return extract_intent_with_llm(query)


def _spacy_extract(query: str) -> ScientificIntent:
    import spacy
    nlp = spacy.load(str(MODEL_PATH))
    doc = nlp(query)

    result: dict = {
        "variable": None, "location": None, "lat": None, "lon": None,
        "depth": None, "time_range": None, "operation": None, "depth_range": None,
    }

    var_map = {
        "temperature": "temp", "temp": "temp",
        "salinity": "psal", "psal": "psal", "salt": "psal",
        "nitrate": "nitrate", "nitrogen": "nitrate",
        "oxygen": "doxy", "doxy": "doxy",
    }

    for ent in doc.ents:
        label = ent.label_
        text  = ent.text.strip()
        if label == "VARIABLE":
            result["variable"] = var_map.get(text.lower(), text.lower())
        elif label == "LOCATION":
            result["location"] = text
        elif label == "DEPTH":
            nums = re.findall(r"[\d.]+", text)
            if nums:
                result["depth"] = float(nums[0])
        elif label == "TIME_RANGE":
            from src.tools.intent_extractor import parse_relative_time
            start, end = parse_relative_time(text)
            if start and end:
                result["time_range"] = (start, end)

    return ScientificIntent(**result)
