"""
One-time model training script.
Run ONCE after collecting ARGO data: python scripts/train_models.py
Trains: Isolation Forest (anomaly) + spaCy NER (intent)
"""
import logging
import random

from src.core.paths import (
    ANOMALY_MODEL_PATH,
    INTENT_NER_MODEL_PATH,
    MODELS_DIR,
    TRAINING_DATA_PATH,
    ensure_runtime_dirs,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ensure_runtime_dirs()
MODELS_DIR.mkdir(exist_ok=True)


def train_anomaly_detector():
    log.info("Training Isolation Forest anomaly detector...")
    try:
        import pandas as pd
        from sklearn.ensemble import IsolationForest
        import joblib
    except ImportError:
        log.warning("scikit-learn or pandas not installed - skipping anomaly model")
        return

    training_data = TRAINING_DATA_PATH
    if not training_data.exists():
        log.warning(f"No training data found at {training_data} - skipping anomaly model")
        return

    df = pd.read_csv(str(training_data))
    features = ["temp", "psal", "pres", "latitude", "longitude"]
    X = df[features].dropna()

    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X)

    joblib.dump(clf, ANOMALY_MODEL_PATH)
    log.info(f"Anomaly detector trained on {len(X)} profiles -> {ANOMALY_MODEL_PATH}")


def train_intent_ner():
    log.info("Training spaCy NER intent extractor...")
    try:
        import spacy
        from spacy.training import Example
    except ImportError:
        log.warning("spaCy not installed - skipping NER model")
        return

    # Minimal training data - expand with real query examples
    TRAIN_DATA = [
        ("What is the salinity at 500m near Arabian Sea in January 2024?", {
            "entities": [(18,26,"VARIABLE"),(30,34,"DEPTH"),(40,51,"LOCATION"),(55,67,"TIME_RANGE")]
        }),
        ("Show temperature trend in Bay of Bengal last month", {
            "entities": [(5,16,"VARIABLE"),(26,40,"LOCATION"),(41,51,"TIME_RANGE")]
        }),
        ("Nitrate levels at 200m depth in Indian Ocean", {
            "entities": [(0,7,"VARIABLE"),(18,22,"DEPTH"),(32,44,"LOCATION")]
        }),
        ("Temperature at 1000m in Arabian Sea over last 30 days", {
            "entities": [(0,11,"VARIABLE"),(15,20,"DEPTH"),(24,36,"LOCATION"),(42,56,"TIME_RANGE")]
        }),
        ("Salinity anomaly near Mumbai in March 2024", {
            "entities": [(0,8,"VARIABLE"),(22,28,"LOCATION"),(32,42,"TIME_RANGE")]
        }),
    ]

    nlp  = spacy.blank("en")
    ner  = nlp.add_pipe("ner")
    for _, annotations in TRAIN_DATA:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    nlp.begin_training()
    for epoch in range(30):
        random.shuffle(TRAIN_DATA)
        for text, annotations in TRAIN_DATA:
            doc     = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example])

    INTENT_NER_MODEL_PATH.mkdir(exist_ok=True)
    nlp.to_disk(INTENT_NER_MODEL_PATH)
    log.info(f"spaCy NER model saved -> {INTENT_NER_MODEL_PATH}/")
    log.info("IMPORTANT: Add more training examples for better accuracy")


if __name__ == "__main__":
    train_anomaly_detector()
    train_intent_ner()
    log.info("All models trained. System ready.")
