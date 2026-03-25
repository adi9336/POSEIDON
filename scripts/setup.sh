#!/bin/bash
# Run once to set up all POSEIDON dependencies and model downloads
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo "Setup complete. Run scripts/train_models.py next."
