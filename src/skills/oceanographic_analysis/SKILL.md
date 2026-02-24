---
name: oceanographic-analysis
description: Analyze oceanographic query outputs with domain checks and trend logic. Use when requests require interpretation of temperature, salinity, depth, anomalies, or basin-aware insights.
---

# Oceanographic Analysis

1. Validate depth bounds against Argo operating range (`0-6000m`).
2. Check time windows for seasonality and report if sample windows are too short.
3. Compare basin/location context before drawing cross-region conclusions.
4. Flag likely anomalies only when variability exceeds expected local spread.
5. Return concise findings with confidence and recommended follow-up data.

