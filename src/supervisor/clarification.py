"""
Clarification Engine for the Supervisor Agent.
Generates structured clarification questions based on missing/ambiguous parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.supervisor.state import ClarificationQuestion

logger = logging.getLogger(__name__)

# Canonical parameter definitions with user-friendly questions
PARAMETER_QUESTIONS = {
    "variable": ClarificationQuestion(
        id="variable",
        text="Which ocean parameter are you interested in?",
        options=[
            "Temperature (temp)",
            "Salinity (psal)",
            "Pressure/Depth (pres)",
            "Nitrate (nitrate)",
            "All available parameters",
        ],
        default="Temperature (temp)",
        required=True,
    ),
    "location": ClarificationQuestion(
        id="location",
        text="Which geographic region are you interested in?",
        options=[
            "Let me type a location name",
            "Show me a map to select",
            "Arabian Sea / Indian Ocean",
            "Pacific Ocean",
            "Atlantic Ocean",
        ],
        required=True,
    ),
    "depth": ClarificationQuestion(
        id="depth",
        text="What depth range are you interested in?",
        options=[
            "Surface (0-10m)",
            "Shallow (10-100m)",
            "Mid-depth (100-500m)",
            "Deep ocean (500-2000m)",
            "Full depth profile",
        ],
        default="Surface (0-10m)",
        required=False,
    ),
    "time_range": ClarificationQuestion(
        id="time_range",
        text="What time period should I analyze?",
        options=[
            "Last 7 days",
            "Last 30 days",
            "Last 3 months",
            "Last year",
            "Custom range",
        ],
        default="Last 30 days",
        required=False,
    ),
    "operation": ClarificationQuestion(
        id="operation",
        text="What type of analysis would you like?",
        options=[
            "Raw data retrieval",
            "Statistical summary (mean, min, max)",
            "Trend analysis over time",
            "Comparison across locations",
        ],
        default="Statistical summary (mean, min, max)",
        required=False,
    ),
}


class ClarificationEngine:
    """
    Rule-based clarification question generator.
    Inspects a parsed intent (or lack thereof) and generates questions
    for any missing/ambiguous parameters.
    """

    def identify_missing_params(
        self,
        intent: Optional[Dict[str, Any]],
        collected: Dict[str, Any],
    ) -> List[str]:
        """
        Compare parsed intent against required fields to find gaps.

        Returns list of missing parameter keys like ["location", "depth"].
        """
        missing: List[str] = []
        intent = intent or {}

        # Variable is always needed
        var = intent.get("variable") or collected.get("variable")
        if not var:
            missing.append("variable")

        # Location: need at least a name or lat/lon
        has_location = (
            intent.get("location")
            or collected.get("location")
            or (intent.get("lat") is not None and intent.get("lon") is not None)
            or (collected.get("lat") is not None and collected.get("lon") is not None)
        )
        if not has_location:
            missing.append("location")

        # Depth is optional but useful — only ask if nothing at all
        has_depth = (
            intent.get("depth") is not None
            or intent.get("depth_range") is not None
            or collected.get("depth") is not None
        )
        if not has_depth and not collected.get("depth_skipped"):
            missing.append("depth")

        # Time range has sensible defaults, only flag if query seems time-specific
        has_time = (
            intent.get("time_range") is not None
            or collected.get("time_range") is not None
        )
        if not has_time and not collected.get("time_range_skipped"):
            missing.append("time_range")

        return missing

    def generate_questions(
        self,
        missing_params: List[str],
        already_asked: Optional[List[str]] = None,
    ) -> List[ClarificationQuestion]:
        """
        Generate ClarificationQuestion objects for each missing param.
        Skips params that were already asked about.
        """
        already_asked = already_asked or []
        questions: List[ClarificationQuestion] = []

        for param_id in missing_params:
            if param_id in already_asked:
                continue
            if param_id in PARAMETER_QUESTIONS:
                questions.append(PARAMETER_QUESTIONS[param_id])

        return questions

    def needs_clarification(
        self,
        intent: Optional[Dict[str, Any]],
        collected: Dict[str, Any],
    ) -> bool:
        """Quick check: are there any missing critical params?"""
        missing = self.identify_missing_params(intent, collected)
        # Only variable and location are truly critical
        critical = {"variable", "location"}
        return bool(critical.intersection(missing))

    def parse_clarification_answer(
        self, question_id: str, answer: str
    ) -> Dict[str, Any]:
        """
        Convert a user's clarification answer into structured data.
        Returns a dict to merge into collected clarifications.
        """
        result: Dict[str, Any] = {}

        if question_id == "variable":
            var_map = {
                "Temperature (temp)": "temp",
                "Salinity (psal)": "psal",
                "Pressure/Depth (pres)": "pres",
                "Nitrate (nitrate)": "nitrate",
                "All available parameters": "all",
            }
            result["variable"] = var_map.get(answer, answer.lower())

        elif question_id == "depth":
            depth_map = {
                "Surface (0-10m)": 5.0,
                "Shallow (10-100m)": 50.0,
                "Mid-depth (100-500m)": 300.0,
                "Deep ocean (500-2000m)": 1000.0,
                "Full depth profile": None,
            }
            result["depth"] = depth_map.get(answer)
            if answer == "Full depth profile":
                result["depth_skipped"] = True

        elif question_id == "time_range":
            time_map = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 3 months": 90,
                "Last year": 365,
            }
            days = time_map.get(answer)
            if days:
                result["time_range_days"] = days
            else:
                result["time_range_skipped"] = True

        elif question_id == "operation":
            op_map = {
                "Raw data retrieval": "retrieval",
                "Statistical summary (mean, min, max)": "summary",
                "Trend analysis over time": "trend",
                "Comparison across locations": "comparison",
            }
            result["operation"] = op_map.get(answer, "summary")

        elif question_id == "location":
            if answer == "Show me a map to select":
                result["wants_map"] = True
            else:
                result["location"] = answer

        return result
