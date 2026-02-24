from __future__ import annotations

from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.state.schemas import AgentResult, AgentTask, QualityReport


class ValidationAgent(BaseAgent):
    name = "validation"
    OUTLIER_RATE_THRESHOLD = 0.10
    TIME_WINDOW_DRIFT_THRESHOLD = 0.05

    def plan(self, context: Dict[str, Any]) -> List[AgentTask]:
        return [
            AgentTask(
                agent=self.name,
                task_type="validate_quality",
                payload={
                    "intent": context.get("intent", {}),
                    "analysis": context.get("analysis", {}),
                    "retrieval": context.get("retrieval", {}),
                },
            )
        ]

    def execute(self, task: AgentTask, context: Dict[str, Any]) -> AgentResult:
        intent = task.payload.get("intent", {}) or {}
        analysis = task.payload.get("analysis", {}) or {}
        data = analysis.get("data", []) or []
        row_count = int(analysis.get("row_count", len(data) if isinstance(data, list) else 0))

        issues: List[str] = []
        metrics: Dict[str, float] = {
            "row_count": float(row_count),
        }
        confidence = 0.9

        # Critical completeness check
        if row_count == 0:
            issues.append("No records returned for current filters.")
            confidence -= 0.35

        # Oceanographic depth sanity
        depth = intent.get("depth")
        if depth is not None:
            try:
                d = float(depth)
                metrics["requested_depth_m"] = d
                if d < 0 or d > 6000:
                    issues.append("Requested depth is outside Argo operating range (0-6000m).")
                    confidence -= 0.2
            except (TypeError, ValueError):
                issues.append("Requested depth value is invalid.")
                confidence -= 0.15

        # Data plausibility checks
        if row_count > 0 and isinstance(data, list):
            temp_values = [r.get("temp") for r in data if isinstance(r, dict) and r.get("temp") is not None]
            psal_values = [r.get("psal") for r in data if isinstance(r, dict) and r.get("psal") is not None]
            pres_values = [r.get("pres") for r in data if isinstance(r, dict) and r.get("pres") is not None]
            lat_values = [r.get("latitude") for r in data if isinstance(r, dict) and r.get("latitude") is not None]
            lon_values = [r.get("longitude") for r in data if isinstance(r, dict) and r.get("longitude") is not None]
            time_values = [r.get("time") for r in data if isinstance(r, dict) and r.get("time")]

            if temp_values:
                tmin = float(min(temp_values))
                tmax = float(max(temp_values))
                metrics["temp_min"] = tmin
                metrics["temp_max"] = tmax
                if tmin < -2.5 or tmax > 40.0:
                    issues.append("Temperature values outside plausible ocean range (-2.5C to 40C).")
                    confidence -= 0.15

            if psal_values:
                smin = float(min(psal_values))
                smax = float(max(psal_values))
                metrics["psal_min"] = smin
                metrics["psal_max"] = smax
                if smin < 0 or smax > 42.0:
                    issues.append("Salinity values outside plausible PSU range (0-42).")
                    confidence -= 0.1

            if lat_values and (min(lat_values) < -90 or max(lat_values) > 90):
                issues.append("Latitude values outside valid range (-90 to 90).")
                confidence -= 0.1

            if lon_values and (min(lon_values) < -180 or max(lon_values) > 180):
                issues.append("Longitude values outside valid range (-180 to 180).")
                confidence -= 0.1

            # Time-window consistency against requested range
            requested_range = intent.get("time_range")
            if requested_range and isinstance(requested_range, (list, tuple)) and len(requested_range) == 2 and time_values:
                start_dt = self._parse_dt(requested_range[0])
                end_dt = self._parse_dt(requested_range[1], end_of_day=True)
                if start_dt and end_dt:
                    parsed = [self._parse_dt(t) for t in time_values]
                    parsed = [t for t in parsed if t is not None]
                    if parsed:
                        outside = [t for t in parsed if t < start_dt or t > end_dt]
                        outside_rate = len(outside) / len(parsed)
                        metrics["time_outside_rate"] = round(outside_rate, 4)
                        if outside_rate > self.TIME_WINDOW_DRIFT_THRESHOLD:
                            issues.append(
                                "Data includes records outside requested time window."
                            )
                            confidence -= 0.15

            # Outlier-rate thresholds (beyond min/max sanity)
            for label, values in (("temp", temp_values), ("psal", psal_values), ("pres", pres_values)):
                numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
                if len(numeric_values) < 8:
                    continue
                outlier_rate = self._robust_outlier_rate(numeric_values)
                metrics[f"{label}_outlier_rate"] = round(outlier_rate, 4)
                if outlier_rate > self.OUTLIER_RATE_THRESHOLD:
                    issues.append(
                        f"High {label} outlier rate ({outlier_rate:.1%}) exceeds threshold ({self.OUTLIER_RATE_THRESHOLD:.0%})."
                    )
                    confidence -= 0.12

        report = QualityReport(
            passed=len(issues) == 0,
            confidence=max(0.05, round(confidence, 3)),
            issues=issues,
            metrics=metrics,
        )

        return AgentResult(
            task_id=task.task_id,
            agent=self.name,
            status="success",
            confidence=report.confidence,
            data={
                "quality_report": report.model_dump(),
                "validated_at": datetime.utcnow().isoformat(),
            },
        )

    def validate(
        self, result: AgentResult, context: Dict[str, Any]
    ) -> QualityReport | None:
        report = result.data.get("quality_report")
        if not report:
            return QualityReport(
                passed=result.status != "error",
                confidence=result.confidence,
                issues=[] if result.status != "error" else [result.error or "unknown error"],
                metrics={},
            )
        return QualityReport(**report)

    def _parse_dt(self, value: Any, end_of_day: bool = False):
        if value is None:
            return None
        try:
            s = str(value).strip()
            if len(s) == 10:
                if end_of_day:
                    s = f"{s}T23:59:59"
                else:
                    s = f"{s}T00:00:00"
            s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            # Normalize to UTC-naive to avoid aware/naive comparison errors.
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    def _robust_outlier_rate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        med = median(values)
        deviations = [abs(x - med) for x in values]
        mad = median(deviations)

        if mad > 0:
            modified_z = [0.6745 * abs(x - med) / mad for x in values]
            outliers = [z for z in modified_z if z > 3.5]
            return len(outliers) / len(values)

        # Fallback when MAD is zero: use IQR, or exact-match deviation when IQR is also zero.
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[(3 * n) // 4]
        iqr = q3 - q1
        if iqr == 0:
            outliers = [x for x in values if x != med]
            return len(outliers) / len(values)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [x for x in values if x < lower or x > upper]
        return len(outliers) / len(values)
