"""Local presentation of non-diagnostic webcam research observations.

This module deliberately reports measurement quality and task summaries without
calculating an autism likelihood, risk category, or referral recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def calibration_summary(profile_path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        validation = profile.get("validation")
        if not isinstance(validation, dict):
            return {"available": False, "message": "No independent calibration-validation result is available."}
        return {
            "available": True,
            "median_error_px": validation.get("median_px"),
            "status": validation.get("status", "UNKNOWN"),
            "normalized_error": validation.get("normalized_median_error"),
            "message": "Calibration is an engineering measurement-quality indicator, not a clinical result.",
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"available": False, "message": "Calibration profile could not be read."}


def load_research_observation(report_path: Path) -> dict[str, Any]:
    """Read a local gaze/attention report into a non-diagnostic presentation model."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read the selected local research report: {error}") from error

    if report.get("task") == "standardized_visual_attention_v1":
        trials = report.get("trials")
        if not isinstance(trials, list) or not trials:
            raise ValueError("The attention-task report contains no trials.")
        valid_seconds = [float(trial.get("valid_gaze_seconds", 0.0) or 0.0) for trial in trials]
        on_target_ratios = [trial.get("on_target_ratio") for trial in trials if trial.get("on_target_ratio") is not None]
        return {
            "kind": "Visual-attention research task",
            "source_name": report_path.name,
            "data_sufficiency": "Task-level only - no overall sufficiency label was recorded.",
            "measures": {
                "Completed trials": len(trials),
                "Total valid gaze time": f"{sum(valid_seconds):.2f} seconds",
                "Mean on-target ratio": (
                    "Not available" if not on_target_ratios
                    else f"{sum(float(value) for value in on_target_ratios) / len(on_target_ratios):.3f}"
                ),
            },
        }

    if "tracking_ratio" in report and "data_quality" in report:
        return {
            "kind": "Webcam engagement research session",
            "source_name": report_path.name,
            "data_sufficiency": str(report.get("data_quality", "not reported")).upper(),
            "measures": {
                "Tracking ratio": f"{float(report.get('tracking_ratio', 0.0)) * 100:.1f}%",
                "Valid gaze samples": report.get("valid_gaze_samples", "Not available"),
                "Session duration": f"{float(report.get('duration_seconds', 0.0)):.1f} seconds",
                "Longest tracking gap": f"{float(report.get('longest_tracking_gap_seconds', 0.0)):.1f} seconds",
                "Fixation ratio": report.get("fixation_ratio", "Not available"),
            },
        }
    raise ValueError("Select a local visual-attention or webcam engagement research report.")


def format_observation(observation: dict[str, Any], calibration: dict[str, Any]) -> str:
    lines = [
        f"Research observation: {observation['kind']}",
        f"Source: {observation['source_name']}",
        f"Data sufficiency: {observation['data_sufficiency']}",
        "",
        "Recorded measures:",
    ]
    lines.extend(f"- {label}: {value}" for label, value in observation["measures"].items())
    lines.extend(["", "Calibration quality:"])
    if calibration.get("available"):
        lines.extend([
            f"- Independent median gaze error: {calibration.get('median_error_px')} px",
            f"- Engineering status: {calibration.get('status')}",
            f"- Normalized median error: {float(calibration.get('normalized_error', 0.0)) * 100:.1f}%",
        ])
    else:
        lines.append(f"- {calibration['message']}")
    lines.extend([
        "",
        "Interpretation boundary:",
        "These are webcam-derived research observations only. They must not be combined with questionnaire answers to generate an ASD likelihood, automated risk flag, diagnosis, or referral decision.",
    ])
    return "\n".join(lines)
