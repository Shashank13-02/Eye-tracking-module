"""Transparent, local workflow export without an automated clinical decision."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


INTERPRETATION_BOUNDARY = (
    "This export records separate webcam research observations and published "
    "M-CHAT-R/F scoring. It does not calculate an ASD/ADHD likelihood, combined "
    "risk flag, diagnosis, automated referral, or treatment decision."
)


def compact_webcam_parameters(report: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    """Keep derived research parameters while excluding raw samples and pseudonyms."""
    if report.get("task") == "standardized_visual_attention_v1":
        return {"source_name": source_name, "kind": "visual_attention_task", "parameters": {
            "completed": report.get("completed"), "trials": deepcopy(report.get("trials", [])),
        }}
    if "tracking_ratio" in report and "data_quality" in report:
        keys = (
            "duration_seconds", "tracking_ratio", "valid_gaze_samples", "mean_gaze_speed_per_second",
            "fixation_ratio", "gaze_dispersion", "gaze_entropy_bits", "gaze_region_transitions",
            "longest_tracking_gap_seconds", "invalid_gaze_frames", "quality_flag_counts", "data_quality",
        )
        return {"source_name": source_name, "kind": "webcam_engagement_session", "parameters": {
            key: deepcopy(report.get(key)) for key in keys if key in report
        }}
    if isinstance(report.get("modules"), list) and "age_band" in report:
        return {"source_name": source_name, "kind": "age_banded_research_tasks", "parameters": {
            "age_months": report.get("age_months"), "age_band": report.get("age_band"),
            "modules": deepcopy(report["modules"]),
        }}
    if isinstance(report.get("samples"), list) and isinstance(report.get("events"), list):
        counts: dict[str, int] = {}
        for event in report["events"]:
            event_type = str(event.get("event_type", "unknown"))
            counts[event_type] = counts.get(event_type, 0) + 1
        valid = sum(1 for sample in report["samples"] if sample.get("valid"))
        return {"source_name": source_name, "kind": "timestamped_gaze_event_summary", "parameters": {
            "duration_seconds": report.get("duration_seconds"), "coordinate_system": report.get("coordinate_system"),
            "total_samples": len(report["samples"]), "valid_samples": valid, "event_counts": counts,
        }}
    raise ValueError("Unsupported webcam report. Select a visual-attention, engagement, age-banded, or gaze-event report.")


def build_final_workflow_export(
    screening_payload: Mapping[str, Any],
    webcam_reports: Sequence[Mapping[str, Any]],
    calibration_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine records structurally, never statistically or diagnostically."""
    result = screening_payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("A scored M-CHAT-R/F result is required for final export.")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_type": "local_developmental_screening_workflow_summary",
        "child_local_id": screening_payload.get("child_local_id"),
        "session_id": screening_payload.get("session_id"),
        "age_months": screening_payload.get("age_months"),
        "webcam_research": {
            "calibration_context": deepcopy(calibration_context) if calibration_context else {
                "available": False, "message": "No calibration context attached."
            },
            "attached_reports": [deepcopy(report) for report in webcam_reports],
            "missing_reports_notice": None if webcam_reports else "No webcam research report was attached to this export.",
            "interpretation": "Research observations only; preserve measurement quality and data sufficiency beside every result.",
        },
        "mchat_rf": {
            "first_stage_answers": deepcopy(screening_payload.get("answers", {})),
            "first_stage_result": deepcopy(result),
            "official_follow_up": deepcopy(screening_payload.get("follow_up")),
            "caregiver_concerns": screening_payload.get("caregiver_concerns"),
        },
        "qchat_research": deepcopy(screening_payload.get("qchat_research")),
        "clinician_review": deepcopy(screening_payload.get("clinician_review")),
        "record_status": screening_payload.get("record_status", "draft_unreviewed"),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "automated_decision": {"available": False, "reason": INTERPRETATION_BOUNDARY},
    }
