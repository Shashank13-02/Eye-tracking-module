"""Age-banded, task-defined research biomarkers for NeuroGaze.

These measures are descriptive task outcomes, not norms, diagnoses, or risk scores.
Age bands use AAP developmental-screening anchors (9, 18, 24, 30 months) to select
short, developmentally conservative task protocols. They require clinical and
empirical validation before any clinical interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AgeBand:
    key: str
    min_months: int
    max_months: int
    task_duration_seconds: float
    enabled_tasks: tuple[str, ...]
    rationale: str


AGE_BANDS = (
    AgeBand("0_to_8_months", 0, 8, 0.0, (), "No self-directed screen-gaze task; use clinician-designed observational research only."),
    AgeBand("9_to_17_months", 9, 17, 3.0, ("face_preference", "moving_tracking"),
            "Short passive, high-contrast social and motion stimuli; no cue-following outcome by default."),
    AgeBand("18_to_24_months", 18, 24, 3.5, ("face_preference", "moving_tracking", "social_gaze_following"),
            "Toddler protocol aligned to the age range where AAP recommends ASD-specific screening alongside standard care."),
    AgeBand("25_to_30_months", 25, 30, 4.0, ("face_preference", "moving_tracking", "social_gaze_following"),
            "Slightly longer toddler protocol; keep caregiver present and allow breaks."),
    AgeBand("31_to_72_months", 31, 72, 5.0, ("face_preference", "moving_tracking", "social_gaze_following"),
            "Preschool protocol; use interactive age-appropriate stimuli and ongoing developmental surveillance."),
)


def select_age_band(age_months: int) -> AgeBand:
    if not 0 <= age_months <= 72:
        raise ValueError("Age must be supplied in whole months from 0 through 72.")
    return next(band for band in AGE_BANDS if band.min_months <= age_months <= band.max_months)


@dataclass(frozen=True)
class BiomarkerResult:
    module: str
    age_band: str
    data_quality: str
    metrics: dict[str, Optional[float]]
    limitations: tuple[str, ...]


def _usable_samples(samples: Sequence[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [s for s in samples if s.get("valid") and str(s.get("task_event") or "").startswith(prefix)
            and s.get("gaze_x") is not None and s.get("gaze_y") is not None]


def _duration_weights(samples: Sequence[dict[str, Any]], cap: float = 0.20) -> list[float]:
    if not samples:
        return []
    times = [float(s["timestamp_seconds"]) for s in samples]
    return [min(cap, max(0.0, b - a)) for a, b in zip(times, times[1:])] + [0.0]


def face_preference(samples: Sequence[dict[str, Any]], age_months: int) -> BiomarkerResult:
    band = select_age_band(age_months)
    values = _usable_samples(samples, "face_preference:")
    if "face_preference" not in band.enabled_tasks:
        return _disabled("face_preference", band, "This protocol is not enabled for this age band.")
    if len(values) < 8:
        return _insufficient("face_preference", band)
    weights = _duration_weights(values)
    face_time = non_social_time = 0.0
    entries = 0
    was_face = False
    for sample, weight in zip(values, weights):
        # Event labels encode counterbalanced side, e.g. face_preference:face_left.
        face_left = str(sample["task_event"]).endswith("face_left")
        in_face = sample["gaze_x"] < 0.5 if face_left else sample["gaze_x"] >= 0.5
        if in_face:
            face_time += weight
            entries += int(not was_face)
        else:
            non_social_time += weight
        was_face = in_face
    total = face_time + non_social_time
    return BiomarkerResult("face_preference", band.key, "adequate" if total >= 1.0 else "low",
                           {"face_dwell_ratio": _ratio(face_time, total), "face_entry_frequency": float(entries),
                            "valid_gaze_seconds": round(total, 3)},
                           ("AOIs are simple left/right regions; replace with stimulus-specific face masks before validation.",))


def moving_tracking(samples: Sequence[dict[str, Any]], age_months: int) -> BiomarkerResult:
    band = select_age_band(age_months)
    all_task = [s for s in samples if str(s.get("task_event") or "").startswith("moving_tracking:")]
    values = [s for s in all_task if s.get("valid") and s.get("gaze_x") is not None and s.get("target_x") is not None]
    if "moving_tracking" not in band.enabled_tasks:
        return _disabled("moving_tracking", band, "This protocol is not enabled for this age band.")
    if len(values) < 8:
        return _insufficient("moving_tracking", band)
    errors = [math.dist((s["gaze_x"], s["gaze_y"]), (s["target_x"], s["target_y"])) for s in values]
    weights = _duration_weights(values)
    valid_seconds = sum(weights)
    task_seconds = max(0.0, float(all_task[-1]["timestamp_seconds"]) - float(all_task[0]["timestamp_seconds"])) if len(all_task) > 1 else 0.0
    lag = _estimate_horizontal_lag(values)
    return BiomarkerResult("moving_tracking", band.key, "adequate" if valid_seconds >= 1.0 else "low",
                           {"target_following_accuracy": round(float(np.mean(np.asarray(errors) <= 0.12)), 4),
                            "mean_tracking_error": round(float(np.mean(errors)), 4), "estimated_lag_seconds": lag,
                            "dropout_ratio": _ratio(max(0.0, task_seconds - valid_seconds), task_seconds),
                            "valid_gaze_seconds": round(valid_seconds, 3)},
                           ("Lag is an exploratory cross-correlation estimate, not reaction time.",))


def social_gaze_following(samples: Sequence[dict[str, Any]], age_months: int) -> BiomarkerResult:
    band = select_age_band(age_months)
    all_task = [s for s in samples if str(s.get("task_event") or "").startswith("social_gaze:")]
    target_samples = [s for s in all_task if str(s.get("task_event") or "").startswith("social_gaze:target")]
    if "social_gaze_following" not in band.enabled_tasks:
        return _disabled("social_gaze_following", band, "This protocol is not enabled for this age band.")
    if len(target_samples) < 4:
        return _insufficient("social_gaze_following", band)
    trials: dict[str, list[dict[str, Any]]] = {}
    for sample in target_samples:
        trials.setdefault(str(sample["task_event"]), []).append(sample)
    latencies: list[float] = []
    successes = 0
    for trial in trials.values():
        onset = float(trial[0]["timestamp_seconds"])
        hit = next((s for s in trial if s.get("valid") and s.get("target_x") is not None and
                    math.dist((s["gaze_x"], s["gaze_y"]), (s["target_x"], s["target_y"])) <= 0.12), None)
        if hit is not None:
            successes += 1
            latencies.append(float(hit["timestamp_seconds"]) - onset)
    return BiomarkerResult("social_gaze_following", band.key, "adequate" if len(trials) >= 2 else "low",
                           {"gaze_following_success_rate": _ratio(successes, len(trials)),
                            "cue_to_target_latency_seconds": None if not latencies else round(float(np.median(latencies)), 3),
                            "successful_trials": float(successes), "total_trials": float(len(trials))},
                           ("A cue animation and stimulus AOIs must be validated with clinicians before interpreting this measure.",))


def build_biomarker_report(stream: dict[str, Any], age_months: int) -> dict[str, Any]:
    samples = stream.get("samples", [])
    return {"schema_version": 1, "age_months": age_months, "age_band": select_age_band(age_months).key,
            "clinical_interpretation": "Research task measures only; not an ASD, cognitive-delay, or developmental diagnosis.",
            "modules": [asdict(face_preference(samples, age_months)), asdict(moving_tracking(samples, age_months)),
                        asdict(social_gaze_following(samples, age_months))]}


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator <= 1e-6 else round(float(numerator / denominator), 4)


def _estimate_horizontal_lag(samples: Sequence[dict[str, Any]]) -> Optional[float]:
    if len(samples) < 12:
        return None
    gaze = np.asarray([s["gaze_x"] for s in samples], dtype=float)
    target = np.asarray([s["target_x"] for s in samples], dtype=float)
    if np.std(gaze) < 1e-6 or np.std(target) < 1e-6:
        return None
    dt = float(np.median(np.diff([s["timestamp_seconds"] for s in samples])))
    shifts = range(-min(20, len(samples) // 3), min(20, len(samples) // 3) + 1)
    correlations = [np.corrcoef(gaze[max(0, shift):len(gaze) + min(0, shift)],
                                target[max(0, -shift):len(target) - max(0, shift)])[0, 1] for shift in shifts]
    return round(float(list(shifts)[int(np.nanargmax(correlations))] * dt), 3)


def _insufficient(module: str, band: AgeBand) -> BiomarkerResult:
    return BiomarkerResult(module, band.key, "insufficient", {}, ("Insufficient valid task gaze data.",))


def _disabled(module: str, band: AgeBand, reason: str) -> BiomarkerResult:
    return BiomarkerResult(module, band.key, "not_applicable", {}, (reason,))
