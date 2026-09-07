"""Privacy-preserving engagement metrics for research screening sessions.

This module intentionally does not diagnose ASD, ADHD, developmental delay, or
any other condition. It summarizes derived gaze and tracking signals only;
clinical risk prediction requires a separately validated model and governance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ScreeningReport:
    schema_version: int
    created_at: str
    duration_seconds: float
    tracking_ratio: float
    valid_gaze_samples: int
    mean_gaze_speed_per_second: Optional[float]
    fixation_ratio: Optional[float]
    gaze_dispersion: Optional[float]
    gaze_entropy_bits: Optional[float]
    gaze_region_transitions: int
    longest_tracking_gap_seconds: float
    data_quality: str
    clinical_interpretation: str
    referral_recommendation: str


class ScreeningSession:
    """Collect derived engagement metrics without storing images or identity data."""

    MIN_DURATION_SECONDS = 20.0
    MIN_GAZE_SAMPLES = 120
    MIN_TRACKING_RATIO = 0.60
    FIXATION_SPEED_THRESHOLD = 0.35  # normalized-screen units per second

    def __init__(self) -> None:
        self._active = False
        self._started_at: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._observations: list[tuple[float, bool, Optional[float], Optional[float]]] = []

    @property
    def active(self) -> bool:
        return self._active

    def start(self, timestamp: float) -> None:
        self._active = True
        self._started_at = float(timestamp)
        self._last_timestamp = None
        self._observations.clear()

    def observe(
        self,
        timestamp: float,
        face_detected: bool,
        gaze_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        """Record one frame's derived state; raw video and facial landmarks are discarded."""
        if not self._active:
            return
        x = y = None
        if gaze_xy is not None:
            candidate_x, candidate_y = gaze_xy
            if math.isfinite(candidate_x) and math.isfinite(candidate_y):
                x = float(np.clip(candidate_x, 0.0, 1.0))
                y = float(np.clip(candidate_y, 0.0, 1.0))
        self._observations.append((float(timestamp), bool(face_detected), x, y))
        self._last_timestamp = float(timestamp)

    def stop(self, timestamp: float) -> ScreeningReport:
        if not self._active or self._started_at is None:
            raise RuntimeError("No screening session is active.")
        self._active = False
        end_time = max(float(timestamp), self._started_at)
        report = self._build_report(end_time)
        self._started_at = None
        return report

    def _build_report(self, end_time: float) -> ScreeningReport:
        assert self._started_at is not None
        duration = max(end_time - self._started_at, 0.0)
        if not self._observations:
            return self._empty_report(duration)

        tracked = np.array([observation[1] for observation in self._observations], dtype=bool)
        tracking_ratio = float(np.mean(tracked))
        timestamps = np.array([observation[0] for observation in self._observations], dtype=float)
        gaze = np.array(
            [(x, y) for _, _, x, y in self._observations if x is not None and y is not None],
            dtype=float,
        )
        longest_gap = self._longest_tracking_gap(timestamps, tracked, end_time)

        if len(gaze) == 0:
            report_data = asdict(self._empty_report(duration))
            report_data.update(
                tracking_ratio=tracking_ratio,
                longest_tracking_gap_seconds=longest_gap,
            )
            return ScreeningReport(**report_data)

        gaze_timestamps = np.array(
            [timestamp for timestamp, _, x, y in self._observations if x is not None and y is not None],
            dtype=float,
        )
        speeds = self._gaze_speeds(gaze, gaze_timestamps)
        mean_speed = float(np.mean(speeds)) if len(speeds) else None
        fixation_ratio = float(np.mean(speeds <= self.FIXATION_SPEED_THRESHOLD)) if len(speeds) else None
        dispersion = float(np.sqrt(np.mean(np.sum((gaze - np.mean(gaze, axis=0)) ** 2, axis=1))))
        entropy = self._spatial_entropy(gaze)
        transitions = self._region_transitions(gaze)
        quality = self._data_quality(duration, tracking_ratio, len(gaze))

        return ScreeningReport(
            schema_version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration, 3),
            tracking_ratio=round(tracking_ratio, 4),
            valid_gaze_samples=len(gaze),
            mean_gaze_speed_per_second=None if mean_speed is None else round(mean_speed, 4),
            fixation_ratio=None if fixation_ratio is None else round(fixation_ratio, 4),
            gaze_dispersion=round(dispersion, 4),
            gaze_entropy_bits=round(entropy, 4),
            gaze_region_transitions=transitions,
            longest_tracking_gap_seconds=round(longest_gap, 3),
            data_quality=quality,
            clinical_interpretation=(
                "Research engagement metrics only; not a diagnosis or clinical risk score."
            ),
            referral_recommendation=(
                "No automated referral recommendation is available because no clinically validated "
                "model is installed. Consult a qualified clinician for developmental concerns."
            ),
        )

    def _empty_report(self, duration: float) -> ScreeningReport:
        return ScreeningReport(
            schema_version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration, 3),
            tracking_ratio=0.0,
            valid_gaze_samples=0,
            mean_gaze_speed_per_second=None,
            fixation_ratio=None,
            gaze_dispersion=None,
            gaze_entropy_bits=None,
            gaze_region_transitions=0,
            longest_tracking_gap_seconds=round(duration, 3),
            data_quality="insufficient",
            clinical_interpretation="No usable gaze data was recorded; not clinically interpretable.",
            referral_recommendation="No automated referral recommendation is available.",
        )

    def _data_quality(self, duration: float, tracking_ratio: float, sample_count: int) -> str:
        if duration < self.MIN_DURATION_SECONDS or sample_count < self.MIN_GAZE_SAMPLES:
            return "insufficient"
        if tracking_ratio < self.MIN_TRACKING_RATIO:
            return "low"
        return "adequate"

    @staticmethod
    def _gaze_speeds(gaze: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        if len(gaze) < 2:
            return np.array([], dtype=float)
        elapsed = np.diff(timestamps)
        distances = np.linalg.norm(np.diff(gaze, axis=0), axis=1)
        valid = elapsed > 1e-6
        return distances[valid] / elapsed[valid]

    @staticmethod
    def _spatial_entropy(gaze: np.ndarray) -> float:
        grid, _, _ = np.histogram2d(gaze[:, 0], gaze[:, 1], bins=3, range=[[0, 1], [0, 1]])
        probabilities = grid.ravel() / len(gaze)
        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def _region_transitions(gaze: np.ndarray) -> int:
        cells = np.minimum((gaze * 3).astype(int), 2)
        regions = cells[:, 1] * 3 + cells[:, 0]
        return int(np.sum(np.diff(regions) != 0)) if len(regions) > 1 else 0

    @staticmethod
    def _longest_tracking_gap(timestamps: np.ndarray, tracked: np.ndarray, end_time: float) -> float:
        if len(timestamps) == 0:
            return 0.0
        longest = current_start = None
        for timestamp, is_tracked in zip(timestamps, tracked):
            if not is_tracked and current_start is None:
                current_start = timestamp
            elif is_tracked and current_start is not None:
                gap = timestamp - current_start
                longest = gap if longest is None else max(longest, gap)
                current_start = None
        if current_start is not None:
            gap = end_time - current_start
            longest = gap if longest is None else max(longest, gap)
        return float(longest or 0.0)


def save_report(report: ScreeningReport, destination: str) -> None:
    """Atomically save a derived-metrics report, never raw frames or identifiers."""
    temp_destination = f"{destination}.tmp"
    with open(temp_destination, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=2)
    os.replace(temp_destination, destination)
