"""Standardized visual-attention task with derived, non-diagnostic metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AttentionTrial:
    trial_id: str
    target_x: float
    target_y: float
    duration_seconds: float


@dataclass(frozen=True)
class AttentionTrialResult:
    trial_id: str
    target_x: float
    target_y: float
    valid_gaze_seconds: float
    on_target_seconds: float
    on_target_ratio: Optional[float]
    first_target_latency_seconds: Optional[float]


class VisualAttentionTask:
    """A local target-following task that stores only derived trial summaries."""

    TARGET_RADIUS = 0.12  # normalized-screen distance

    def __init__(self, trials: list[AttentionTrial]) -> None:
        if not trials:
            raise ValueError("At least one attention trial is required.")
        self.trials = trials
        self.active = False
        self.completed = False
        self._index = 0
        self._trial_started_at: Optional[float] = None
        self._last_observed_at: Optional[float] = None
        self._valid_seconds = 0.0
        self._on_target_seconds = 0.0
        self._first_target_at: Optional[float] = None
        self._results: list[AttentionTrialResult] = []

    @classmethod
    def default(cls) -> "VisualAttentionTask":
        # Center first, then symmetric edge/corner targets in a predictable order.
        locations = [
            ("center", 0.50, 0.50), ("upper_left", 0.20, 0.20),
            ("upper_right", 0.80, 0.20), ("lower_right", 0.80, 0.80),
            ("lower_left", 0.20, 0.80), ("left", 0.15, 0.50),
            ("right", 0.85, 0.50), ("top", 0.50, 0.15), ("bottom", 0.50, 0.85),
        ]
        return cls([AttentionTrial(name, x, y, 3.0) for name, x, y in locations])

    @property
    def current_trial(self) -> Optional[AttentionTrial]:
        return self.trials[self._index] if self.active else None

    @property
    def results(self) -> list[AttentionTrialResult]:
        return list(self._results)

    def start(self, timestamp: float) -> None:
        self.active = True
        self.completed = False
        self._index = 0
        self._results.clear()
        self._begin_trial(float(timestamp))

    def observe(self, timestamp: float, gaze_xy: Optional[tuple[float, float]]) -> bool:
        """Record a gaze observation. Returns true only when the task completes."""
        if not self.active or self._trial_started_at is None:
            return False
        now = float(timestamp)
        if self._last_observed_at is not None:
            elapsed = max(0.0, min(now - self._last_observed_at, 0.25))
            if gaze_xy is not None:
                self._valid_seconds += elapsed
                if self._is_on_target(gaze_xy):
                    self._on_target_seconds += elapsed
                    if self._first_target_at is None:
                        self._first_target_at = now
        self._last_observed_at = now

        if now - self._trial_started_at >= self.trials[self._index].duration_seconds:
            self._finish_current_trial(now)
            if self._index >= len(self.trials):
                self.active = False
                self.completed = True
                return True
            self._begin_trial(now)
        return False

    def report(self) -> dict:
        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": "standardized_visual_attention_v1",
            "completed": self.completed,
            "clinical_interpretation": "Research engagement task only; not a diagnosis or clinical risk score.",
            "trials": [asdict(result) for result in self._results],
        }

    def _begin_trial(self, timestamp: float) -> None:
        self._trial_started_at = timestamp
        self._last_observed_at = None
        self._valid_seconds = 0.0
        self._on_target_seconds = 0.0
        self._first_target_at = None

    def _finish_current_trial(self, timestamp: float) -> None:
        trial = self.trials[self._index]
        latency = None if self._first_target_at is None else self._first_target_at - self._trial_started_at
        ratio = None if self._valid_seconds <= 0 else self._on_target_seconds / self._valid_seconds
        self._results.append(AttentionTrialResult(
            trial_id=trial.trial_id,
            target_x=trial.target_x,
            target_y=trial.target_y,
            valid_gaze_seconds=round(self._valid_seconds, 3),
            on_target_seconds=round(self._on_target_seconds, 3),
            on_target_ratio=None if ratio is None else round(ratio, 4),
            first_target_latency_seconds=None if latency is None else round(latency, 3),
        ))
        self._index += 1

    def _is_on_target(self, gaze_xy: tuple[float, float]) -> bool:
        trial = self.trials[self._index]
        gaze = np.asarray(gaze_xy, dtype=float)
        return bool(np.linalg.norm(gaze - np.array([trial.target_x, trial.target_y])) <= self.TARGET_RADIUS)


def save_task_report(report: dict, destination: str) -> None:
    temp_destination = f"{destination}.tmp"
    with open(temp_destination, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    os.replace(temp_destination, destination)
