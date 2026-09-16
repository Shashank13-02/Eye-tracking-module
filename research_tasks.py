"""Simple, age-banded visual research tasks with explicit task-event metadata."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from age_biomarkers import AgeBand


@dataclass(frozen=True)
class ActiveStimulus:
    task_event: str
    target_xy: Optional[tuple[float, float]]
    title: str


class AgeResearchTask:
    def __init__(self, module: str, band: AgeBand, started_at: float) -> None:
        if module not in band.enabled_tasks:
            raise ValueError(f"{module} is not enabled for {band.key}.")
        self.module, self.band, self.started_at = module, band, started_at
        self.finished = False

    @property
    def total_duration(self) -> float:
        return self.band.task_duration_seconds * (4 if self.module != "moving_tracking" else 5)

    def stimulus(self, timestamp: float) -> ActiveStimulus:
        elapsed = max(0.0, timestamp - self.started_at)
        if elapsed >= self.total_duration:
            self.finished = True
        trial = min(int(elapsed / self.band.task_duration_seconds), 3 if self.module != "moving_tracking" else 0)
        local = elapsed % self.band.task_duration_seconds
        if self.module == "face_preference":
            face_left = trial % 2 == 0
            return ActiveStimulus(f"face_preference:trial_{trial + 1}:{'face_left' if face_left else 'face_right'}", None,
                                  "Look naturally at either picture")
        if self.module == "moving_tracking":
            phase = elapsed * (2 * math.pi / max(self.total_duration, 1.0))
            target = (0.5 + 0.32 * math.sin(phase), 0.5 + 0.18 * math.sin(phase * 2))
            return ActiveStimulus("moving_tracking:trial_1", target, "Follow the moving object")
        target_left = trial % 2 == 0
        if local < 1.0:
            return ActiveStimulus(f"social_gaze:cue:trial_{trial + 1}", None, "Watch the social cue")
        target = (0.22 if target_left else 0.78, 0.52)
        return ActiveStimulus(f"social_gaze:target:trial_{trial + 1}", target, "Look where the cue points")

    def render(self, width: int, height: int, timestamp: float) -> np.ndarray:
        stimulus = self.stimulus(timestamp)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, f"{self.band.key} | {stimulus.title} | Esc cancels", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 45, 45), 2, cv2.LINE_AA)
        if self.module == "face_preference":
            face_left = stimulus.task_event.endswith("face_left")
            self._face(canvas, (int(width * (.25 if face_left else .75)), height // 2))
            self._non_social(canvas, (int(width * (.75 if face_left else .25)), height // 2))
        elif self.module == "moving_tracking":
            x, y = stimulus.target_xy
            cv2.circle(canvas, (int(x * width), int(y * height)), max(18, width // 35), (40, 120, 255), -1)
        else:
            self._face(canvas, (width // 2, height // 2))
            if stimulus.target_xy is not None:
                x, y = stimulus.target_xy
                cv2.arrowedLine(canvas, (width // 2, height // 2), (int(x * width), int(y * height)), (60, 60, 60), 6)
                cv2.circle(canvas, (int(x * width), int(y * height)), max(18, width // 35), (40, 180, 70), -1)
        return canvas

    @staticmethod
    def _face(canvas: np.ndarray, center: tuple[int, int]) -> None:
        cv2.circle(canvas, center, 85, (185, 220, 255), -1)
        for dx in (-30, 30):
            cv2.circle(canvas, (center[0] + dx, center[1] - 22), 8, (40, 40, 40), -1)
        cv2.ellipse(canvas, (center[0], center[1] + 28), (32, 17), 0, 0, 180, (40, 40, 40), 4)

    @staticmethod
    def _non_social(canvas: np.ndarray, center: tuple[int, int]) -> None:
        for radius, color in ((82, (255, 190, 70)), (58, (80, 90, 240)), (32, (70, 220, 100))):
            cv2.circle(canvas, center, radius, color, -1)
