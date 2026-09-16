"""Frame-level, non-diagnostic camera and landmark quality checks.

The checks in this module decide whether a frame is suitable for gaze *measurement*.
They do not infer child behaviour, cognition, or a health condition.  FaceMesh does
not expose a detector confidence score, so that field is deliberately ``None``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    face_detected: bool
    face_confidence: Optional[float]
    face_confidence_available: bool
    left_eye_visible: bool
    right_eye_visible: bool
    blink: bool
    head_yaw_deg: Optional[float]
    head_pitch_deg: Optional[float]
    head_roll_deg: Optional[float]
    face_size_ratio: Optional[float]
    distance_ratio: Optional[float]
    illumination: Optional[float]
    landmark_jitter: Optional[float]
    calibration_age_seconds: Optional[float]
    usable_for_gaze: bool
    flags: tuple[str, ...]


class TrackingQualityMonitor:
    """Evaluate quality from MediaPipe FaceMesh landmarks without retaining frames."""

    MIN_FACE_WIDTH_RATIO = 0.18
    MAX_FACE_WIDTH_RATIO = 0.80
    MIN_ILLUMINATION = 35.0
    MAX_ILLUMINATION = 220.0
    MIN_EYE_APERTURE = 0.035
    MAX_JITTER = 0.12  # iris movement / inter-pupil distance, rolling median
    MAX_HEAD_YAW = 30.0
    MAX_HEAD_PITCH = 25.0
    MAX_HEAD_ROLL = 20.0
    MAX_CALIBRATION_AGE_SECONDS = 10 * 60
    MAX_DISTANCE_CHANGE_RATIO = 0.15
    MAX_HEAD_MOVEMENT_FROM_CALIBRATION_DEG = 12.0
    PROXY_POSE_HISTORY = 5
    PROXY_POSE_CONFIRMATION_FRAMES = 3

    # FaceMesh landmark indices.  These are used only for geometric quality proxies.
    LEFT_OUTER, LEFT_INNER, LEFT_UPPER, LEFT_LOWER = 33, 133, 159, 145
    RIGHT_OUTER, RIGHT_INNER, RIGHT_UPPER, RIGHT_LOWER = 263, 362, 386, 374
    NOSE, MOUTH_UPPER, MOUTH_LOWER = 1, 13, 14
    LEFT_IRIS, RIGHT_IRIS = 468, 473

    def __init__(self) -> None:
        self._iris_history: deque[np.ndarray] = deque(maxlen=8)
        self._calibrated_at: Optional[float] = None
        self._calibration_face_size_ratio: Optional[float] = None
        self._calibration_head_pose_deg: Optional[tuple[float, float, float]] = None
        self._proxy_pose_history: deque[tuple[float, float, float]] = deque(maxlen=self.PROXY_POSE_HISTORY)
        self._proxy_pose_excessive_frames = {"yaw": 0, "pitch": 0, "roll": 0}

    def mark_calibrated(self, timestamp: float, face_size_ratio: Optional[float] = None,
                        head_pose_deg: Optional[tuple[float, float, float]] = None) -> None:
        self._calibrated_at = float(timestamp)
        self._calibration_face_size_ratio = face_size_ratio
        self._calibration_head_pose_deg = head_pose_deg
        self._iris_history.clear()
        self._proxy_pose_history.clear()
        self._proxy_pose_excessive_frames = {"yaw": 0, "pitch": 0, "roll": 0}

    def evaluate(
        self, frame_bgr: np.ndarray, landmarks: Optional[Sequence[object]], timestamp: float,
        head_pose_deg: Optional[tuple[float, float, float]] = None,
    ) -> FrameQuality:
        if not landmarks:
            return self._missing_face(timestamp)

        h, w = frame_bgr.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=float)
        face_min, face_max = pts.min(axis=0), pts.max(axis=0)
        face_width = float(face_max[0] - face_min[0])
        face_ratio = face_width / max(float(w), 1.0)
        face_ok = self.MIN_FACE_WIDTH_RATIO <= face_ratio <= self.MAX_FACE_WIDTH_RATIO

        left_visible, left_aperture = self._eye_visible(pts, self.LEFT_OUTER, self.LEFT_INNER,
                                                        self.LEFT_UPPER, self.LEFT_LOWER)
        right_visible, right_aperture = self._eye_visible(pts, self.RIGHT_OUTER, self.RIGHT_INNER,
                                                           self.RIGHT_UPPER, self.RIGHT_LOWER)
        blink = left_aperture < self.MIN_EYE_APERTURE and right_aperture < self.MIN_EYE_APERTURE

        using_proxy_pose = head_pose_deg is None
        if using_proxy_pose:
            yaw, pitch, roll = self._stabilized_head_pose_proxy(pts)
        else:
            yaw, pitch, roll = head_pose_deg
            self._proxy_pose_history.clear()
            self._proxy_pose_excessive_frames = {"yaw": 0, "pitch": 0, "roll": 0}
        illumination = self._face_illumination(frame_bgr, face_min, face_max)
        jitter = self._update_jitter(pts)
        calibration_age = None if self._calibrated_at is None else max(0.0, timestamp - self._calibrated_at)
        distance_change = None
        if self._calibration_face_size_ratio and self._calibration_face_size_ratio > 1e-6:
            distance_change = abs(face_ratio / self._calibration_face_size_ratio - 1.0)
        head_change = None
        if self._calibration_head_pose_deg is not None:
            head_change = max(abs(a - b) for a, b in zip((yaw, pitch, roll), self._calibration_head_pose_deg))

        flags: list[str] = []
        if not face_ok:
            flags.append("face_distance_out_of_range")
        if not left_visible:
            flags.append("left_eye_not_visible")
        if not right_visible:
            flags.append("right_eye_not_visible")
        if blink:
            flags.append("blink")
        if illumination < self.MIN_ILLUMINATION:
            flags.append("illumination_too_dark")
        elif illumination > self.MAX_ILLUMINATION:
            flags.append("illumination_too_bright")
        if self._pose_excessive("yaw", abs(yaw) > self.MAX_HEAD_YAW, using_proxy_pose):
            flags.append("head_yaw_excessive")
        if self._pose_excessive("pitch", abs(pitch) > self.MAX_HEAD_PITCH, using_proxy_pose):
            flags.append("head_pitch_excessive")
        if self._pose_excessive("roll", abs(roll) > self.MAX_HEAD_ROLL, using_proxy_pose):
            flags.append("head_roll_excessive")
        if jitter is not None and jitter > self.MAX_JITTER:
            flags.append("landmark_jitter_high")
        if distance_change is not None and distance_change > self.MAX_DISTANCE_CHANGE_RATIO:
            flags.append("distance_changed_since_calibration")
        if head_change is not None and head_change > self.MAX_HEAD_MOVEMENT_FROM_CALIBRATION_DEG:
            flags.append("head_moved_since_calibration")
        if calibration_age is None:
            flags.append("not_eye_calibrated")
        elif calibration_age > self.MAX_CALIBRATION_AGE_SECONDS:
            flags.append("calibration_stale")

        # Calibration uses the same physical-quality gates but is allowed before C locks
        # eye spheres.  The caller separately enforces eye-sphere lock when appropriate.
        blocking = {flag for flag in flags if flag != "not_eye_calibrated"}
        return FrameQuality(
            face_detected=True, face_confidence=None, face_confidence_available=False,
            left_eye_visible=left_visible, right_eye_visible=right_visible, blink=blink,
            head_yaw_deg=round(yaw, 2), head_pitch_deg=round(pitch, 2), head_roll_deg=round(roll, 2),
            face_size_ratio=round(face_ratio, 4), distance_ratio=round(face_ratio, 4),
            illumination=round(illumination, 1), landmark_jitter=None if jitter is None else round(jitter, 4),
            calibration_age_seconds=None if calibration_age is None else round(calibration_age, 1),
            usable_for_gaze=not blocking, flags=tuple(flags),
        )

    def _missing_face(self, timestamp: float) -> FrameQuality:
        self._iris_history.clear()
        self._proxy_pose_history.clear()
        self._proxy_pose_excessive_frames = {"yaw": 0, "pitch": 0, "roll": 0}
        age = None if self._calibrated_at is None else max(0.0, timestamp - self._calibrated_at)
        return FrameQuality(False, None, False, False, False, False, None, None, None,
                            None, None, None, None, age, False, ("no_face",))

    @staticmethod
    def _eye_visible(pts: np.ndarray, outer: int, inner: int, upper: int, lower: int) -> tuple[bool, float]:
        width = float(np.linalg.norm(pts[outer] - pts[inner]))
        aperture = float(np.linalg.norm(pts[upper] - pts[lower]) / max(width, 1e-6))
        on_screen = all(np.all(np.isfinite(pts[i])) for i in (outer, inner, upper, lower))
        return bool(on_screen and width > 3.0 and aperture >= TrackingQualityMonitor.MIN_EYE_APERTURE), aperture

    @staticmethod
    def _face_illumination(frame: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        h, w = frame.shape[:2]
        x0, y0 = np.maximum(lower.astype(int), 0)
        x1, y1 = np.minimum(upper.astype(int), [w - 1, h - 1])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        gray = cv2.cvtColor(frame[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2GRAY)
        return float(np.median(gray))

    def _update_jitter(self, pts: np.ndarray) -> Optional[float]:
        iris_mid = (pts[self.LEFT_IRIS] + pts[self.RIGHT_IRIS]) / 2.0
        pupil_distance = float(np.linalg.norm(pts[self.LEFT_IRIS] - pts[self.RIGHT_IRIS]))
        if pupil_distance < 1e-6:
            return None
        self._iris_history.append(iris_mid)
        if len(self._iris_history) < 3:
            return None
        history = np.asarray(self._iris_history)
        return float(np.median(np.linalg.norm(history - np.median(history, axis=0), axis=1)) / pupil_distance)

    @staticmethod
    def _head_pose_proxy(pts: np.ndarray) -> tuple[float, float, float]:
        """Return image-geometry pose proxies, not camera-calibrated 3D pose."""
        left, right, nose = pts[33], pts[263], pts[1]
        eye_mid = (left + right) / 2.0
        eye_span = max(float(np.linalg.norm(right - left)), 1e-6)
        # Nose offset from eye midpoint is a stable, deliberately conservative proxy.
        yaw = float(np.clip((nose[0] - eye_mid[0]) / (eye_span * 0.35) * 30.0, -60.0, 60.0))
        mouth_mid = (pts[13] + pts[14]) / 2.0
        vertical_span = max(float(mouth_mid[1] - eye_mid[1]), 1e-6)
        pitch = float(np.clip(((nose[1] - eye_mid[1]) / vertical_span - 0.45) * 55.0, -60.0, 60.0))
        roll = float(np.degrees(math.atan2(right[1] - left[1], right[0] - left[0])))
        return yaw, pitch, roll

    def _stabilized_head_pose_proxy(self, pts: np.ndarray) -> tuple[float, float, float]:
        """Smooth image-landmark pose proxies without changing calibrated pose."""
        self._proxy_pose_history.append(self._head_pose_proxy(pts))
        pose = np.asarray(self._proxy_pose_history, dtype=float)
        return tuple(float(value) for value in np.median(pose, axis=0))

    def _pose_excessive(self, axis: str, excessive: bool, using_proxy_pose: bool) -> bool:
        """Require persistence before a non-calibrated image proxy pauses gaze."""
        if not using_proxy_pose:
            return excessive
        self._proxy_pose_excessive_frames[axis] = (
            self._proxy_pose_excessive_frames[axis] + 1 if excessive else 0
        )
        return self._proxy_pose_excessive_frames[axis] >= self.PROXY_POSE_CONFIRMATION_FRAMES


def summarize_quality_flags(flags: Sequence[str]) -> dict[str, int]:
    return dict(Counter(flags))
