"""Camera, head-pose, eye-geometry, and screen-setup support for NeuroGaze.

This module is deliberately measurement-oriented.  It never labels behaviour or
health status.  A missing camera profile uses an explicitly marked approximation;
it is not silently presented as calibrated geometry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    image_width: int
    image_height: int
    camera_matrix: list[list[float]]
    distortion: list[float]
    reprojection_error_px: float
    created_at: str


@dataclass(frozen=True)
class ScreenGeometry:
    screen_width_mm: float
    screen_height_mm: float
    camera_to_screen_mm: float
    camera_offset_x_mm: float
    camera_offset_y_mm: float
    created_at: str


@dataclass(frozen=True)
class EyeGeometry:
    center: np.ndarray
    radius: float
    visible: bool
    iris_radius_residual: float


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    reprojection_error_px: float
    calibrated_intrinsics: bool
    rotation_matrix: np.ndarray


def save_json_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(tmp, path)


def load_camera_intrinsics(path: str, width: int, height: int) -> Optional[CameraIntrinsics]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if (data["image_width"], data["image_height"]) != (width, height):
            return None
        matrix = np.asarray(data["camera_matrix"], dtype=float)
        distortion = np.asarray(data["distortion"], dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
            return None
        return CameraIntrinsics(**data)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_camera_intrinsics(path: str, matrix: np.ndarray, distortion: np.ndarray, width: int, height: int,
                           reprojection_error_px: float) -> CameraIntrinsics:
    profile = CameraIntrinsics(width, height, matrix.tolist(), distortion.ravel().tolist(),
                               float(reprojection_error_px), datetime.now(timezone.utc).isoformat())
    save_json_atomic(path, asdict(profile))
    return profile


def load_screen_geometry(path: str) -> Optional[ScreenGeometry]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        geometry = ScreenGeometry(**data)
        if min(geometry.screen_width_mm, geometry.screen_height_mm, geometry.camera_to_screen_mm) <= 0:
            return None
        return geometry
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_screen_geometry(path: str, width_mm: float, height_mm: float, camera_to_screen_mm: float,
                         offset_x_mm: float = 0.0, offset_y_mm: float = 0.0) -> ScreenGeometry:
    geometry = ScreenGeometry(float(width_mm), float(height_mm), float(camera_to_screen_mm),
                              float(offset_x_mm), float(offset_y_mm), datetime.now(timezone.utc).isoformat())
    save_json_atomic(path, asdict(geometry))
    return geometry


class FaceGeometryEstimator:
    """Estimate head pose and eye centres from FaceMesh geometry.

    Eye centres are inferred from the canthi midpoint, eyelid aperture, iris
    location, and head-forward direction. They are more person/pose responsive
    than a fixed pixel radius but remain webcam estimates, not anatomical truth.
    """

    # A conventional sparse face model in millimetres for solvePnP.
    POSE_INDICES = (1, 152, 33, 263, 61, 291)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    def __init__(self, intrinsics: Optional[CameraIntrinsics], width: int, height: int) -> None:
        self.width, self.height = width, height
        self.intrinsics = intrinsics

    def _camera_parameters(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.intrinsics is not None:
            return (np.asarray(self.intrinsics.camera_matrix, dtype=np.float64),
                    np.asarray(self.intrinsics.distortion, dtype=np.float64), True)
        focal = float(max(self.width, self.height))
        return np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]], dtype=np.float64), np.zeros(5), False

    def estimate_head_pose(self, landmarks: Sequence[object]) -> Optional[HeadPose]:
        if len(landmarks) <= max(self.POSE_INDICES):
            return None
        image_points = np.array([(landmarks[i].x * self.width, landmarks[i].y * self.height)
                                 for i in self.POSE_INDICES], dtype=np.float64)
        matrix, distortion, calibrated = self._camera_parameters()
        ok, rvec, tvec = cv2.solvePnP(self.MODEL_POINTS, image_points, matrix, distortion,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(self.MODEL_POINTS, rvec, tvec, matrix, distortion)
        error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
        # OpenCV decomposition supplies a stable camera-coordinate Euler representation.
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation)
        pitch, yaw, roll = (float(angles[0]), float(angles[1]), float(angles[2]))
        return HeadPose(yaw, pitch, roll, error, calibrated, rotation)

    def estimate_eyes(self, landmarks: Sequence[object], pose: Optional[HeadPose]) -> tuple[EyeGeometry, EyeGeometry]:
        points = np.array([(lm.x * self.width, lm.y * self.height, lm.z * self.width) for lm in landmarks], dtype=float)
        forward = np.array([0.0, 0.0, 1.0]) if pose is None else pose.rotation_matrix[:, 2]
        if forward[2] < 0:
            forward = -forward
        forward /= max(np.linalg.norm(forward), 1e-9)
        return (self._eye(points, 33, 133, 159, 145, 468, forward),
                self._eye(points, 263, 362, 386, 374, 473, forward))

    @staticmethod
    def _eye(points: np.ndarray, outer: int, inner: int, upper: int, lower: int, iris: int,
             forward: np.ndarray) -> EyeGeometry:
        canthus_mid = (points[outer] + points[inner]) / 2.0
        eye_width = float(np.linalg.norm(points[outer] - points[inner]))
        aperture = float(np.linalg.norm(points[upper] - points[lower]))
        radius = max(eye_width * 0.48, aperture * 1.6, 1.0)
        # The centre lies behind the visible iris, in the head-forward direction.
        center = canthus_mid + forward * radius * 0.55
        iris_distance = float(np.linalg.norm(points[iris] - center))
        residual = abs(iris_distance - radius) / radius
        visible = bool(np.all(np.isfinite(center)) and eye_width > 3.0 and aperture / eye_width > 0.035)
        return EyeGeometry(center=center, radius=radius, visible=visible, iris_radius_residual=residual)


def calibrate_camera_from_chessboards(image_points: list[np.ndarray], board_cols: int, board_rows: int,
                                      square_size_mm: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate an OpenCV camera from checkerboard corner observations."""
    if len(image_points) < 12:
        raise ValueError("At least 12 varied checkerboard views are required.")
    object_points = np.zeros((board_rows * board_cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_mm
    rms, matrix, distortion, _, _ = cv2.calibrateCamera([object_points] * len(image_points), image_points,
                                                          (width, height), None, None)
    return matrix, distortion, float(rms)
