"""Calibration validation and evaluation engine for NeuroGaze eye tracking.

Provides independent validation targets, robust multi-metric error analysis,
spatial error mapping, prototype engineering quality gates, and profile serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Thirteen training targets: the original centre/corner/edge grid plus four
# intermediate upper/lower points.  The added points make the vertical mapping
# substantially less dependent on extrapolation, which is the weakest axis for
# a webcam-only iris estimate.
CALIBRATION_TARGETS: list[tuple[float, float]] = [
    (0.50, 0.50),  # Center
    (0.10, 0.10),  # Top-Left
    (0.90, 0.10),  # Top-Right
    (0.90, 0.90),  # Bottom-Right
    (0.10, 0.90),  # Bottom-Left
    (0.50, 0.10),  # Top-Center
    (0.90, 0.50),  # Right-Center
    (0.50, 0.90),  # Bottom-Center
    (0.10, 0.50),  # Left-Center
    (0.30, 0.20),  # Upper-left intermediate
    (0.70, 0.20),  # Upper-right intermediate
    (0.70, 0.80),  # Lower-right intermediate
    (0.30, 0.80),  # Lower-left intermediate
]

# Six independent validation targets, deliberately disjoint from training.  The
# upper and lower centre checks expose vertical bias that quadrant-only tests can
# otherwise conceal.
VALIDATION_TARGETS: list[tuple[float, float]] = [
    (0.30, 0.35),  # Upper-left quadrant
    (0.70, 0.35),  # Upper-right quadrant
    (0.70, 0.65),  # Lower-right quadrant
    (0.30, 0.65),  # Lower-left quadrant
    (0.50, 0.25),  # Upper-centre interpolation check
    (0.50, 0.75),  # Lower-centre interpolation check
]

# Prototype engineering thresholds (in pixels)
# NOTE: These are engineering prototype quality gates, not clinical accuracy thresholds.
QUALITY_THRESHOLDS = {
    "EXCELLENT": 60.0,
    "GOOD": 110.0,
    "FAIR": 180.0,
}


@dataclass(frozen=True)
class SpatialTargetError:
    target_index: int
    target_norm: tuple[float, float]
    target_px: tuple[float, float]
    predicted_px: tuple[float, float]
    error_px: float
    dx_px: float
    dy_px: float


@dataclass(frozen=True)
class ValidationMetrics:
    mae_px: float
    median_px: float
    rmse_px: float
    p95_px: float

    horizontal_mae_px: float
    horizontal_median_px: float

    vertical_mae_px: float
    vertical_median_px: float

    normalized_median_error: float
    angular_median_error_deg: float

    status: str
    num_targets: int
    num_samples: int
    spatial_errors: list[SpatialTargetError]


@dataclass(frozen=True)
class CalibrationModelSelection:
    """Training-only model-selection evidence; validation targets remain untouched."""
    selected_ridge_alpha: float
    leave_one_target_out_median_error_norm: float
    candidate_median_errors_norm: dict[str, float]


def vector_to_yaw_pitch_deg(direction_vector: np.ndarray) -> tuple[float, float]:
    """Convert a 3D gaze direction vector to continuous signed yaw and pitch in degrees.

    Coordinate conventions:
      Forward is -Z (towards screen).
      Right is +X -> positive yaw.
      Left is -X  -> negative yaw.
      Up is -Y    -> positive pitch.
      Down is +Y  -> negative pitch.
    """
    v = np.asarray(direction_vector, dtype=float).ravel()
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return 0.0, 0.0
    x, y, z = v[0] / norm, v[1] / norm, v[2] / norm

    # Continuous yaw: rotation around Y-axis from -Z reference
    yaw_rad = math.atan2(x, -z)

    # Continuous pitch: rotation around X-axis (upwards is positive, so -y)
    xz_plane_dist = math.sqrt(x * x + z * z)
    pitch_rad = math.atan2(-y, xz_plane_dist)

    return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))


def compute_calibration_features(yaw_deg: float, pitch_deg: float, degree: int = 2) -> np.ndarray:
    """Compute polynomial gaze-to-screen features for a per-user calibration.

    Degree two remains supported for legacy profiles.  New calibrations use the
    degree-three representation: its extra pitch/cross terms let a user's
    vertical gaze response bend independently at the top and bottom of screen.
    """
    y = float(yaw_deg)
    p = float(pitch_deg)
    if degree == 2:
        return np.array([y, p, y * y, y * p, p * p, 1.0], dtype=float)
    if degree == 3:
        return np.array([
            y, p, y * y, y * p, p * p,
            y * y * y, y * y * p, y * p * p, p * p * p,
            1.0,
        ], dtype=float)
    raise ValueError(f"Unsupported calibration polynomial degree: {degree}")


def fit_gaze_model(
    training_observations: list[tuple[float, float, tuple[float, float]]]
) -> tuple[np.ndarray, float]:
    """Fit 2nd-order polynomial regression model on training observations only.

    training_observations: list of (mean_yaw, mean_pitch, (target_x_norm, target_y_norm))
    Returns (model_weights, training_rmse_px).
    """
    if len(training_observations) < 6:
        raise ValueError(f"Need at least 6 training observations, got {len(training_observations)}")

    features = np.vstack([
        compute_calibration_features(yaw, pitch)
        for yaw, pitch, _ in training_observations
    ])
    targets = np.vstack([
        np.asarray(target, dtype=float)
        for _, _, target in training_observations
    ])

    model, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
    pred = features @ model
    residuals_norm = pred - targets
    training_rmse_norm = float(np.sqrt(np.mean(residuals_norm ** 2)))
    return model, training_rmse_norm


def fit_regularized_gaze_model(
    training_observations: list[tuple[float, float, tuple[float, float]]], ridge_alpha: float,
    degree: int = 3,
) -> tuple[np.ndarray, float]:
    """Fit a standardized, regularized per-user polynomial gaze mapping.

    The intercept is never regularized.  Degree three needs at least ten target
    observations; degree two is retained for legacy-profile compatibility.
    """
    feature_count = 10 if degree == 3 else 6 if degree == 2 else 0
    if feature_count == 0:
        raise ValueError(f"Unsupported calibration polynomial degree: {degree}")
    if len(training_observations) < feature_count:
        raise ValueError(f"Need at least {feature_count} training observations, got {len(training_observations)}")
    alpha = max(0.0, float(ridge_alpha))
    features = np.vstack([compute_calibration_features(yaw, pitch, degree) for yaw, pitch, _ in training_observations])
    targets = np.vstack([np.asarray(target, dtype=float) for _, _, target in training_observations])
    means = np.mean(features[:, :-1], axis=0)
    scales = np.std(features[:, :-1], axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    standardized = np.column_stack(((features[:, :-1] - means) / scales, np.ones(len(features))))
    penalty = np.diag([alpha] * (feature_count - 1) + [0.0])
    system = standardized.T @ standardized + penalty
    rhs = standardized.T @ targets
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        # Degenerate training layouts can occur when a target is left out during
        # cross-validation. Preserve a deterministic least-squares fallback.
        coefficients, _, _, _ = np.linalg.lstsq(standardized, targets, rcond=None)
    model = np.empty((feature_count, 2), dtype=float)
    model[:-1] = coefficients[:-1] / scales[:, None]
    model[-1] = coefficients[-1] - (means / scales) @ coefficients[:-1]
    residuals = features @ model - targets
    return model, float(np.sqrt(np.mean(residuals ** 2)))


def select_regularized_gaze_model(
    training_observations: list[tuple[float, float, tuple[float, float]]],
    ridge_alphas: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0),
    degree: int = 3,
) -> tuple[np.ndarray, float, CalibrationModelSelection]:
    """Choose a regularized model using leave-one-training-target-out error only.

    This deliberately does not inspect the independent five validation targets.
    """
    min_targets = 11 if degree == 3 else 7 if degree == 2 else 0
    if len(training_observations) < min_targets:
        raise ValueError(f"Need at least {min_targets} training targets for model selection.")
    candidate_scores: dict[str, float] = {}
    for alpha in ridge_alphas:
        held_out_errors: list[float] = []
        for index, held_out in enumerate(training_observations):
            subset = training_observations[:index] + training_observations[index + 1:]
            model, _ = fit_regularized_gaze_model(subset, alpha, degree)
            prediction = compute_calibration_features(held_out[0], held_out[1], degree) @ model
            held_out_errors.append(float(np.linalg.norm(prediction - np.asarray(held_out[2], dtype=float))))
        candidate_scores[f"ridge_{alpha:g}"] = float(np.median(held_out_errors))
    selected_alpha = min(ridge_alphas, key=lambda alpha: candidate_scores[f"ridge_{alpha:g}"])
    model, training_rmse = fit_regularized_gaze_model(training_observations, selected_alpha, degree)
    return model, training_rmse, CalibrationModelSelection(
        selected_ridge_alpha=float(selected_alpha),
        leave_one_target_out_median_error_norm=round(candidate_scores[f"ridge_{selected_alpha:g}"], 6),
        candidate_median_errors_norm={key: round(value, 6) for key, value in candidate_scores.items()},
    )


def predict_screen_coordinates(
    yaw_deg: float,
    pitch_deg: float,
    model: np.ndarray,
    monitor_width: int,
    monitor_height: int,
    clip: bool = True
) -> tuple[int, int, float, float]:
    """Predict screen pixel coordinates given raw yaw/pitch and fitted model."""
    degree = 2 if model.shape[0] == 6 else 3 if model.shape[0] == 10 else 0
    if degree == 0:
        raise ValueError(f"Unsupported calibration model shape: {model.shape}")
    features = compute_calibration_features(yaw_deg, pitch_deg, degree)
    mapped_pos = features @ model
    norm_x = float(mapped_pos[0])
    norm_y = float(mapped_pos[1])

    if clip:
        norm_x_clipped = max(0.0, min(1.0, norm_x))
        norm_y_clipped = max(0.0, min(1.0, norm_y))
    else:
        norm_x_clipped = norm_x
        norm_y_clipped = norm_y

    px_x = int(norm_x_clipped * (monitor_width - 1))
    px_y = int(norm_y_clipped * (monitor_height - 1))
    return px_x, px_y, norm_x, norm_y


def robust_target_estimate(
    samples: list[tuple[float, float]],
    minimum_inliers: int,
    minimum_tolerance_deg: float = 0.08,
    maximum_mad_deg: float = 1.0,
) -> tuple[float, float, list[tuple[float, float]]] | None:
    """Return a robust per-target gaze estimate or ``None`` when it is unstable.

    Calibration targets are held for a short interval, so their gaze angles
    should form one compact cluster.  A coordinate-wise median is less affected
    by a bad iris frame than a mean.  MAD-based inlier selection is retained,
    with a small floor so nearly-identical floating-point samples do not cause
    a zero-width acceptance window.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < minimum_inliers:
        return None
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < minimum_inliers:
        return None
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    tolerance = np.maximum(3.5 * 1.4826 * mad, float(minimum_tolerance_deg))
    inliers = values[np.all(np.abs(values - median) <= tolerance, axis=1)]
    if len(inliers) < minimum_inliers:
        return None
    # A calibration target should be a fixation, not a slow sweep. Use the
    # robust spread rather than the total range: a single imperfect webcam
    # frame can have a wide range but should not discard an otherwise stable
    # target collection.
    if float(np.max(mad)) > maximum_mad_deg:
        return None
    estimate = np.median(inliers, axis=0)
    return float(estimate[0]), float(estimate[1]), [(float(y), float(p)) for y, p in inliers]


def classify_calibration_quality(median_error_px: float) -> str:
    """Classify calibration quality using prototype engineering thresholds."""
    if median_error_px <= QUALITY_THRESHOLDS["EXCELLENT"]:
        return "EXCELLENT"
    elif median_error_px <= QUALITY_THRESHOLDS["GOOD"]:
        return "GOOD"
    elif median_error_px <= QUALITY_THRESHOLDS["FAIR"]:
        return "FAIR"
    else:
        return "POOR"


def evaluate_calibration(
    model: np.ndarray,
    validation_target_observations: list[tuple[float, float, tuple[float, float], list[tuple[float, float]]]],
    monitor_width: int,
    monitor_height: int,
    viewing_distance_mm: float = 600.0,
    estimated_dpi: float = 96.0,
) -> ValidationMetrics:
    """Perform independent multi-metric evaluation of the calibration model.

    validation_target_observations: list of:
      (mean_yaw, mean_pitch, target_norm, list_of_raw_samples)
    """
    if not validation_target_observations:
        raise ValueError("Cannot evaluate calibration without validation observations")

    spatial_errors: list[SpatialTargetError] = []
    all_sample_errors_px: list[float] = []
    all_sample_dx_px: list[float] = []
    all_sample_dy_px: list[float] = []

    total_samples = 0

    for idx, (mean_yaw, mean_pitch, target_norm, raw_samples) in enumerate(validation_target_observations):
        tx_px = target_norm[0] * (monitor_width - 1)
        ty_px = target_norm[1] * (monitor_height - 1)

        # Evaluate representative mean for spatial report
        pred_x_px, pred_y_px, _, _ = predict_screen_coordinates(
            mean_yaw, mean_pitch, model, monitor_width, monitor_height, clip=False
        )
        target_dx = pred_x_px - tx_px
        target_dy = pred_y_px - ty_px
        target_dist = math.sqrt(target_dx * target_dx + target_dy * target_dy)

        spatial_errors.append(
            SpatialTargetError(
                target_index=idx,
                target_norm=target_norm,
                target_px=(float(tx_px), float(ty_px)),
                predicted_px=(float(pred_x_px), float(pred_y_px)),
                error_px=float(target_dist),
                dx_px=float(target_dx),
                dy_px=float(target_dy),
            )
        )

        # If raw inlier samples are available, evaluate over all independent samples
        samples_to_eval = raw_samples if raw_samples else [(mean_yaw, mean_pitch)]
        total_samples += len(samples_to_eval)

        for s_yaw, s_pitch in samples_to_eval:
            sx_px, sy_px, _, _ = predict_screen_coordinates(
                s_yaw, s_pitch, model, monitor_width, monitor_height, clip=False
            )
            dx = sx_px - tx_px
            dy = sy_px - ty_px
            dist = math.sqrt(dx * dx + dy * dy)
            all_sample_errors_px.append(dist)
            all_sample_dx_px.append(abs(dx))
            all_sample_dy_px.append(abs(dy))

    errors = np.asarray(all_sample_errors_px, dtype=float)
    dxs = np.asarray(all_sample_dx_px, dtype=float)
    dys = np.asarray(all_sample_dy_px, dtype=float)

    mae_px = float(np.mean(errors))
    median_px = float(np.median(errors))
    rmse_px = float(np.sqrt(np.mean(errors ** 2)))
    p95_px = float(np.percentile(errors, 95))

    horizontal_mae_px = float(np.mean(dxs))
    horizontal_median_px = float(np.median(dxs))

    vertical_mae_px = float(np.mean(dys))
    vertical_median_px = float(np.median(dys))

    # Normalized error (% of screen diagonal)
    diagonal_px = math.sqrt(monitor_width ** 2 + monitor_height ** 2)
    normalized_median_error = float(median_px / max(diagonal_px, 1e-6))

    # Visual angle in degrees: theta = 2 * atan(size_mm / (2 * D))
    # Using 96 DPI: 1 inch = 25.4 mm => 1 px = 25.4 / 96 mm
    mm_per_pixel = 25.4 / max(estimated_dpi, 1.0)
    median_error_mm = median_px * mm_per_pixel
    angular_median_error_deg = float(
        np.degrees(2.0 * math.atan(median_error_mm / (2.0 * max(viewing_distance_mm, 100.0))))
    )

    status = classify_calibration_quality(median_px)

    return ValidationMetrics(
        mae_px=round(mae_px, 1),
        median_px=round(median_px, 1),
        rmse_px=round(rmse_px, 1),
        p95_px=round(p95_px, 1),
        horizontal_mae_px=round(horizontal_mae_px, 1),
        horizontal_median_px=round(horizontal_median_px, 1),
        vertical_mae_px=round(vertical_mae_px, 1),
        vertical_median_px=round(vertical_median_px, 1),
        normalized_median_error=round(normalized_median_error, 4),
        angular_median_error_deg=round(angular_median_error_deg, 2),
        status=status,
        num_targets=len(validation_target_observations),
        num_samples=total_samples,
        spatial_errors=spatial_errors,
    )


def save_calibration_profile_v2(
    file_path: str,
    monitor_width: int,
    monitor_height: int,
    model: np.ndarray,
    training_rmse_px: float,
    validation_metrics: Optional[ValidationMetrics] = None,
    model_selection: Optional[CalibrationModelSelection] = None,
) -> bool:
    """Save calibration profile using Version 2 schema with training and validation metadata."""
    try:
        profile: dict[str, Any] = {
            "version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "monitor_size": [monitor_width, monitor_height],
            "model_type": "polynomial_degree_3" if model.shape[0] == 10 else "polynomial_degree_2",
            "model": model.tolist(),
            "training": {
                "points": len(CALIBRATION_TARGETS),
                "rmse_px": round(float(training_rmse_px), 2),
            },
        }

        if validation_metrics is not None:
            spatial_summary = [
                {
                    "target_index": s.target_index,
                    "target_norm": list(s.target_norm),
                    "target_px": [round(s.target_px[0], 1), round(s.target_px[1], 1)],
                    "predicted_px": [round(s.predicted_px[0], 1), round(s.predicted_px[1], 1)],
                    "error_px": round(s.error_px, 1),
                    "dx_px": round(s.dx_px, 1),
                    "dy_px": round(s.dy_px, 1),
                }
                for s in validation_metrics.spatial_errors
            ]
            profile["validation"] = {
                "points": validation_metrics.num_targets,
                "samples": validation_metrics.num_samples,
                "mae_px": validation_metrics.mae_px,
                "median_px": validation_metrics.median_px,
                "rmse_px": validation_metrics.rmse_px,
                "p95_px": validation_metrics.p95_px,
                "horizontal_mae_px": validation_metrics.horizontal_mae_px,
                "horizontal_median_px": validation_metrics.horizontal_median_px,
                "vertical_mae_px": validation_metrics.vertical_mae_px,
                "vertical_median_px": validation_metrics.vertical_median_px,
                "normalized_median_error": validation_metrics.normalized_median_error,
                "angular_median_error_deg": validation_metrics.angular_median_error_deg,
                "status": validation_metrics.status,
                "spatial_errors": spatial_summary,
            }
        if model_selection is not None:
            profile["training"]["model_selection"] = asdict(model_selection)

        temp_file = f"{file_path}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        os.replace(temp_file, file_path)
        return True
    except OSError:
        return False


def load_calibration_profile_v2(
    file_path: str,
    monitor_width: int,
    monitor_height: int,
) -> Optional[dict[str, Any]]:
    """Load calibration profile, handling both Version 2 and legacy Version 1 schemas."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        if profile.get("monitor_size") != [monitor_width, monitor_height]:
            return None

        raw_model = profile.get("model")
        if raw_model is None:
            return None
        model = np.asarray(raw_model, dtype=float)
        if model.shape not in ((6, 2), (10, 2)) or not np.all(np.isfinite(model)):
            return None

        version = profile.get("version", 1)
        res: dict[str, Any] = {
            "version": version,
            "model": model,
            "monitor_size": profile["monitor_size"],
            "is_valid": True,
        }

        if version >= 2 and "validation" in profile:
            val = profile["validation"]
            res["validation"] = val
            res["status_text"] = (
                f"Calibration: VALID | Median Error: {val.get('median_px', 0):.0f}px | "
                f"Status: {val.get('status', 'UNKNOWN')}"
            )
            res["model_selection"] = profile.get("training", {}).get("model_selection")
        else:
            rmse = profile.get("rmse_px", profile.get("training", {}).get("rmse_px", 0.0))
            res["legacy_rmse_px"] = rmse
            res["status_text"] = f"Calibration: UNKNOWN (Legacy profile without independent validation, fit RMSE {rmse:.0f}px)"

        return res
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
