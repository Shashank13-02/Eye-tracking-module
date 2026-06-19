"""
Gaze Features — Digital Biomarker Extraction
=============================================
Computes clinical gaze features from scanpath and session data.
Inspired by open source gaze features — adapted for real-time screening.

Feature categories:
  1. Scanpath features (fixation count, duration stats, saccade stats)
  2. Spatial features (distance to center, spatial dispersion, gaze range)
  3. Temporal features (time-to-first-fixation, dwell time on ROIs)
  4. Social attention features (face vs. object, eye region fixation)
  5. Cognitive features (pursuit accuracy, inhibition, sustained attention)
"""

import math
import numpy as np
from collections import OrderedDict


def compute_scanpath_features(fixations, saccades, frame_size=(640, 480)):
    """
    Compute basic scanpath features from fixation and saccade lists.

    Parameters
    ----------
    fixations : list of Fixation
    saccades : list of Saccade
    frame_size : tuple (width, height)

    Returns
    -------
    dict : feature_name → value
    """
    features = OrderedDict()
    w, h = frame_size

    if not fixations:
        return _empty_scanpath_features()

    durations = np.array([f.duration_ms for f in fixations])
    xs = np.array([f.x for f in fixations])
    ys = np.array([f.y for f in fixations])

    # ── Fixation count ────────────────────────────────────────────────
    features["fix_count"] = len(fixations)

    # ── Duration statistics ───────────────────────────────────────────
    features["fix_duration_total_ms"] = float(np.sum(durations))
    features["fix_duration_mean_ms"] = float(np.mean(durations))
    features["fix_duration_median_ms"] = float(np.median(durations))
    features["fix_duration_std_ms"] = float(np.std(durations))
    features["fix_duration_max_ms"] = float(np.max(durations))
    features["fix_duration_min_ms"] = float(np.min(durations))

    # ── Saccade features ──────────────────────────────────────────────
    if saccades:
        amplitudes = np.array([s.amplitude for s in saccades])
        sacc_durations = np.array([s.duration_ms for s in saccades])
        features["saccade_count"] = len(saccades)
        features["saccade_amplitude_mean_px"] = float(np.mean(amplitudes))
        features["saccade_amplitude_std_px"] = float(np.std(amplitudes))
        features["saccade_amplitude_max_px"] = float(np.max(amplitudes))
        features["saccade_duration_mean_ms"] = float(np.mean(sacc_durations))
        features["scanpath_length_px"] = float(np.sum(amplitudes))
    else:
        features["saccade_count"] = 0
        features["saccade_amplitude_mean_px"] = 0.0
        features["saccade_amplitude_std_px"] = 0.0
        features["saccade_amplitude_max_px"] = 0.0
        features["saccade_duration_mean_ms"] = 0.0
        features["scanpath_length_px"] = 0.0

    # ── Spatial dispersion ────────────────────────────────────────────
    center_x, center_y = w / 2.0, h / 2.0
    dist_to_center = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    features["fix_distance_to_center_mean_px"] = float(np.mean(dist_to_center))
    features["fix_distance_to_center_std_px"] = float(np.std(dist_to_center))

    mean_x, mean_y = np.mean(xs), np.mean(ys)
    dist_to_mean = np.sqrt((xs - mean_x) ** 2 + (ys - mean_y) ** 2)
    features["fix_spatial_dispersion_px"] = float(np.mean(dist_to_mean))

    # ── Gaze range ────────────────────────────────────────────────────
    features["gaze_range_x_px"] = float(np.ptp(xs))
    features["gaze_range_y_px"] = float(np.ptp(ys))
    features["gaze_range_ratio"] = (
        float(np.ptp(xs) * np.ptp(ys)) / (w * h) if w * h > 0 else 0.0
    )

    # ── Fixation rate ─────────────────────────────────────────────────
    total_time_s = features["fix_duration_total_ms"] / 1000.0
    features["fixation_rate_per_sec"] = (
        features["fix_count"] / total_time_s if total_time_s > 0 else 0.0
    )

    return features


def compute_roi_features(fixations, rois, frame_size=(640, 480)):
    """
    Compute Region-of-Interest (ROI) dwell features.

    Parameters
    ----------
    fixations : list of Fixation
    rois : dict of {name: (x1, y1, x2, y2)} — bounding boxes in pixels
    frame_size : tuple (width, height)

    Returns
    -------
    dict : feature_name → value
    """
    features = OrderedDict()
    total_dwell = sum(f.duration_ms for f in fixations) if fixations else 1.0

    for roi_name, (x1, y1, x2, y2) in rois.items():
        roi_fixations = [
            f for f in fixations
            if x1 <= f.x <= x2 and y1 <= f.y <= y2
        ]
        roi_dwell = sum(f.duration_ms for f in roi_fixations)
        roi_count = len(roi_fixations)

        # First fixation latency to this ROI
        first_fix_time = None
        if roi_fixations and fixations:
            session_start = fixations[0].start_time
            first_fix_time = (roi_fixations[0].start_time - session_start) * 1000

        features[f"roi_{roi_name}_dwell_ms"] = float(roi_dwell)
        features[f"roi_{roi_name}_dwell_ratio"] = float(roi_dwell / total_dwell)
        features[f"roi_{roi_name}_fix_count"] = roi_count
        features[f"roi_{roi_name}_first_fix_latency_ms"] = (
            float(first_fix_time) if first_fix_time is not None else -1.0
        )

    return features


def compute_social_attention_features(fixations, rois, frame_size=(640, 480)):
    """
    Compute social attention features — specifically for face/object preference tasks.

    Parameters
    ----------
    fixations : list of Fixation
    rois : dict with expected keys like 'face', 'object', 'eyes', 'mouth', etc.
    frame_size : tuple

    Returns
    -------
    dict
    """
    features = OrderedDict()
    total_dwell = sum(f.duration_ms for f in fixations) if fixations else 1.0

    # Face preference ratio
    face_dwell = 0.0
    object_dwell = 0.0
    for f in fixations:
        if "face" in rois:
            x1, y1, x2, y2 = rois["face"]
            if x1 <= f.x <= x2 and y1 <= f.y <= y2:
                face_dwell += f.duration_ms
        if "object" in rois:
            x1, y1, x2, y2 = rois["object"]
            if x1 <= f.x <= x2 and y1 <= f.y <= y2:
                object_dwell += f.duration_ms

    combined = face_dwell + object_dwell
    features["face_preference_ratio"] = float(face_dwell / combined) if combined > 0 else 0.5
    features["face_dwell_ratio"] = float(face_dwell / total_dwell)
    features["object_dwell_ratio"] = float(object_dwell / total_dwell)

    # Eye region fixation (if 'eyes' ROI is defined)
    if "eyes" in rois:
        x1, y1, x2, y2 = rois["eyes"]
        eye_dwell = sum(
            f.duration_ms for f in fixations
            if x1 <= f.x <= x2 and y1 <= f.y <= y2
        )
        features["eye_region_fixation_ratio"] = float(eye_dwell / total_dwell)
    else:
        features["eye_region_fixation_ratio"] = 0.0

    return features


def compute_pursuit_features(gaze_points, target_positions):
    """
    Compute smooth pursuit tracking accuracy.

    Parameters
    ----------
    gaze_points : list of GazePoint
    target_positions : list of (x, y, t) — target position at each timestamp

    Returns
    -------
    dict
    """
    features = OrderedDict()

    if not gaze_points or not target_positions:
        features["pursuit_accuracy_mean_px"] = -1.0
        features["pursuit_accuracy_std_px"] = -1.0
        features["pursuit_gain"] = -1.0
        return features

    # Match gaze points to nearest target position by time
    errors = []
    for gp in gaze_points:
        # Find closest target position by time
        closest = min(target_positions, key=lambda tp: abs(tp[2] - gp.t))
        error = math.hypot(gp.x - closest[0], gp.y - closest[1])
        errors.append(error)

    errors = np.array(errors)
    features["pursuit_accuracy_mean_px"] = float(np.mean(errors))
    features["pursuit_accuracy_std_px"] = float(np.std(errors))

    # Pursuit gain (ratio of eye movement amplitude to target movement amplitude)
    if len(gaze_points) >= 2 and len(target_positions) >= 2:
        gaze_travel = sum(
            math.hypot(gaze_points[i].x - gaze_points[i-1].x,
                        gaze_points[i].y - gaze_points[i-1].y)
            for i in range(1, len(gaze_points))
        )
        target_travel = sum(
            math.hypot(target_positions[i][0] - target_positions[i-1][0],
                        target_positions[i][1] - target_positions[i-1][1])
            for i in range(1, len(target_positions))
        )
        features["pursuit_gain"] = float(gaze_travel / target_travel) if target_travel > 0 else 0.0
    else:
        features["pursuit_gain"] = 0.0

    return features


def compute_inhibition_features(gaze_points, stimulus_side, response_window_ms=800):
    """
    Compute anti-saccade (inhibition) features.

    Parameters
    ----------
    gaze_points : list of GazePoint
    stimulus_side : str — 'left' or 'right'
    response_window_ms : int — time window to detect response

    Returns
    -------
    dict
    """
    features = OrderedDict()

    if not gaze_points:
        features["anti_saccade_correct"] = 0
        features["anti_saccade_error"] = 0
        features["anti_saccade_latency_ms"] = -1.0
        return features

    # Check if first gaze shift goes to correct side (opposite of stimulus)
    correct_side = "left" if stimulus_side == "right" else "right"

    start_time = gaze_points[0].t
    center_x = 320  # half of 640

    first_shift_direction = None
    first_shift_time = None

    for gp in gaze_points:
        elapsed_ms = (gp.t - start_time) * 1000
        if elapsed_ms > response_window_ms:
            break
        if gp.gaze_direction and gp.gaze_direction != "center":
            if "left" in gp.gaze_direction:
                first_shift_direction = "left"
            elif "right" in gp.gaze_direction:
                first_shift_direction = "right"
            if first_shift_direction:
                first_shift_time = elapsed_ms
                break

    is_correct = first_shift_direction == correct_side if first_shift_direction else False

    features["anti_saccade_correct"] = 1 if is_correct else 0
    features["anti_saccade_error"] = 0 if is_correct else 1
    features["anti_saccade_latency_ms"] = float(first_shift_time) if first_shift_time else -1.0

    return features


def compute_sustained_attention_features(responses, targets):
    """
    Compute sustained attention (CPT) features.

    Parameters
    ----------
    responses : list of dict {'time_ms': float, 'responded': bool}
    targets : list of dict {'time_ms': float, 'is_target': bool}

    Returns
    -------
    dict
    """
    features = OrderedDict()

    if not targets:
        features["sustained_hits"] = 0
        features["sustained_misses"] = 0
        features["sustained_false_alarms"] = 0
        features["sustained_correct_rejections"] = 0
        features["sustained_omission_rate"] = 0.0
        features["sustained_commission_rate"] = 0.0
        features["sustained_d_prime"] = 0.0
        return features

    hits = 0
    misses = 0
    false_alarms = 0
    correct_rejections = 0
    reaction_times = []

    for target in targets:
        matching_response = None
        for resp in responses:
            if abs(resp["time_ms"] - target["time_ms"]) < 1500:  # 1.5s window
                matching_response = resp
                break

        if target["is_target"]:
            if matching_response and matching_response["responded"]:
                hits += 1
                reaction_times.append(
                    abs(matching_response["time_ms"] - target["time_ms"])
                )
            else:
                misses += 1
        else:
            if matching_response and matching_response["responded"]:
                false_alarms += 1
            else:
                correct_rejections += 1

    total_targets = hits + misses
    total_non_targets = false_alarms + correct_rejections

    features["sustained_hits"] = hits
    features["sustained_misses"] = misses
    features["sustained_false_alarms"] = false_alarms
    features["sustained_correct_rejections"] = correct_rejections
    features["sustained_omission_rate"] = (
        float(misses / total_targets) if total_targets > 0 else 0.0
    )
    features["sustained_commission_rate"] = (
        float(false_alarms / total_non_targets) if total_non_targets > 0 else 0.0
    )
    features["sustained_mean_rt_ms"] = (
        float(np.mean(reaction_times)) if reaction_times else -1.0
    )

    # d-prime (signal detection theory)
    hit_rate = max(0.01, min(0.99, hits / total_targets if total_targets > 0 else 0.5))
    fa_rate = max(0.01, min(0.99, false_alarms / total_non_targets if total_non_targets > 0 else 0.5))
    from scipy.stats import norm
    features["sustained_d_prime"] = float(norm.ppf(hit_rate) - norm.ppf(fa_rate))

    return features


def compute_repetitive_pattern_index(fixations, frame_size=(640, 480)):
    """
    Detect repetitive/stereotypical gaze patterns.

    Measures how often the scan path revisits the same spatial regions
    in a repetitive manner (indicator of restricted interests in ASD).

    Returns
    -------
    dict
    """
    features = OrderedDict()

    if len(fixations) < 4:
        features["repetitive_scan_index"] = 0.0
        features["revisit_ratio"] = 0.0
        return features

    w, h = frame_size
    grid_cols, grid_rows = 4, 3
    cell_w, cell_h = w / grid_cols, h / grid_rows

    # Map fixations to grid cells
    cells = []
    for f in fixations:
        col = min(int(f.x / cell_w), grid_cols - 1)
        row = min(int(f.y / cell_h), grid_rows - 1)
        cells.append((row, col))

    # Count transitions that return to a previously visited cell
    visited = set()
    revisits = 0
    for cell in cells:
        if cell in visited:
            revisits += 1
        visited.add(cell)

    features["revisit_ratio"] = float(revisits / len(cells))

    # Detect repeated subsequences (length 2-3)
    pair_counts = {}
    for i in range(len(cells) - 1):
        pair = (cells[i], cells[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    max_repeat = max(pair_counts.values()) if pair_counts else 0
    features["repetitive_scan_index"] = float(max_repeat / len(cells))

    return features


def _empty_scanpath_features():
    """Return zeroed scanpath features when no fixations detected."""
    return OrderedDict([
        ("fix_count", 0),
        ("fix_duration_total_ms", 0.0),
        ("fix_duration_mean_ms", 0.0),
        ("fix_duration_median_ms", 0.0),
        ("fix_duration_std_ms", 0.0),
        ("fix_duration_max_ms", 0.0),
        ("fix_duration_min_ms", 0.0),
        ("saccade_count", 0),
        ("saccade_amplitude_mean_px", 0.0),
        ("saccade_amplitude_std_px", 0.0),
        ("saccade_amplitude_max_px", 0.0),
        ("saccade_duration_mean_ms", 0.0),
        ("scanpath_length_px", 0.0),
        ("fix_distance_to_center_mean_px", 0.0),
        ("fix_distance_to_center_std_px", 0.0),
        ("fix_spatial_dispersion_px", 0.0),
        ("gaze_range_x_px", 0.0),
        ("gaze_range_y_px", 0.0),
        ("gaze_range_ratio", 0.0),
        ("fixation_rate_per_sec", 0.0),
    ])


def aggregate_all_features(session_data, task_rois=None, task_type=None):
    """
    Master aggregation — compute all relevant features for a session.

    Parameters
    ----------
    session_data : dict from GazeAnalyzer.get_session_data()
    task_rois : dict of ROIs (optional)
    task_type : str (optional)

    Returns
    -------
    dict : all features
    """
    fixations = session_data.get("fixations", [])
    saccades = session_data.get("saccades", [])
    blinks = session_data.get("blinks", [])
    frame_size = session_data.get("frame_size", (640, 480))

    all_features = OrderedDict()

    # Scanpath features
    all_features.update(compute_scanpath_features(fixations, saccades, frame_size))

    # Blink features
    all_features["blink_count"] = len(blinks)
    if blinks:
        blink_durs = [b.duration_ms for b in blinks]
        all_features["blink_duration_mean_ms"] = float(np.mean(blink_durs))
    else:
        all_features["blink_duration_mean_ms"] = 0.0

    # Repetitive pattern features
    all_features.update(compute_repetitive_pattern_index(fixations, frame_size))

    # ROI features (if provided)
    if task_rois:
        all_features.update(compute_roi_features(fixations, task_rois, frame_size))

        # Social attention features (if face/object ROIs present)
        if "face" in task_rois or "eyes" in task_rois:
            all_features.update(
                compute_social_attention_features(fixations, task_rois, frame_size)
            )

    # Session duration
    all_features["session_duration_s"] = session_data.get("duration_seconds", 0.0)

    return all_features
