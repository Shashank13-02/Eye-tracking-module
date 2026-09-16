import cv2
import numpy as np
import os
import json
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard
from screening_metrics import ScreeningSession, save_report
from attention_task import VisualAttentionTask, save_task_report
from tracking_quality import FrameQuality, TrackingQualityMonitor
from geometry_calibration import FaceGeometryEstimator, load_camera_intrinsics, load_screen_geometry
from gaze_events import ResearchGazeRecorder
from age_biomarkers import build_biomarker_report, select_age_band
from research_tasks import AgeResearchTask
from calibration_validator import (
    CALIBRATION_TARGETS,
    VALIDATION_TARGETS,
    QUALITY_THRESHOLDS,
    ValidationMetrics,
    vector_to_yaw_pitch_deg,
    compute_calibration_features,
    fit_gaze_model,
    select_regularized_gaze_model,
    robust_target_estimate,
    predict_screen_coordinates,
    classify_calibration_quality,
    evaluate_calibration,
    save_calibration_profile_v2,
    load_calibration_profile_v2,
)

# Screen and mouse control setup (from old script)
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False
f7_was_pressed = False
filter_length = 10
gaze_length = 350

# --- Orbit camera state for the debug view ---
orbit_yaw   = math.radians(-151.0)  # radians, left/right
orbit_pitch = 00.0          # radians, up/down
orbit_radius = 1500.0       # distance from head center
orbit_fov_deg = 50.0       # horizontal FOV for projection

# --- Debug-view world freeze (pivot fixed after center calibration) ---
debug_world_frozen = False
orbit_pivot_frozen = None  # world-space point the debug camera orbits (monitor center at calib)

# Stored gaze markers on the monitor plane (as (a,b) in plane coords)
# a = 0..1 across width (p0->p1), b = 0..1 down height (p0->p3)
gaze_markers = []

# --- 3D monitor plane state (world space) ---
monitor_corners = None   # list of 4 world points (p0..p3)
monitor_center_w = None  # world center of the plane
monitor_normal_w = None  # world normal
units_per_cm = None      # world units per centimeter (computed at calibration)

# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets for screen mapping
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# --- Multi-point screen calibration state ---
CALIBRATION_WINDOW = "Gaze Calibration"
ATTENTION_WINDOW = "Visual Attention Task"
BIOMARKER_WINDOW = "Age-Banded Research Task"
AGE_ENTRY_WINDOW = "NeuroGaze Age Setup"
WEBCAM_WINDOW = "Webcam Feed"
GAZE_WINDOW = "Integrated Eye Tracking"
DEBUG_WINDOW = "Head/Eye Debug"
CALIBRATION_PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaze_calibration.json")
CAMERA_INTRINSICS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_intrinsics.json")
SCREEN_GEOMETRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_geometry.json")
# Calibration state machine
CALIB_STAGE_IDLE = 0
CALIB_STAGE_TRAINING = 1
CALIB_STAGE_VALIDATING = 2
CALIB_STAGE_RESULTS = 3

calibration_stage = CALIB_STAGE_IDLE
multi_calibration_active = False

CALIBRATION_SETTLE_FRAMES = 20
CALIBRATION_SAMPLES_PER_TARGET = 36
CALIBRATION_MIN_INLIERS = 20

VALIDATION_SETTLE_FRAMES = 18
VALIDATION_SAMPLES_PER_TARGET = 32
VALIDATION_MIN_INLIERS = 18

calibration_target_index = 0
calibration_settle_frames = 0
calibration_samples = []

# Strictly isolated buffers to ensure validation samples are NEVER reused in training
training_observations = []
validation_observations = []

training_rmse_val = 0.0
training_model_selection = None
calibration_model = None
latest_validation_metrics = None
calibration_result_show_time = 0.0
calibration_status = "Press M after eye calibration to run 9-point screen calibration."

# --- Simple monitor-edge calibration state ---
# 0 = waiting for center, 1 = waiting for left edge, 2 = done
calib_step = 0

# Buffers to store recent gaze data for smoothing
combined_gaze_directions = deque(maxlen=filter_length)

# reference matrices to fix coordinate flipping issue
# These help keep the axes consistent from frame to frame by stabilizing eigenvector directions
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Open webcam ===
print("[Startup] Opening webcam...", flush=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"[Startup] Webcam opened: {cap.isOpened()}", flush=True)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Startup] Resolution: {w}x{h}", flush=True)
camera_intrinsics = load_camera_intrinsics(CAMERA_INTRINSICS_FILE, w, h)
screen_geometry = load_screen_geometry(SCREEN_GEOMETRY_FILE)
face_geometry = FaceGeometryEstimator(camera_intrinsics, w, h)
print(f"[Geometry] Camera intrinsics: {'CALIBRATED' if camera_intrinsics else 'APPROXIMATE'} | "
      f"screen geometry: {'MEASURED' if screen_geometry else 'DEFAULT'}", flush=True)
cv2.namedWindow(GAZE_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(WEBCAM_WINDOW, cv2.WINDOW_NORMAL)

# Window display mode & Always-On-Top configuration
always_on_top = True

def arrange_main_windows():
    """Position tracking windows cleanly so they float clearly in front of other applications."""
    win_w = min(640, max(360, (MONITOR_WIDTH - 60) // 2))
    win_h = int(win_w * 3 / 4)

    cv2.resizeWindow(GAZE_WINDOW, win_w, win_h)
    cv2.resizeWindow(DEBUG_WINDOW, win_w, win_h)
    cv2.resizeWindow(WEBCAM_WINDOW, win_w, win_h)

    # Primary tracking window on top-left, 3D debug view beside it
    cv2.moveWindow(GAZE_WINDOW, 30, 40)
    cv2.moveWindow(DEBUG_WINDOW, 30 + win_w + 20, 40)
    y_webcam = 40 + win_h + 40
    if y_webcam + win_h <= MONITOR_HEIGHT:
        cv2.moveWindow(WEBCAM_WINDOW, 30, y_webcam)
    else:
        cv2.moveWindow(WEBCAM_WINDOW, 60, 60)


arrange_main_windows()

# Force windows to foreground and configure Always-On-Top using Windows API
import ctypes
import ctypes.wintypes

def bring_windows_to_front(topmost=True):
    """Find OpenCV windows, bring them to the foreground, and set Always-On-Top (topmost)."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    SW_RESTORE = 9
    SWP_NOSIZE = 0x0001
    SWP_NOMOVE = 0x0002
    SWP_SHOWWINDOW = 0x0040
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2

    # Simulate ALT key to bypass Windows SetForegroundWindow restrictions
    VK_MENU = 0x12
    KEYEVENTF_KEYUP = 0x0002
    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

    curr_thread = kernel32.GetCurrentThreadId()
    fore_hwnd = user32.GetForegroundWindow()
    fore_thread = user32.GetWindowThreadProcessId(fore_hwnd, None) if fore_hwnd else 0

    # Order: Webcam first, then Debug, then GAZE_WINDOW last so the main tracker is on top
    target_titles = [WEBCAM_WINDOW, DEBUG_WINDOW, GAZE_WINDOW]
    for title in target_titles:
        try:
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1 if topmost else 0)
        except Exception:
            pass
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.ShowWindow(hwnd, SW_RESTORE)
            z_order = HWND_TOPMOST if topmost else HWND_NOTOPMOST
            user32.SetWindowPos(hwnd, z_order, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            if fore_thread and fore_thread != curr_thread:
                user32.AttachThreadInput(curr_thread, fore_thread, True)
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                user32.AttachThreadInput(curr_thread, fore_thread, False)
            else:
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)

bring_windows_to_front(topmost=always_on_top)
print(f"[Startup] Tracking windows initialized and set Always-On-Top={always_on_top}.", flush=True)

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
# These landmarks are near the nose and are less affected by lateral head movement
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ===== NEW: File writing for screen position =====
screen_position_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_position.txt")
screening_report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screening_reports")
research_stream_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_gaze_streams")
screening_session = ScreeningSession()
attention_task = VisualAttentionTask.default()
research_task = None
age_entry_active = False
age_entry_text = ""
selected_age_months = None
selected_age_band = None
age_plan_show_until = 0.0
quality_monitor = TrackingQualityMonitor()
event_recorder = ResearchGazeRecorder(research_stream_dir)

def write_screen_position(x, y):
    """Atomically publish the latest screen position for external readers."""
    try:
        temp_file = f"{screen_position_file}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"{x},{y}\n")
        os.replace(temp_file, screen_position_file)
    except OSError:
        pass


def _calibration_features(yaw_deg, pitch_deg):
    """Second-order feature vector for the gaze-to-screen regression."""
    return compute_calibration_features(yaw_deg, pitch_deg)


def load_calibration_profile():
    """Load a calibration only when it was made for this display size, supporting v1 and v2 schemas."""
    global calibration_model, calibration_status, latest_validation_metrics
    res = load_calibration_profile_v2(CALIBRATION_PROFILE_FILE, MONITOR_WIDTH, MONITOR_HEIGHT)
    if res is None:
        return
    calibration_model = res["model"]
    calibration_status = res["status_text"]
    print(f"[Calibration] {calibration_status}")


def _show_calibration_target():
    """Render the active calibration target or the results card."""
    global calibration_stage, multi_calibration_active
    if calibration_stage == CALIB_STAGE_IDLE:
        return

    canvas = np.zeros((MONITOR_HEIGHT, MONITOR_WIDTH, 3), dtype=np.uint8)

    if calibration_stage == CALIB_STAGE_TRAINING:
        x_norm, y_norm = CALIBRATION_TARGETS[calibration_target_index]
        x = int(x_norm * (MONITOR_WIDTH - 1))
        y = int(y_norm * (MONITOR_HEIGHT - 1))
        cv2.circle(canvas, (x, y), 24, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 7, (0, 255, 0), -1, cv2.LINE_AA)
        message = f"TRAINING: Look at dot {calibration_target_index + 1}/{len(CALIBRATION_TARGETS)}"
        progress = f"Collecting {len(calibration_samples)}/{CALIBRATION_SAMPLES_PER_TARGET} stable samples"
        cv2.putText(canvas, message, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, progress, (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Keep your head steady. Fixate gaze on the center of the dot.", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press Esc to cancel.", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

    elif calibration_stage == CALIB_STAGE_VALIDATING:
        x_norm, y_norm = VALIDATION_TARGETS[calibration_target_index]
        x = int(x_norm * (MONITOR_WIDTH - 1))
        y = int(y_norm * (MONITOR_HEIGHT - 1))
        cv2.circle(canvas, (x, y), 24, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 7, (255, 200, 0), -1, cv2.LINE_AA)
        message = f"INDEPENDENT VALIDATION: Look at dot {calibration_target_index + 1}/{len(VALIDATION_TARGETS)}"
        progress = f"Evaluating novel target positions {len(calibration_samples)}/{VALIDATION_SAMPLES_PER_TARGET}"
        cv2.putText(canvas, message, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, progress, (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Independent evaluation in progress. Keep looking directly at the dot.", (40, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press Esc to cancel.", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

    elif calibration_stage == CALIB_STAGE_RESULTS:
        if latest_validation_metrics is not None:
            m = latest_validation_metrics
            # 1. Draw spatial error vectors on display
            for s in m.spatial_errors:
                tx, ty = int(s.target_px[0]), int(s.target_px[1])
                px, py = int(s.predicted_px[0]), int(s.predicted_px[1])
                cv2.circle(canvas, (tx, ty), 10, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 3, (255, 200, 0), -1, cv2.LINE_AA)
                cv2.drawMarker(canvas, (px, py), (0, 140, 255), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
                cv2.line(canvas, (tx, ty), (px, py), (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"T{s.target_index + 1}: {s.error_px:.0f}px", (tx + 14, ty - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            # 2. Draw centered Glassmorphic Card
            card_w, card_h = 720, 430
            cx, cy = MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2
            x1, y1 = cx - card_w // 2, cy - card_h // 2
            x2, y2 = cx + card_w // 2, cy + card_h // 2

            card_overlay = canvas.copy()
            cv2.rectangle(card_overlay, (x1, y1), (x2, y2), (24, 24, 30), -1)
            cv2.rectangle(card_overlay, (x1, y1), (x2, y2), (90, 90, 110), 2)
            cv2.addWeighted(card_overlay, 0.90, canvas, 0.10, 0, canvas)

            # Card Header
            cv2.putText(canvas, "CALIBRATION QUALITY", (x1 + 30, y1 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x1 + 30, y1 + 54), (x2 - 30, y1 + 54), (80, 80, 100), 1, cv2.LINE_AA)

            # Primary Metric: MEDIAN GAZE ERROR
            cv2.putText(canvas, "MEDIAN GAZE ERROR", (x1 + 30, y1 + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 190), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{m.median_px:.1f} px", (x1 + 30, y1 + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

            # STATUS BADGE
            status_col = (0, 255, 120) if m.status in ("EXCELLENT", "GOOD") else (0, 215, 255) if m.status == "FAIR" else (50, 50, 255)
            cv2.putText(canvas, f"STATUS: {m.status}", (x1 + 350, y1 + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_col, 2, cv2.LINE_AA)
            cv2.putText(canvas, "(Prototype engineering quality gate)", (x1 + 350, y1 + 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.line(canvas, (x1 + 30, y1 + 160), (x2 - 30, y1 + 160), (60, 60, 80), 1, cv2.LINE_AA)

            # Detailed Error Statistics
            row1 = f"RMSE: {m.rmse_px:.1f} px        MAE: {m.mae_px:.1f} px        P95: {m.p95_px:.1f} px"
            cv2.putText(canvas, row1, (x1 + 30, y1 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

            row2 = f"Horizontal MAE: {m.horizontal_mae_px:.1f} px    Vertical MAE: {m.vertical_mae_px:.1f} px"
            cv2.putText(canvas, row2, (x1 + 30, y1 + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

            row3 = f"Normalized Error: {m.normalized_median_error * 100:.1f} %        Visual Angle: {m.angular_median_error_deg:.1f} deg"
            cv2.putText(canvas, row3, (x1 + 30, y1 + 265), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 200), 1, cv2.LINE_AA)

            # Spatial breakdown summary
            spat_str = "Targets: " + "  ".join([f"T{s.target_index + 1}={s.error_px:.0f}px" for s in m.spatial_errors])
            cv2.putText(canvas, spat_str, (x1 + 30, y1 + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)

            # Legend
            cv2.putText(canvas, "Spatial Map: (O) Known Target  -->  (X) Predicted Gaze", (x1 + 30, y1 + 345),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA)

            # Dismiss hint
            remaining = max(0.0, 7.0 - (time.monotonic() - calibration_result_show_time))
            cv2.putText(canvas, f"[ SPACE / ENTER ] Return to Tracking (auto-closes in {remaining:.0f}s)",
                        (x1 + 120, y1 + 395), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # Auto close after timeout
            if remaining <= 0.0:
                calibration_stage = CALIB_STAGE_IDLE
                multi_calibration_active = False
                cv2.destroyWindow(CALIBRATION_WINDOW)

    cv2.imshow(CALIBRATION_WINDOW, canvas)


def _finish_training_and_start_validation():
    """Fit 2nd-order model on training observations and start independent validation."""
    global calibration_stage, calibration_model, calibration_status
    global calibration_target_index, calibration_settle_frames, calibration_samples
    global multi_calibration_active, training_rmse_val, training_model_selection

    features_obs = training_observations
    model, train_rmse_norm, training_model_selection = select_regularized_gaze_model(features_obs)
    diag_px = math.sqrt(MONITOR_WIDTH ** 2 + MONITOR_HEIGHT ** 2)
    training_rmse_val = float(train_rmse_norm * diag_px)
    calibration_model = model

    # Transition to independent validation stage
    calibration_stage = CALIB_STAGE_VALIDATING
    calibration_target_index = 0
    calibration_settle_frames = 0
    calibration_samples = []
    calibration_status = f"Validating calibration against {len(VALIDATION_TARGETS)} independent targets..."
    print(f"[Calibration] {len(CALIBRATION_TARGETS)}-point training fit complete (training RMSE {training_rmse_val:.1f}px; "
          f"ridge={training_model_selection.selected_ridge_alpha:g}, "
          f"LOTO median={training_model_selection.leave_one_target_out_median_error_norm:.4f} normalized).")
    print(f"[Calibration] Starting independent {len(VALIDATION_TARGETS)}-target validation (testing novel screen positions)...")


def _finish_validation():
    """Evaluate fitted model against independent validation observations and compute metrics."""
    global calibration_stage, calibration_status, latest_validation_metrics
    global calibration_result_show_time, multi_calibration_active

    metrics = evaluate_calibration(
        calibration_model,
        validation_observations,
        MONITOR_WIDTH,
        MONITOR_HEIGHT,
        viewing_distance_mm=600.0,
    )
    latest_validation_metrics = metrics

    # Save to gaze_calibration.json with Version 2 schema
    save_calibration_profile_v2(
        CALIBRATION_PROFILE_FILE,
        MONITOR_WIDTH,
        MONITOR_HEIGHT,
        calibration_model,
        training_rmse_val,
        metrics,
        training_model_selection,
    )

    calibration_stage = CALIB_STAGE_RESULTS
    calibration_result_show_time = time.monotonic()
    calibration_status = (
        f"Calibration Quality: Median Error {metrics.median_px:.0f}px | Status: {metrics.status}"
    )

    print("=" * 60)
    print("CALIBRATION QUALITY EVALUATION (Independent Validation)")
    print(f"  Median Gaze Error:   {metrics.median_px:.1f} px")
    print(f"  Status:              {metrics.status}")
    print(f"  RMSE:                {metrics.rmse_px:.1f} px")
    print(f"  MAE:                 {metrics.mae_px:.1f} px")
    print(f"  95th Percentile:     {metrics.p95_px:.1f} px")
    print(f"  Horizontal MAE:      {metrics.horizontal_mae_px:.1f} px")
    print(f"  Vertical MAE:        {metrics.vertical_mae_px:.1f} px")
    print(f"  Normalized Error:    {metrics.normalized_median_error * 100:.2f} %")
    print(f"  Angular Error:       {metrics.angular_median_error_deg:.2f} deg")
    for s in metrics.spatial_errors:
        print(f"    Target {s.target_index + 1} at ({s.target_norm[0]:.2f}, {s.target_norm[1]:.2f}): "
              f"Error {s.error_px:.1f}px (dx: {s.dx_px:+.1f}px, dy: {s.dy_px:+.1f}px)")
    print(f"  Saved Version 2 profile to: {CALIBRATION_PROFILE_FILE}")
    print("=" * 60)


def _record_calibration_sample(yaw_deg, pitch_deg, left_locked=True, right_locked=True, left_dir=None, right_dir=None,
                               frame_quality_ok=True):
    """Collect stable readings with quality gate filtering for training and validation."""
    global calibration_target_index, calibration_settle_frames, calibration_samples
    global training_observations, validation_observations, calibration_stage

    if calibration_stage not in (CALIB_STAGE_TRAINING, CALIB_STAGE_VALIDATING):
        return

    # 1. Quality Gate: Ensure finite numbers and locked eye spheres
    if not (np.isfinite(yaw_deg) and np.isfinite(pitch_deg)):
        return
    if not (left_locked and right_locked):
        return
    if not frame_quality_ok:
        return

    # 2. Quality Gate: Binocular agreement
    if left_dir is not None and right_dir is not None:
        dot = float(np.clip(np.dot(left_dir, right_dir), -1.0, 1.0))
        # A desk-mounted display is far enough away that the two eye rays should
        # agree closely.  Rejecting readings below this threshold prevents a
        # partially occluded iris from disproportionately corrupting vertical
        # calibration samples.
        if dot < 0.90:  # > ~26 deg difference between left and right gaze
            return

    # 3. Settle frames check
    settle_threshold = CALIBRATION_SETTLE_FRAMES if calibration_stage == CALIB_STAGE_TRAINING else VALIDATION_SETTLE_FRAMES
    if calibration_settle_frames < settle_threshold:
        calibration_settle_frames += 1
        return

    # 4. Collect samples
    samples_needed = CALIBRATION_SAMPLES_PER_TARGET if calibration_stage == CALIB_STAGE_TRAINING else VALIDATION_SAMPLES_PER_TARGET
    calibration_samples.append((yaw_deg, pitch_deg))
    if len(calibration_samples) < samples_needed:
        return

    # 5. Outlier rejection using Median & Median Absolute Deviation (MAD)
    min_inliers = CALIBRATION_MIN_INLIERS if calibration_stage == CALIB_STAGE_TRAINING else VALIDATION_MIN_INLIERS
    target_estimate = robust_target_estimate(calibration_samples, min_inliers)
    if target_estimate is None:
        # Do not throw away an entire target after one noisy capture window.
        # Retain the newest half-window, so a user who is already holding the
        # dot can recover with a short additional fixation instead of starting
        # again from zero samples.
        retain_count = max(min_inliers, samples_needed // 2)
        calibration_samples = calibration_samples[-retain_count:]
        stage_name = "Training" if calibration_stage == CALIB_STAGE_TRAINING else "Validation"
        print(f"[{stage_name}] Gaze jitter detected; retaining {len(calibration_samples)} recent samples and waiting for a stable segment.")
        return
    mean_yaw, mean_pitch, inliers = target_estimate

    if calibration_stage == CALIB_STAGE_TRAINING:
        target = CALIBRATION_TARGETS[calibration_target_index]
        training_observations.append((float(mean_yaw), float(mean_pitch), target))
        calibration_target_index += 1
        calibration_samples = []
        calibration_settle_frames = 0
        if calibration_target_index == len(CALIBRATION_TARGETS):
            _finish_training_and_start_validation()

    elif calibration_stage == CALIB_STAGE_VALIDATING:
        target = VALIDATION_TARGETS[calibration_target_index]
        validation_observations.append((float(mean_yaw), float(mean_pitch), target, inliers))
        calibration_target_index += 1
        calibration_samples = []
        calibration_settle_frames = 0
        if calibration_target_index == len(VALIDATION_TARGETS):
            _finish_validation()


def start_multi_point_calibration():
    global calibration_stage, multi_calibration_active
    global calibration_target_index, calibration_settle_frames
    global calibration_samples, training_observations, validation_observations
    global latest_validation_metrics, calibration_status, training_model_selection

    calibration_stage = CALIB_STAGE_TRAINING
    multi_calibration_active = True
    calibration_target_index = 0
    calibration_settle_frames = 0
    calibration_samples = []
    training_observations = []
    validation_observations = []
    latest_validation_metrics = None
    training_model_selection = None
    calibration_status = "9-point calibration in progress (training)..."

    cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(CALIBRATION_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("[Calibration] Starting 9-point training calibration. Keep your head still and follow the dot.")



def _show_attention_target():
    """Render the active visual-attention target in a full-screen task window."""
    trial = attention_task.current_trial
    if trial is None:
        return
    canvas = np.zeros((MONITOR_HEIGHT, MONITOR_WIDTH, 3), dtype=np.uint8)
    x = int(trial.target_x * (MONITOR_WIDTH - 1))
    y = int(trial.target_y * (MONITOR_HEIGHT - 1))
    cv2.circle(canvas, (x, y), 28, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 9, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "Look at the dot", (40, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Target {len(attention_task.results) + 1}/{len(attention_task.trials)}  |  Esc cancels",
                (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imshow(ATTENTION_WINDOW, canvas)


def start_attention_task():
    global attention_task
    attention_task = VisualAttentionTask.default()
    attention_task.start(time.monotonic())


def start_research_task(module: str):
    """Start an age-banded research stimulus only during a consented event session."""
    global research_task
    if not event_recorder.active:
        print("[Research Task] Start a consented T research session before starting a task.")
        return
    try:
        if selected_age_months is None:
            raise ValueError("Enter an age with T before starting a task")
        band = select_age_band(selected_age_months)
        research_task = AgeResearchTask(module, band, time.monotonic())
    except (ValueError, TypeError) as error:
        print(f"[Research Task] Cannot start: {error}.")
        return
    cv2.namedWindow(BIOMARKER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(BIOMARKER_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print(f"[Research Task] Started {module} for {band.key}.")


def _show_research_task():
    global research_task
    if research_task is None:
        return
    canvas = research_task.render(MONITOR_WIDTH, MONITOR_HEIGHT, time.monotonic())
    cv2.imshow(BIOMARKER_WINDOW, canvas)
    if research_task.finished:
        cv2.destroyWindow(BIOMARKER_WINDOW)
        print(f"[Research Task] {research_task.module} complete; end the T session to save its biomarker report.")
        research_task = None


def _show_age_entry():
    if not age_entry_active:
        return
    canvas = np.full((430, 760, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "NeuroGaze research session", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2)
    cv2.putText(canvas, "Enter child age in whole months (0-72)", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (50, 50, 50), 1)
    cv2.rectangle(canvas, (45, 160), (710, 230), (255, 255, 255), -1)
    cv2.rectangle(canvas, (45, 160), (710, 230), (80, 80, 80), 2)
    cv2.putText(canvas, age_entry_text or "_", (65, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 2)
    cv2.putText(canvas, "Enter = continue | Backspace = edit | Esc = cancel", (45, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 80), 1)
    cv2.putText(canvas, "Research measures only: no diagnosis or risk category.", (45, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 110, 200), 1)
    cv2.imshow(AGE_ENTRY_WINDOW, canvas)


def _show_age_plan():
    if time.monotonic() >= age_plan_show_until or selected_age_band is None:
        return
    canvas = np.full((430, 900, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, f"Age band: {selected_age_band.key}", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2)
    if selected_age_band.enabled_tasks:
        labels = {"face_preference": "1 = face versus non-social preference", "moving_tracking": "2 = moving-object tracking",
                  "social_gaze_following": "3 = social-cue gaze following"}
        cv2.putText(canvas, "Available research tasks:", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1)
        for index, task in enumerate(selected_age_band.enabled_tasks):
            cv2.putText(canvas, labels[task], (60, 175 + index * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 110, 40), 1)
    else:
        cv2.putText(canvas, "No self-directed screen task is enabled for this age band.", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 110, 200), 1)
    cv2.putText(canvas, "These produce research measures, not a diagnosis or risk category.", (40, 365),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 110, 200), 1)
    cv2.imshow(AGE_ENTRY_WINDOW, canvas)


def start_research_session_for_age(age_months: int):
    """Validate explicit in-app age entry, then start a consented research session."""
    global selected_age_months, selected_age_band, age_plan_show_until
    consent_confirmed = os.environ.get("NEUROGAZE_CONSENT", "").upper() == "YES"
    pseudonym = os.environ.get("NEUROGAZE_PARTICIPANT_PSEUDONYM", "")
    if not consent_confirmed or not pseudonym:
        print("[Research Stream] Not started. Set NEUROGAZE_CONSENT=YES and a non-identifying "
              "NEUROGAZE_PARTICIPANT_PSEUDONYM before launching the tracker.")
        return False
    selected_age_band = select_age_band(age_months)
    selected_age_months = age_months
    session_id = event_recorder.start(time.monotonic(), pseudonym, consent_confirmed=True)
    screening_session.start(time.monotonic())
    age_plan_show_until = time.monotonic() + 8.0
    print(f"[Screening] Research session started ({session_id}) for {selected_age_band.key}. "
          f"Allowed tasks: {selected_age_band.enabled_tasks or 'none'}; no diagnosis or risk category.")
    return True
    cv2.namedWindow(ATTENTION_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(ATTENTION_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("[Attention Task] Started. Confirm guardian consent and use this only for research metrics, not diagnosis.")

def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    # horizontal pinhole focal length
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    Build a measured monitor plane when setup data exists, otherwise use the legacy
    60cm x 40cm / 50cm debug-plane fallback.
    Monitor is oriented horizontally like a real monitor (top edge parallel to global X-axis).
    """
    # 1) Estimate scale from chin<->forehead distance
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0  # units per cm
    except Exception:
        upc = 5.0
    
    # 2) Monitor geometry in world units
    if screen_geometry is not None:
        dist_cm = screen_geometry.camera_to_screen_mm / 10.0
        mon_w_cm, mon_h_cm = screen_geometry.screen_width_mm / 10.0, screen_geometry.screen_height_mm / 10.0
    else:
        dist_cm = 50.0
        mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    # Head forward vector
    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    # --- NEW: use gaze ray intersection ---
    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)

        # Place the monitor so its center is exactly at some point on the gaze ray
        # For simplicity: choose intersection at 50 cm from head_center along head_forward
        plane_point = head_center + head_forward * (dist_cm * upc)
        plane_normal = head_forward

        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            # fallback: use fixed distance
            center_w = head_center + head_forward * (dist_cm * upc)
    else:
        # fallback: original placement
        center_w = head_center + head_forward * (dist_cm * upc)

    # Compute right/up using head orientation
    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)
    if screen_geometry is not None:
        center_w += head_right * (screen_geometry.camera_offset_x_mm / 10.0) * upc
        center_w -= head_up * (screen_geometry.camera_offset_y_mm / 10.0) * upc

    # Corners
    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc




def update_orbit_from_keys():
    """Keyboard orbit controls that PRINT every frame while a key is held."""
    global orbit_yaw, orbit_pitch, orbit_radius
    yaw_step   = math.radians(1.5)
    pitch_step = math.radians(1.5)
    zoom_step  = 12.0

    changed = False

    # Rotate
    if keyboard.is_pressed('j'):  # yaw left
        orbit_yaw -= yaw_step; changed = True
    if keyboard.is_pressed('l'):  # yaw right
        orbit_yaw += yaw_step; changed = True
    if keyboard.is_pressed('i'):  # pitch up
        orbit_pitch += pitch_step; changed = True
    if keyboard.is_pressed('k'):  # pitch down
        orbit_pitch -= pitch_step; changed = True

    # Zoom
    if keyboard.is_pressed('['):  # zoom out
        orbit_radius += zoom_step; changed = True
    if keyboard.is_pressed(']'):  # zoom in
        orbit_radius = max(80.0, orbit_radius - zoom_step); changed = True

    # Reset (prints every frame while held)
    if keyboard.is_pressed('r'):
        orbit_yaw = 0.0
        orbit_pitch = 0.0
        orbit_radius = 600.0
        changed = True

    # Clamp pitch & radius
    orbit_pitch = max(math.radians(-89), min(math.radians(89), orbit_pitch))
    orbit_radius = max(80.0, orbit_radius)

    if changed:
        print(f"[Orbit Debug] yaw={math.degrees(orbit_yaw):.2f}°, "
              f"pitch={math.degrees(orbit_pitch):.2f}°, "
              f"radius={orbit_radius:.2f}, "
              f"fov={orbit_fov_deg:.1f}°")




def compute_scale(points_3d):
    # Use average pairwise distance for robustness
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    # Gaze vector
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    # Segment points
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    # ---- PART 1: back segment (behind iris) ----
    cv2.line(
        frame,
        (int(eye_center[0]), int(eye_center[1])),
        (int(iris_offset[0]), int(iris_offset[1])),
        color,
        1
    )

    # ---- IRIS (occludes part of the ray) ----
    up_dir = np.array([0, -1, 0])
    right_dir = np.cross(gaze_direction, up_dir)
    if np.linalg.norm(right_dir) < 1e-6:
        right_dir = np.array([1, 0, 0])
    up_dir = np.cross(right_dir, gaze_direction)
    up_dir /= np.linalg.norm(up_dir)
    right_dir /= np.linalg.norm(right_dir)
    ellipse_axes = (
        int((eye_radius / 3) * np.linalg.norm(right_dir[:2])),
        int((eye_radius / 3) * np.linalg.norm(up_dir[:2]))
    )
    angle = math.degrees(math.atan2(gaze_direction[1], gaze_direction[0]))

    # ---- PART 2: front segment (on top of iris) ----
    cv2.line(
        frame,
        (int(iris_offset[0]), int(iris_offset[1])),
        (int(gaze_endpoint[0]), int(gaze_endpoint[1])),
        color,
        1
    )

def draw_wireframe_cube(frame, center, R, size=80):
    # Given a center and rotation matrix, draw a cube aligned to that orientation
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    # 8 corners of the cube
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # Edges connecting the corners
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    # Extract 3D positions of selected landmarks
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    # Compute the average position as the center of this substructure
    center = np.mean(points_3d, axis=0)

    # Draw the raw 2D landmark points
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # PCA-based orientation: Compute eigenvectors of the covariance matrix
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

    # Ensure the orientation matrix is right-handed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Convert to Euler angles and re-construct rotation matrix (optional but clarifies the transform)
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # === Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change ===
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    # Draw cube and orientation axes on the image
    draw_wireframe_cube(frame, center, R_final, size)

    # Draw X (green), Y (blue), Z (red) axes
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    Convert 3D gaze direction vector to continuous yaw/pitch and screen coordinates.
    Uses atan2 for continuous signed representation:
      yaw = atan2(x, -z)
      pitch = atan2(-y, sqrt(x^2 + z^2))
    """
    raw_yaw_deg, raw_pitch_deg = vector_to_yaw_pitch_deg(combined_gaze_direction)

    # Specify degrees at which screen border will be reached (legacy fallback)
    yawDegrees = 5 * 3  # x degrees left or right
    pitchDegrees = 2.0 * 2.5  # x degrees up or down

    # Apply calibration offsets
    yaw_deg = raw_yaw_deg + calibration_offset_yaw
    pitch_deg = raw_pitch_deg + calibration_offset_pitch

    if calibration_model is not None:
        screen_x, screen_y, _, _ = predict_screen_coordinates(
            raw_yaw_deg, raw_pitch_deg, calibration_model, MONITOR_WIDTH, MONITOR_HEIGHT, clip=True
        )
    else:
        # Fall back to the original fixed-angle mapping until a profile exists
        screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH)
        screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

    # Clamp screen position to monitor bounds
    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))

    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg


def render_debug_view_orbit(
    h, w,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None,
    combined_dir=None,
    gaze_len=430,
    monitor_corners=None,
    monitor_center=None,
    monitor_normal=None,
    gaze_markers=None,
):
    if head_center3d is None:
        return

    debug = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Choose orbit pivot ---
    head_w = np.asarray(head_center3d, dtype=float)

    # NEW: if we've frozen the world, orbit around the frozen pivot (monitor center at calib)
    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    # --- Camera pose (orbit around pivot_w) ---
    f_px = _focal_px(w, orbit_fov_deg)
    cam_offset = _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch) @ np.array([0.0, 0.0, orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])   # image-space up is -Y
    fwd = _normalize(pivot_w - cam_pos)     # look at pivot
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)

    def project_point(P):
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    # --- helper draws ---
    def draw_poly_3d(pts, color=(0, 200, 255), thickness=2):
        projs = [project_point(p) for p in pts]
        if any(p is None for p in projs): return
        p2 = [p[0] for p in projs]
        for a, b in zip(p2, p2[1:] + [p2[0]]):
            cv2.line(debug, a, b, color, thickness)

    def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
        res = project_point(P)
        if res is None: return
        (x, y), _ = res
        cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
        cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

    def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
        a = project_point(P0); b = project_point(P1)
        if a is None or b is None: return
        p0, p1 = a[0], b[0]
        cv2.line(debug, p0, p1, color, thickness)
        v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n
            l = np.array([-v[1], v[0]])
            ah = 10
            a1 = (int(p1[0] - v[0]*ah + l[0]*ah*0.6), int(p1[1] - v[1]*ah + l[1]*ah*0.6))
            a2 = (int(p1[0] - v[0]*ah - l[0]*ah*0.6), int(p1[1] - v[1]*ah - l[1]*ah*0.6))
            cv2.line(debug, p1, a1, color, thickness)
            cv2.line(debug, p1, a2, color, thickness)

    # --- Landmarks ---
    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    # --- Head center ---
    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Pivot visual (small cross + line to head and monitor) ---
    draw_cross_3d(pivot_w, size=8, color=(180, 120, 255), thickness=2)
    if monitor_center is not None:
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if mc2d is not None and pv2d is not None and hc2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

    # --- Eyes + per-eye gaze (unchanged from your version) ---
    left_dir = None
    right_dir = None

    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                p1 = project_point(np.asarray(sphere_world_l) + _normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + _normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None and (left_dir is not None or right_dir is not None):
            parts = []
            if left_dir is not None:  parts.append(_normalize(left_dir))
            if right_dir is not None: parts.append(_normalize(right_dir))
            if parts:
                combined_dir = _normalize(np.mean(parts, axis=0))
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + _normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    # --- Monitor plane ---
    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            draw_cross_3d(monitor_center, size=8, color=(0, 200, 255), thickness=2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                draw_arrow_3d(monitor_center, tip, color=(0, 220, 255), thickness=2)

    # --- Stored gaze markers on the monitor plane (green circles) ---
    if (gaze_markers and monitor_corners is not None):
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0  # width direction
        v = p3 - p0  # height direction
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world  # 2% of width
            for (a, b) in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)



    # --- Gaze hit on monitor plane (circle at intersection) ---
    if (monitor_corners is not None and monitor_center is not None and monitor_normal is not None
        and combined_dir is not None
        and sphere_world_l is not None and sphere_world_r is not None):

        # Ray: origin at midpoint between eyes; direction = combined gaze
        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = _normalize(np.asarray(combined_dir, dtype=float))

        # Plane: through monitor_center with normal = monitor_normal
        C = np.asarray(monitor_center, dtype=float)
        N = _normalize(np.asarray(monitor_normal, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D  # world-space intersection point

                # Inside-quad test using monitor's local axes (top-left p0, top-right p1, bottom-left p3)
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                u = p1 - p0             # horizontal (width) vector
                v = p3 - p0             # vertical (height) vector
                wv = P  - p0

                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)  # 0..1 across width
                    b = float(np.dot(wv, v) / v_len2)  # 0..1 across height

                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        # Project center to pixels
                        projP = project_point(P)
                        if projP is not None:
                            center_px = projP[0]

                            # Circle radius = 5% of monitor width (world), projected to pixels
                            width_world = math.sqrt(u_len2)
                            r_world = 0.05 * width_world
                            u_hat = u / max(width_world, 1e-9)

                            projR = project_point(P + u_hat * r_world)
                            if projR is not None:
                                r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)


    # --- Key command help text in lower-left ---
    help_text = [
        "C = calibrate screen center",
        "O = toggle window always-on-top",
        "J = yaw left",
        "L = yaw right",
        "I = pitch up",
        "K = pitch down",
        "[ = zoom out",
        "] = zoom in",
        "R = reset view",
        "X = add marker",
        "M = 9-point calibration",
        "Esc = cancel calibration",
        "T = start/stop metrics session",
        "V = visual-attention task",
        "q = quit",
        "F7 = toggle mouse control"
    ]

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.5
    thickness   = 1
    line_height = 18  # pixels between lines

    # Start a bit above the bottom-left corner
    y0 = h - (len(help_text) * line_height) - 10
    x0 = 10

    for i, text in enumerate(help_text):
        y = y0 + i * line_height
        cv2.putText(debug, text, (x0, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


    cv2.imshow(DEBUG_WINDOW, debug)


def draw_efficiency_hud(frame, fps, latency_ms, face_detected, left_locked, right_locked, screen_coords=None,
                        mouse_on=False, quality: FrameQuality | None = None):
    """Draw a modern HUD overlay displaying real-time efficiency and tracking status."""
    h_f, w_f = frame.shape[:2]
    hud_h = 94
    hud_w = min(w_f - 20, 620)
    x0, y0 = 10, 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (80, 80, 80), 1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Line 1: FPS, latency, and Always-On-Top indicator
    fps_color = (0, 255, 120) if fps >= 25 else (0, 215, 255) if fps >= 15 else (50, 50, 255)
    fps_text = f"FPS: {fps:4.1f} ({latency_ms:4.1f}ms)"
    cv2.putText(frame, fps_text, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 1, cv2.LINE_AA)

    ontop_text = "On-Top: ON [O]" if always_on_top else "On-Top: OFF [O]"
    ontop_color = (0, 255, 200) if always_on_top else (160, 160, 160)
    cv2.putText(frame, ontop_text, (x0 + 205, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ontop_color, 1, cv2.LINE_AA)

    # Tracking state
    if not face_detected:
        status_text = "Face: SEARCHING..."
        status_color = (0, 120, 255)
    elif not (left_locked and right_locked):
        status_text = "Eyes: PREVIEW (Press C to lock)"
        status_color = (0, 215, 255)
    else:
        status_text = "Tracking: ACTIVE"
        status_color = (0, 255, 0)
    cv2.putText(frame, status_text, (x0 + 365, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 1, cv2.LINE_AA)

    # Line 2: Gaze coords, mouse status & controls hint
    if screen_coords:
        prefix = "Gaze Screen" if (left_locked and right_locked) else "Gaze [Est]"
        gaze_text = f"{prefix}: ({screen_coords[0]}, {screen_coords[1]})"
    else:
        gaze_text = "Gaze Screen: (Searching...)"
    cv2.putText(frame, gaze_text, (x0 + 10, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    mouse_text = f"Mouse: {'ON' if mouse_on else 'OFF'} [F7]"
    mouse_color = (0, 255, 0) if mouse_on else (160, 160, 160)
    cv2.putText(frame, mouse_text, (x0 + 265, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mouse_color, 1, cv2.LINE_AA)

    if quality is not None:
        if quality.usable_for_gaze:
            quality_text, quality_color = "Frame quality: USABLE", (0, 230, 0)
        else:
            quality_text = "Frame paused: " + ", ".join(quality.flags[:2])
            quality_color = (0, 160, 255)
        cv2.putText(frame, quality_text, (x0 + 10, y0 + 64), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, quality_color, 1, cv2.LINE_AA)

    hint_text = "[C] Lock  [M] Calib  [Q] Quit"
    cv2.putText(frame, hint_text, (x0 + 410, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

    # Line 3: Calibration Quality Status
    if latest_validation_metrics is not None:
        q_text = f"Calib Quality: Median Error {latest_validation_metrics.median_px:.0f}px | Status: {latest_validation_metrics.status}"
        q_color = (0, 255, 120) if latest_validation_metrics.status in ("EXCELLENT", "GOOD") else (0, 215, 255) if latest_validation_metrics.status == "FAIR" else (50, 50, 255)
    elif calibration_model is not None:
        q_text = "Calib Quality: Active (Calibrated Profile)"
        q_color = (0, 215, 255)
    else:
        q_text = "Calib Quality: Uncalibrated [Press M to Calibrate]"
        q_color = (160, 160, 160)
    cv2.putText(frame, q_text, (x0 + 10, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, q_color, 1, cv2.LINE_AA)




def mouse_mover():
    """Mouse movement thread from old script"""
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)  # adjust for responsiveness

# Start mouse movement thread
threading.Thread(target=mouse_mover, daemon=True).start()

# Eye sphere tracking variables (from new script)
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None
left_calibration_radius = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None
right_calibration_radius = None

load_calibration_profile()

frame_count = 0
fps_history = deque(maxlen=30)
c_was_pressed = False
m_was_pressed = False

while cap.isOpened():
    loop_start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        print(f"[Error] Failed to read frame at frame_count={frame_count}", flush=True)
        break

    frame_count += 1
    if frame_count % 100 == 1:
        print(f"[Running] Frame {frame_count}, shape={frame.shape}", flush=True)

    raw_frame = frame.copy()  # preserve the raw webcam image before any overlays

    combined_dir = None  # will be filled once you compute a smoothed direction
    face_detected = False
    session_gaze_xy = None
    screen_coords_for_hud = None
    frame_quality = quality_monitor.evaluate(frame, None, time.monotonic())

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_detected = True
        face_landmarks = results.multi_face_landmarks[0].landmark
        head_pose = face_geometry.estimate_head_pose(face_landmarks)
        left_eye_geometry, right_eye_geometry = face_geometry.estimate_eyes(face_landmarks, head_pose)
        # An uncalibrated solvePnP camera matrix is useful for drawing approximate
        # geometry, but its Euler angles are not reliable enough to hard-block the
        # first eye lock.  Let TrackingQualityMonitor use its image-landmark pose
        # proxy until a measured intrinsics profile is available.
        pose_degrees = (
            None if head_pose is None or not head_pose.calibrated_intrinsics
            else (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg)
        )
        frame_quality = quality_monitor.evaluate(frame, face_landmarks, time.monotonic(), pose_degrees)

        # Index for left iris center point (from MediaPipe's iris model)
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

        # Compute and draw stabilized coordinate frame from nose region
        head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
            frame,
            face_landmarks,
            nose_indices,
            R_ref_nose,
            color=(0, 255, 0),
            size=80
        )

        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)

        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

        current_nose_scale = compute_scale(nose_points_3d)

        # Eye sphere positions (locked if calibrated with C, or estimated preview)
        camera_dir_world = np.array([0, 0, 1])
        if left_sphere_locked:
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            scaled_radius_l = int((left_calibration_radius or left_eye_geometry.radius) * scale_ratio)
        else:
            sphere_world_l = left_eye_geometry.center
            scaled_radius_l = int(left_eye_geometry.radius)

        x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
        sphere_color_l = (255, 255, 25) if left_sphere_locked else (220, 180, 40)
        cv2.circle(frame, (x_sphere_l, y_sphere_l), scaled_radius_l, sphere_color_l, 2)
        cv2.circle(frame, (x_iris_l, y_iris_l), 8, (255, 25, 25), 2)

        if right_sphere_locked:
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            scaled_radius_r = int((right_calibration_radius or right_eye_geometry.radius) * scale_ratio_r)
        else:
            sphere_world_r = right_eye_geometry.center
            scaled_radius_r = int(right_eye_geometry.radius)

        x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
        sphere_color_r = (25, 255, 255) if right_sphere_locked else (220, 180, 40)
        cv2.circle(frame, (x_sphere_r, y_sphere_r), scaled_radius_r, sphere_color_r, 2)
        cv2.circle(frame, (x_iris_r, y_iris_r), 8, (25, 255, 25), 2)

        # Automatically create default 3D virtual monitor plane if not yet created
        if monitor_corners is None:
            monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
                head_center, R_final, face_landmarks, w, h
            )

        # ==== DRAW LEFT AND RIGHT GAZE ====
        gaze_color = (55, 255, 0) if (left_sphere_locked and right_sphere_locked) else (0, 215, 255)
        draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, gaze_color, 130)   
        draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, gaze_color, 130)  

        # ==== COMPUTE COMBINED GAZE DIRECTION FOR SCREEN MAPPING ====
        left_gaze_dir = iris_3d_left - sphere_world_l
        norm_l = np.linalg.norm(left_gaze_dir)
        if norm_l > 1e-6: left_gaze_dir /= norm_l
        
        right_gaze_dir = iris_3d_right - sphere_world_r
        norm_r = np.linalg.norm(right_gaze_dir)
        if norm_r > 1e-6: right_gaze_dir /= norm_r
        
        raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        raw_norm = np.linalg.norm(raw_combined_direction)
        if raw_norm > 1e-6: raw_combined_direction /= raw_norm

        combined_gaze_directions.append(raw_combined_direction)
        avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
        avg_norm = np.linalg.norm(avg_combined_direction)
        if avg_norm > 1e-6: avg_combined_direction /= avg_norm

        combined_dir = avg_combined_direction

        # ==== CONVERT GAZE TO SCREEN COORDINATES ====
        screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            avg_combined_direction, 
            calibration_offset_yaw, 
            calibration_offset_pitch
        )
        _record_calibration_sample(
            raw_yaw, raw_pitch,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            left_dir=left_gaze_dir,
            right_dir=right_gaze_dir,
            frame_quality_ok=frame_quality.usable_for_gaze,
        )
        if frame_quality.usable_for_gaze:
            session_gaze_xy = (screen_x / max(MONITOR_WIDTH - 1, 1), screen_y / max(MONITOR_HEIGHT - 1, 1))
            screen_coords_for_hud = (screen_x, screen_y)

        if mouse_control_enabled and (left_sphere_locked and right_sphere_locked) and frame_quality.usable_for_gaze:
            with mouse_lock:
                mouse_target[0] = screen_x
                mouse_target[1] = screen_y

        if frame_quality.usable_for_gaze:
            write_screen_position(screen_x, screen_y)

        # Draw combined gaze ray
        combined_origin = (sphere_world_l + sphere_world_r) / 2
        combined_target = combined_origin + avg_combined_direction * gaze_length
        cv2.line(
            frame,
            (int(combined_origin[0]), int(combined_origin[1])),
            (int(combined_target[0]), int(combined_target[1])),
            (255, 255, 10), 3
        )

        # Draw all landmark points in white
        for idx, lm in enumerate(face_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

        # Smooth orbit controls each frame
        update_orbit_from_keys()

        # Build 3D landmarks in your existing scale (x*w, y*h, z*w)
        landmarks3d = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

        render_debug_view_orbit(
            h, w,
            head_center3d=head_center if 'head_center' in locals() else None,
            sphere_world_l=sphere_world_l if 'sphere_world_l' in locals() else None,
            scaled_radius_l=scaled_radius_l if 'scaled_radius_l' in locals() else None,
            sphere_world_r=sphere_world_r if 'sphere_world_r' in locals() else None,
            scaled_radius_r=scaled_radius_r if 'scaled_radius_r' in locals() else None,
            iris3d_l=iris_3d_left if 'iris_3d_left' in locals() else None,
            iris3d_r=iris_3d_right if 'iris_3d_right' in locals() else None,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            landmarks3d=landmarks3d,
            combined_dir=avg_combined_direction if 'avg_combined_direction' in locals() else None,
            gaze_len=5230,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers
        )
    else:
        debug_placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(debug_placeholder, "Head/Eye 3D Debug View", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_placeholder, "Searching for face... Position head in front of webcam", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(DEBUG_WINDOW, debug_placeholder)

    frame_duration = time.perf_counter() - loop_start_time
    fps_history.append(frame_duration)
    avg_duration = sum(fps_history) / len(fps_history) if fps_history else 0.033
    current_fps = 1.0 / avg_duration if avg_duration > 0 else 0.0
    latency_ms = avg_duration * 1000.0

    observation_time = time.monotonic()
    screening_session.observe(observation_time, face_detected, session_gaze_xy, frame_quality.flags)
    if research_task is not None:
        research_stimulus = research_task.stimulus(observation_time)
        task_event, target_xy = research_stimulus.task_event, research_stimulus.target_xy
    else:
        active_trial = attention_task.current_trial
        task_event = None if active_trial is None else f"visual_attention:{active_trial.trial_id}"
        target_xy = None if active_trial is None else (active_trial.target_x, active_trial.target_y)
    event_recorder.observe(observation_time, session_gaze_xy, session_gaze_xy is not None,
                           frame_quality.flags, task_event, target_xy)
    if attention_task.observe(observation_time, session_gaze_xy):
        os.makedirs(screening_report_dir, exist_ok=True)
        task_report_path = os.path.join(screening_report_dir, f"attention_task_{int(time.time())}.json")
        save_task_report(attention_task.report(), task_report_path)
        cv2.destroyWindow(ATTENTION_WINDOW)
        print(f"[Attention Task] Complete. Derived report: {task_report_path}")

    # Draw real-time efficiency HUD on tracking window
    draw_efficiency_hud(
        frame,
        fps=current_fps,
        latency_ms=latency_ms,
        face_detected=face_detected,
        left_locked=left_sphere_locked,
        right_locked=right_sphere_locked,
        screen_coords=screen_coords_for_hud,
        mouse_on=mouse_control_enabled,
        quality=frame_quality,
    )

    cv2.imshow(GAZE_WINDOW, frame)
    cv2.imshow(WEBCAM_WINDOW, raw_frame)
    _show_calibration_target()
    _show_attention_target()
    _show_research_task()
    _show_age_entry()
    _show_age_plan()

    # Ensure windows pop to the front on early rendered frames
    if frame_count in (1, 3, 5):
        bring_windows_to_front(topmost=always_on_top)

    # Handle global keyboard input
    f7_pressed = keyboard.is_pressed('f7')
    if f7_pressed and not f7_was_pressed:
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
    f7_was_pressed = f7_pressed

    c_pressed = keyboard.is_pressed('c')
    c_triggered = (c_pressed and not c_was_pressed)
    c_was_pressed = c_pressed
    # Calibration previously relied only on cv2.waitKey(), so pressing M while
    # another window had focus appeared to do nothing.  Match the global C-key
    # behavior while retaining normal OpenCV keyboard handling.
    m_pressed = keyboard.is_pressed('m')
    m_triggered = (m_pressed and not m_was_pressed)
    m_was_pressed = m_pressed

    key = cv2.waitKey(1) & 0xFF
    if age_entry_active:
        if key == 27:
            age_entry_active = False
            age_entry_text = ""
            cv2.destroyWindow(AGE_ENTRY_WINDOW)
            print("[Research Session] Age entry cancelled.")
        elif key in (8, 127):
            age_entry_text = age_entry_text[:-1]
        elif key in (13, 10):
            try:
                entered_age = int(age_entry_text)
                if not 0 <= entered_age <= 72:
                    raise ValueError
                if start_research_session_for_age(entered_age):
                    age_entry_active = False
                    age_entry_text = ""
                else:
                    age_entry_active = False
                    age_entry_text = ""
                    cv2.destroyWindow(AGE_ENTRY_WINDOW)
            except ValueError:
                print("[Research Session] Enter a whole number from 0 through 72.")
        elif ord('0') <= key <= ord('9') and len(age_entry_text) < 2:
            age_entry_text += chr(key)
        continue
    if key == 27 and (calibration_stage != CALIB_STAGE_IDLE or multi_calibration_active):
        calibration_stage = CALIB_STAGE_IDLE
        multi_calibration_active = False
        calibration_status = "Calibration cancelled."
        cv2.destroyWindow(CALIBRATION_WINDOW)
        print("[Calibration] Calibration cancelled.")
    elif (key in (13, 32)) and (calibration_stage == CALIB_STAGE_RESULTS):
        calibration_stage = CALIB_STAGE_IDLE
        multi_calibration_active = False
        cv2.destroyWindow(CALIBRATION_WINDOW)
        print("[Calibration] Result card dismissed; returning to tracking.")
    elif key == 27 and attention_task.active:
        attention_task.active = False
        cv2.destroyWindow(ATTENTION_WINDOW)
        print("[Attention Task] Cancelled; no report was saved.")
    elif key == 27 and research_task is not None:
        research_task = None
        cv2.destroyWindow(BIOMARKER_WINDOW)
        print("[Research Task] Cancelled; event stream remains available until the T session ends.")
    elif key == ord('q'):
        break
    elif key == ord('o'):
        always_on_top = not always_on_top
        bring_windows_to_front(topmost=always_on_top)
        print(f"[Window] Always-on-top: {'ENABLED' if always_on_top else 'DISABLED'}", flush=True)
    elif (key in (ord('m'), ord('M')) or m_triggered) and face_detected and frame_quality.usable_for_gaze and left_sphere_locked and right_sphere_locked and not attention_task.active and research_task is None:
        start_multi_point_calibration()
    elif key in (ord('m'), ord('M')) or m_triggered:
        reasons = list(frame_quality.flags)
        if not face_detected:
            reasons.append("no_face")
        if not (left_sphere_locked and right_sphere_locked):
            reasons.append("press_C_to_lock_eyes")
        if attention_task.active:
            reasons.append("attention_task_active")
        if research_task is not None:
            reasons.append("research_task_active")
        print("[Calibration] Cannot start yet: " + ", ".join(dict.fromkeys(reasons)))
    elif key == ord('t'):
        if screening_session.active:
            report = screening_session.stop(time.monotonic())
            os.makedirs(screening_report_dir, exist_ok=True)
            report_path = os.path.join(screening_report_dir, f"screening_report_{int(time.time())}.json")
            save_report(report, report_path)
            print(f"[Screening] Session stopped. Data quality: {report.data_quality}. Report: {report_path}")
            if event_recorder.active:
                event_path, event_summary = event_recorder.stop_and_save(time.monotonic())
                print(f"[Research Stream] Saved consented timestamped gaze events: {event_path} | {event_summary}")
                try:
                    if selected_age_months is None:
                        raise ValueError("No age was selected for this session")
                    age_months = selected_age_months
                    with open(event_path, encoding="utf-8") as handle:
                        biomarker_report = build_biomarker_report(json.load(handle), age_months)
                    biomarker_path = event_path.replace(".json", "_biomarkers.json")
                    with open(biomarker_path, "w", encoding="utf-8") as handle:
                        json.dump(biomarker_report, handle, indent=2)
                    print(f"[Biomarkers] Saved age-banded research report: {biomarker_path}")
                except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
                    print(f"[Biomarkers] No age-banded report created: {error}")
        else:
            age_entry_active = True
            age_entry_text = ""
            cv2.namedWindow(AGE_ENTRY_WINDOW, cv2.WINDOW_NORMAL)
    elif key == ord('v') and face_detected and frame_quality.usable_for_gaze and left_sphere_locked and right_sphere_locked and not multi_calibration_active and research_task is None:
        start_attention_task()
    elif key == ord('v'):
        print("[Attention Task] Complete eye calibration first and do not run it during gaze calibration.")
    elif key in (ord('1'), ord('2'), ord('3')) and face_detected and frame_quality.usable_for_gaze and left_sphere_locked and right_sphere_locked and not multi_calibration_active and not attention_task.active and research_task is None:
        start_research_task({ord('1'): "face_preference", ord('2'): "moving_tracking", ord('3'): "social_gaze_following"}[key])
    elif (key in (ord('c'), ord('C')) or c_triggered) and face_detected and frame_quality.usable_for_gaze:
        current_nose_scale = compute_scale(nose_points_3d)
        # Lock LEFT eye
        left_sphere_local_offset = R_final.T @ (left_eye_geometry.center - head_center)
        left_calibration_nose_scale = current_nose_scale
        left_calibration_radius = left_eye_geometry.radius
        left_sphere_locked = True

        # Lock RIGHT eye
        right_sphere_local_offset = R_final.T @ (right_eye_geometry.center - head_center)
        right_calibration_nose_scale = current_nose_scale
        right_calibration_radius = right_eye_geometry.radius
        right_sphere_locked = True
        quality_monitor.mark_calibrated(
            time.monotonic(), frame_quality.face_size_ratio,
            (frame_quality.head_yaw_deg, frame_quality.head_pitch_deg, frame_quality.head_roll_deg),
        )

        # === Create 3D monitor plane at calibration ===
        # Compute instantaneous sphere positions at calibration distance (scale=1)
        sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

        # Estimate a forward gaze direction from the two eyes
        left_dir  = iris_3d_left  - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        # Normalize (guard zero)
        if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None  # fallback to head frame

        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint  # already normalized

        monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
            head_center, R_final, face_landmarks, w, h,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir
        )

        # Freeze the debug world's orbit pivot at the calibrated monitor center
        #global debug_world_frozen, orbit_pivot_frozen
        debug_world_frozen = True
        orbit_pivot_frozen = monitor_center_w.copy()
        print("[Debug View] World pivot frozen at monitor center.")

        print(f"[Monitor] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")


        print("[Both Spheres Locked] Eye sphere calibration complete.")
    elif key == ord('c') and not face_detected:
        print("[Calibration] No face detected; wait for tracking before calibrating.")
    elif key == ord('s') and face_detected and left_sphere_locked and right_sphere_locked:
        # Screen calibration - user should look at center of screen when pressing 's'
        # Get current gaze direction
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)
        
        # Calculate what the raw angles would be without calibration
        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0  # no calibration offset
        )
        
        # Set calibration offsets to center the gaze
        calibration_offset_yaw = 0 - raw_yaw
        calibration_offset_pitch = 0 - raw_pitch
        
        print(f"[Screen Calibrated] Offset Yaw: {calibration_offset_yaw:.2f}, Offset Pitch: {calibration_offset_pitch:.2f}")
    elif key == ord('s') and not face_detected:
        print("[Screen Calibration] No face detected; wait for tracking before calibrating.")
    elif key == ord('x') and face_detected:
        # Drop a marker at the current gaze∩monitor point
        if (monitor_corners is not None and monitor_center_w is not None and monitor_normal_w is not None
            and left_sphere_locked and right_sphere_locked):
            # Recompute current eye-sphere positions (scale-aware)
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)

            # Combined gaze direction (use smoothed if available; otherwise instantaneous)
            if 'avg_combined_direction' in locals() and avg_combined_direction is not None:
                D = _normalize(np.asarray(avg_combined_direction, dtype=float))
            else:
                lg = iris_3d_left  - sphere_world_l_now
                rg = iris_3d_right - sphere_world_r_now
                if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                    print("[Marker] Gaze direction invalid; try again.")
                    D = None
                else:
                    lg /= np.linalg.norm(lg)
                    rg /= np.linalg.norm(rg)
                    D = _normalize(lg + rg)

            if D is not None:
                O = (sphere_world_l_now + sphere_world_r_now) * 0.5
                C = np.asarray(monitor_center_w, dtype=float)
                N = _normalize(np.asarray(monitor_normal_w, dtype=float))
                denom = float(np.dot(N, D))
                if abs(denom) < 1e-6:
                    print("[Marker] Gaze ray parallel to monitor; no marker.")
                else:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t <= 0.0:
                        print("[Marker] Intersection behind/at eye; no marker.")
                    else:
                        P = O + t * D  # world-space intersection
                        # Map P to monitor local (a,b), then store if inside the quad
                        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                        u = p1 - p0
                        v = p3 - p0
                        u_len2 = float(np.dot(u, u))
                        v_len2 = float(np.dot(v, v))
                        if u_len2 > 1e-9 and v_len2 > 1e-9:
                            wv = P - p0
                            a = float(np.dot(wv, u) / u_len2)
                            b = float(np.dot(wv, v) / v_len2)
                            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                                gaze_markers.append((a, b))
                                print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
                            else:
                                print("[Marker] Gaze not on monitor; no marker.")
                        else:
                            print("[Marker] Monitor dimensions degenerate; no marker.")
        else:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")
    elif key == ord('x') and not face_detected:
        print("[Marker] No face detected; wait for tracking before adding a marker.")


cap.release()

cv2.destroyAllWindows()
