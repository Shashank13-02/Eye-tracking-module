"""
Gaze Analyzer — Real-time Gaze Analysis Engine
===============================================
Wraps MediaPipe Face Mesh to provide:
  - Gaze direction estimation (iris offset relative to eye contour)
  - Fixation detection (I-VT velocity-threshold algorithm)
  - Saccade detection (rapid movements between fixations)
  - Blink detection (Eye Aspect Ratio)
  - Pupil/iris metrics
  - Scanpath recording with timestamps
"""

import time
import math
import numpy as np
import cv2
import mediapipe as mp

from config import (
    FACE_MESH_MAX_FACES, FACE_MESH_REFINE_LANDMARKS,
    FACE_MESH_MIN_DETECTION_CONFIDENCE, FACE_MESH_MIN_TRACKING_CONFIDENCE,
    LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES,
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
    LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES,
    FIXATION_VELOCITY_THRESHOLD, FIXATION_MIN_DURATION_MS,
    BLINK_EAR_THRESHOLD, BLINK_CONSECUTIVE_FRAMES,
    GAZE_LEFT_THRESHOLD, GAZE_RIGHT_THRESHOLD,
    GAZE_UP_THRESHOLD, GAZE_DOWN_THRESHOLD,
)


class GazePoint:
    """A single gaze sample with timestamp."""
    __slots__ = ("x", "y", "t", "left_iris", "right_iris",
                 "left_ear", "right_ear", "gaze_direction")

    def __init__(self, x, y, t, left_iris=None, right_iris=None,
                 left_ear=1.0, right_ear=1.0, gaze_direction="center"):
        self.x = x
        self.y = y
        self.t = t
        self.left_iris = left_iris
        self.right_iris = right_iris
        self.left_ear = left_ear
        self.right_ear = right_ear
        self.gaze_direction = gaze_direction


class Fixation:
    """A detected fixation event."""
    __slots__ = ("x", "y", "start_time", "end_time", "duration_ms", "points")

    def __init__(self, x, y, start_time, end_time, points):
        self.x = x
        self.y = y
        self.start_time = start_time
        self.end_time = end_time
        self.duration_ms = (end_time - start_time) * 1000
        self.points = points


class Saccade:
    """A detected saccade event."""
    __slots__ = ("start_x", "start_y", "end_x", "end_y",
                 "amplitude", "start_time", "end_time", "duration_ms")

    def __init__(self, start_x, start_y, end_x, end_y, start_time, end_time):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.amplitude = math.hypot(end_x - start_x, end_y - start_y)
        self.start_time = start_time
        self.end_time = end_time
        self.duration_ms = (end_time - start_time) * 1000


class BlinkEvent:
    """A detected blink event."""
    __slots__ = ("start_time", "end_time", "duration_ms")

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration_ms = (end_time - start_time) * 1000


class GazeAnalyzer:
    """
    Real-time gaze analysis engine built on MediaPipe Face Mesh.

    Usage:
        analyzer = GazeAnalyzer()
        analyzer.start_session()
        # In frame loop:
        gaze_point = analyzer.process_frame(frame)
        # After session:
        analyzer.end_session()
        results = analyzer.get_session_data()
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None

        # Session state
        self._session_active = False
        self._gaze_points = []
        self._fixations = []
        self._saccades = []
        self._blinks = []

        # Fixation detection state
        self._fixation_buffer = []

        # Blink detection state
        self._blink_counter = 0
        self._blink_start_time = None
        self._is_blinking = False

        # Previous gaze for velocity calculation
        self._prev_gaze = None

        # Frame dimensions (set on first frame)
        self._frame_w = 0
        self._frame_h = 0

    def start_session(self):
        """Initialize a new tracking session."""
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=FACE_MESH_MAX_FACES,
            refine_landmarks=FACE_MESH_REFINE_LANDMARKS,
            min_detection_confidence=FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )
        self._session_active = True
        self._gaze_points = []
        self._fixations = []
        self._saccades = []
        self._blinks = []
        self._fixation_buffer = []
        self._blink_counter = 0
        self._blink_start_time = None
        self._is_blinking = False
        self._prev_gaze = None

    def end_session(self):
        """Finalize the current session — flush any pending fixation."""
        if self._fixation_buffer:
            self._emit_fixation()
        self._session_active = False
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None

    def process_frame(self, frame):
        """
        Process a single BGR frame. Returns a GazePoint or None if no face.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from camera.

        Returns
        -------
        GazePoint or None
        """
        if not self._session_active:
            return None

        h, w, _ = frame.shape
        self._frame_w = w
        self._frame_h = h
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        to_px = lambda lm: (lm.x * w, lm.y * h)

        # ── Iris centers ──────────────────────────────────────────────
        left_iris_center = self._landmark_center(landmarks, LEFT_IRIS_INDICES, w, h)
        right_iris_center = self._landmark_center(landmarks, RIGHT_IRIS_INDICES, w, h)

        # Average gaze point (midpoint of both irises)
        gaze_x = (left_iris_center[0] + right_iris_center[0]) / 2
        gaze_y = (left_iris_center[1] + right_iris_center[1]) / 2

        # ── Eye Aspect Ratio (EAR) for blink detection ────────────────
        left_ear = self._compute_ear(landmarks, LEFT_EYE_EAR_INDICES, w, h)
        right_ear = self._compute_ear(landmarks, RIGHT_EYE_EAR_INDICES, w, h)

        # ── Gaze direction ────────────────────────────────────────────
        left_eye_center = self._landmark_center(landmarks, LEFT_EYE_CONTOUR, w, h)
        right_eye_center = self._landmark_center(landmarks, RIGHT_EYE_CONTOUR, w, h)
        gaze_dir = self._classify_gaze_direction(
            left_iris_center, left_eye_center,
            right_iris_center, right_eye_center,
            landmarks, w, h
        )

        # ── Create gaze point ────────────────────────────────────────
        gp = GazePoint(
            x=gaze_x, y=gaze_y, t=now,
            left_iris=left_iris_center, right_iris=right_iris_center,
            left_ear=left_ear, right_ear=right_ear,
            gaze_direction=gaze_dir,
        )
        self._gaze_points.append(gp)

        # ── Blink detection ───────────────────────────────────────────
        avg_ear = (left_ear + right_ear) / 2
        self._detect_blink(avg_ear, now)

        # ── Fixation / saccade detection ──────────────────────────────
        self._detect_fixation_saccade(gp)

        self._prev_gaze = gp
        return gp

    # ─── Visualization ────────────────────────────────────────────────────

    def draw_overlay(self, frame, gaze_point):
        """Draw gaze tracking overlay on frame."""
        if gaze_point is None:
            return frame

        vis = frame.copy()
        h, w, _ = vis.shape

        # Draw iris circles
        if gaze_point.left_iris:
            cx, cy = int(gaze_point.left_iris[0]), int(gaze_point.left_iris[1])
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 1, cv2.LINE_AA)
        if gaze_point.right_iris:
            cx, cy = int(gaze_point.right_iris[0]), int(gaze_point.right_iris[1])
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw gaze crosshair
        gx, gy = int(gaze_point.x), int(gaze_point.y)
        cv2.drawMarker(vis, (gx, gy), (0, 200, 255), cv2.MARKER_CROSS, 15, 2)

        # Draw gaze direction text
        cv2.putText(vis, f"Gaze: {gaze_point.gaze_direction}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        # Draw EAR
        avg_ear = (gaze_point.left_ear + gaze_point.right_ear) / 2
        color = (0, 0, 255) if avg_ear < BLINK_EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(vis, f"EAR: {avg_ear:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw fixation count
        cv2.putText(vis, f"Fixations: {len(self._fixations)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(vis, f"Blinks: {len(self._blinks)}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return vis

    # ─── Session Data Retrieval ───────────────────────────────────────────

    def get_session_data(self):
        """Return all session data as a dictionary."""
        return {
            "gaze_points": self._gaze_points,
            "fixations": self._fixations,
            "saccades": self._saccades,
            "blinks": self._blinks,
            "duration_seconds": self._get_session_duration(),
            "frame_size": (self._frame_w, self._frame_h),
        }

    def get_scanpath_array(self):
        """Return scanpath as a numpy structured array."""
        if not self._fixations:
            return np.array([], dtype=[("x", float), ("y", float), ("duration", float)])

        data = [(f.x, f.y, f.duration_ms) for f in self._fixations]
        return np.array(data, dtype=[("x", float), ("y", float), ("duration", float)])

    def _get_session_duration(self):
        if not self._gaze_points:
            return 0.0
        return self._gaze_points[-1].t - self._gaze_points[0].t

    # ─── Internal Methods ─────────────────────────────────────────────────

    @staticmethod
    def _landmark_center(landmarks, indices, w, h):
        """Compute the pixel center of a set of landmarks."""
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _compute_ear(landmarks, ear_indices, w, h):
        """Compute the Eye Aspect Ratio (EAR)."""
        def dist(i1, i2):
            p1 = landmarks[i1]
            p2 = landmarks[i2]
            return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

        # Vertical distances
        v1 = dist(ear_indices["top"][0], ear_indices["bottom"][0])
        v2 = dist(ear_indices["top"][1], ear_indices["bottom"][1])
        # Horizontal distance
        hz = dist(ear_indices["left"], ear_indices["right"])

        if hz < 1e-6:
            return 1.0
        return (v1 + v2) / (2.0 * hz)

    def _classify_gaze_direction(self, l_iris, l_eye, r_iris, r_eye, landmarks, w, h):
        """Classify gaze direction based on iris offset from eye center."""
        # Average offsets for both eyes
        # Compute eye widths for normalization
        l_width = math.hypot(
            (landmarks[LEFT_EYE_CONTOUR[0]].x - landmarks[LEFT_EYE_CONTOUR[8]].x) * w,
            (landmarks[LEFT_EYE_CONTOUR[0]].y - landmarks[LEFT_EYE_CONTOUR[8]].y) * h,
        )
        r_width = math.hypot(
            (landmarks[RIGHT_EYE_CONTOUR[0]].x - landmarks[RIGHT_EYE_CONTOUR[8]].x) * w,
            (landmarks[RIGHT_EYE_CONTOUR[0]].y - landmarks[RIGHT_EYE_CONTOUR[8]].y) * h,
        )

        # Normalized iris offset (negative = left, positive = right)
        if l_width > 1 and r_width > 1:
            l_offset_x = (l_iris[0] - l_eye[0]) / l_width
            r_offset_x = (r_iris[0] - r_eye[0]) / r_width
            l_offset_y = (l_iris[1] - l_eye[1]) / l_width
            r_offset_y = (r_iris[1] - r_eye[1]) / r_width

            avg_x = (l_offset_x + r_offset_x) / 2
            avg_y = (l_offset_y + r_offset_y) / 2
        else:
            avg_x = 0
            avg_y = 0

        # Classify
        directions = []
        if avg_y < GAZE_UP_THRESHOLD:
            directions.append("up")
        elif avg_y > GAZE_DOWN_THRESHOLD:
            directions.append("down")
        if avg_x < GAZE_LEFT_THRESHOLD:
            directions.append("left")
        elif avg_x > GAZE_RIGHT_THRESHOLD:
            directions.append("right")

        if not directions:
            return "center"
        return "-".join(directions)

    def _detect_blink(self, avg_ear, now):
        """Detect blinks via EAR threshold crossing."""
        if avg_ear < BLINK_EAR_THRESHOLD:
            if not self._is_blinking:
                self._is_blinking = True
                self._blink_start_time = now
            self._blink_counter += 1
        else:
            if self._is_blinking and self._blink_counter >= BLINK_CONSECUTIVE_FRAMES:
                self._blinks.append(BlinkEvent(self._blink_start_time, now))
            self._is_blinking = False
            self._blink_counter = 0

    def _detect_fixation_saccade(self, gp):
        """I-VT fixation detection: classify sample as fixation or saccade."""
        if self._prev_gaze is None:
            self._fixation_buffer.append(gp)
            return

        dt = gp.t - self._prev_gaze.t
        if dt < 1e-6:
            dt = 1e-6

        # Velocity in pixels per second → convert to per-sample at ~30 fps
        velocity = math.hypot(gp.x - self._prev_gaze.x,
                              gp.y - self._prev_gaze.y) / dt

        # Convert to px/frame equivalent (velocity threshold is calibrated per-frame)
        velocity_per_frame = velocity / 30.0  # normalize to ~30fps

        if velocity_per_frame < FIXATION_VELOCITY_THRESHOLD:
            # Part of a fixation
            self._fixation_buffer.append(gp)
        else:
            # Saccade — emit any accumulated fixation, then record saccade
            if self._fixation_buffer:
                self._emit_fixation()
            # Record saccade
            self._saccades.append(Saccade(
                self._prev_gaze.x, self._prev_gaze.y,
                gp.x, gp.y,
                self._prev_gaze.t, gp.t
            ))
            self._fixation_buffer = [gp]

    def _emit_fixation(self):
        """Convert the fixation buffer into a Fixation event if long enough."""
        if not self._fixation_buffer:
            return

        xs = [p.x for p in self._fixation_buffer]
        ys = [p.y for p in self._fixation_buffer]
        start_t = self._fixation_buffer[0].t
        end_t = self._fixation_buffer[-1].t
        duration_ms = (end_t - start_t) * 1000

        if duration_ms >= FIXATION_MIN_DURATION_MS:
            self._fixations.append(Fixation(
                x=sum(xs) / len(xs),
                y=sum(ys) / len(ys),
                start_time=start_t,
                end_time=end_t,
                points=len(self._fixation_buffer),
            ))

        self._fixation_buffer = []
