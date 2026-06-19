"""
Centralized Configuration for NeuroScreen System
=========================================================================
All tunable parameters, thresholds, and reference norms in one place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
STIMULI_DIR = os.path.join(BASE_DIR, "screening_tasks", "stimuli")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STYLES_DIR = os.path.join(BASE_DIR, "styles")

# ─── Camera Settings ─────────────────────────────────────────────────────────
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ─── MediaPipe Face Mesh Parameters ──────────────────────────────────────────
FACE_MESH_MAX_FACES = 1
FACE_MESH_REFINE_LANDMARKS = True
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# ─── Landmark Indices ────────────────────────────────────────────────────────
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

LEFT_EYE_CONTOUR = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]
RIGHT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

# Upper/lower eyelid landmarks for EAR (Eye Aspect Ratio)
LEFT_EYE_EAR_INDICES = {
    "top": [386, 385],       # Upper eyelid
    "bottom": [374, 380],    # Lower eyelid
    "left": 263,             # Left corner
    "right": 362,            # Right corner
}
RIGHT_EYE_EAR_INDICES = {
    "top": [159, 160],
    "bottom": [145, 144],
    "left": 33,
    "right": 133,
}

# ─── Gaze Analysis Thresholds ────────────────────────────────────────────────
# Fixation detection (I-VT: velocity threshold)
FIXATION_VELOCITY_THRESHOLD = 30.0   # pixels per sample; below = fixation
FIXATION_MIN_DURATION_MS = 100       # minimum fixation duration

# Blink detection
BLINK_EAR_THRESHOLD = 0.21          # Eye Aspect Ratio below this = blink
BLINK_CONSECUTIVE_FRAMES = 2        # Minimum consecutive frames

# Gaze direction thresholds (relative iris offset as fraction of eye width)
GAZE_LEFT_THRESHOLD = -0.15
GAZE_RIGHT_THRESHOLD = 0.15
GAZE_UP_THRESHOLD = -0.12
GAZE_DOWN_THRESHOLD = 0.12

# ─── Screening Task Configuration ────────────────────────────────────────────
TASK_CONFIGS = {
    "face_preference": {
        "name": "Face vs. Object Preference",
        "description": "Measures preferential looking time toward faces vs. geometric patterns",
        "duration_seconds": 10,
        "indicator": "ASD",
        "icon": "👤",
    },
    "social_scene": {
        "name": "Social Scene Scanning",
        "description": "Tracks fixation on eyes, mouths, bodies, and background in social scenes",
        "duration_seconds": 12,
        "indicator": "ASD",
        "icon": "👥",
    },
    "smooth_pursuit": {
        "name": "Smooth Pursuit Tracking",
        "description": "Animated object moves across screen; measures tracking accuracy",
        "duration_seconds": 10,
        "indicator": "Cognitive",
        "icon": "🎯",
    },
    "joint_attention": {
        "name": "Joint Attention Response",
        "description": "Gaze cue directs attention to a target; measures response latency",
        "duration_seconds": 8,
        "indicator": "ASD/Social",
        "icon": "👀",
    },
    "anti_saccade": {
        "name": "Inhibition Task (Anti-Saccade)",
        "description": "Stimulus appears on one side, child must look to opposite side",
        "duration_seconds": 8,
        "indicator": "ADHD",
        "icon": "⚡",
    },
    "sustained_attention": {
        "name": "Sustained Attention",
        "description": "Continuous performance test measuring vigilance and attention",
        "duration_seconds": 15,
        "indicator": "ADHD",
        "icon": "⏱️",
    },
    "pattern_recognition": {
        "name": "Visual Pattern Recognition",
        "description": "Visual search for targets among distractors",
        "duration_seconds": 12,
        "indicator": "Cognitive",
        "icon": "🔍",
    },
}

# Task sequencing
TASK_ORDER = [
    "face_preference",
    "social_scene",
    "smooth_pursuit",
    "joint_attention",
    "anti_saccade",
    "sustained_attention",
    "pattern_recognition",
]

# ─── Risk Scoring Configuration ──────────────────────────────────────────────
# Weights for each sub-score (higher = more influence)
RISK_WEIGHTS = {
    "asd": {
        "face_preference_ratio": 0.25,
        "eye_region_fixation_ratio": 0.20,
        "joint_attention_latency": 0.20,
        "social_scene_face_dwell": 0.15,
        "repetitive_scan_index": 0.10,
        "gaze_range_restriction": 0.10,
    },
    "adhd": {
        "sustained_attention_score": 0.25,
        "anti_saccade_error_rate": 0.20,
        "fixation_duration_variance": 0.20,
        "omission_error_rate": 0.15,
        "commission_error_rate": 0.10,
        "hyperkinetic_gaze_index": 0.10,
    },
    "cognitive": {
        "smooth_pursuit_accuracy": 0.30,
        "visual_search_efficiency": 0.25,
        "pattern_recognition_score": 0.20,
        "fixation_count_deviation": 0.15,
        "processing_speed_index": 0.10,
    },
    "social_communication": {
        "joint_attention_score": 0.30,
        "gaze_following_accuracy": 0.25,
        "social_scene_scanning_pattern": 0.20,
        "eye_contact_duration": 0.15,
        "face_preference_ratio": 0.10,
    },
}

# Risk level thresholds (0-100 scale)
RISK_THRESHOLDS = {
    "low": (0, 30),
    "moderate": (30, 60),
    "elevated": (60, 100),
}

# Risk level colors
RISK_COLORS = {
    "low": "#22c55e",        # Green
    "moderate": "#f59e0b",   # Amber
    "elevated": "#ef4444",   # Red
}

# ─── Age-Group Reference Norms ───────────────────────────────────────────────
# Normative values by age group (placeholder ranges - should be calibrated)
AGE_GROUP_NORMS = {
    "2-4": {
        "fixation_duration_mean_ms": (200, 500),
        "fixation_count_per_task": (8, 25),
        "saccade_amplitude_mean_px": (30, 150),
        "face_preference_ratio": (0.55, 0.80),
    },
    "5-7": {
        "fixation_duration_mean_ms": (180, 450),
        "fixation_count_per_task": (10, 30),
        "saccade_amplitude_mean_px": (40, 180),
        "face_preference_ratio": (0.55, 0.80),
    },
    "8-12": {
        "fixation_duration_mean_ms": (150, 400),
        "fixation_count_per_task": (12, 35),
        "saccade_amplitude_mean_px": (50, 200),
        "face_preference_ratio": (0.55, 0.80),
    },
    "13-15": {
        "fixation_duration_mean_ms": (120, 350),
        "fixation_count_per_task": (15, 40),
        "saccade_amplitude_mean_px": (60, 220),
        "face_preference_ratio": (0.50, 0.75),
    },
}

# ─── UI Theme ─────────────────────────────────────────────────────────────────
APP_TITLE = "NeuroScreen"
APP_ICON = "🧠"
APP_SUBTITLE = "AI-Powered Neurodevelopmental Screening via Gaze Analytics"

MEDICAL_DISCLAIMER = """
⚠️ **Important Medical Disclaimer**

This tool is designed as a **screening aid only** and is **NOT** a diagnostic instrument. 
Results generated by this system should be interpreted by qualified healthcare professionals 
and must not be used as the sole basis for clinical decisions. A positive screening result 
indicates a need for further comprehensive evaluation by a specialist. This system has not 
been validated in a clinical trial and should be used for research and educational purposes only.
"""

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(SESSIONS_DIR, "screenings.db")
