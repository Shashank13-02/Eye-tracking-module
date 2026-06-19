"""
Screening Tasks — Age-Appropriate Visual Tasks for Gaze Assessment
==================================================================
Each task presents a visual stimulus and defines ROIs (regions of interest)
for the gaze feature extraction system.
"""

import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import TASK_CONFIGS


class ScreeningTask:
    """Base class for all screening tasks."""

    def __init__(self, task_id):
        self.task_id = task_id
        self.config = TASK_CONFIGS[task_id]
        self.name = self.config["name"]
        self.duration = self.config["duration_seconds"]
        self.description = self.config["description"]
        self.icon = self.config["icon"]
        self.indicator = self.config["indicator"]

        self.start_time = None
        self.is_active = False
        self.rois = {}
        self.stimulus_image = None
        self.target_positions = []  # For pursuit tasks
        self.task_events = []       # For sustained attention tasks

    def start(self):
        self.start_time = time.time()
        self.is_active = True

    def stop(self):
        self.is_active = False

    @property
    def elapsed_seconds(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def remaining_seconds(self):
        return max(0, self.duration - self.elapsed_seconds)

    @property
    def is_complete(self):
        return self.elapsed_seconds >= self.duration

    @property
    def progress(self):
        return min(1.0, self.elapsed_seconds / self.duration)

    def get_stimulus(self, canvas_size=(800, 600)):
        """Generate the stimulus image. Override in subclasses."""
        raise NotImplementedError

    def get_rois(self):
        """Return ROIs as dict of {name: (x1, y1, x2, y2)}."""
        return self.rois

    def get_instructions(self):
        """Return task instructions for the child/clinician."""
        return self.description


class FacePreferenceTask(ScreeningTask):
    """
    Face vs. Object Preference Task
    --------------------------------
    Split screen: face on one side, geometric pattern on the other.
    Measures preferential looking time toward faces.
    ASD indicator: reduced face preference.
    """

    def __init__(self):
        super().__init__("face_preference")

    def get_stimulus(self, canvas_size=(800, 600)):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (20, 20, 35))
        draw = ImageDraw.Draw(img)

        half_w = w // 2

        # Left side: Stylized face
        face_cx, face_cy = half_w // 2, h // 2
        face_r = min(half_w, h) // 3

        # Face outline
        draw.ellipse(
            (face_cx - face_r, face_cy - face_r, face_cx + face_r, face_cy + face_r),
            fill=(255, 220, 180), outline=(200, 170, 140), width=3
        )

        # Eyes
        eye_offset_x = face_r // 3
        eye_y = face_cy - face_r // 5
        eye_r = face_r // 6
        for ex in [face_cx - eye_offset_x, face_cx + eye_offset_x]:
            draw.ellipse((ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r),
                         fill="white", outline=(100, 100, 100))
            pupil_r = eye_r // 2
            draw.ellipse((ex - pupil_r, eye_y - pupil_r, ex + pupil_r, eye_y + pupil_r),
                         fill=(50, 50, 80))

        # Nose
        nose_y = face_cy + face_r // 8
        draw.polygon([(face_cx, nose_y - 8), (face_cx - 8, nose_y + 8), (face_cx + 8, nose_y + 8)],
                     fill=(230, 190, 160))

        # Smile
        mouth_y = face_cy + face_r // 3
        draw.arc(
            (face_cx - face_r // 3, mouth_y - 10, face_cx + face_r // 3, mouth_y + 20),
            0, 180, fill=(200, 100, 100), width=3
        )

        # Right side: Geometric pattern
        pattern_cx = half_w + half_w // 2
        pattern_cy = h // 2
        pat_r = min(half_w, h) // 3

        colors = [(65, 105, 225), (220, 60, 60), (50, 180, 50),
                  (255, 200, 0), (180, 80, 220)]
        for i, color in enumerate(colors):
            angle = i * (2 * math.pi / len(colors)) - math.pi / 2
            cx = pattern_cx + int(pat_r * 0.5 * math.cos(angle))
            cy = pattern_cy + int(pat_r * 0.5 * math.sin(angle))
            size = pat_r // 3
            if i % 3 == 0:
                draw.rectangle((cx - size, cy - size, cx + size, cy + size), fill=color)
            elif i % 3 == 1:
                draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=color)
            else:
                pts = []
                for j in range(6):
                    a = j * (2 * math.pi / 6) - math.pi / 2
                    pts.append((cx + int(size * math.cos(a)), cy + int(size * math.sin(a))))
                draw.polygon(pts, fill=color)

        # Additional concentric circles
        for r_mult in [0.8, 0.6, 0.4]:
            r = int(pat_r * r_mult)
            draw.ellipse(
                (pattern_cx - r, pattern_cy - r, pattern_cx + r, pattern_cy + r),
                outline=(80, 80, 120), width=2
            )

        # Divider line
        draw.line([(half_w, 0), (half_w, h)], fill=(60, 60, 80), width=2)

        # ROIs
        margin = 20
        self.rois = {
            "face": (margin, margin, half_w - margin, h - margin),
            "object": (half_w + margin, margin, w - margin, h - margin),
            "eyes": (
                face_cx - eye_offset_x - eye_r * 2,
                eye_y - eye_r * 2,
                face_cx + eye_offset_x + eye_r * 2,
                eye_y + eye_r * 2
            ),
        }

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Look at the screen naturally. There's no right or wrong answer!"


class SocialSceneTask(ScreeningTask):
    """
    Social Scene Scanning Task
    ---------------------------
    Scene with people interacting. Tracks fixation on social elements.
    ASD indicator: less eye region fixation, more object/background fixation.
    """

    def __init__(self):
        super().__init__("social_scene")

    def get_stimulus(self, canvas_size=(800, 600)):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (135, 195, 235))
        draw = ImageDraw.Draw(img)

        # Ground
        draw.rectangle((0, h * 2 // 3, w, h), fill=(120, 180, 80))

        # Sun
        sun_x, sun_y = w - 80, 60
        draw.ellipse((sun_x - 40, sun_y - 40, sun_x + 40, sun_y + 40), fill=(255, 220, 50))

        # Person 1 (left) — adult looking at child
        p1_x = w // 4
        p1_y = h * 2 // 3 - 20

        # Body
        draw.rectangle((p1_x - 25, p1_y - 80, p1_x + 25, p1_y), fill=(70, 130, 180))
        # Head
        draw.ellipse((p1_x - 20, p1_y - 120, p1_x + 20, p1_y - 80), fill=(255, 220, 180))
        # Eyes
        draw.ellipse((p1_x - 10, p1_y - 107, p1_x - 4, p1_y - 101), fill="white")
        draw.ellipse((p1_x + 4, p1_y - 107, p1_x + 10, p1_y - 101), fill="white")
        draw.ellipse((p1_x - 8, p1_y - 105, p1_x - 5, p1_y - 102), fill=(40, 40, 60))
        draw.ellipse((p1_x + 5, p1_y - 105, p1_x + 8, p1_y - 102), fill=(40, 40, 60))
        # Smile
        draw.arc((p1_x - 8, p1_y - 95, p1_x + 8, p1_y - 85), 0, 180, fill=(180, 80, 80), width=2)
        # Arms (reaching toward child)
        draw.line([(p1_x + 25, p1_y - 60), (p1_x + 60, p1_y - 70)], fill=(255, 220, 180), width=5)

        # Person 2 (center-right) — child
        p2_x = w // 2 + 30
        p2_y = h * 2 // 3 - 10

        # Body (smaller)
        draw.rectangle((p2_x - 18, p2_y - 55, p2_x + 18, p2_y), fill=(220, 80, 80))
        # Head
        draw.ellipse((p2_x - 15, p2_y - 85, p2_x + 15, p2_y - 55), fill=(255, 220, 180))
        # Eyes
        draw.ellipse((p2_x - 8, p2_y - 75, p2_x - 3, p2_y - 70), fill="white")
        draw.ellipse((p2_x + 3, p2_y - 75, p2_x + 8, p2_y - 70), fill="white")
        draw.ellipse((p2_x - 6, p2_y - 74, p2_x - 4, p2_y - 71), fill=(40, 40, 60))
        draw.ellipse((p2_x + 4, p2_y - 74, p2_x + 6, p2_y - 71), fill=(40, 40, 60))

        # Object: toy/ball on the ground
        ball_x, ball_y = w * 3 // 4, h * 2 // 3 + 15
        draw.ellipse((ball_x - 18, ball_y - 18, ball_x + 18, ball_y + 18), fill=(255, 100, 50))
        draw.arc((ball_x - 18, ball_y - 18, ball_x + 18, ball_y + 18), 0, 360, fill=(200, 60, 30), width=2)

        # Tree (background object)
        tree_x = w - 100
        tree_y = h * 2 // 3 - 20
        draw.rectangle((tree_x - 8, tree_y - 40, tree_x + 8, tree_y), fill=(139, 90, 43))
        draw.ellipse((tree_x - 35, tree_y - 90, tree_x + 35, tree_y - 30), fill=(34, 139, 34))

        # ROIs
        self.rois = {
            "face": (p1_x - 30, p1_y - 130, p2_x + 25, p2_y - 50),
            "eyes": (p1_x - 15, p1_y - 110, p2_x + 12, p2_y - 68),
            "object": (ball_x - 30, ball_y - 30, ball_x + 30, ball_y + 30),
            "background": (w - 160, 0, w, h * 2 // 3),
            "person1_face": (p1_x - 25, p1_y - 125, p1_x + 25, p1_y - 75),
            "person2_face": (p2_x - 20, p2_y - 90, p2_x + 20, p2_y - 50),
        }

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Look at what's happening in this picture."


class SmoothPursuitTask(ScreeningTask):
    """
    Smooth Pursuit Tracking Task
    ----------------------------
    Animated target moves across screen.
    Measures tracking accuracy and coordination.
    """

    def __init__(self):
        super().__init__("smooth_pursuit")
        self._target_path = []

    def get_stimulus(self, canvas_size=(800, 600), t=0.0):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (15, 15, 30))
        draw = ImageDraw.Draw(img)

        # Draw subtle grid
        for x in range(0, w, 50):
            draw.line([(x, 0), (x, h)], fill=(30, 30, 50), width=1)
        for y in range(0, h, 50):
            draw.line([(0, y), (w, y)], fill=(30, 30, 50), width=1)

        # Target position — figure-8 pattern
        period = self.duration
        phase = (t % period) / period * 2 * math.pi
        target_x = w // 2 + int((w // 3) * math.sin(phase))
        target_y = h // 2 + int((h // 4) * math.sin(2 * phase))

        # Draw target — glowing circle
        for r in range(25, 5, -3):
            alpha = int(255 * (1 - r / 25))
            color = (0, min(255, 150 + alpha), min(255, 200 + alpha // 2))
            draw.ellipse(
                (target_x - r, target_y - r, target_x + r, target_y + r),
                fill=color
            )

        # Inner bright core
        draw.ellipse(
            (target_x - 6, target_y - 6, target_x + 6, target_y + 6),
            fill=(200, 255, 255)
        )

        # Trail
        trail_len = 15
        for i in range(trail_len):
            trail_t = t - i * 0.05
            if trail_t < 0:
                continue
            trail_phase = (trail_t % period) / period * 2 * math.pi
            tx = w // 2 + int((w // 3) * math.sin(trail_phase))
            ty = h // 2 + int((h // 4) * math.sin(2 * trail_phase))
            alpha_val = int(80 * (1 - i / trail_len))
            draw.ellipse(
                (tx - 3, ty - 3, tx + 3, ty + 3),
                fill=(0, alpha_val, alpha_val + 20)
            )

        self.rois = {
            "target": (target_x - 40, target_y - 40, target_x + 40, target_y + 40),
        }

        # Record target position for pursuit analysis
        now = time.time()
        self.target_positions.append((target_x, target_y, now))

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Follow the moving dot with your eyes!"


class JointAttentionTask(ScreeningTask):
    """
    Joint Attention Response Task
    -----------------------------
    Face with gaze cue (arrow/eyes pointing) directs attention to target.
    Measures response latency.
    ASD/social communication indicator.
    """

    def __init__(self):
        super().__init__("joint_attention")
        self._cue_side = "right"  # Target appears on right

    def get_stimulus(self, canvas_size=(800, 600), phase="cue"):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (25, 25, 45))
        draw = ImageDraw.Draw(img)

        center_x, center_y = w // 2, h // 2

        # Central face (looking toward target)
        face_r = 60
        draw.ellipse(
            (center_x - face_r, center_y - face_r, center_x + face_r, center_y + face_r),
            fill=(255, 220, 180), outline=(200, 170, 140), width=2
        )

        # Eyes looking to the right
        for eye_offset in [-20, 20]:
            ex = center_x + eye_offset
            ey = center_y - 12
            draw.ellipse((ex - 10, ey - 7, ex + 10, ey + 7), fill="white")
            # Pupil shifted to the right (cue direction)
            pupil_shift = 5 if self._cue_side == "right" else -5
            draw.ellipse((ex + pupil_shift - 4, ey - 4, ex + pupil_shift + 4, ey + 4),
                         fill=(40, 40, 60))

        # Arrow cue
        if self._cue_side == "right":
            arrow_x = center_x + face_r + 30
        else:
            arrow_x = center_x - face_r - 30

        draw.polygon([
            (arrow_x, center_y - 15),
            (arrow_x + 30, center_y),
            (arrow_x, center_y + 15),
        ], fill=(255, 220, 50))

        # Target (only in "target" phase)
        if phase == "target":
            if self._cue_side == "right":
                target_x = w - 100
            else:
                target_x = 100
            target_y = center_y

            # Star target
            for r in [30, 20, 10]:
                draw.ellipse(
                    (target_x - r, target_y - r, target_x + r, target_y + r),
                    fill=(255, 215, 0) if r == 30 else (255, 235, 100)
                )

            self.rois = {
                "face": (center_x - face_r - 10, center_y - face_r - 10,
                         center_x + face_r + 10, center_y + face_r + 10),
                "target": (target_x - 40, target_y - 40, target_x + 40, target_y + 40),
                "cue": (arrow_x - 10, center_y - 20, arrow_x + 40, center_y + 20),
            }
        else:
            self.rois = {
                "face": (center_x - face_r - 10, center_y - face_r - 10,
                         center_x + face_r + 10, center_y + face_r + 10),
                "cue": (arrow_x - 10, center_y - 20, arrow_x + 40, center_y + 20),
            }

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Look where the face is looking!"


class AntiSaccadeTask(ScreeningTask):
    """
    Anti-Saccade (Inhibition) Task
    -------------------------------
    Stimulus appears on one side; child must look to the opposite side.
    ADHD/executive function indicator.
    """

    def __init__(self):
        super().__init__("anti_saccade")
        self._stimulus_side = "left"

    def get_stimulus(self, canvas_size=(800, 600), phase="fixation"):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (15, 15, 30))
        draw = ImageDraw.Draw(img)

        center_x, center_y = w // 2, h // 2

        if phase == "fixation":
            # Central fixation cross
            cross_size = 20
            draw.line([(center_x - cross_size, center_y),
                       (center_x + cross_size, center_y)],
                      fill=(200, 200, 200), width=3)
            draw.line([(center_x, center_y - cross_size),
                       (center_x, center_y + cross_size)],
                      fill=(200, 200, 200), width=3)

            # Instruction text area
            self.rois = {
                "center": (center_x - 50, center_y - 50, center_x + 50, center_y + 50),
            }

        elif phase == "stimulus":
            # Bright stimulus on one side
            if self._stimulus_side == "left":
                stim_x = 100
                correct_x = w - 100
            else:
                stim_x = w - 100
                correct_x = 100

            # Flashing stimulus
            for r in range(40, 10, -5):
                draw.ellipse(
                    (stim_x - r, center_y - r, stim_x + r, center_y + r),
                    fill=(255, 50, 50)
                )

            # Correct side marker (subtle)
            draw.rectangle(
                (correct_x - 30, center_y - 30, correct_x + 30, center_y + 30),
                outline=(50, 150, 50), width=2
            )

            self.rois = {
                "stimulus": (stim_x - 50, center_y - 50, stim_x + 50, center_y + 50),
                "correct": (correct_x - 50, center_y - 50, correct_x + 50, center_y + 50),
                "center": (center_x - 50, center_y - 50, center_x + 50, center_y + 50),
            }

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Look at the center cross. When a dot appears, look to the OPPOSITE side!"


class SustainedAttentionTask(ScreeningTask):
    """
    Sustained Attention (Continuous Performance) Task
    --------------------------------------------------
    Shapes appear sequentially. Respond (look at) targets, ignore non-targets.
    ADHD indicator: omission and commission errors.
    """

    def __init__(self):
        super().__init__("sustained_attention")
        self._current_shape = None
        self._is_target = False

    def get_stimulus(self, canvas_size=(800, 600), shape="circle", is_target=True):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (15, 15, 30))
        draw = ImageDraw.Draw(img)

        center_x, center_y = w // 2, h // 2
        self._current_shape = shape
        self._is_target = is_target

        # Shape color: green for target, blue for non-target
        color = (50, 220, 100) if is_target else (80, 120, 200)

        if shape == "circle":
            draw.ellipse(
                (center_x - 50, center_y - 50, center_x + 50, center_y + 50),
                fill=color, outline=(255, 255, 255), width=2
            )
        elif shape == "square":
            draw.rectangle(
                (center_x - 45, center_y - 45, center_x + 45, center_y + 45),
                fill=color, outline=(255, 255, 255), width=2
            )
        elif shape == "triangle":
            draw.polygon([
                (center_x, center_y - 55),
                (center_x - 50, center_y + 40),
                (center_x + 50, center_y + 40),
            ], fill=color, outline=(255, 255, 255), width=2)
        elif shape == "star":
            points = []
            for i in range(10):
                angle = i * math.pi / 5 - math.pi / 2
                r = 50 if i % 2 == 0 else 25
                points.append((
                    center_x + int(r * math.cos(angle)),
                    center_y + int(r * math.sin(angle))
                ))
            draw.polygon(points, fill=color, outline=(255, 255, 255), width=2)

        # Label (subtle)
        label = "TARGET" if is_target else ""
        if label:
            draw.text((center_x - 30, h - 40), label, fill=(100, 200, 100))

        self.rois = {
            "stimulus": (center_x - 60, center_y - 60, center_x + 60, center_y + 60),
        }

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Look at the GREEN shapes quickly! Ignore the BLUE ones."


class PatternRecognitionTask(ScreeningTask):
    """
    Visual Pattern Recognition Task
    --------------------------------
    Find the target shape among distractors.
    Cognitive delay indicator: search efficiency and strategy.
    """

    def __init__(self):
        super().__init__("pattern_recognition")

    def get_stimulus(self, canvas_size=(800, 600)):
        w, h = canvas_size
        img = Image.new("RGB", (w, h), (20, 20, 40))
        draw = ImageDraw.Draw(img)

        np.random.seed(42)

        # Target: red circle
        target_x = int(np.random.randint(100, w - 100))
        target_y = int(np.random.randint(100, h - 100))

        # Distractors: blue circles and red squares
        distractors = []
        for _ in range(12):
            dx = int(np.random.randint(50, w - 50))
            dy = int(np.random.randint(50, h - 50))
            # Avoid overlap with target
            if abs(dx - target_x) < 60 and abs(dy - target_y) < 60:
                continue
            dtype = np.random.choice(["blue_circle", "red_square"])
            distractors.append((dx, dy, dtype))

        # Draw distractors
        for dx, dy, dtype in distractors:
            if dtype == "blue_circle":
                draw.ellipse((dx - 20, dy - 20, dx + 20, dy + 20),
                             fill=(60, 100, 200), outline=(100, 140, 240), width=2)
            else:
                draw.rectangle((dx - 18, dy - 18, dx + 18, dy + 18),
                               fill=(200, 60, 60), outline=(240, 100, 100), width=2)

        # Draw target (red circle — unique conjunction)
        draw.ellipse((target_x - 22, target_y - 22, target_x + 22, target_y + 22),
                     fill=(220, 50, 50), outline=(255, 100, 100), width=3)

        # Hint border glow
        for r in range(30, 22, -1):
            c = int(50 * (30 - r) / 8)
            draw.ellipse((target_x - r, target_y - r, target_x + r, target_y + r),
                         outline=(c, 0, 0))

        # Instruction at top
        draw.text((w // 2 - 100, 15), "Find the RED CIRCLE", fill=(200, 200, 200))

        self.rois = {
            "target": (target_x - 35, target_y - 35, target_x + 35, target_y + 35),
        }

        self.stimulus_image = img
        return img

    def get_instructions(self):
        return "Find the red circle among the other shapes!"


# ─── Task Factory ──────────────────────────────────────────────────────────

TASK_CLASSES = {
    "face_preference": FacePreferenceTask,
    "social_scene": SocialSceneTask,
    "smooth_pursuit": SmoothPursuitTask,
    "joint_attention": JointAttentionTask,
    "anti_saccade": AntiSaccadeTask,
    "sustained_attention": SustainedAttentionTask,
    "pattern_recognition": PatternRecognitionTask,
}


def create_task(task_id):
    """Factory function to create a screening task by ID."""
    if task_id not in TASK_CLASSES:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_CLASSES.keys())}")
    return TASK_CLASSES[task_id]()
