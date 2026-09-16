import unittest

import numpy as np

from tracking_quality import TrackingQualityMonitor


class Landmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


def face_landmarks(*, blink=False):
    points = [Landmark(0.5, 0.5) for _ in range(478)]
    # A centered, well-lit synthetic face with open eyes.
    values = {
        33: (.30, .42), 133: (.44, .42), 159: (.37, .40), 145: (.37, .45),
        263: (.70, .42), 362: (.56, .42), 386: (.63, .40), 374: (.63, .45),
        1: (.50, .54), 13: (.50, .67), 14: (.50, .69), 468: (.37, .425), 473: (.63, .425),
    }
    if blink:
        values[159] = values[145] = (.37, .425)
        values[386] = values[374] = (.63, .425)
    for index, (x, y) in values.items():
        points[index] = Landmark(x, y)
    # Set face bounds without changing eye geometry.
    points[10], points[152] = Landmark(.42, .18), Landmark(.58, .82)
    points[234], points[454] = Landmark(.20, .50), Landmark(.80, .50)
    return points


class TestTrackingQuality(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    def test_missing_face_is_not_usable(self):
        q = TrackingQualityMonitor().evaluate(self.frame, None, 1.0)
        self.assertFalse(q.usable_for_gaze)
        self.assertEqual(q.flags, ("no_face",))

    def test_open_eyes_produce_usable_frame_before_calibration(self):
        q = TrackingQualityMonitor().evaluate(self.frame, face_landmarks(), 1.0)
        self.assertTrue(q.usable_for_gaze)
        self.assertFalse(q.blink)
        self.assertIn("not_eye_calibrated", q.flags)
        self.assertIsNone(q.face_confidence)

    def test_blink_pauses_gaze(self):
        q = TrackingQualityMonitor().evaluate(self.frame, face_landmarks(blink=True), 1.0)
        self.assertTrue(q.blink)
        self.assertFalse(q.usable_for_gaze)
        self.assertIn("blink", q.flags)

    def test_stale_calibration_pauses_gaze(self):
        monitor = TrackingQualityMonitor()
        monitor.mark_calibrated(0.0)
        q = monitor.evaluate(self.frame, face_landmarks(), monitor.MAX_CALIBRATION_AGE_SECONDS + 1)
        self.assertFalse(q.usable_for_gaze)
        self.assertIn("calibration_stale", q.flags)

    def test_distance_change_since_calibration_pauses_gaze(self):
        monitor = TrackingQualityMonitor()
        monitor.mark_calibrated(0.0, face_size_ratio=0.30, head_pose_deg=(0.0, 0.0, 0.0))
        # Synthetic face occupies 60% of the frame rather than the calibrated 30%.
        landmarks = face_landmarks()
        landmarks[234], landmarks[454] = Landmark(.05, .5), Landmark(.95, .5)
        q = monitor.evaluate(self.frame, landmarks, 1.0, (0.0, 0.0, 0.0))
        self.assertFalse(q.usable_for_gaze)
        self.assertIn("distance_changed_since_calibration", q.flags)

    def test_proxy_pose_requires_persistence_and_stays_a_nonblocking_warning(self):
        monitor = TrackingQualityMonitor()
        landmarks = face_landmarks()
        landmarks[1] = Landmark(.50, .75)
        first = monitor.evaluate(self.frame, landmarks, 1.0)
        second = monitor.evaluate(self.frame, landmarks, 1.1)
        third = monitor.evaluate(self.frame, landmarks, 1.2)
        self.assertNotIn("head_pitch_proxy_warning", first.flags)
        self.assertNotIn("head_pitch_proxy_warning", second.flags)
        self.assertIn("head_pitch_proxy_warning", third.flags)
        self.assertTrue(third.usable_for_gaze)

    def test_calibrated_pose_remains_an_immediate_safety_gate(self):
        monitor = TrackingQualityMonitor()
        q = monitor.evaluate(self.frame, face_landmarks(), 1.0, (0.0, 30.0, 0.0))
        self.assertIn("head_pitch_excessive", q.flags)
        self.assertFalse(q.usable_for_gaze)


if __name__ == "__main__":
    unittest.main()
