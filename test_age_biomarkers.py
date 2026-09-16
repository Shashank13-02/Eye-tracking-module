import unittest

from age_biomarkers import face_preference, moving_tracking, select_age_band, social_gaze_following


def sample(t, event, x, y, target=None, valid=True):
    return {"timestamp_seconds": t, "task_event": event, "gaze_x": x if valid else None,
            "gaze_y": y if valid else None, "target_x": None if target is None else target[0],
            "target_y": None if target is None else target[1], "valid": valid}


class TestAgeBiomarkers(unittest.TestCase):
    def test_age_band_disables_screen_tasks_for_young_infants(self):
        self.assertEqual(select_age_band(8).key, "0_to_8_months")
        result = face_preference([], 8)
        self.assertEqual(result.data_quality, "not_applicable")

    def test_face_preference_metric(self):
        samples = [sample(t / 10, "face_preference:trial_1:face_left", .25, .5) for t in range(12)]
        result = face_preference(samples, 18)
        self.assertEqual(result.metrics["face_dwell_ratio"], 1.0)

    def test_tracking_and_social_metrics(self):
        moving = [sample(t / 10, "moving_tracking:trial_1", .5, .5, (.5, .5)) for t in range(15)]
        self.assertEqual(moving_tracking(moving, 24).metrics["target_following_accuracy"], 1.0)
        social = [sample(t / 10, "social_gaze:target:trial_1", .2, .5, (.2, .5)) for t in range(5)]
        social += [sample(.5 + t / 10, "social_gaze:target:trial_2", .8, .5, (.8, .5)) for t in range(5)]
        self.assertEqual(social_gaze_following(social, 24).metrics["gaze_following_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
