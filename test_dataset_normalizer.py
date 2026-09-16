import unittest

from dataset_normalizer import normalize_neurogaze_stream


class TestDatasetNormalizer(unittest.TestCase):
    def test_normalizes_research_stream_with_provenance(self):
        stream = {"coordinate_system": "normalized_screen_xy_0_to_1", "session_id": "s1",
                  "participant_pseudonym": "study-001", "protocol_version": "v1",
                  "samples": [{"timestamp_seconds": 0.1, "gaze_x": 0.2, "gaze_y": 0.3, "valid": True,
                               "quality_flags": ["ok"], "task_event": "moving_tracking:trial_1",
                               "target_x": 0.2, "target_y": 0.3}]}
        record = normalize_neurogaze_stream(stream)[0]
        self.assertEqual(record.source_dataset, "neurogaze_local_research")
        self.assertEqual(record.quality_flags, "ok")


if __name__ == "__main__":
    unittest.main()
