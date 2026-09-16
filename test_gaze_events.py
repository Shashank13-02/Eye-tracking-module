import json
import tempfile
import unittest

from gaze_events import ResearchGazeRecorder


class TestResearchGazeRecorder(unittest.TestCase):
    def test_consent_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ResearchGazeRecorder(directory)
            with self.assertRaises(PermissionError):
                recorder.start(0.0, "study-001", consent_confirmed=False)

    def test_writes_pseudonymous_samples_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ResearchGazeRecorder(directory)
            recorder.start(0.0, "study-001", consent_confirmed=True)
            # A stable target dwell, followed by an invalid tracking gap.
            for t in (0.0, 0.1, 0.2, 0.3):
                recorder.observe(t, (0.5, 0.5), True, (), "target:center", (0.5, 0.5))
            recorder.observe(0.4, None, False, ("blink",), "target:center", (0.5, 0.5))
            recorder.observe(0.8, None, False, ("no_face",), "target:center", (0.5, 0.5))
            path, summary = recorder.stop_and_save(0.8)
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["participant_pseudonym"], "study-001")
            self.assertEqual(len(saved["samples"]), 6)
            self.assertGreaterEqual(summary["event_counts"].get("fixation", 0), 1)
            self.assertGreaterEqual(summary["event_counts"].get("dwell", 0), 1)
            self.assertGreaterEqual(summary["event_counts"].get("missing_data", 0), 1)


if __name__ == "__main__":
    unittest.main()
