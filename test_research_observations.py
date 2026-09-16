import json
import tempfile
import unittest
from pathlib import Path

from research_observations import calibration_summary, format_observation, load_research_observation


class TestResearchObservations(unittest.TestCase):
    def test_attention_report_is_presented_without_risk_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            path.write_text(json.dumps({
                "task": "standardized_visual_attention_v1",
                "trials": [
                    {"valid_gaze_seconds": 2.0, "on_target_ratio": 0.5},
                    {"valid_gaze_seconds": 3.0, "on_target_ratio": 0.7},
                ],
            }), encoding="utf-8")
            observation = load_research_observation(path)
            self.assertEqual(observation["kind"], "Visual-attention research task")
            text = format_observation(observation, {"available": False, "message": "Missing"})
            self.assertIn("must not be combined", text)

    def test_calibration_summary_reads_independent_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"validation": {"median_px": 100, "status": "GOOD", "normalized_median_error": 0.04}}), encoding="utf-8")
            summary = calibration_summary(path)
            self.assertTrue(summary["available"])
            self.assertEqual(summary["status"], "GOOD")


if __name__ == "__main__":
    unittest.main()
