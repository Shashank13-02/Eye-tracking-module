import unittest

from workflow_export import build_final_workflow_export, compact_webcam_parameters


class WorkflowExportTests(unittest.TestCase):
    def test_export_preserves_separate_records_without_automated_decision(self):
        webcam = compact_webcam_parameters({
            "task": "standardized_visual_attention_v1", "completed": True,
            "trials": [{"trial_id": "center", "on_target_ratio": 0.8}],
        }, "attention.json")
        screening = {
            "child_local_id": "child_001", "session_id": "session_001", "age_months": 24,
            "answers": {str(item): "yes" for item in range(1, 21)},
            "result": {"score": 3, "failed_items": [1, 2, 5], "category": "medium"},
            "follow_up": {"failed_items": 1, "outcome": "Follow-Up screen negative"},
        }
        exported = build_final_workflow_export(screening, [webcam], {"available": True, "status": "GOOD"})
        self.assertEqual(exported["mchat_rf"]["first_stage_result"]["score"], 3)
        self.assertEqual(exported["webcam_research"]["attached_reports"][0]["kind"], "visual_attention_task")
        self.assertFalse(exported["automated_decision"]["available"])
        self.assertNotIn("risk_score", exported)

    def test_event_stream_is_summarized_without_pseudonym_or_raw_coordinates(self):
        compact = compact_webcam_parameters({
            "participant_pseudonym": "do-not-export", "duration_seconds": 10,
            "coordinate_system": "normalized", "samples": [{"valid": True, "gaze_x": 0.2, "gaze_y": 0.3}],
            "events": [{"event_type": "fixation"}, {"event_type": "fixation"}],
        }, "events.json")
        self.assertEqual(compact["parameters"]["event_counts"]["fixation"], 2)
        self.assertNotIn("participant_pseudonym", str(compact))
        self.assertNotIn("gaze_x", str(compact))


if __name__ == "__main__":
    unittest.main()
