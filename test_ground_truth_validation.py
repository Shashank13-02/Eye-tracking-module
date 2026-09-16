import unittest

from ground_truth_validation import validate_manifest


def row(participant, session, split="train"):
    return {"participant_pseudonym": participant, "session_id": session, "age_months": "24",
            "protocol_version": "v1", "assessment_instrument": "independent assessment",
            "assessment_date": "2026-09-16", "assessor_role": "qualified clinician",
            "ground_truth_status": "independent_assessment_complete", "split": split}


class TestGroundTruthValidation(unittest.TestCase):
    def test_valid_participant_level_split(self):
        self.assertTrue(validate_manifest([row("study-001", "s1"), row("study-002", "s2", "test")]).valid)

    def test_rejects_participant_split_leakage(self):
        report = validate_manifest([row("study-001", "s1", "train"), row("study-001", "s2", "test")])
        self.assertFalse(report.valid)
        self.assertTrue(any("multiple splits" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
