import unittest

from asd_screening import failed_items, score_follow_up, score_mchat, storage_token


class TestMChatScoring(unittest.TestCase):
    def answers_with_failures(self, failures):
        answers = {item: True for item in range(1, 21)}
        for item in (2, 5, 12):
            answers[item] = False
        for item in failures:
            answers[item] = True if item in (2, 5, 12) else False
        return answers

    def test_reverse_scored_items(self):
        self.assertEqual(failed_items(self.answers_with_failures({2, 5, 12})), (2, 5, 12))

    def test_first_stage_bands(self):
        self.assertEqual(score_mchat(self.answers_with_failures({1, 3}), 20).category, "low")
        medium = score_mchat(self.answers_with_failures({1, 3, 4}), 20)
        self.assertEqual(medium.category, "medium")
        self.assertEqual(score_mchat(self.answers_with_failures(set(range(1, 9))), 20).category, "high")

    def test_age_guard_and_follow_up_threshold(self):
        with self.assertRaises(ValueError):
            score_mchat(self.answers_with_failures(set()), 15)
        failures, text = score_follow_up({1: False, 3: False, 4: True}, (1, 3, 4))
        self.assertEqual(failures, 2)
        self.assertIn("positive", text)

    def test_local_storage_identifiers_are_pseudonymous_and_safe(self):
        self.assertEqual(storage_token("child_023", "Child local ID"), "child_023")
        with self.assertRaises(ValueError):
            storage_token("Child Name", "Child local ID")


if __name__ == "__main__":
    unittest.main()
