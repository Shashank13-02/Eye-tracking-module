import unittest

from qchat_research import score_qchat_research


class TestQChatResearch(unittest.TestCase):
    def test_documented_encoding(self):
        answers = {item: "A" for item in range(1, 11)}
        result = score_qchat_research(answers, 24)
        self.assertEqual(result.score, 1)  # A10 is encoded as one by the source rule.
        answers.update({1: "C", 2: "D", 9: "E", 10: "D"})
        result = score_qchat_research(answers, 24)
        self.assertEqual(result.score, 3)
        self.assertIn("does not establish", result.interpretation)

    def test_age_and_completion_guards(self):
        with self.assertRaises(ValueError):
            score_qchat_research({item: "A" for item in range(1, 11)}, 11)
        with self.assertRaises(ValueError):
            score_qchat_research({1: "A"}, 24)


if __name__ == "__main__":
    unittest.main()
