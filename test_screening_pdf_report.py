from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asd_screening import score_mchat
from qchat_research import score_qchat_research
from screening_pdf_report import create_qchat_research_pdf, create_screening_pdf


class ScreeningPdfReportTests(unittest.TestCase):
    def test_creates_pdf_with_scoring_and_completed_answers(self) -> None:
        answers = {item: True for item in range(1, 21)}
        answers[2] = False
        answers[5] = False
        answers[12] = False
        result = score_mchat(answers, 24)
        payload = {
            "created_at": "2026-09-17T00:00:00+00:00",
            "child_local_id": "child_001",
            "session_id": "session_001",
            "age_months": 24,
            "answers": {str(item): "yes" if answer else "no" for item, answer in answers.items()},
            "result": {"score": result.score, "failed_items": list(result.failed_items), "category": result.category, "next_step": result.next_step},
            "caregiver_concerns": "No direct identifiers.",
            "follow_up": None,
            "clinician_review": None,
            "record_status": "draft_unreviewed",
        }
        with TemporaryDirectory() as directory:
            destination = create_screening_pdf(payload, Path(directory) / "report.pdf")
            content = destination.read_bytes()
        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertGreater(len(content), 2_000)

    def test_creates_separate_qchat_research_pdf(self) -> None:
        answers = {item: "C" for item in range(1, 11)}
        result = score_qchat_research(answers, 24)
        payload = {
            "created_at": "2026-09-17T00:00:00+00:00",
            "child_local_id": "child_001",
            "session_id": "session_001",
            "age_months": 24,
            "answers": {str(item): answer for item, answer in answers.items()},
            "score": result.score,
            "interpretation": result.interpretation,
        }
        with TemporaryDirectory() as directory:
            destination = create_qchat_research_pdf(payload, Path(directory) / "qchat.pdf")
            content = destination.read_bytes()
        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertGreater(len(content), 2_000)


if __name__ == "__main__":
    unittest.main()
