"""Dataset-described Q-CHAT-10-style research scoring.

The bundled Saudi toddler dataset documents an A-E response form and a binary
feature encoding.  This module preserves that transparent encoding as research
context only; it is not a diagnosis, a combined ASD likelihood, or a referral
rule for an individual child.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


QCHAT_ITEMS: tuple[tuple[str, tuple[str, str, str, str, str]], ...] = (
    ("Does your child look at you when you call his/her name?", ("Always", "Usually", "Sometimes", "Rarely", "Never")),
    ("How easy is it for you to get eye contact with your child?", ("Very easy", "Quite easy", "Quite difficult", "Very difficult", "Impossible")),
    ("Does your child point to indicate that he/she wants something?", ("Many times a day", "Few times a day", "Few times a week", "Less than once a week", "Never")),
    ("Does your child point to share interest with you?", ("Many times a day", "Few times a day", "Few times a week", "Less than once a week", "Never")),
    ("Does your child pretend?", ("Many times a day", "Few times a day", "Few times a week", "Less than once a week", "Never")),
    ("Does your child follow where you are looking?", ("Many times a day", "Few times a day", "Few times a week", "Less than once a week", "Never")),
    ("If you or someone else is visibly upset, does your child show signs of wanting to comfort them?", ("Always", "Usually", "Sometimes", "Rarely", "Never")),
    ("Would you describe your child's first word as?", ("Very typical", "Quite typical", "Slightly unusual", "Very unusual", "My child does not speak")),
    ("Does your child use simple gestures?", ("Many times a day", "Few times a day", "Few times a week", "Less than once a week", "Never")),
    ("Does your child stare at nothing with no apparent purpose?", ("Many times a day", "Few times a day", "Few times a week", "Less than once a week", "Never")),
)


@dataclass(frozen=True)
class QChatResearchResult:
    score: int
    item_codes: tuple[int, ...]
    interpretation: str


def score_qchat_research(answers: Mapping[int, str], age_months: int) -> QChatResearchResult:
    """Encode A-E responses using the bundled dataset's documented feature logic."""
    if not 12 <= int(age_months) <= 36:
        raise ValueError("The bundled dataset describes toddlers aged 12 through 36 months.")
    missing = set(range(1, 11)) - set(answers)
    if missing:
        raise ValueError(f"All 10 A-E responses are required; missing: {sorted(missing)}")
    codes: list[int] = []
    for item in range(1, 11):
        response = answers[item].upper()
        if response not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"Question {item} must use A, B, C, D, or E.")
        position = ord(response) - ord("A")
        # The source document encodes C/D/E as 1 for items 1-9, and A/B/C as
        # 1 for item 10. This is retained only to match the documented dataset.
        codes.append(int(position >= 2) if item < 10 else int(position <= 2))
    score = sum(codes)
    return QChatResearchResult(
        score=score,
        item_codes=tuple(codes),
        interpretation=(
            "Dataset-derived research score only. It is shown separately from M-CHAT-R/F and webcam observations; "
            "it does not establish an ASD likelihood, diagnosis, referral, or treatment decision."
        ),
    )
