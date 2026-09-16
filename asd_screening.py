"""Local M-CHAT-R/F scoring support for clinician/research use.

This module implements the published first-stage M-CHAT-R scoring convention and
the follow-up pass/fail threshold.  It does not diagnose autism or produce a
machine-learning prediction from the bundled toddler dataset.
"""

# FUTURE CLINICAL VALIDATION GATE:
# Do not add ASD/ADHD likelihood scores, cross-instrument risk flags, or automated
# referrals here. Any future model would require an ethics-approved prospective
# study, independently clinician-established outcomes, external validation,
# calibration/fairness analysis, age-appropriate validated instruments, and
# clinician-governed deployment. ADHD additionally requires information across
# settings (for example caregiver and teacher reports), not this toddler form alone.

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


MCHAT_QUESTIONS: tuple[str, ...] = (
    "If you point at something across the room, does your child look at it?",
    "Have you ever wondered if your child might be deaf?",
    "Does your child play pretend or make-believe?",
    "Does your child like climbing on things?",
    "Does your child make unusual finger movements near his or her eyes?",
    "Does your child point with one finger to ask for something or to get help?",
    "Does your child point with one finger to show you something interesting?",
    "Is your child interested in other children?",
    "Does your child show you things by bringing them to you or holding them up for you to see - not to get help, but just to share?",
    "Does your child respond when you call his or her name?",
    "When you smile at your child, does he or she smile back at you?",
    "Does your child get upset by everyday noises?",
    "Does your child walk?",
    "Does your child look you in the eye when you are talking to him or her, playing with him or her, or dressing him or her?",
    "Does your child try to copy what you do?",
    "If you turn your head to look at something, does your child look around to see what you are looking at?",
    "Does your child try to get you to watch him or her?",
    "Does your child understand when you tell him or her to do something?",
    "If something new happens, does your child look at your face to see how you feel about it?",
    "Does your child like movement activities?",
)

# On these three items a Yes response is scored as a failed item.  On every
# other item a No response is scored as a failed item.
REVERSE_SCORED_ITEMS = frozenset({2, 5, 12})


def storage_token(value: str, label: str) -> str:
    """Validate a local pseudonymous identifier used in a report filename."""
    token = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", token):
        raise ValueError(f"{label} must be 3-64 characters: letters, numbers, hyphens, or underscores only.")
    return token


@dataclass(frozen=True)
class MChatResult:
    score: int
    failed_items: tuple[int, ...]
    category: str
    next_step: str


def failed_items(answers: Mapping[int, bool]) -> tuple[int, ...]:
    """Return M-CHAT-R failed item numbers from complete Yes/No answers.

    ``True`` represents Yes and ``False`` represents No.
    """
    missing = set(range(1, 21)) - set(answers)
    if missing:
        raise ValueError(f"All 20 answers are required; missing: {sorted(missing)}")
    failures = []
    for item in range(1, 21):
        answer_yes = bool(answers[item])
        failed = answer_yes if item in REVERSE_SCORED_ITEMS else not answer_yes
        if failed:
            failures.append(item)
    return tuple(failures)


def score_mchat(answers: Mapping[int, bool], age_months: int) -> MChatResult:
    """Score the M-CHAT-R first stage using the published 0-2/3-7/8-20 bands."""
    if not 16 <= int(age_months) <= 30:
        raise ValueError("M-CHAT-R/F is intended for toddlers aged 16 through 30 months.")
    failures = failed_items(answers)
    score = len(failures)
    if score <= 2:
        next_step = (
            "First-stage screen negative. Continue routine developmental surveillance; "
            "if the child is younger than 24 months, rescreen after the second birthday. "
            "Discuss any developmental concern with a qualified clinician."
        )
        category = "low"
    elif score <= 7:
        next_step = (
            "Follow-Up is indicated. A qualified clinician should administer the official "
            "M-CHAT-R/F interview only for the failed items before interpreting the screen."
        )
        category = "medium"
    else:
        next_step = (
            "The published tool identifies this as a high-likelihood screen. Prompt clinical "
            "developmental and early-intervention evaluation is recommended; this is not a diagnosis."
        )
        category = "high"
    return MChatResult(score, failures, category, next_step)


def score_follow_up(pass_fail: Mapping[int, bool], initial_failed_items: tuple[int, ...]) -> tuple[int, str]:
    """Score clinician-recorded official Follow-Up outcomes for initially failed items.

    ``True`` represents Pass, ``False`` represents Fail.  The official detailed
    interview remains in the supplied M-CHAT-R/F PDF; this function intentionally
    records its final item outcomes rather than recreating its decision trees.
    """
    if not initial_failed_items:
        raise ValueError("Follow-Up is only applicable after one or more failed first-stage items.")
    if set(pass_fail) != set(initial_failed_items):
        raise ValueError("Record one Pass/Fail outcome for every initially failed item.")
    failures = sum(1 for passed in pass_fail.values() if not bool(passed))
    outcome = (
        "Follow-Up screen positive: two or more Follow-Up items failed; clinical evaluation is indicated."
        if failures >= 2
        else "Follow-Up screen negative: continue developmental surveillance and address any caregiver concern."
    )
    return failures, outcome
