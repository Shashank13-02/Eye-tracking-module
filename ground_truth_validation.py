"""Validate a participant-level ground-truth manifest before model development.

No participant data is collected by this module. It blocks the common leakage error
of placing sessions from one child in multiple train/validation/test partitions.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


REQUIRED_COLUMNS = {
    "participant_pseudonym", "session_id", "age_months", "protocol_version",
    "assessment_instrument", "assessment_date", "assessor_role", "ground_truth_status", "split",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_STATUS = {"pending", "independent_assessment_complete", "not_eligible"}


@dataclass(frozen=True)
class ManifestValidation:
    valid: bool
    errors: tuple[str, ...]
    participants: int
    sessions: int


def read_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header row.")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def validate_manifest(rows: Iterable[dict[str, str]]) -> ManifestValidation:
    rows = list(rows)
    errors: list[str] = []
    participant_splits: dict[str, set[str]] = defaultdict(set)
    seen_sessions: set[str] = set()
    for index, row in enumerate(rows, start=2):
        participant = row.get("participant_pseudonym", "").strip()
        session_id = row.get("session_id", "").strip()
        split = row.get("split", "").strip().lower()
        status = row.get("ground_truth_status", "").strip().lower()
        if not participant:
            errors.append(f"Row {index}: participant_pseudonym is required.")
        if not session_id or session_id in seen_sessions:
            errors.append(f"Row {index}: session_id must be unique and non-empty.")
        seen_sessions.add(session_id)
        if split not in VALID_SPLITS:
            errors.append(f"Row {index}: split must be train, validation, or test.")
        else:
            participant_splits[participant].add(split)
        if status not in VALID_STATUS:
            errors.append(f"Row {index}: invalid ground_truth_status.")
        if status == "independent_assessment_complete" and (not row.get("assessment_instrument", "").strip() or
                                                            not row.get("assessor_role", "").strip()):
            errors.append(f"Row {index}: completed independent assessment requires instrument and assessor role.")
        try:
            age = int(row.get("age_months", ""))
            if not 0 <= age <= 72:
                raise ValueError
        except ValueError:
            errors.append(f"Row {index}: age_months must be an integer from 0 through 72.")
    for participant, splits in participant_splits.items():
        if participant and len(splits) > 1:
            errors.append(f"Participant {participant!r} appears in multiple splits: {sorted(splits)}.")
    return ManifestValidation(not errors, tuple(errors), len(participant_splits), len(rows))
