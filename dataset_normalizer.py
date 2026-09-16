"""Provenance-preserving normalizer for consented NeuroGaze event streams.

External datasets must be adapted in separate, documented adapters; this module does
not pretend their hardware or task semantics match the NeuroGaze protocol.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from typing import Any


@dataclass(frozen=True)
class NormalizedGazeRecord:
    source_dataset: str
    source_session_id: str
    participant_pseudonym: str
    timestamp_seconds: float
    gaze_x_norm: float | None
    gaze_y_norm: float | None
    valid: bool
    quality_flags: str
    task_event: str | None
    target_x_norm: float | None
    target_y_norm: float | None
    protocol_version: str | None


def normalize_neurogaze_stream(stream: dict[str, Any]) -> list[NormalizedGazeRecord]:
    if stream.get("coordinate_system") != "normalized_screen_xy_0_to_1":
        raise ValueError("Only normalized NeuroGaze event streams are supported.")
    session_id = str(stream.get("session_id") or "")
    participant = str(stream.get("participant_pseudonym") or "")
    if not session_id or not participant:
        raise ValueError("Stream must have a session ID and study pseudonym.")
    records: list[NormalizedGazeRecord] = []
    for sample in stream.get("samples", []):
        records.append(NormalizedGazeRecord(
            source_dataset="neurogaze_local_research", source_session_id=session_id,
            participant_pseudonym=participant, timestamp_seconds=float(sample["timestamp_seconds"]),
            gaze_x_norm=sample.get("gaze_x"), gaze_y_norm=sample.get("gaze_y"), valid=bool(sample.get("valid")),
            quality_flags="|".join(sample.get("quality_flags", [])), task_event=sample.get("task_event"),
            target_x_norm=sample.get("target_x"), target_y_norm=sample.get("target_y"),
            protocol_version=stream.get("protocol_version"),
        ))
    return records


def normalize_neurogaze_file(input_path: str, output_csv_path: str) -> int:
    with open(input_path, encoding="utf-8") as handle:
        records = normalize_neurogaze_stream(json.load(handle))
    with open(output_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else list(NormalizedGazeRecord.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return len(records)
