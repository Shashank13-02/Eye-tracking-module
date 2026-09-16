"""Consented, pseudonymous timestamped gaze-event recording for research.

This module stores gaze coordinates only after the application receives an explicit
local consent signal. It never creates identifiers from faces, images, or names.
The output is for research analysis, not diagnostic interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from typing import Optional, Sequence
import uuid

import numpy as np


@dataclass(frozen=True)
class GazeSample:
    timestamp_seconds: float
    gaze_x: Optional[float]
    gaze_y: Optional[float]
    valid: bool
    quality_flags: tuple[str, ...]
    task_event: Optional[str]
    target_x: Optional[float]
    target_y: Optional[float]


@dataclass(frozen=True)
class TemporalEvent:
    event_type: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    task_event: Optional[str]
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None
    details: Optional[dict] = None


class ResearchGazeRecorder:
    """Records a single consented research session and derives temporal events."""

    MAX_MISSING_GAP_SECONDS = 0.25
    FIXATION_SPEED_MAX = 0.45  # normalized-screen units / second
    FIXATION_MIN_SECONDS = 0.10
    SACCADE_SPEED_MIN = 0.80
    DWELL_RADIUS = 0.12

    def __init__(self, destination_dir: str) -> None:
        self.destination_dir = destination_dir
        self.active = False
        self._started_at: Optional[float] = None
        self._session_id: Optional[str] = None
        self._participant_pseudonym: Optional[str] = None
        self._samples: list[GazeSample] = []
        self._task_targets: dict[str, tuple[float, float]] = {}

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def start(self, timestamp: float, participant_pseudonym: str, consent_confirmed: bool) -> str:
        if not consent_confirmed:
            raise PermissionError("Consent must be explicitly confirmed before recording timestamped gaze data.")
        pseudonym = participant_pseudonym.strip()
        if not pseudonym or len(pseudonym) > 80:
            raise ValueError("Provide a non-empty study pseudonym of 80 characters or fewer.")
        self.active = True
        self._started_at = float(timestamp)
        self._session_id = f"neurogaze-{uuid.uuid4().hex}"
        self._participant_pseudonym = pseudonym
        self._samples.clear()
        self._task_targets.clear()
        return self._session_id

    def observe(self, timestamp: float, gaze_xy: Optional[tuple[float, float]], valid: bool,
                quality_flags: Sequence[str] = (), task_event: Optional[str] = None,
                target_xy: Optional[tuple[float, float]] = None) -> None:
        if not self.active or self._started_at is None:
            return
        x = y = None
        if valid and gaze_xy is not None and all(math.isfinite(v) for v in gaze_xy):
            x, y = (float(np.clip(gaze_xy[0], 0.0, 1.0)), float(np.clip(gaze_xy[1], 0.0, 1.0)))
        if task_event is not None and target_xy is not None:
            self._task_targets[task_event] = (float(target_xy[0]), float(target_xy[1]))
        target_x_value = target_y_value = None
        if target_xy is not None and all(math.isfinite(v) for v in target_xy):
            target_x_value, target_y_value = float(target_xy[0]), float(target_xy[1])
        self._samples.append(GazeSample(round(max(0.0, float(timestamp) - self._started_at), 6), x, y,
                                        bool(valid and x is not None and y is not None), tuple(quality_flags), task_event,
                                        target_x_value, target_y_value))

    def stop_and_save(self, timestamp: float) -> tuple[str, dict]:
        if not self.active or self._started_at is None or self._session_id is None:
            raise RuntimeError("No event-recording session is active.")
        duration = max(0.0, float(timestamp) - self._started_at)
        events = self._derive_events(duration)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "participant_pseudonym": self._participant_pseudonym,
            "consent_required": True,
            "duration_seconds": round(duration, 3),
            "coordinate_system": "normalized_screen_xy_0_to_1",
            "clinical_interpretation": "Research event stream only; not a diagnosis or clinical risk score.",
            "samples": [asdict(sample) for sample in self._samples],
            "events": [asdict(event) for event in events],
        }
        os.makedirs(self.destination_dir, exist_ok=True)
        path = os.path.join(self.destination_dir, f"{self._session_id}.json")
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
        summary = self._event_summary(events)
        self.active = False
        self._started_at = None
        return path, summary

    def _derive_events(self, duration: float) -> list[TemporalEvent]:
        events = self._missing_data_events(duration)
        valid_runs = self._valid_runs()
        for run in valid_runs:
            events.extend(self._fixations_and_saccades(run))
        events.extend(self._dwell_revisit_latency_events())
        return sorted(events, key=lambda event: (event.start_seconds, event.event_type))

    def _valid_runs(self) -> list[list[GazeSample]]:
        runs: list[list[GazeSample]] = []
        current: list[GazeSample] = []
        previous: Optional[GazeSample] = None
        for sample in self._samples:
            discontinuity = previous is not None and sample.timestamp_seconds - previous.timestamp_seconds > self.MAX_MISSING_GAP_SECONDS
            if not sample.valid or discontinuity:
                if current:
                    runs.append(current)
                current = []
            if sample.valid:
                current.append(sample)
            previous = sample
        if current:
            runs.append(current)
        return runs

    def _missing_data_events(self, duration: float) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        started: Optional[float] = None
        previous: Optional[GazeSample] = None
        for sample in self._samples:
            if previous is not None and sample.timestamp_seconds - previous.timestamp_seconds >= self.MAX_MISSING_GAP_SECONDS:
                events.append(self._event("missing_data", previous.timestamp_seconds, sample.timestamp_seconds, previous.task_event))
            if not sample.valid and started is None:
                started = sample.timestamp_seconds
            elif sample.valid and started is not None:
                if sample.timestamp_seconds - started >= self.MAX_MISSING_GAP_SECONDS:
                    events.append(self._event("missing_data", started, sample.timestamp_seconds, None))
                started = None
            previous = sample
        if started is not None and duration - started >= self.MAX_MISSING_GAP_SECONDS:
            events.append(self._event("missing_data", started, duration, None))
        return events

    def _fixations_and_saccades(self, samples: list[GazeSample]) -> list[TemporalEvent]:
        if len(samples) < 2:
            return []
        events: list[TemporalEvent] = []
        group = [samples[0]]
        speeds: list[float] = []
        for previous, sample in zip(samples, samples[1:]):
            elapsed = sample.timestamp_seconds - previous.timestamp_seconds
            speed = math.inf if elapsed <= 1e-6 else math.dist((previous.gaze_x, previous.gaze_y), (sample.gaze_x, sample.gaze_y)) / elapsed
            if speed <= self.FIXATION_SPEED_MAX:
                group.append(sample)
                speeds.append(speed)
            else:
                events.extend(self._finish_fixation(group))
                group, speeds = [sample], []
                if speed >= self.SACCADE_SPEED_MIN:
                    events.append(self._event("saccade", previous.timestamp_seconds, sample.timestamp_seconds, sample.task_event,
                                              details={"speed_norm_per_second": round(speed, 4)}))
        events.extend(self._finish_fixation(group))
        return events

    def _finish_fixation(self, samples: list[GazeSample]) -> list[TemporalEvent]:
        if len(samples) < 2:
            return []
        duration = samples[-1].timestamp_seconds - samples[0].timestamp_seconds
        if duration < self.FIXATION_MIN_SECONDS:
            return []
        xy = np.array([(sample.gaze_x, sample.gaze_y) for sample in samples], dtype=float)
        return [self._event("fixation", samples[0].timestamp_seconds, samples[-1].timestamp_seconds,
                            samples[0].task_event, float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])),
                            {"dispersion": round(float(np.max(np.linalg.norm(xy - np.mean(xy, axis=0), axis=1))), 4)})]

    def _dwell_revisit_latency_events(self) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []
        for task_event, target in self._task_targets.items():
            all_matching = [sample for sample in self._samples if sample.task_event == task_event]
            matching = [sample for sample in all_matching if sample.valid]
            if not matching:
                continue
            inside = [math.dist((sample.gaze_x, sample.gaze_y), target) <= self.DWELL_RADIUS for sample in matching]
            dwell_start: Optional[GazeSample] = None
            dwell_count = 0
            first_hit: Optional[float] = None
            for sample, is_inside in zip(matching, inside):
                if is_inside and dwell_start is None:
                    dwell_start = sample
                    first_hit = sample.timestamp_seconds if first_hit is None else first_hit
                elif not is_inside and dwell_start is not None:
                    events.append(self._event("dwell", dwell_start.timestamp_seconds, sample.timestamp_seconds, task_event))
                    dwell_count += 1
                    dwell_start = None
            if dwell_start is not None:
                events.append(self._event("dwell", dwell_start.timestamp_seconds, matching[-1].timestamp_seconds, task_event))
                dwell_count += 1
            onset = all_matching[0].timestamp_seconds
            if first_hit is not None:
                events.append(self._event("response_latency", onset, first_hit, task_event,
                                          details={"latency_seconds": round(first_hit - onset, 4)}))
            if dwell_count > 1:
                events.append(self._event("revisit", onset, matching[-1].timestamp_seconds, task_event,
                                          details={"revisit_count": dwell_count - 1}))
        return events

    @staticmethod
    def _event(event_type: str, start: float, end: float, task_event: Optional[str], x: Optional[float] = None,
               y: Optional[float] = None, details: Optional[dict] = None) -> TemporalEvent:
        return TemporalEvent(event_type, round(start, 6), round(end, 6), round(max(0.0, end - start), 6), task_event,
                             None if x is None else round(x, 5), None if y is None else round(y, 5), details)

    @staticmethod
    def _event_summary(events: list[TemporalEvent]) -> dict:
        counts: dict[str, int] = {}
        for event in events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return {"event_counts": counts}
