"""
Task Manager — Screening Session Orchestrator
==============================================
Manages the sequence of screening tasks, collects gaze data per task,
and aggregates results for the risk engine.
"""

import time
from collections import OrderedDict

from config import TASK_ORDER, TASK_CONFIGS
from screening_tasks.tasks import create_task


class TaskManager:
    """
    Orchestrates a complete screening session.

    Usage:
        manager = TaskManager()
        manager.start_session(child_info)
        while not manager.is_complete:
            task = manager.current_task
            stimulus = task.get_stimulus(...)
            # ... present stimulus, collect gaze ...
            manager.record_task_data(task_id, data)
            manager.advance_to_next_task()
        results = manager.get_session_results()
    """

    def __init__(self, task_order=None):
        self.task_order = task_order or TASK_ORDER
        self.tasks = OrderedDict()
        self.current_task_index = 0
        self.session_start_time = None
        self.session_end_time = None
        self.child_info = {}
        self.task_results = OrderedDict()  # task_id → {features, gaze_data, ...}
        self._session_active = False

        # Pre-create all tasks
        for task_id in self.task_order:
            self.tasks[task_id] = create_task(task_id)

    def start_session(self, child_info=None):
        """Start a new screening session."""
        self.child_info = child_info or {}
        self.session_start_time = time.time()
        self.current_task_index = 0
        self.task_results = OrderedDict()
        self._session_active = True

        # Reset all tasks
        for task in self.tasks.values():
            task.start_time = None
            task.is_active = False
            task.target_positions = []
            task.task_events = []

    def end_session(self):
        """End the current session."""
        self.session_end_time = time.time()
        self._session_active = False

    @property
    def is_active(self):
        return self._session_active

    @property
    def is_complete(self):
        return self.current_task_index >= len(self.task_order)

    @property
    def current_task_id(self):
        if self.is_complete:
            return None
        return self.task_order[self.current_task_index]

    @property
    def current_task(self):
        tid = self.current_task_id
        if tid is None:
            return None
        return self.tasks[tid]

    @property
    def progress(self):
        """Overall session progress (0.0 to 1.0)."""
        total = len(self.task_order)
        if total == 0:
            return 1.0
        return self.current_task_index / total

    @property
    def total_tasks(self):
        return len(self.task_order)

    @property
    def completed_tasks(self):
        return self.current_task_index

    def get_task_info(self, task_id):
        """Get config info for a task."""
        return TASK_CONFIGS.get(task_id, {})

    def record_task_data(self, task_id, features, session_data=None):
        """
        Record results for a completed task.

        Parameters
        ----------
        task_id : str
        features : dict — computed gaze features
        session_data : dict — raw session data (optional, for storage)
        """
        self.task_results[task_id] = {
            "features": features,
            "session_data_summary": {
                "n_fixations": len(session_data.get("fixations", [])) if session_data else 0,
                "n_saccades": len(session_data.get("saccades", [])) if session_data else 0,
                "n_blinks": len(session_data.get("blinks", [])) if session_data else 0,
                "duration_s": session_data.get("duration_seconds", 0) if session_data else 0,
            },
            "task_config": TASK_CONFIGS.get(task_id, {}),
            "timestamp": time.time(),
        }

    def advance_to_next_task(self):
        """Move to the next task in the sequence."""
        if not self.is_complete:
            self.current_task_index += 1

    def get_session_results(self):
        """
        Return complete session results.

        Returns
        -------
        dict with session metadata and per-task results
        """
        return {
            "child_info": self.child_info,
            "session_start": self.session_start_time,
            "session_end": self.session_end_time,
            "session_duration_s": (
                (self.session_end_time or time.time()) - self.session_start_time
                if self.session_start_time else 0
            ),
            "tasks_completed": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "task_results": dict(self.task_results),
        }

    def get_all_features_flat(self):
        """
        Flatten all task features into a single dictionary.

        Returns
        -------
        dict : {task_id__feature_name: value}
        """
        flat = OrderedDict()
        for task_id, result in self.task_results.items():
            for feat_name, feat_val in result.get("features", {}).items():
                flat[f"{task_id}__{feat_name}"] = feat_val
        return flat
