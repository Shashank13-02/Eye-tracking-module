"""
Report Generator — PDF & Structured Report Output
==================================================
Generates clinical screening reports with:
  - Child demographics
  - Per-task performance summaries
  - Risk score visualizations
  - Recommendations
  - PDF export via fpdf2
"""

import os
import time
import json
import sqlite3
from datetime import datetime
from collections import OrderedDict
from config import (
    SESSIONS_DIR, DB_PATH, RISK_COLORS, MEDICAL_DISCLAIMER,
    APP_TITLE, TASK_CONFIGS,
)


def ensure_db():
    """Ensure the sessions database and table exist."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            child_name TEXT,
            child_age INTEGER,
            child_gender TEXT,
            age_group TEXT,
            session_date TEXT,
            session_duration_s REAL,
            tasks_completed INTEGER,
            total_tasks INTEGER,
            risk_scores_json TEXT,
            task_features_json TEXT,
            recommendations_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_session(session_id, child_info, session_results, risk_scores, recommendations):
    """Save a screening session to the database."""
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, child_name, child_age, child_gender, age_group,
                session_date, session_duration_s, tasks_completed, total_tasks,
                risk_scores_json, task_features_json, recommendations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            child_info.get("name", "Unknown"),
            child_info.get("age", 0),
            child_info.get("gender", ""),
            child_info.get("age_group", "5-7"),
            datetime.now().isoformat(),
            session_results.get("session_duration_s", 0),
            session_results.get("tasks_completed", 0),
            session_results.get("total_tasks", 0),
            json.dumps(_make_serializable(risk_scores)),
            json.dumps(_make_serializable(session_results.get("task_results", {}))),
            json.dumps(_make_serializable(recommendations)),
        ))
        conn.commit()
    finally:
        conn.close()


def load_all_sessions():
    """Load all sessions from the database."""
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        session = dict(row)
        session["risk_scores"] = json.loads(session.pop("risk_scores_json", "{}"))
        session["task_features"] = json.loads(session.pop("task_features_json", "{}"))
        session["recommendations"] = json.loads(session.pop("recommendations_json", "[]"))
        sessions.append(session)

    return sessions


def load_session(session_id):
    """Load a single session by ID."""
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    session = dict(row)
    session["risk_scores"] = json.loads(session.pop("risk_scores_json", "{}"))
    session["task_features"] = json.loads(session.pop("task_features_json", "{}"))
    session["recommendations"] = json.loads(session.pop("recommendations_json", "[]"))
    return session


def delete_session(session_id):
    """Delete a session by ID."""
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def generate_report_data(child_info, session_results, risk_scores, recommendations):
    """
    Generate a structured report dictionary.

    Returns
    -------
    dict : complete report data for rendering
    """
    report = OrderedDict()

    # Header
    report["title"] = f"{APP_TITLE} — Screening Report"
    report["date"] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    report["disclaimer"] = MEDICAL_DISCLAIMER

    # Child Information
    report["child"] = {
        "name": child_info.get("name", "Not provided"),
        "age": child_info.get("age", "N/A"),
        "gender": child_info.get("gender", "N/A"),
        "age_group": child_info.get("age_group", "N/A"),
    }

    # Session Summary
    report["session"] = {
        "duration_s": session_results.get("session_duration_s", 0),
        "tasks_completed": session_results.get("tasks_completed", 0),
        "total_tasks": session_results.get("total_tasks", 0),
    }

    # Risk Scores
    report["risk_scores"] = {
        "asd": {
            "score": risk_scores.get("asd", 0),
            "level": risk_scores.get("levels", {}).get("asd", "unknown"),
            "color": RISK_COLORS.get(risk_scores.get("levels", {}).get("asd", "low")),
            "label": "Autism Spectrum Disorder",
        },
        "adhd": {
            "score": risk_scores.get("adhd", 0),
            "level": risk_scores.get("levels", {}).get("adhd", "unknown"),
            "color": RISK_COLORS.get(risk_scores.get("levels", {}).get("adhd", "low")),
            "label": "Attention Deficit Hyperactivity Disorder",
        },
        "cognitive": {
            "score": risk_scores.get("cognitive", 0),
            "level": risk_scores.get("levels", {}).get("cognitive", "unknown"),
            "color": RISK_COLORS.get(risk_scores.get("levels", {}).get("cognitive", "low")),
            "label": "Cognitive Delay",
        },
        "social_communication": {
            "score": risk_scores.get("social_communication", 0),
            "level": risk_scores.get("levels", {}).get("social_communication", "unknown"),
            "color": RISK_COLORS.get(
                risk_scores.get("levels", {}).get("social_communication", "low")
            ),
            "label": "Social Communication Disorder",
        },
        "overall": {
            "score": risk_scores.get("overall", 0),
            "level": risk_scores.get("levels", {}).get("overall", "unknown"),
            "color": RISK_COLORS.get(risk_scores.get("levels", {}).get("overall", "low")),
            "label": "Overall Neurodevelopmental Risk",
        },
    }

    # Per-Task Results
    report["tasks"] = OrderedDict()
    for task_id, task_result in session_results.get("task_results", {}).items():
        config = TASK_CONFIGS.get(task_id, {})
        report["tasks"][task_id] = {
            "name": config.get("name", task_id),
            "icon": config.get("icon", "📋"),
            "indicator": config.get("indicator", ""),
            "summary": task_result.get("session_data_summary", {}),
            "key_features": _extract_key_features(task_id, task_result.get("features", {})),
        }

    # Recommendations
    report["recommendations"] = recommendations

    return report


def generate_pdf_bytes(report_data):
    """
    Generate a PDF report as bytes.

    Returns
    -------
    bytes : PDF content
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, report_data["title"], new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, report_data["date"], new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 50, 50)
    pdf.multi_cell(0, 4, "SCREENING AID ONLY - Not a diagnostic instrument. "
                         "Results require clinical interpretation.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Child Information
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Child Information", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    child = report_data.get("child", {})
    for key, val in child.items():
        pdf.cell(0, 7, f"  {key.replace('_', ' ').title()}: {val}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Risk Scores
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Risk Assessment Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    for key, score_data in report_data.get("risk_scores", {}).items():
        level = score_data.get("level", "unknown").upper()
        score = score_data.get("score", 0)
        label = score_data.get("label", key)
        pdf.cell(0, 7,
                 f"  {label}: {score:.1f}/100 [{level}]",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Task Results
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Task Performance Details", new_x="LMARGIN", new_y="NEXT")

    for task_id, task_data in report_data.get("tasks", {}).items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8,
                 f"  {task_data.get('icon', '')} {task_data.get('name', task_id)}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        summary = task_data.get("summary", {})
        pdf.cell(0, 6,
                 f"    Fixations: {summary.get('n_fixations', 0)} | "
                 f"Saccades: {summary.get('n_saccades', 0)} | "
                 f"Duration: {summary.get('duration_s', 0):.1f}s",
                 new_x="LMARGIN", new_y="NEXT")

        for feat_name, feat_val in task_data.get("key_features", {}).items():
            pdf.cell(0, 5,
                     f"    - {feat_name}: {feat_val}",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Recommendations
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    for rec in report_data.get("recommendations", []):
        priority = rec.get("priority", "")
        text = rec.get("text", "")
        pdf.multi_cell(0, 6, f"  [{priority}] {text}")
        pdf.ln(3)

    # Footer disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4,
                   "Generated by NeuroScreen. This report is for screening "
                   "purposes only and should not replace professional clinical evaluation.")

    return pdf.output()


def _extract_key_features(task_id, features):
    """Extract the most clinically relevant features for a task."""
    key_feats = OrderedDict()

    if not features:
        return key_feats

    # Common features
    if "fix_count" in features:
        key_feats["Fixation Count"] = features["fix_count"]
    if "fix_duration_mean_ms" in features:
        key_feats["Mean Fixation Duration"] = f"{features['fix_duration_mean_ms']:.0f} ms"
    if "scanpath_length_px" in features:
        key_feats["Scanpath Length"] = f"{features['scanpath_length_px']:.0f} px"

    # Task-specific features
    if task_id == "face_preference":
        if "face_preference_ratio" in features:
            key_feats["Face Preference"] = f"{features['face_preference_ratio']:.1%}"
    elif task_id == "social_scene":
        if "eye_region_fixation_ratio" in features:
            key_feats["Eye Region Fixation"] = f"{features['eye_region_fixation_ratio']:.1%}"
    elif task_id == "smooth_pursuit":
        if "pursuit_accuracy_mean_px" in features:
            key_feats["Pursuit Accuracy"] = f"{features['pursuit_accuracy_mean_px']:.1f} px"
        if "pursuit_gain" in features:
            key_feats["Pursuit Gain"] = f"{features['pursuit_gain']:.2f}"
    elif task_id == "anti_saccade":
        if "anti_saccade_correct" in features:
            key_feats["Correct Response"] = "Yes" if features["anti_saccade_correct"] else "No"
    elif task_id == "sustained_attention":
        if "sustained_omission_rate" in features:
            key_feats["Omission Rate"] = f"{features['sustained_omission_rate']:.1%}"
        if "sustained_d_prime" in features:
            key_feats["d' (Sensitivity)"] = f"{features['sustained_d_prime']:.2f}"

    return key_feats


def _make_serializable(obj):
    """Convert numpy types and other non-serializable objects to Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return str(obj)
