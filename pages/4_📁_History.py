"""
History Page — Session History and Longitudinal Tracking
========================================================
"""

import os
import sys
import json
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APP_TITLE, STYLES_DIR, RISK_COLORS
from report_generator import load_all_sessions, delete_session

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title=f"{APP_TITLE} — History", page_icon="📁", layout="wide")

css_path = os.path.join(STYLES_DIR, "theme.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 20px;">
    <h1 style="font-size: 2rem; margin: 0;
               background: linear-gradient(135deg, #8b5cf6, #06b6d4);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📁 Screening History
    </h1>
    <p style="color: #94a3b8; margin-top: 6px;">
        View past screening sessions and track progress over time
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load Sessions ─────────────────────────────────────────────────────
sessions = load_all_sessions()

if not sessions:
    st.markdown("""
    <div style="text-align: center; padding: 60px 0;">
        <div style="font-size: 3rem; margin-bottom: 12px;">📭</div>
        <h3 style="color: #94a3b8;">No Screening Sessions Yet</h3>
        <p style="color: #64748b;">Complete a screening to see results here.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("👶 Start New Screening", use_container_width=False):
        st.switch_page("pages/2_👶_New_Screening.py")
    st.stop()

# ── Summary Stats ─────────────────────────────────────────────────────
stat_cols = st.columns(4)
with stat_cols[0]:
    st.metric("Total Sessions", len(sessions))
with stat_cols[1]:
    unique_children = len(set(s.get("child_name", "") for s in sessions))
    st.metric("Unique Children", unique_children)
with stat_cols[2]:
    avg_score = sum(
        s.get("risk_scores", {}).get("overall", 0) for s in sessions
    ) / len(sessions) if sessions else 0
    st.metric("Avg. Overall Score", f"{avg_score:.0f}")
with stat_cols[3]:
    latest_date = sessions[0].get("session_date", "N/A")
    if latest_date != "N/A":
        try:
            dt = datetime.fromisoformat(latest_date)
            latest_date = dt.strftime("%b %d, %Y")
        except (ValueError, TypeError):
            pass
    st.metric("Latest Session", latest_date)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sessions Table ────────────────────────────────────────────────────
st.markdown("### All Sessions")

for i, session in enumerate(sessions):
    risk_scores = session.get("risk_scores", {})
    overall = risk_scores.get("overall", 0)
    levels = risk_scores.get("levels", {})
    overall_level = levels.get("overall", "unknown")
    color = RISK_COLORS.get(overall_level, "#888")

    session_date = session.get("session_date", "Unknown")
    try:
        dt = datetime.fromisoformat(session_date)
        session_date = dt.strftime("%B %d, %Y • %I:%M %p")
    except (ValueError, TypeError):
        pass

    with st.expander(
        f"{'🟢' if overall_level == 'low' else '🟡' if overall_level == 'moderate' else '🔴'} "
        f"{session.get('child_name', 'Unknown')} — "
        f"Score: {overall:.0f} ({overall_level.upper()}) — {session_date}",
        expanded=(i == 0)
    ):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown("**Child Information**")
            st.write(f"- **Name:** {session.get('child_name', 'N/A')}")
            st.write(f"- **Age:** {session.get('child_age', 'N/A')} years")
            st.write(f"- **Gender:** {session.get('child_gender', 'N/A')}")
            st.write(f"- **Age Group:** {session.get('age_group', 'N/A')}")

        with col2:
            st.markdown("**Risk Scores**")
            for key, label in [("asd", "ASD"), ("adhd", "ADHD"),
                               ("cognitive", "Cognitive"), 
                               ("social_communication", "Social Comm."),
                               ("overall", "Overall")]:
                score_val = risk_scores.get(key, 0)
                level = levels.get(key, "unknown")
                lcolor = RISK_COLORS.get(level, "#888")
                st.markdown(
                    f"<span style='color:{lcolor};font-weight:600;'>{label}: "
                    f"{score_val:.0f} ({level})</span>",
                    unsafe_allow_html=True
                )

        with col3:
            st.markdown("**Session**")
            st.write(f"Tasks: {session.get('tasks_completed', 0)}/{session.get('total_tasks', 7)}")
            duration = session.get("session_duration_s", 0)
            st.write(f"Duration: {duration:.0f}s")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", key=f"del_{session.get('session_id', i)}"):
                delete_session(session.get("session_id", ""))
                st.rerun()

# ── Longitudinal Chart (if multiple sessions for same child) ──────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📈 Longitudinal Tracking")

# Group sessions by child name
from collections import defaultdict
child_sessions = defaultdict(list)
for s in sessions:
    child_sessions[s.get("child_name", "Unknown")].append(s)

children_with_multiple = {k: v for k, v in child_sessions.items() if len(v) > 1}

if children_with_multiple:
    selected_child = st.selectbox(
        "Select Child", list(children_with_multiple.keys())
    )

    child_data = sorted(
        children_with_multiple[selected_child],
        key=lambda x: x.get("session_date", "")
    )

    dates = []
    asd_scores = []
    adhd_scores = []
    cog_scores = []
    social_scores = []
    overall_scores = []

    for s in child_data:
        try:
            dt = datetime.fromisoformat(s.get("session_date", ""))
            dates.append(dt)
        except (ValueError, TypeError):
            dates.append(None)

        rs = s.get("risk_scores", {})
        asd_scores.append(rs.get("asd", 0))
        adhd_scores.append(rs.get("adhd", 0))
        cog_scores.append(rs.get("cognitive", 0))
        social_scores.append(rs.get("social_communication", 0))
        overall_scores.append(rs.get("overall", 0))

    fig = go.Figure()
    for name, vals, color in [
        ("ASD", asd_scores, "#3b82f6"),
        ("ADHD", adhd_scores, "#f59e0b"),
        ("Cognitive", cog_scores, "#8b5cf6"),
        ("Social Comm.", social_scores, "#06b6d4"),
        ("Overall", overall_scores, "#ef4444"),
    ]:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vals) + 1)),
            y=vals,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        xaxis=dict(title="Session #", color='#94a3b8', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Risk Score", range=[0, 100], color='#94a3b8',
                   gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
    )

    # Risk threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.3)",
                  annotation_text="Low", annotation_position="right")
    fig.add_hline(y=60, line_dash="dash", line_color="rgba(239,68,68,0.3)",
                  annotation_text="Elevated", annotation_position="right")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Longitudinal tracking requires multiple sessions for the same child. "
            "Run additional screenings to see trends over time.")
