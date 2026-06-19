"""
Results Page — Detailed Screening Results Dashboard
====================================================
Displays comprehensive results with:
  - Risk score radar chart
  - Traffic-light risk summary cards
  - Per-task feature details
  - Clinical recommendations
  - PDF export
"""

import os
import sys
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APP_TITLE, STYLES_DIR, TASK_CONFIGS, RISK_COLORS, MEDICAL_DISCLAIMER
from risk_engine import RiskEngine
from report_generator import generate_report_data, generate_pdf_bytes, load_all_sessions

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title=f"{APP_TITLE} — Results", page_icon="📊", layout="wide")

css_path = os.path.join(STYLES_DIR, "theme.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 20px;">
    <h1 style="font-size: 2rem; margin: 0;
               background: linear-gradient(135deg, #3b82f6, #06b6d4);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📊 Screening Results
    </h1>
</div>
""", unsafe_allow_html=True)

# ── Check for Data ───────────────────────────────────────────────────
scores = st.session_state.get("risk_scores")
child_info = st.session_state.get("child_info", {})
task_features = st.session_state.get("all_task_features", {})

if not scores:
    # Try loading from most recent session
    sessions = load_all_sessions()
    if sessions:
        st.info("Loading results from the most recent screening session...")
        latest = sessions[0]
        scores = latest.get("risk_scores", {})
        child_info = {
            "name": latest.get("child_name", "Unknown"),
            "age": latest.get("child_age", 0),
            "gender": latest.get("child_gender", ""),
            "age_group": latest.get("age_group", "5-7"),
        }
    else:
        st.warning("No screening results available. Please run a screening first.")
        if st.button("👶 Start New Screening"):
            st.switch_page("pages/2_👶_New_Screening.py")
        st.stop()

if not scores:
    st.warning("No risk scores available.")
    st.stop()

# ── Medical Disclaimer ───────────────────────────────────────────────
st.markdown("""
<div style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.12);
            border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;">
    <span style="color: #f87171; font-size: 0.8rem;">
        ⚠️ <strong>Screening Aid Only</strong> — Results require professional clinical interpretation
    </span>
</div>
""", unsafe_allow_html=True)

# ── Child Info Bar ────────────────────────────────────────────────────
info_cols = st.columns(4)
with info_cols[0]:
    st.metric("Child", child_info.get("name", "N/A"))
with info_cols[1]:
    st.metric("Age", f"{child_info.get('age', 'N/A')} years")
with info_cols[2]:
    st.metric("Age Group", child_info.get("age_group", "N/A"))
with info_cols[3]:
    st.metric("Session", st.session_state.get("session_id", "N/A"))

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# RISK OVERVIEW
# ══════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Risk Assessment Overview")

tab_overview, tab_details, tab_tasks, tab_report = st.tabs([
    "📈 Overview", "🔬 Detailed Scores", "📋 Task Results", "📄 Report"
])

with tab_overview:
    col_radar, col_cards = st.columns([3, 2])

    with col_radar:
        # Radar Chart
        categories = ["ASD", "ADHD", "Cognitive\nDelay", "Social\nComm."]
        values = [
            scores.get("asd", 0),
            scores.get("adhd", 0),
            scores.get("cognitive", 0),
            scores.get("social_communication", 0),
        ]

        fig = go.Figure()

        # Risk zones (background)
        fig.add_trace(go.Scatterpolar(
            r=[100, 100, 100, 100],
            theta=categories,
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.05)',
            line=dict(color='rgba(239, 68, 68, 0.15)', width=1),
            name='Elevated Zone',
            showlegend=True,
        ))
        fig.add_trace(go.Scatterpolar(
            r=[60, 60, 60, 60],
            theta=categories,
            fill='toself',
            fillcolor='rgba(245, 158, 11, 0.05)',
            line=dict(color='rgba(245, 158, 11, 0.15)', width=1),
            name='Moderate Zone',
            showlegend=True,
        ))
        fig.add_trace(go.Scatterpolar(
            r=[30, 30, 30, 30],
            theta=categories,
            fill='toself',
            fillcolor='rgba(34, 197, 94, 0.05)',
            line=dict(color='rgba(34, 197, 94, 0.15)', width=1),
            name='Low Risk Zone',
            showlegend=True,
        ))

        # Actual scores
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.15)',
            line=dict(color='#3b82f6', width=3),
            name='Risk Profile',
            marker=dict(size=8, color='#3b82f6'),
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor='rgba(255,255,255,0.06)',
                    linecolor='rgba(255,255,255,0.06)',
                    tickfont=dict(color='#64748b', size=10),
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.06)',
                    linecolor='rgba(255,255,255,0.06)',
                    tickfont=dict(color='#e2e8f0', size=12),
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
            showlegend=True,
            legend=dict(
                font=dict(color='#94a3b8', size=10),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=60, r=60, t=30, b=30),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_cards:
        # Overall Score Card
        overall = scores.get("overall", 0)
        overall_level = scores.get("levels", {}).get("overall", "low")
        overall_color = RISK_COLORS.get(overall_level, "#888")

        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.85); border: 2px solid {overall_color}40;
                    border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 16px;
                    box-shadow: 0 0 25px {overall_color}15;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">🧠</div>
            <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
                        letter-spacing: 0.05em;">Overall Risk</div>
            <div style="color: {overall_color}; font-size: 3rem; font-weight: 800;
                        margin: 8px 0;">{overall:.0f}</div>
            <div style="background: {overall_color}20; color: {overall_color};
                        display: inline-block; padding: 4px 16px; border-radius: 20px;
                        font-weight: 700; text-transform: uppercase; font-size: 0.8rem;
                        border: 1px solid {overall_color}30;">
                {overall_level}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Individual Score Cards
        sub_scores = [
            ("🧩", "ASD", "asd"),
            ("⚡", "ADHD", "adhd"),
            ("🧮", "Cognitive", "cognitive"),
            ("💬", "Social Comm.", "social_communication"),
        ]

        for icon, label, key in sub_scores:
            score_val = scores.get(key, 0)
            level = scores.get("levels", {}).get(key, "low")
            color = RISK_COLORS.get(level, "#888")

            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
                        display: flex; align-items: center; justify-content: space-between;
                        border-left: 4px solid {color};">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <span style="color: #e2e8f0; font-weight: 500; font-size: 0.9rem;">{label}</span>
                </div>
                <div style="text-align: right;">
                    <span style="color: {color}; font-weight: 700; font-size: 1.2rem;">{score_val:.0f}</span>
                    <span style="color: {color}; font-size: 0.65rem; font-weight: 600;
                                text-transform: uppercase; margin-left: 6px;">{level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab_details:
    st.markdown("### Detailed Risk Indicators")

    # Show each dimension's indicators
    indicators = scores.get("indicators", {})

    for dim_key, dim_label, dim_icon in [
        ("asd", "Autism Spectrum Disorder", "🧩"),
        ("adhd", "ADHD", "⚡"),
        ("cognitive", "Cognitive Delay", "🧮"),
        ("social_communication", "Social Communication", "💬"),
    ]:
        dim_indicators = indicators.get(dim_key, {})
        dim_score = scores.get(dim_key, 0)
        dim_level = scores.get("levels", {}).get(dim_key, "low")
        dim_color = RISK_COLORS.get(dim_level, "#888")

        with st.expander(f"{dim_icon} {dim_label} — Score: {dim_score:.0f} ({dim_level.upper()})", 
                        expanded=False):
            if dim_indicators:
                # Bar chart of indicators
                fig_bar = go.Figure()
                names = [k.replace("_", " ").title() for k in dim_indicators.keys()]
                vals = list(dim_indicators.values())

                colors = []
                for v in vals:
                    if v < 30:
                        colors.append("#22c55e")
                    elif v < 60:
                        colors.append("#f59e0b")
                    else:
                        colors.append("#ef4444")

                fig_bar.add_trace(go.Bar(
                    x=vals, y=names,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.0f}" for v in vals],
                    textposition='outside',
                    textfont=dict(color='#e2e8f0'),
                ))

                fig_bar.update_layout(
                    xaxis=dict(
                        range=[0, 110], title="Risk Score",
                        gridcolor='rgba(255,255,255,0.05)',
                        color='#94a3b8',
                    ),
                    yaxis=dict(color='#e2e8f0', automargin=True),
                    height=max(200, len(names) * 40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=40, t=10, b=10),
                    showlegend=False,
                )

                # Add threshold lines
                fig_bar.add_vline(x=30, line_dash="dash", line_color="rgba(34,197,94,0.3)")
                fig_bar.add_vline(x=60, line_dash="dash", line_color="rgba(239,68,68,0.3)")

                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No detailed indicators available for this dimension.")

with tab_tasks:
    st.markdown("### Per-Task Performance")

    task_results = st.session_state.get("all_task_features", {})
    
    if not task_results:
        # Try loading from session data
        sessions = load_all_sessions()
        if sessions:
            latest_features = sessions[0].get("task_features", {})
            for tid, tdata in latest_features.items():
                task_results[tid] = tdata.get("features", {})

    if task_results:
        for task_id, features in task_results.items():
            cfg = TASK_CONFIGS.get(task_id, {})
            with st.expander(f"{cfg.get('icon', '📋')} {cfg.get('name', task_id)}", expanded=False):
                col1, col2 = st.columns(2)
                
                feature_items = list(features.items()) if isinstance(features, dict) else []
                mid = len(feature_items) // 2

                with col1:
                    for k, v in feature_items[:mid]:
                        display_name = k.replace("_", " ").title()
                        if isinstance(v, float):
                            st.metric(display_name, f"{v:.2f}")
                        else:
                            st.metric(display_name, str(v))

                with col2:
                    for k, v in feature_items[mid:]:
                        display_name = k.replace("_", " ").title()
                        if isinstance(v, float):
                            st.metric(display_name, f"{v:.2f}")
                        else:
                            st.metric(display_name, str(v))
    else:
        st.info("No per-task data available. Run a screening to see detailed results.")

with tab_report:
    st.markdown("### Clinical Report")

    # Recommendations
    engine = RiskEngine(child_info.get("age_group", "5-7"))
    recommendations = engine.get_recommendations(scores)

    st.markdown("#### Recommendations")
    for rec in recommendations:
        icon = rec.get("icon", "ℹ️")
        priority = rec.get("priority", "")
        text = rec.get("text", "")

        if priority == "HIGH":
            bg = "rgba(239, 68, 68, 0.08)"
            border = "rgba(239, 68, 68, 0.2)"
        elif priority == "MEDIUM":
            bg = "rgba(245, 158, 11, 0.08)"
            border = "rgba(245, 158, 11, 0.2)"
        else:
            bg = "rgba(34, 197, 94, 0.08)"
            border = "rgba(34, 197, 94, 0.2)"

        st.markdown(f"""
        <div style="background: {bg}; border: 1px solid {border};
                    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;">
            <div style="display: flex; align-items: start; gap: 10px;">
                <span style="font-size: 1.2rem;">{icon}</span>
                <div>
                    <span style="color: #94a3b8; font-size: 0.7rem; font-weight: 600;
                                text-transform: uppercase;">[{priority}]</span>
                    <p style="color: #e2e8f0; font-size: 0.85rem; margin: 4px 0 0; line-height: 1.5;">
                        {text}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PDF Export
    st.markdown("#### Export Report")

    tm_results = {}
    if st.session_state.get("task_manager"):
        tm_results = st.session_state.task_manager.get_session_results()
    else:
        tm_results = {
            "session_duration_s": 0,
            "tasks_completed": len(task_features),
            "total_tasks": 7,
            "task_results": {},
        }

    report_data = generate_report_data(child_info, tm_results, scores, recommendations)

    pdf_bytes = generate_pdf_bytes(report_data)
    if pdf_bytes:
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"NeuroScreen_Report_{st.session_state.get('session_id', 'session')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("PDF generation requires the `fpdf2` package. Install with: `pip install fpdf2`")
