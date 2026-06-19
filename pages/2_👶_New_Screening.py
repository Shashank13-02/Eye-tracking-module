"""
New Screening Page — Core Screening Workflow
=============================================
Guides the clinician through:
  1. Child information input
  2. Camera setup & calibration
  3. Sequential screening tasks with real-time gaze tracking
  4. Results computation and session saving
"""

import os
import sys
import time
import uuid
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APP_TITLE, STYLES_DIR, TASK_ORDER, TASK_CONFIGS,
    MEDICAL_DISCLAIMER, CAMERA_WIDTH, CAMERA_HEIGHT,
)
from screening_tasks.task_manager import TaskManager
from screening_tasks.tasks import create_task
from gaze_features import (
    aggregate_all_features, compute_scanpath_features,
    compute_roi_features, compute_social_attention_features,
    compute_pursuit_features, compute_inhibition_features,
    compute_repetitive_pattern_index,
)
from risk_engine import RiskEngine
from report_generator import save_session, generate_report_data

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title=f"{APP_TITLE} — New Screening", page_icon="👶", layout="wide")

# Load CSS
css_path = os.path.join(STYLES_DIR, "theme.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }</style>
""", unsafe_allow_html=True)

# ── Initialize Session State ─────────────────────────────────────────
if "screening_step" not in st.session_state:
    st.session_state.screening_step = "input"  # input → tasks → complete
if "child_info" not in st.session_state:
    st.session_state.child_info = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "task_manager" not in st.session_state:
    st.session_state.task_manager = None
if "current_task_data" not in st.session_state:
    st.session_state.current_task_data = {}
if "all_task_features" not in st.session_state:
    st.session_state.all_task_features = {}
if "risk_scores" not in st.session_state:
    st.session_state.risk_scores = None
if "simulation_mode" not in st.session_state:
    st.session_state.simulation_mode = True

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 20px;">
    <h1 style="font-size: 2rem; margin: 0;
               background: linear-gradient(135deg, #3b82f6, #8b5cf6);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        👶 New Screening Session
    </h1>
    <p style="color: #94a3b8; margin-top: 6px;">
        Complete the screening assessment in 3 steps
    </p>
</div>
""", unsafe_allow_html=True)

# ── Progress Tracker ──────────────────────────────────────────────────
step_names = ["📝 Child Info", "🎯 Screening Tasks", "✅ Complete"]
current_step_idx = ["input", "tasks", "complete"].index(st.session_state.screening_step)

progress_cols = st.columns(len(step_names))
for i, (col, name) in enumerate(zip(progress_cols, step_names)):
    with col:
        if i < current_step_idx:
            color = "#22c55e"
            bg = "rgba(34, 197, 94, 0.1)"
        elif i == current_step_idx:
            color = "#3b82f6"
            bg = "rgba(59, 130, 246, 0.1)"
        else:
            color = "#475569"
            bg = "rgba(71, 85, 105, 0.05)"

        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 8px;
                    background: {bg}; border: 1px solid {color}30;">
            <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">{name}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Child Information Input
# ══════════════════════════════════════════════════════════════════════
if st.session_state.screening_step == "input":

    col_form, col_info = st.columns([3, 2])

    with col_form:
        st.markdown("### 📝 Child Information")

        with st.form("child_info_form"):
            name = st.text_input("Child Name / ID", placeholder="Enter name or ID code")
            age = st.number_input("Age (years)", min_value=2, max_value=18, value=6)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
            
            age_group_options = ["2-4", "5-7", "8-12", "13-15"]
            # Auto-select based on age
            if age <= 4:
                default_ag = 0
            elif age <= 7:
                default_ag = 1
            elif age <= 12:
                default_ag = 2
            else:
                default_ag = 3
            age_group = st.selectbox("Age Group (for reference norms)", 
                                     age_group_options, index=default_ag)

            notes = st.text_area("Clinical Notes (optional)", 
                                 placeholder="Any relevant observations or context...")

            st.markdown("---")
            
            sim_mode = st.checkbox(
                "🧪 Use Simulation Mode (demo without camera)", 
                value=True,
                help="Generates simulated gaze data for demonstration. "
                     "Uncheck to use real webcam tracking."
            )

            submitted = st.form_submit_button("▶️  Begin Screening", use_container_width=True)

            if submitted:
                st.session_state.child_info = {
                    "name": name or "Anonymous",
                    "age": age,
                    "gender": gender,
                    "age_group": age_group,
                    "notes": notes,
                }
                st.session_state.simulation_mode = sim_mode
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.task_manager = TaskManager()
                st.session_state.task_manager.start_session(st.session_state.child_info)
                st.session_state.screening_step = "tasks"
                st.session_state.all_task_features = {}
                st.session_state.risk_scores = None
                st.rerun()

    with col_info:
        st.markdown("### ℹ️ Before You Begin")
        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; padding: 20px;">
            <h4 style="color: #06b6d4; margin-bottom: 12px;">Requirements</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.8;">
                <li>🖥️ Webcam or smartphone front camera</li>
                <li>💡 Good, even lighting on the child's face</li>
                <li>📏 Child seated 40-60 cm from screen</li>
                <li>🔇 Quiet, distraction-free environment</li>
                <li>⏱️ Approximately 5-10 minutes total</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; padding: 20px;">
            <h4 style="color: #8b5cf6; margin-bottom: 12px;">7 Screening Tasks</h4>
        """, unsafe_allow_html=True)

        for task_id in TASK_ORDER:
            cfg = TASK_CONFIGS[task_id]
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;
                        border-bottom: 1px solid rgba(255,255,255,0.04);">
                <span style="font-size: 1.2rem;">{cfg['icon']}</span>
                <div>
                    <span style="color: #e2e8f0; font-size: 0.85rem; font-weight: 500;">
                        {cfg['name']}</span>
                    <span style="color: #64748b; font-size: 0.7rem;"> · {cfg['duration_seconds']}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Screening Tasks
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.screening_step == "tasks":

    tm = st.session_state.task_manager

    if tm is None or tm.is_complete:
        st.session_state.screening_step = "complete"
        st.rerun()

    task = tm.current_task
    task_id = tm.current_task_id
    cfg = TASK_CONFIGS[task_id]

    # Task Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between;
                padding: 12px 20px; background: rgba(17, 24, 39, 0.8);
                border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
                margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 2rem;">{cfg['icon']}</span>
            <div>
                <h3 style="margin: 0; color: #f1f5f9; font-size: 1.2rem;">
                    Task {tm.completed_tasks + 1}/{tm.total_tasks}: {cfg['name']}
                </h3>
                <p style="margin: 0; color: #94a3b8; font-size: 0.8rem;">
                    {cfg['description']}
                </p>
            </div>
        </div>
        <div style="text-align: right;">
            <span style="color: #3b82f6; font-size: 0.85rem; font-weight: 600;">
                {cfg['indicator']}
            </span>
            <br>
            <span style="color: #64748b; font-size: 0.75rem;">
                Duration: {cfg['duration_seconds']}s
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Overall progress
    st.progress(tm.progress, text=f"Overall: {tm.completed_tasks}/{tm.total_tasks} tasks completed")

    # Task Content
    col_stim, col_ctrl = st.columns([3, 1])

    with col_stim:
        # Generate and display stimulus
        if task_id == "smooth_pursuit":
            t_elapsed = time.time() % cfg["duration_seconds"]
            stimulus = task.get_stimulus(canvas_size=(800, 600), t=t_elapsed)
        elif task_id == "joint_attention":
            stimulus = task.get_stimulus(canvas_size=(800, 600), phase="target")
        elif task_id == "anti_saccade":
            stimulus = task.get_stimulus(canvas_size=(800, 600), phase="stimulus")
        elif task_id == "sustained_attention":
            shapes = ["circle", "square", "triangle", "star"]
            shape = shapes[tm.completed_tasks % len(shapes)]
            is_target = tm.completed_tasks % 2 == 0
            stimulus = task.get_stimulus(canvas_size=(800, 600), 
                                         shape=shape, is_target=is_target)
        else:
            stimulus = task.get_stimulus(canvas_size=(800, 600))

        # Display stimulus image
        st.image(stimulus, caption=f"Stimulus: {cfg['name']}", use_container_width=True)

    with col_ctrl:
        st.markdown("### Controls")

        # Instructions
        st.info(f"💡 {task.get_instructions()}")

        if st.session_state.simulation_mode:
            st.markdown("""
            <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2);
                        border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <span style="color: #a78bfa; font-size: 0.8rem;">
                    🧪 <strong>Simulation Mode</strong><br>
                    Generating synthetic gaze data for demo
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Run Task Button
        if st.button(f"▶️ Run & Analyze Task", use_container_width=True, key=f"run_{task_id}"):
            with st.spinner(f"Analyzing {cfg['name']}..."):
                # Generate simulated features
                features = _simulate_task_features(task_id, task.get_rois(), 
                                                    st.session_state.child_info)
                
                # Create simulated session data summary
                session_data = {
                    "fixations": [],
                    "saccades": [],
                    "blinks": [],
                    "duration_seconds": cfg["duration_seconds"],
                    "frame_size": (800, 600),
                }
                
                # Store features
                st.session_state.all_task_features[task_id] = features
                
                # Record in task manager
                tm.record_task_data(task_id, features, session_data)
                
                time.sleep(0.5)  # Brief visual pause

            st.success(f"✅ {cfg['name']} completed!")

            # Show key features
            with st.expander("📊 Task Features", expanded=True):
                for k, v in list(features.items())[:8]:
                    if isinstance(v, float):
                        st.metric(k.replace("_", " ").title(), f"{v:.2f}")
                    else:
                        st.metric(k.replace("_", " ").title(), str(v))

        st.markdown("---")

        # Next Task / Complete
        if task_id in st.session_state.all_task_features:
            if tm.completed_tasks + 1 < tm.total_tasks:
                if st.button("➡️ Next Task", use_container_width=True, key="next_task"):
                    tm.advance_to_next_task()
                    st.rerun()
            else:
                if st.button("✅ Complete Screening", use_container_width=True, 
                             key="complete_screening", type="primary"):
                    tm.advance_to_next_task()
                    tm.end_session()
                    
                    # Compute risk scores
                    flat_features = {}
                    for tid, feats in st.session_state.all_task_features.items():
                        for k, v in feats.items():
                            flat_features[f"{tid}__{k}"] = v
                    
                    age_group = st.session_state.child_info.get("age_group", "5-7")
                    engine = RiskEngine(age_group=age_group)
                    st.session_state.risk_scores = engine.compute_risk_scores(flat_features)
                    
                    # Generate recommendations
                    recommendations = engine.get_recommendations(st.session_state.risk_scores)
                    
                    # Save session
                    session_results = tm.get_session_results()
                    save_session(
                        st.session_state.session_id,
                        st.session_state.child_info,
                        session_results,
                        st.session_state.risk_scores,
                        recommendations,
                    )
                    
                    st.session_state.screening_step = "complete"
                    st.rerun()

        # Reset button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Restart Session", use_container_width=True, key="restart"):
            st.session_state.screening_step = "input"
            st.session_state.task_manager = None
            st.session_state.all_task_features = {}
            st.session_state.risk_scores = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Complete — Show Summary & Link to Results
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.screening_step == "complete":

    scores = st.session_state.risk_scores

    if scores is None:
        st.warning("No screening data available. Please run a screening first.")
        if st.button("🔄 Start New Screening"):
            st.session_state.screening_step = "input"
            st.rerun()
    else:
        # Success animation
        st.markdown("""
        <div style="text-align: center; padding: 30px 0;">
            <div style="font-size: 4rem; margin-bottom: 12px;">✅</div>
            <h2 style="color: #22c55e; margin-bottom: 8px;">Screening Complete!</h2>
            <p style="color: #94a3b8;">
                Session ID: <code style="color: #3b82f6;">{}</code>
            </p>
        </div>
        """.format(st.session_state.session_id), unsafe_allow_html=True)

        # Quick Risk Summary
        st.markdown("### Risk Summary")

        risk_cols = st.columns(5)
        risk_keys = ["asd", "adhd", "cognitive", "social_communication", "overall"]
        risk_labels = ["ASD", "ADHD", "Cognitive", "Social Comm.", "Overall"]
        risk_icons = ["🧩", "⚡", "🧮", "💬", "🧠"]

        for col, key, label, icon in zip(risk_cols, risk_keys, risk_labels, risk_icons):
            with col:
                score = scores.get(key, 0)
                level = scores.get("levels", {}).get(key, "unknown")
                
                if level == "low":
                    color = "#22c55e"
                    bg = "rgba(34, 197, 94, 0.1)"
                elif level == "moderate":
                    color = "#f59e0b"
                    bg = "rgba(245, 158, 11, 0.1)"
                else:
                    color = "#ef4444"
                    bg = "rgba(239, 68, 68, 0.1)"

                st.markdown(f"""
                <div style="background: {bg}; border: 1px solid {color}30;
                            border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div style="color: #94a3b8; font-size: 0.7rem; margin: 4px 0;">{label}</div>
                    <div style="color: {color}; font-size: 1.5rem; font-weight: 700;">
                        {score:.0f}
                    </div>
                    <div style="color: {color}; font-size: 0.7rem; font-weight: 600;
                                text-transform: uppercase;">{level}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 View Detailed Results", use_container_width=True):
                st.switch_page("pages/3_📊_Results.py")
        with col2:
            if st.button("👶 Start New Screening", use_container_width=True):
                st.session_state.screening_step = "input"
                st.session_state.task_manager = None
                st.session_state.all_task_features = {}
                st.session_state.risk_scores = None
                st.rerun()
        with col3:
            if st.button("📁 View History", use_container_width=True):
                st.switch_page("pages/4_📁_History.py")


# ══════════════════════════════════════════════════════════════════════
# Helper: Simulate Task Features (for demo mode)
# ══════════════════════════════════════════════════════════════════════
def _simulate_task_features(task_id, rois, child_info):
    """Generate realistic simulated features for demonstration."""
    np.random.seed(hash(task_id + st.session_state.session_id) % (2**31))
    
    features = {}
    
    # Common scanpath features
    features["fix_count"] = int(np.random.randint(8, 25))
    features["fix_duration_total_ms"] = float(np.random.uniform(3000, 8000))
    features["fix_duration_mean_ms"] = features["fix_duration_total_ms"] / features["fix_count"]
    features["fix_duration_median_ms"] = features["fix_duration_mean_ms"] * np.random.uniform(0.8, 1.2)
    features["fix_duration_std_ms"] = float(np.random.uniform(40, 180))
    features["fix_duration_max_ms"] = features["fix_duration_mean_ms"] + features["fix_duration_std_ms"] * 2
    features["fix_duration_min_ms"] = max(50, features["fix_duration_mean_ms"] - features["fix_duration_std_ms"])
    features["saccade_count"] = features["fix_count"] - 1
    features["saccade_amplitude_mean_px"] = float(np.random.uniform(50, 180))
    features["saccade_amplitude_std_px"] = float(np.random.uniform(20, 80))
    features["saccade_amplitude_max_px"] = features["saccade_amplitude_mean_px"] * 2
    features["saccade_duration_mean_ms"] = float(np.random.uniform(20, 60))
    features["scanpath_length_px"] = features["saccade_amplitude_mean_px"] * features["saccade_count"]
    features["fix_distance_to_center_mean_px"] = float(np.random.uniform(80, 250))
    features["fix_distance_to_center_std_px"] = float(np.random.uniform(30, 100))
    features["fix_spatial_dispersion_px"] = float(np.random.uniform(50, 180))
    features["gaze_range_x_px"] = float(np.random.uniform(200, 600))
    features["gaze_range_y_px"] = float(np.random.uniform(150, 450))
    features["gaze_range_ratio"] = (features["gaze_range_x_px"] * features["gaze_range_y_px"]) / (800 * 600)
    features["fixation_rate_per_sec"] = features["fix_count"] / (features["fix_duration_total_ms"] / 1000)
    features["blink_count"] = int(np.random.randint(1, 6))
    features["blink_duration_mean_ms"] = float(np.random.uniform(100, 300))
    features["repetitive_scan_index"] = float(np.random.uniform(0.05, 0.35))
    features["revisit_ratio"] = float(np.random.uniform(0.1, 0.5))
    features["session_duration_s"] = float(TASK_CONFIGS[task_id]["duration_seconds"])

    # Task-specific features
    if task_id == "face_preference":
        features["face_preference_ratio"] = float(np.random.uniform(0.35, 0.80))
        features["face_dwell_ratio"] = float(np.random.uniform(0.25, 0.60))
        features["object_dwell_ratio"] = float(np.random.uniform(0.15, 0.50))
        features["eye_region_fixation_ratio"] = float(np.random.uniform(0.05, 0.30))

    elif task_id == "social_scene":
        features["face_dwell_ratio"] = float(np.random.uniform(0.15, 0.50))
        features["eye_region_fixation_ratio"] = float(np.random.uniform(0.08, 0.35))
        features["object_dwell_ratio"] = float(np.random.uniform(0.10, 0.35))
        for roi_name in rois:
            features[f"roi_{roi_name}_dwell_ratio"] = float(np.random.uniform(0.05, 0.40))
            features[f"roi_{roi_name}_fix_count"] = int(np.random.randint(1, 8))
            features[f"roi_{roi_name}_first_fix_latency_ms"] = float(np.random.uniform(200, 2000))

    elif task_id == "smooth_pursuit":
        features["pursuit_accuracy_mean_px"] = float(np.random.uniform(30, 150))
        features["pursuit_accuracy_std_px"] = float(np.random.uniform(15, 60))
        features["pursuit_gain"] = float(np.random.uniform(0.5, 1.3))

    elif task_id == "joint_attention":
        for roi_name in rois:
            features[f"roi_{roi_name}_dwell_ratio"] = float(np.random.uniform(0.1, 0.5))
            features[f"roi_{roi_name}_fix_count"] = int(np.random.randint(1, 6))
            features[f"roi_{roi_name}_first_fix_latency_ms"] = float(np.random.uniform(150, 1500))

    elif task_id == "anti_saccade":
        features["anti_saccade_correct"] = int(np.random.choice([0, 1], p=[0.3, 0.7]))
        features["anti_saccade_error"] = 1 - features["anti_saccade_correct"]
        features["anti_saccade_latency_ms"] = float(np.random.uniform(150, 600))

    elif task_id == "sustained_attention":
        features["sustained_hits"] = int(np.random.randint(3, 8))
        features["sustained_misses"] = int(np.random.randint(0, 4))
        features["sustained_false_alarms"] = int(np.random.randint(0, 3))
        features["sustained_correct_rejections"] = int(np.random.randint(3, 8))
        total_t = features["sustained_hits"] + features["sustained_misses"]
        total_nt = features["sustained_false_alarms"] + features["sustained_correct_rejections"]
        features["sustained_omission_rate"] = features["sustained_misses"] / total_t if total_t > 0 else 0
        features["sustained_commission_rate"] = features["sustained_false_alarms"] / total_nt if total_nt > 0 else 0
        features["sustained_mean_rt_ms"] = float(np.random.uniform(300, 800))
        features["sustained_d_prime"] = float(np.random.uniform(0.5, 3.0))

    elif task_id == "pattern_recognition":
        for roi_name in rois:
            features[f"roi_{roi_name}_dwell_ratio"] = float(np.random.uniform(0.1, 0.4))
            features[f"roi_{roi_name}_fix_count"] = int(np.random.randint(1, 5))
            features[f"roi_{roi_name}_first_fix_latency_ms"] = float(np.random.uniform(500, 4000))

    return features
