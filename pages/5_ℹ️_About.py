"""
About Page — Scientific Background & System Information
========================================================
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import APP_TITLE, STYLES_DIR, MEDICAL_DISCLAIMER

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(page_title=f"{APP_TITLE} — About", page_icon="ℹ️", layout="wide")

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
               background: linear-gradient(135deg, #06b6d4, #8b5cf6);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        ℹ️ About NeuroScreen
    </h1>
</div>
""", unsafe_allow_html=True)

# ── Scientific Background ────────────────────────────────────────────
st.markdown("## 🔬 Scientific Background")

st.markdown("""
<div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px; padding: 24px; margin-bottom: 20px; line-height: 1.8;">
    <p style="color: #e2e8f0; font-size: 0.95rem;">
        Visual attention, gaze behavior, and eye-tracking metrics have been extensively studied as
        biomarkers for neurodevelopmental disorders. Research has shown that children with 
        <strong>Autism Spectrum Disorder (ASD)</strong> exhibit distinct gaze patterns compared to 
        typically developing children, including:
    </p>
    <ul style="color: #94a3b8; font-size: 0.9rem; padding-left: 20px;">
        <li>Reduced preference for faces and social stimuli</li>
        <li>Less fixation on the eye region of faces</li>
        <li>Delayed joint attention response</li>
        <li>More restricted gaze patterns with less exploration</li>
        <li>Atypical saccade patterns during social scenes</li>
    </ul>
    <p style="color: #e2e8f0; font-size: 0.95rem;">
        Similarly, <strong>ADHD</strong> is associated with difficulties in sustained visual attention,
        higher rates of inhibition errors (anti-saccade tasks), and more variable fixation patterns.
        <strong>Cognitive delays</strong> can manifest as reduced smooth pursuit accuracy and slower 
        visual processing speeds.
    </p>
</div>
""", unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────
st.markdown("## ⚙️ How the System Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px; padding: 24px;">
        <h3 style="color: #3b82f6; margin-bottom: 12px;">👁️ Gaze Tracking</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.7;">
            The system uses <strong>Google MediaPipe Face Mesh</strong> to detect 478 facial landmarks
            in real-time, including iris landmarks (indices 468-477). This enables:
        </p>
        <ul style="color: #94a3b8; font-size: 0.85rem;">
            <li><strong>Gaze direction estimation</strong> from iris offset relative to eye contour</li>
            <li><strong>Fixation detection</strong> using the I-VT (Velocity-Threshold) algorithm</li>
            <li><strong>Saccade detection</strong> between fixation events</li>
            <li><strong>Blink detection</strong> via Eye Aspect Ratio (EAR)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px; padding: 24px;">
        <h3 style="color: #8b5cf6; margin-bottom: 12px;">📊 Feature Extraction</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.7;">
            From the raw gaze data, the system extracts <strong>digital biomarkers</strong> including:
        </p>
        <ul style="color: #94a3b8; font-size: 0.85rem;">
            <li>Fixation count, duration statistics, and spatial dispersion</li>
            <li>Saccade amplitude, frequency, and scanpath length</li>
            <li>Region-of-Interest (ROI) dwell time and first-fixation latency</li>
            <li>Social attention metrics (face preference ratio, eye region fixation)</li>
            <li>Pursuit accuracy, inhibition scores, and sustained attention d-prime</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Screening Tasks ──────────────────────────────────────────────────
st.markdown("## 🎯 Screening Tasks Explained")

tasks_info = [
    {
        "icon": "👤", "name": "Face vs. Object Preference", "color": "#3b82f6",
        "indicator": "ASD",
        "description": "A split-screen stimulus presents a face on one side and a geometric "
                       "pattern on the other. Children with ASD tend to show reduced preference "
                       "for looking at faces compared to objects.",
        "biomarkers": ["Face preference ratio", "Face dwell time", "Eye region fixation"],
    },
    {
        "icon": "👥", "name": "Social Scene Scanning", "color": "#3b82f6",
        "indicator": "ASD",
        "description": "A naturalistic scene with people interacting is displayed. The system "
                       "tracks where the child looks — eyes, faces, bodies, objects, or background.",
        "biomarkers": ["Eye fixation ratio", "Social vs. non-social dwell", "Scan pattern complexity"],
    },
    {
        "icon": "🎯", "name": "Smooth Pursuit Tracking", "color": "#8b5cf6",
        "indicator": "Cognitive",
        "description": "An animated target moves in a figure-8 pattern. The child follows it "
                       "with their eyes. Accuracy reflects oculomotor control and visual processing.",
        "biomarkers": ["Pursuit accuracy", "Pursuit gain", "Tracking consistency"],
    },
    {
        "icon": "👀", "name": "Joint Attention Response", "color": "#06b6d4",
        "indicator": "ASD / Social",
        "description": "A face with a gaze cue (looking to one side) is shown, followed by "
                       "a target appearing in the cued direction. Joint attention is a key "
                       "early social communication skill.",
        "biomarkers": ["Response latency", "Gaze following accuracy", "Target dwell time"],
    },
    {
        "icon": "⚡", "name": "Anti-Saccade (Inhibition)", "color": "#f59e0b",
        "indicator": "ADHD",
        "description": "A stimulus appears on one side and the child must look to the "
                       "OPPOSITE side. This tests inhibitory control — a core executive "
                       "function often impaired in ADHD.",
        "biomarkers": ["Correct/error rate", "Response latency", "Inhibition accuracy"],
    },
    {
        "icon": "⏱️", "name": "Sustained Attention (CPT)", "color": "#f59e0b",
        "indicator": "ADHD",
        "description": "Shapes appear one at a time. The child must look at target shapes "
                       "(green) and ignore non-targets (blue). Tests sustained visual attention.",
        "biomarkers": ["Hit rate", "Omission errors", "Commission errors", "d-prime", "Response variability"],
    },
    {
        "icon": "🔍", "name": "Visual Pattern Recognition", "color": "#8b5cf6",
        "indicator": "Cognitive",
        "description": "Find a specific target among distractors. Search strategy and "
                       "efficiency reflect cognitive processing and visual attention organization.",
        "biomarkers": ["Time to target", "Search efficiency", "Fixation count", "Strategy pattern"],
    },
]

for task in tasks_info:
    with st.expander(f"{task['icon']} {task['name']} — {task['indicator']}", expanded=False):
        st.markdown(f"""
        <div style="padding: 8px 0;">
            <p style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.7;">
                {task['description']}
            </p>
            <div style="margin-top: 12px;">
                <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; 
                            font-weight: 600;">Digital Biomarkers:</span>
                <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;">
                    {''.join(f'<span style="background: {task["color"]}15; color: {task["color"]}; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; border: 1px solid {task["color"]}25;">{bm}</span>' for bm in task['biomarkers'])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Technology Stack ──────────────────────────────────────────────────
st.markdown("## 🛠️ Technology Stack")

tech_cols = st.columns(4)

techs = [
    ("🤖", "MediaPipe", "Google's ML framework for face mesh and iris detection with 478 landmarks"),
    ("🐍", "Python + NumPy", "Core computation engine for feature extraction and signal processing"),
    ("📊", "Plotly + Streamlit", "Interactive visualization and responsive web application framework"),
    ("🧪", "Scikit-learn", "Machine learning pipeline interface for pluggable trained models"),
]

for col, (icon, name, desc) in zip(tech_cols, techs):
    with col:
        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; padding: 20px; text-align: center; min-height: 160px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
            <h4 style="color: #f1f5f9; margin-bottom: 6px; font-size: 0.95rem;">{name}</h4>
            <p style="color: #64748b; font-size: 0.75rem; line-height: 1.4;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── References ────────────────────────────────────────────────────────
st.markdown("## 📚 Key References")

references = [
    "Duan, H., et al. (2019). *A Dataset of Eye Movements for the Children with Autism "
    "Spectrum Disorder*. ACM Multimedia Systems Conference (MMSys'19).",
    "Klin, A., et al. (2002). *Visual fixation patterns during viewing of naturalistic social "
    "situations as predictors of social competence*. Archives of General Psychiatry.",
    "Falck-Ytter, T., et al. (2013). *Eye tracking in early autism research*. "
    "Journal of Neurodevelopmental Disorders.",
    "Lupu, R. G., & Ungureanu, F. (2013). *A survey of eye tracking methods and applications*. "
    "Bulletin of the Transilvania University of Brasov.",
    "Guillon, Q., et al. (2014). *Visual social attention in autism spectrum disorder*. "
    "Autism Research.",

]

for ref in references:
    st.markdown(f"- {ref}")

# ── Medical Disclaimer ───────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px; padding: 20px; margin: 20px 0;">
    <h4 style="color: #f87171; margin-bottom: 8px;">⚠️ Important Medical Disclaimer</h4>
    <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6; margin: 0;">
        This tool is designed as a <strong>screening aid only</strong> and is <strong>NOT</strong> 
        a diagnostic instrument. Results generated by this system should be interpreted by qualified 
        healthcare professionals and must not be used as the sole basis for clinical decisions.
        This system has not been validated in a clinical trial and should be used for 
        research and educational purposes only.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <p style="color: #475569; font-size: 0.8rem;">
        Built with ❤️ using MediaPipe, Streamlit, Plotly, and AI
    </p>
</div>
""", unsafe_allow_html=True)
