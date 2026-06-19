"""
NeuroScreen — Main Application Entry Point
===================================================
AI-Powered Neurodevelopmental Screening via Gaze Analytics
"""

import os
import streamlit as st
from config import APP_TITLE, APP_ICON, APP_SUBTITLE, STYLES_DIR

# ── Page Configuration (MUST be first Streamlit command) ──────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"## {APP_TITLE}\n{APP_SUBTITLE}\n\n"
                 "AI-powered screening tool for neurodevelopmental disorders "
                 "using gaze analytics. For research and educational purposes only.",
    },
)

# ── Load Custom CSS ───────────────────────────────────────────────────
css_path = os.path.join(STYLES_DIR, "theme.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Google Fonts ──────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0 10px;">
        <div style="font-size: 3rem; margin-bottom: 8px;">{APP_ICON}</div>
        <h2 style="margin: 0; background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-weight: 800; font-size: 1.5rem;">NeuroScreen</h2>
        <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">App</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 8px 0;">
        <p style="color: #64748b; font-size: 0.75rem; text-align: center;">
            AI-Powered Screening Tool<br>
            <span style="color: #ef4444;">⚠️ Research Use Only</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content — Home Page ──────────────────────────────────────────

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 40px 0 20px;">
    <div style="font-size: 4.5rem; margin-bottom: 16px; 
                text-shadow: 0 0 40px rgba(59, 130, 246, 0.3);">🧠</div>
    <h1 style="font-size: 2.8rem; margin: 0; padding: 0;
               background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-weight: 800; letter-spacing: -0.03em;">
        NeuroScreen
    </h1>
    <p style="color: #94a3b8; font-size: 1.15rem; margin-top: 12px; max-width: 700px; 
              margin-left: auto; margin-right: auto; line-height: 1.6;">
        AI-Powered Neurodevelopmental Screening via Gaze Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px; padding: 28px; text-align: center; min-height: 220px;
                backdrop-filter: blur(12px); transition: all 0.3s ease;
                border-top: 3px solid #3b82f6;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">📱</div>
        <h3 style="color: #f1f5f9; font-size: 1.1rem; margin-bottom: 8px;">Non-Invasive</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
            Uses standard smartphone or webcam — no specialized eye-tracking hardware needed
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px; padding: 28px; text-align: center; min-height: 220px;
                backdrop-filter: blur(12px); transition: all 0.3s ease;
                border-top: 3px solid #8b5cf6;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">🔬</div>
        <h3 style="color: #f1f5f9; font-size: 1.1rem; margin-bottom: 8px;">AI-Powered Analysis</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
            Advanced gaze analytics extract objective digital biomarkers from visual attention patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px; padding: 28px; text-align: center; min-height: 220px;
                backdrop-filter: blur(12px); transition: all 0.3s ease;
                border-top: 3px solid #06b6d4;">
        <div style="font-size: 2.5rem; margin-bottom: 12px;">📊</div>
        <h3 style="color: #f1f5f9; font-size: 1.1rem; margin-bottom: 8px;">Multi-Dimensional</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
            Screens across ASD, ADHD, cognitive delays, and social communication disorders
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# How It Works
st.markdown("## How It Works")

cols = st.columns(4)
steps = [
    ("1️⃣", "Input", "Enter child info and calibrate camera"),
    ("2️⃣", "Screen", "7 age-appropriate visual tasks with gaze tracking"),
    ("3️⃣", "Analyze", "AI extracts digital biomarkers from gaze patterns"),
    ("4️⃣", "Report", "Multi-dimensional risk scores and clinical report"),
]

for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 16px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
            <h4 style="color: #f1f5f9; margin-bottom: 6px;">{title}</h4>
            <p style="color: #64748b; font-size: 0.8rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Screening Dimensions
st.markdown("## Screening Dimensions")

dim_col1, dim_col2 = st.columns(2)

dimensions = [
    ("🧩", "Autism Spectrum Disorder", "Face preference, social attention, joint attention, repetitive gaze patterns", "#3b82f6"),
    ("⚡", "ADHD", "Sustained attention, inhibition control, gaze hyperactivity, response consistency", "#f59e0b"),
    ("🧮", "Cognitive Delays", "Visual tracking, smooth pursuit accuracy, pattern recognition, processing speed", "#8b5cf6"),
    ("💬", "Social Communication", "Gaze following, joint attention response, eye contact duration, social scene scanning", "#06b6d4"),
]

for i, (icon, name, desc, color) in enumerate(dimensions):
    with [dim_col1, dim_col2][i % 2]:
        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; padding: 20px; margin-bottom: 12px;
                    border-left: 4px solid {color};">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong style="color: #f1f5f9; font-size: 1rem;">{name}</strong>
            </div>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0; line-height: 1.5;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Medical Disclaimer
st.markdown("""
<div style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px; padding: 20px; margin: 20px 0;">
    <h4 style="color: #f87171; margin-bottom: 8px;">⚠️ Important Medical Disclaimer</h4>
    <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6; margin: 0;">
        This tool is designed as a <strong>screening aid only</strong> and is <strong>NOT</strong> 
        a diagnostic instrument. Results generated by this system should be interpreted by qualified 
        healthcare professionals and must not be used as the sole basis for clinical decisions. 
        A positive screening result indicates a need for further comprehensive evaluation by a 
        specialist. This system has not been validated in a clinical trial and should be used for 
        research and educational purposes only.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Start CTA
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p style="color: #94a3b8; margin-bottom: 16px;">
            Ready to begin? Navigate to <strong>New Screening</strong> in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px 0;">
    <p style="color: #475569; font-size: 0.75rem;">
        Built with ❤️ using MediaPipe, Streamlit, and AI · 
        For Research Use Only
    </p>
</div>
""", unsafe_allow_html=True)
