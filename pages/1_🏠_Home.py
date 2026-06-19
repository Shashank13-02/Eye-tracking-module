"""
Home Page — Redirect to main app
"""
import streamlit as st

st.set_page_config(page_title="NeuroScreen — Home", page_icon="🏠", layout="wide")

st.markdown("""
<meta http-equiv="refresh" content="0; url=/">
<div style="text-align: center; padding: 60px 0;">
    <h2 style="color: #94a3b8;">🏠 Redirecting to Home...</h2>
    <p style="color: #64748b;">If not redirected, click the app logo in the sidebar.</p>
</div>
""", unsafe_allow_html=True)

st.switch_page("app.py")
