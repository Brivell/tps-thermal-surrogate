import streamlit as st
import os, sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

st.set_page_config(
    page_title="TPS Thermal Surrogate",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load global CSS
with open(os.path.join(_ROOT, "assets", "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='text-align:center; padding: 1rem 0 1.5rem;'>
  <div style='font-size:2rem;'>🔥</div>
  <div style='font-size:1.1rem; font-weight:700; color:#00b4d8; letter-spacing:0.05em;'>
    TPS Surrogate
  </div>
  <div style='font-size:0.72rem; color:#8892a4; margin-top:0.2rem;'>
    Thermal Protection System
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

pages = {
    "🏠  Home": os.path.join(_ROOT, "pages", "1_home.py"),
    "⚙️  How It Works": os.path.join(_ROOT, "pages", "2_how_it_works.py"),
    "📊  Benchmarks": os.path.join(_ROOT, "pages", "3_benchmarks.py"),
    "⚡  Live Demo": os.path.join(_ROOT, "pages", "4_live_demo.py"),
    "🗺️  Design Space": os.path.join(_ROOT, "pages", "5_design_space.py"),
    "🎯  Optimization": os.path.join(_ROOT, "pages", "6_optimization.py"),
}

page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.72rem; color:#8892a4; text-align:center;'>"
    "FDM → FEM → GP + MLP<br>Validated vs COMSOL"
    "</div>",
    unsafe_allow_html=True,
)

# Route to page
import importlib.util, sys, os

page_file = pages[page]
spec = importlib.util.spec_from_file_location("page_module", page_file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
