import sys
from pathlib import Path

# Add repo root to sys.path so `ai_safety` / `dashboard` packages import under Streamlit.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

base_dir = Path(__file__).parent

pages = {
    "Overview": [
        st.Page(str(base_dir / "pages/02_architecture.py"), title="Architecture"),
        st.Page(str(base_dir / "pages/06_failure_analysis.py"), title="Failure Analysis"),
        st.Page(str(base_dir / "pages/05_inference_demo.py"), title="Inference Demo")
    ],
    "Data": [
        st.Page(str(base_dir / "pages/01_data_registry.py"), title="Data Registry")
    ],
    "Experiments": [
        st.Page(str(base_dir / "pages/03_diagnostic_performance.py"), title="Diagnostic Performance"),
        st.Page(str(base_dir / "pages/04_monitor_performance.py"), title="Monitor Performance")
    ]
}

pg = st.navigation(pages)
pg.run()
