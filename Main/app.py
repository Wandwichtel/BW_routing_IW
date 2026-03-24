"""
BW Navigation - Emergency Routing App
Cloud-optimized version for Streamlit Cloud deployment
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Set up data paths for cloud environment
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(".streamlit/cache") if os.path.exists(".streamlit") else Path("/tmp/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Environment variable overrides (for flexibility)
os.environ.setdefault("BW_BUILDINGS_GPKG", str(DATA_DIR / "buildings.gpkg"))
os.environ.setdefault("BW_VEGETATION_TIF", str(DATA_DIR / "vegetation.tif"))
os.environ.setdefault("BW_ROADS_GPKG", str(DATA_DIR / "roads.gpkg"))
os.environ.setdefault("BW_PARKS_GPKG", str(DATA_DIR / "parks.gpkg"))

# Import the main app logic
# Note: Replace this with your actual BW_navigation_ui_V2 implementation
try:
    from navigation_logic import run_app
    run_app()
except ImportError:
    st.error("""
    ### Setup Required
    
    Please ensure the following files are in the cloud repo:
    - `navigation_logic.py` (your Streamlit UI code)
    - `data/buildings.gpkg`, `data/vegetation.tif`, `data/roads.gpkg`, `data/parks.gpkg`
    
    Or upload sample data and update the paths above.
    """)
