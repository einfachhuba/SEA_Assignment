#!/bin/bash
set -e

# Activate the virtual environment
source .venv/bin/activate

# Run the Streamlit application
exec streamlit run app/Home.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false