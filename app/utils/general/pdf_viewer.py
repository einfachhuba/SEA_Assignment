import os
from streamlit_pdf_viewer import pdf_viewer
import streamlit as st

def display_pdf_with_controls(pdf_path: str):
    if os.path.exists(pdf_path):

        st.markdown("### Assignment Document")
        st.write("This document contains the details of the assignment, including the problem statement and requirements.")
        
        # Add display controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            zoom_level = st.selectbox("Zoom Level", 
                options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], 
                index=3,
                format_func=lambda x: f"{int(x*100)}%",
                help="Controls the actual PDF zoom level"
            )

        with open(pdf_path, "rb") as pdf_file:
            pdf_viewer(
                input=pdf_file.read(), 
                width=1000,
                height=1000,
                zoom_level=zoom_level,
                rendering="unwrap",
                pages_vertical_spacing=2,
                annotation_outline_size=1,
                viewer_align="center"
            )
    else:
        st.error(f"PDF file not found at: {pdf_path}")