
import re
import streamlit as st


def render_html(raw: str):
    compact = re.sub(r"\s+", " ", raw.strip())
    st.markdown(compact, unsafe_allow_html=True)
