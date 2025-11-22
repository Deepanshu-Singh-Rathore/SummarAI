import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path so imports from parent folder work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summarizer import summarize_text

st.title("Auto Summary Bot 📝")

text = st.text_area("Enter your text here")
if st.button("Summarize"):
    summary = summarize_text(text)
    st.write(summary)
