import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path so imports from parent folder work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summarizer import summarize_text, abstractive_summarize, evaluate_summary

st.set_page_config(layout="wide")

st.title("SummarAI: Advanced Text Summarizer 📝")
st.markdown("""
    This app generates a summary of your text using either an extractive or abstractive approach. 
    You can also evaluate the summary against a reference text using ROUGE scores.
""")

col_left, col_right = st.columns([3,2])

with col_left:
    st.header("Input Text")
    uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"], help="If provided, its contents override the text area.")
    default_text = "" if uploaded else ""
    text_area_value = st.text_area("Enter or paste text to summarize:", value=default_text, height=320)
    if uploaded is not None:
        try:
            text = uploaded.read().decode("utf-8", errors="ignore")
        except Exception:
            st.error("Failed to decode file; falling back to text area content.")
            text = text_area_value
    else:
        text = text_area_value

    st.subheader("Reference Summary (Optional)")
    reference_summary = st.text_area("Provide a reference summary for ROUGE evaluation:", height=140)

with col_right:
    st.header("Settings")
    summary_mode = st.radio("Mode", ("Extractive (TextRank+MMR)", "Abstractive (Placeholder)"))
    use_ratio = st.checkbox("Use ratio instead of fixed sentence count", value=False)
    if use_ratio:
        ratio = st.slider("Summary ratio (fraction of sentences)", min_value=0.05, max_value=0.9, value=0.3)
        summary_length = None
    else:
        summary_length = st.slider("Number of sentences", min_value=1, max_value=15, value=3)
        ratio = None
    lambda_mmr = st.slider("Redundancy vs relevance (MMR λ)", min_value=0.3, max_value=0.95, value=0.7)

    generate = st.button("Generate Summary", type="primary")

if generate:
    if not text.strip():
        st.warning("Please enter or upload text to summarize.")
    else:
        if summary_mode.startswith("Extractive"):
            summary = summarize_text(text, summary_length=summary_length or 3, ratio=ratio, lambda_mmr=lambda_mmr)
        else:
            summary = abstractive_summarize(text)

        st.header("Generated Summary")
        st.write(summary)
        st.download_button("Download Summary", summary, file_name="summary.txt")

        if reference_summary.strip():
            st.subheader("ROUGE Evaluation")
            scores = evaluate_summary(summary, reference_summary)
            # Display metrics in a table
            display_rows = []
            for metric, vals in scores.items():
                display_rows.append({"Metric": metric, "Precision": vals['p'], "Recall": vals['r'], "F1": vals['f']})
            st.table(display_rows)

        st.caption("MMR reduces redundancy; lower λ increases novelty, higher λ increases relevance.")
