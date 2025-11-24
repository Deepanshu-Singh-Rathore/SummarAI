# SummarAI: Advanced Text Summarizer

SummarAI is a Python application that provides advanced text summarization capabilities. It features both extractive and abstractive summarization methods, a user-friendly web interface built with Streamlit, and evaluation metrics to assess summary quality.

## Features

- **Extractive Summarization (TextRank + MMR)**: Ranks sentences via PageRank on a TF‑IDF cosine graph, then applies Maximal Marginal Relevance to reduce redundancy and encourage section coverage.
- **Abstractive Summarization (Placeholder)**: Hook for integrating an LLM (e.g., BART/T5) later.
- **Interactive UI**: Streamlit app with file upload, ratio vs fixed length, MMR λ tuning, and ROUGE table.
- **Customizable Length Controls**: Fixed sentence count or fraction of total sentences.
- **Detailed ROUGE Metrics**: Precision / Recall / F1 for ROUGE‑1/2/L.
- **Baseline Comparison Script**: `scripts/compare_baseline.py` contrasts old frequency method vs advanced approach.
- **Report Template**: `reports/report.md` outlines methodology, equations, limitations, and references.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Deepanshu-Singh-Rathore/SummarAI.git
    cd SummarAI
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

To run the Streamlit application, use the following command:

```bash
streamlit run ui/app.py
```

This will open the application in your web browser, where you can enter text, choose summarization settings, and generate summaries.

## Quick Demo (CLI)

Generate a 3-sentence summary of the sample file:
```bash
python -c "import pathlib, summarizer; print(summarizer.summarize_text(pathlib.Path('data/sample.txt').read_text(), summary_length=3))"
```

Compare baseline vs advanced (add a reference summary for ROUGE):
```bash
python scripts/compare_baseline.py data/sample.txt --sentences 3 --ref "Your reference summary here"
```

## Tests

Minimal behavioral tests:
```bash
pytest -q
```
(Install pytest if needed: `pip install pytest`.)

## Screenshots

*(Screenshots placeholder – add UI captures of input panel and ROUGE table.)*

## Future Improvements

 - **Real Abstractive Model**: Integrate BART/T5 or API-based LLM.
 - **Chunking Long Docs**: Hierarchical summarization for very large inputs.
 - **Semantic Embeddings**: Replace TF‑IDF with Sentence‑BERT for deeper semantics.
 - **Highlighting**: Visualize which sentences were selected (color overlay).
 - **Additional Metrics**: BERTScore, QuestEval, coverage & compression ratios.
 - **Bias / Safety Checks**: Flag potentially sensitive or misrepresented content.

## License

Educational / academic use. Add a formal license if distributing widely.
