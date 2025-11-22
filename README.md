# SummarAI

Simple automatic text summarizer with a Streamlit UI.

## Features
- Extractive summarization using NLTK (tokenization, stopwords, sentence scoring).
- Streamlit web UI at `ui/app.py`.

## Requirements
- Python 3.8+ (this project used Python 3.13 in the workspace virtualenv)
- See `requirements.txt` for pinned packages. Add `streamlit` if missing.

## Setup (recommended)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
C:\Path\To\Project\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
pip install streamlit
```

3. Run the Streamlit app:

```powershell
C:/Users/deepa/Gunjan/SummarAI/.venv/Scripts/python.exe -m streamlit run ui\app.py
```

Or, after activating the venv:

```powershell
streamlit run ui\app.py
```

## Usage
Type or paste text into the input box and click **Summarize**. The app uses `summarizer.summarize_text` to generate an extractive summary.

## Notes
- NLTK data (punkt, stopwords) are downloaded at runtime if missing.
- The `ui/app.py` file adds the project root to `sys.path` so `summarizer` can be imported when running from the `ui` folder.

## Repository
https://github.com/Deepanshu-Singh-Rathore/SummarAI

---
If you'd like, I can also add `streamlit` to `requirements.txt` and commit the change.
