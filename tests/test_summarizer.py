from summarizer import summarize_text

SAMPLE = """Natural language processing enables computers to understand human language.\n\nIt involves techniques from linguistics, computer science, and AI.\n\nApplications include chatbots, machine translation, and summarization."""

def test_summary_sentence_count():
    s = summarize_text(SAMPLE, summary_length=2)
    # Count sentences by splitting on period heuristically
    count = sum(1 for part in s.split('.') if part.strip())
    assert count <= 2

def test_no_duplicate_sentences():
    s = summarize_text(SAMPLE, summary_length=3)
    parts = [p.strip() for p in s.split('.') if p.strip()]
    assert len(parts) == len(set(parts))

def test_short_text_returns_original():
    short = "Short text example."
    assert summarize_text(short, summary_length=2) == short.strip()
