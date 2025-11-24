from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk

from preprocessing import tokenize_sentences, clean_text


def _ensure_nltk() -> None:
    """Ensure NLTK resources required for tokenization are available."""
    try:
        # Try a quick tokenize to detect missing resources early
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _build_tfidf(sentences: Sequence[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Vectorize sentences with TF-IDF, returning the fitted vectorizer and matrix."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    return vectorizer, tfidf


def _similarity_matrix(tfidf) -> np.ndarray:
    """Compute cosine similarity matrix with diagonal zeroed to avoid self-loops."""
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    return sim


def _textrank_scores(sim_matrix: np.ndarray) -> Dict[int, float]:
    """Run PageRank on the similarity graph to obtain TextRank scores."""
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph) if graph.number_of_nodes() else {}
    return {int(i): float(s) for i, s in scores.items()}


def _split_sections(original_text: str) -> List[Tuple[str, str, int, int]]:
    """Split text into sections.

    Heuristics:
    - A heading is a line with Title Case (most words capitalized) or all caps
      and length <= 80 characters.
    - Sections are separated by blank lines or headings.
    Returns list of tuples: (heading, section_text, char_start, char_end).
    """
    lines = original_text.splitlines()
    sections: List[Tuple[str, str, int, int]] = []
    current_heading = ""
    buffer: List[str] = []
    char_pos = 0
    segment_start = 0

    def is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped) > 80:
            return False
        tokens = [t for t in stripped.split() if t.isalpha()]
        if not tokens:
            return False
        # Title case or all caps majority
        cap_ratio = sum(1 for t in tokens if t[0].isupper()) / len(tokens)
        all_caps = sum(1 for t in tokens if t.isupper()) / len(tokens)
        return cap_ratio >= 0.8 or all_caps >= 0.8

    for line in lines:
        line_len = len(line) + 1  # account for newline
        if is_heading(line) and buffer:
            # flush previous section
            section_text = "\n".join(buffer).strip()
            if section_text:
                sections.append((current_heading, section_text, segment_start, char_pos))
            current_heading = line.strip()
            buffer = []
            segment_start = char_pos
        elif line.strip() == "" and buffer:
            # blank line boundary
            section_text = "\n".join(buffer).strip()
            if section_text:
                sections.append((current_heading, section_text, segment_start, char_pos))
            current_heading = ""
            buffer = []
            segment_start = char_pos
        else:
            buffer.append(line)
        char_pos += line_len
    # flush trailing buffer
    if buffer:
        section_text = "\n".join(buffer).strip()
        if section_text:
            sections.append((current_heading, section_text, segment_start, char_pos))
    return sections if sections else [("", original_text.strip(), 0, len(original_text))]


def _sentence_section_mapping(text: str, sentences: Sequence[str]) -> Dict[int, int]:
    """Map sentence indices to section indices using character spans from _split_sections."""
    sections = _split_sections(text)
    spans = [(start, end) for _, _, start, end in sections]
    mapping: Dict[int, int] = {}
    search_from = 0
    for i, s in enumerate(sentences):
        idx = text.find(s, search_from)
        if idx == -1:
            idx = text.find(s.strip(), search_from)
        if idx == -1:
            idx = search_from
        search_from = idx + len(s)
        sec_id = 0
        for k, (a, b) in enumerate(spans):
            if a <= idx < b:
                sec_id = k
                break
        mapping[i] = sec_id
    return mapping


def _mmr_select(
    sentences: Sequence[str],
    scores: Dict[int, float],
    sim_matrix: np.ndarray,
    k: int,
    lambda_: float = 0.7,
    section_by_idx: Optional[Dict[int, int]] = None,
) -> List[int]:
    """Select k sentence indices using Maximal Marginal Relevance (MMR).

    Balances relevance (TextRank score) with novelty (low similarity to already
    selected sentences). If a section mapping is provided, try to cover as many
    sections as possible before filling remaining slots.
    """
    if not sentences:
        return []
    k = max(1, min(k, len(sentences)))

    # Start by taking the single best-scoring sentence
    remaining = set(range(len(sentences)))
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: List[int] = [ordered[0][0]] if ordered else []
    remaining.discard(selected[0]) if selected else None

    # Encourage section coverage early on
    if section_by_idx and selected:
        covered = {section_by_idx[selected[0]]}
    else:
        covered = set()

    while len(selected) < k and remaining:
        best_idx = None
        best_val = -1.0
        for i in remaining:
            rel = scores.get(i, 0.0)
            # similarity to the closest selected sentence (max sim)
            sim_to_sel = 0.0
            if selected:
                sim_to_sel = float(np.max(sim_matrix[i, selected]))

            novelty = 1.0 - sim_to_sel
            mmr = lambda_ * rel + (1 - lambda_) * novelty

            # Soft preference for new sections when ties occur
            if section_by_idx and section_by_idx.get(i) not in covered:
                mmr += 1e-4

            if mmr > best_val:
                best_val = mmr
                best_idx = i

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
        if section_by_idx:
            covered.add(section_by_idx.get(best_idx))

    return selected


def summarize_text(
    text: str,
    summary_length: int = 3,
    ratio: Optional[float] = None,
    lambda_mmr: float = 0.7,
    ensure_section_coverage: bool = True,
) -> str:
    """Return a concise, section-aware extractive summary.

    Pipeline:
      1. Global TextRank scores over TF-IDF cosine similarity graph.
      2. Proportional allocation: at least one sentence per detected section if
         `ensure_section_coverage` is True.
      3. Redundancy reduction: remaining budget filled via MMR.
      4. Output sentences restored to original order for readability.

    Args:
        text: Raw input text.
        summary_length: Target sentence count when `ratio` is None.
        ratio: Optional ratio (0<ratio<=1) overriding `summary_length`.
        lambda_mmr: Relevance vs novelty trade-off for MMR (higher favors relevance).
        ensure_section_coverage: Guarantee each section contributes at least one
            sentence (when feasible) to improve thematic completeness.

    Returns:
        Extractive summary string.
    """
    if not isinstance(text, str) or not text.strip():
        return "Input text is empty."

    # Guard against extremely short inputs
    if len(text.split()) < 10:
        return text.strip()

    _ensure_nltk()
    orig_sentences = tokenize_sentences(text)
    if len(orig_sentences) == 0:
        return ""

    # Clean sentences only for vectorization; keep originals for output
    cleaned = [clean_text(s) for s in orig_sentences]

    _, tfidf = _build_tfidf(cleaned)
    sim = _similarity_matrix(tfidf)
    scores = _textrank_scores(sim)

    # Decide target length
    if ratio is not None and 0 < ratio <= 1:
        k = max(1, int(round(len(orig_sentences) * ratio)))
    else:
        k = max(1, min(int(summary_length), len(orig_sentences)))

    # Section-aware, redundancy-reduced selection
    section_map = _sentence_section_mapping(text, orig_sentences)

    # Allocate per-section minimum representation when coverage desired
    if not ensure_section_coverage or k <= 1:
        # Fall back to pure MMR selection without enforced coverage
        section_map_simple = _sentence_section_mapping(text, orig_sentences)
        selected_idx = _mmr_select(orig_sentences, scores, sim, k, lambda_mmr, section_map_simple)
    elif k == len(orig_sentences):
        selected_idx = list(range(len(orig_sentences)))
    else:
        section_counts: Dict[int, int] = {}
        for sec in section_map.values():
            section_counts[sec] = section_counts.get(sec, 0) + 1
        unique_sections = sorted(section_counts.keys())

        # Proportional allocation with at least 1 sentence per section (if possible)
        alloc: Dict[int, int] = {}
        remaining = k
        for sec in unique_sections:
            share = max(1, round(section_counts[sec] / len(orig_sentences) * k))
            alloc[sec] = share
        # Normalize if over-allocated
        total_alloc = sum(alloc.values())
        if total_alloc > k:
            # reduce extras from largest allocations
            for sec in sorted(alloc.keys(), key=lambda s: alloc[s], reverse=True):
                if total_alloc <= k:
                    break
                if alloc[sec] > 1:
                    alloc[sec] -= 1
                    total_alloc -= 1
        remaining = k - sum(alloc.values())

        # Rank sentences globally once
        ranked_global = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Index sentences per section by global score order
        per_section_rank: Dict[int, List[int]] = {sec: [] for sec in unique_sections}
        for idx, _ in ranked_global:
            per_section_rank[section_map[idx]].append(idx)

        prelim: List[int] = []
        for sec in unique_sections:
            take = per_section_rank[sec][: alloc[sec]]
            prelim.extend(take)
        prelim = sorted(set(prelim))

        # Fill remaining slots using MMR to reduce redundancy
        prelim_set = set(prelim)
        remaining_candidates = [idx for idx, _ in ranked_global if idx not in prelim_set]
        if remaining > 0 and remaining_candidates:
            # Build reduced similarity matrix referencing indices
            selected_idx = prelim.copy()
            while len(selected_idx) < k and remaining_candidates:
                # compute MMR on candidate list with already selected
                best_c = None
                best_val = -1.0
                for cand in remaining_candidates:
                    rel = scores.get(cand, 0.0)
                    sim_to_sel = 0.0
                    if selected_idx:
                        sim_to_sel = float(np.max(sim[cand, selected_idx]))
                    novelty = 1.0 - sim_to_sel
                    mmr_val = lambda_mmr * rel + (1 - lambda_mmr) * novelty
                    if mmr_val > best_val:
                        best_val = mmr_val
                        best_c = cand
                if best_c is None:
                    break
                selected_idx.append(best_c)
                remaining_candidates.remove(best_c)
        else:
            selected_idx = prelim

    # Preserve original order for readability
    selected_idx = sorted(selected_idx)

    # Preserve original order for readability
    selected_idx = sorted(selected_idx)
    summary = " ".join(orig_sentences[i] for i in selected_idx)
    return summary


def abstractive_summarize(text: str) -> str:
    """Placeholder for abstractive summarization using an external LLM API."""
    return "This is a placeholder for an abstractive summary."


def evaluate_summary(generated_summary: str, reference_summary: str) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1/2/L precision, recall, and F1 scores.

    Returns nested dictionaries per metric, e.g.:
    {
        'rouge1': {'p': ..., 'r': ..., 'f': ...},
        'rouge2': {...},
        'rougeL': {...}
    }
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary or "", generated_summary or "")
    out: Dict[str, Dict[str, float]] = {}
    for key in ["rouge1", "rouge2", "rougeL"]:
        res = scores[key]
        out[key] = {"p": float(res.precision), "r": float(res.recall), "f": float(res.fmeasure)}
    return out

