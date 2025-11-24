# SummarAI Project Report

## 1. Introduction
SummarAI is an extractive text summarization system leveraging TextRank with TF-IDF sentence embeddings, Maximal Marginal Relevance (MMR) for redundancy reduction, and a section coverage heuristic. It includes evaluation via ROUGE and a Streamlit interface.

## 2. Problem Statement
Manual summarization is time-consuming. Basic frequency-based methods overweight repeated words and miss discourse structure. We aim for a concise summary retaining key ideas from each section while minimizing redundancy.

## 3. Methodology
### 3.1 Preprocessing
- Sentence segmentation (NLTK Punkt)
- Light cleaning: lowercasing & punctuation stripping for vectorization only

### 3.2 Sentence Representation
- TF-IDF sparse vectors using scikit-learn (`stop_words='english'`)
- Cosine similarity matrix forms weighted edges of sentence similarity graph

### 3.3 TextRank
- Construct graph G where nodes = sentences; weights = cosine similarity (diagonal zeroed)
- Apply PageRank to obtain importance scores

### 3.4 Section Coverage Heuristic
- Split original text by blank lines (double newlines) into coarse sections
- Map each sentence to a section span
- During selection favor sentences from previously uncovered sections early in MMR loop

### 3.5 Maximal Marginal Relevance (MMR)
MMR balances relevance and novelty:
$$\text{MMR}(d_i) = \lambda \cdot \text{Rel}(d_i) + (1-\lambda) \cdot (1 - \max_{d_j \in S} \text{Sim}(d_i,d_j))$$
Where Rel is TextRank score, Sim is cosine similarity, S is selected sentences set, and \(\lambda\) tunes relevance vs diversity.

### 3.6 Summary Assembly
- Determine target sentence count via fixed length or ratio
- Select with MMR; restore original order for readability

## 4. Evaluation
ROUGE metrics (1,2,L) computed with stemming; we report precision, recall, F1. Future work: add BERTScore & coverage metrics.

## 5. Baselines
- Frequency-based scoring baseline implemented (see `scripts/compare_baseline.py`)
- Advanced model shows improved structural coverage & redundancy reduction in manual inspection

## 6. Limitations
- Coarse section detection (blank-line heuristic) may misclassify complex formatting
- TF-IDF ignores word order and deeper semantics; abstractive mode is placeholder only
- No handling for extremely long texts (chunking not yet implemented)

## 7. Ethical Considerations
Summaries of sensitive socio-cultural content risk misrepresentation. Users should verify outputs for:
- Loss of nuance
- Omission of qualifiers or context
- Potential bias amplification

## 8. Future Work
- Integrate transformer embeddings (e.g., Sentence-BERT) for richer similarity
- Hierarchical chunking for long documents
- Real abstractive summarization (BART/T5)
- Additional metrics (BERTScore, QuestEval) & visualization of sentence contributions
- GUI enhancements: highlight selected sentences in source text

## 9. References
- Mihalcea & Tarau (2004). TextRank: Bringing Order into Texts.
- Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
- Carbonell & Goldstein (1998). The Use of MMR in IR.

## 10. Appendix
### A. Run Commands
```bash
pip install -r requirements.txt
streamlit run ui/app.py
python scripts/compare_baseline.py data/sample.txt --sentences 3 --ref "Your reference summary"
pytest -q  # if pytest installed
```
