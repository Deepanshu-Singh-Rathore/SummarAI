"""Compare baseline frequency summarizer vs TextRank+MMR summarizer using ROUGE.

Run:
    python scripts/compare_baseline.py data/sample.txt --ref "Your reference summary here"
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

from rouge_score import rouge_scorer
from summarizer import summarize_text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string


def baseline_frequency(text: str, k: int) -> str:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stops = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    freq: Dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if wl in stops or wl in string.punctuation:
            continue
        freq[wl] = freq.get(wl, 0) + 1
    max_freq = max(freq.values() or [1])
    for kf in list(freq.keys()):
        freq[kf] /= max_freq
    scores: Dict[int, float] = {}
    for i, s in enumerate(sentences):
        tokens = [t.lower() for t in word_tokenize(s)]
        score = sum(freq.get(t, 0) for t in tokens) / (len(tokens) + 1)
        scores[i] = score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    idxs = sorted(i for i,_ in ranked)
    return " ".join(sentences[i] for i in idxs)


def rouge(reference: str, generated: str) -> Dict[str, Dict[str, float]]:
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(reference or '', generated or '')
    out: Dict[str, Dict[str, float]] = {}
    for m, r in scores.items():
        out[m] = {'p': r.precision, 'r': r.recall, 'f': r.fmeasure}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file', help='Input text file')
    ap.add_argument('--sentences', type=int, default=3, help='Summary length (sentences)')
    ap.add_argument('--ref', type=str, default='', help='Reference summary for ROUGE')
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding='utf-8', errors='ignore')

    baseline_sum = baseline_frequency(text, args.sentences)
    advanced_sum = summarize_text(text, summary_length=args.sentences, lambda_mmr=0.7)

    print('\n=== Baseline (Frequency) ===')
    print(baseline_sum)
    print('\n=== Advanced (TextRank+MMR) ===')
    print(advanced_sum)

    if args.ref.strip():
        b_scores = rouge(args.ref, baseline_sum)
        a_scores = rouge(args.ref, advanced_sum)
        print('\nROUGE (Baseline):', b_scores)
        print('ROUGE (Advanced):', a_scores)


if __name__ == '__main__':
    main()
