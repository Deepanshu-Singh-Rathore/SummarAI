import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def summarize_text(text, summary_ratio=0.3):
    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)

    freq_table = {}
    for word in words:
        word = word.lower()
        if word not in stop_words:
            freq_table[word] = freq_table.get(word, 0) + 1

    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary_length = int(len(sentences) * summary_ratio)
    return " ".join(summary_sentences[:summary_length])
