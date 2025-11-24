import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def clean_text(text):
    """
    Cleans the input text by converting to lowercase, removing punctuation,
    and stripping extra whitespace.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    """
    Tokenizes the text into sentences.

    Args:
        text (str): The input text.

    Returns:
        list: A list of sentences.
    """
    return sent_tokenize(text)

def tokenize_words(sentence):
    """
    Tokenizes a sentence into words.

    Args:
        sentence (str): The input sentence.

    Returns:
        list: A list of words.
    """
    return word_tokenize(sentence)

def remove_stopwords(words):
    """
    Removes stopwords from a list of words.

    Args:
        words (list): A list of words.

    Returns:
        list: A list of words without stopwords.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]
