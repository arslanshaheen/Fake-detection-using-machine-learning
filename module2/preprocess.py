"""Module 2a: Text preprocessing & cleaning."""

from __future__ import annotations

import re

import pandas as pd

# Fallback list so the app works even if NLTK data is missing
_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
}

_STOP_WORDS: set[str] | None = None


def _get_stopwords() -> set[str]:
    global _STOP_WORDS
    if _STOP_WORDS is not None:
        return _STOP_WORDS
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        _STOP_WORDS = set(_FALLBACK_STOPWORDS)
    return _STOP_WORDS


def preprocess_text(text: str) -> str:
    """Lowercase, strip URLs / non-letters, remove English stopwords."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    stop_words = _get_stopwords()
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, content_col: str = "content") -> pd.DataFrame:
    """Drop empty/duplicate content and add clean_content column."""
    out = df.copy()
    out = out.dropna(subset=[content_col]).drop_duplicates(subset=[content_col]).reset_index(drop=True)
    out["clean_content"] = out[content_col].apply(preprocess_text)
    return out
