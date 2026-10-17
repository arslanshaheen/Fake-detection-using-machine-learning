"""Module 2b: TF-IDF feature extraction."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(texts, max_features: int = 5000) -> tuple[TfidfVectorizer, object]:
    """Fit TF-IDF on training texts and return (vectorizer, matrix)."""
    tfidf = TfidfVectorizer(max_features=max_features)
    matrix = tfidf.fit_transform(texts)
    return tfidf, matrix


def transform_tfidf(tfidf: TfidfVectorizer, texts):
    """Transform texts with a fitted TF-IDF vectorizer."""
    return tfidf.transform(texts)
