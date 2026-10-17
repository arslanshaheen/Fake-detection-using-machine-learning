"""Module 3: Baseline model training, persistence, and batch prediction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from module2.features import build_tfidf, transform_tfidf
from module2.preprocess import preprocess_dataframe


LABEL_MAP = {0: "REAL", 1: "FAKE"}


def train_and_select_best(
    df: pd.DataFrame,
    max_features: int = 5000,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Preprocess, TF-IDF, train baselines, return best model + metrics."""
    cleaned = preprocess_dataframe(df)
    X = cleaned["clean_content"]
    y = cleaned["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    tfidf, X_train_tfidf = build_tfidf(X_train, max_features=max_features)
    X_test_tfidf = transform_tfidf(tfidf, X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(X_train_tfidf, y_train)
        fitted[name] = model

        start = time.time()
        y_pred = model.predict(X_test_tfidf)
        elapsed = time.time() - start
        inference_ms = (elapsed / max(len(X_test), 1)) * 1000

        results.append(
            {
                "Algorithm Name": name,
                "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
                "Precision (Test)": round(precision_score(y_test, y_pred), 4),
                "Recall (Test)": round(recall_score(y_test, y_pred), 4),
                "F1-Score": round(f1_score(y_test, y_pred), 4),
                "Inference Speed (ms)": round(inference_ms, 4),
            }
        )

    metrics_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Algorithm Name"]
    best_model = fitted[best_name]

    return {
        "model": best_model,
        "vectorizer": tfidf,
        "best_name": best_name,
        "metrics": metrics_df,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_cleaned": len(cleaned),
    }


def save_artifacts(model, vectorizer, models_dir: str | Path) -> Path:
    """Persist model + vectorizer to models/."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "fake_news_model.joblib"
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return model_path


def load_artifacts(models_dir: str | Path) -> tuple[Any, Any]:
    """Load trained model and TF-IDF vectorizer."""
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "fake_news_model.joblib")
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    return model, vectorizer


def predict_dataframe(df: pd.DataFrame, model, vectorizer) -> pd.DataFrame:
    """Run fake/real predictions on an ingested prediction CSV dataframe."""
    cleaned = preprocess_dataframe(df)
    X = transform_tfidf(vectorizer, cleaned["clean_content"])
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)

    out = cleaned.copy()
    out["prediction"] = preds
    out["label_name"] = out["prediction"].map(LABEL_MAP)
    if probs is not None:
        # Class order is typically [0, 1] = REAL, FAKE
        classes = list(getattr(model, "classes_", [0, 1]))
        fake_idx = classes.index(1) if 1 in classes else -1
        real_idx = classes.index(0) if 0 in classes else 0
        out["confidence_fake"] = [round(float(p[fake_idx]) * 100, 2) for p in probs]
        out["confidence_real"] = [round(float(p[real_idx]) * 100, 2) for p in probs]
        out["confidence"] = [
            round(float(p[classes.index(pred)]) * 100, 2) for p, pred in zip(probs, preds)
        ]
    return out
