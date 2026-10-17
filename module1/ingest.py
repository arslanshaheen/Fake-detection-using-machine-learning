"""Module 1: Data Ingestion & Analysis.

Loads Fake.csv / True.csv (or a single prediction CSV), labels rows,
combines content, and returns basic dataset stats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_text_column(df: pd.DataFrame) -> pd.Series:
    """Build a single text field from common CSV column layouts."""
    cols = {c.lower(): c for c in df.columns}

    title_col = cols.get("title")
    text_col = cols.get("text") or cols.get("content") or cols.get("article") or cols.get("news")
    body_col = cols.get("body")

    if title_col and text_col:
        return df[title_col].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str)
    if text_col:
        return df[text_col].fillna("").astype(str)
    if title_col:
        return df[title_col].fillna("").astype(str)
    if body_col:
        return df[body_col].fillna("").astype(str)

    # Fallback: concatenate all object columns
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if not object_cols:
        raise ValueError(
            "CSV must contain a text column such as title, text, content, article, or news."
        )
    return df[object_cols].fillna("").astype(str).agg(" ".join, axis=1)


def load_and_combine(
    fake_path: str | Path,
    true_path: str | Path,
) -> pd.DataFrame:
    """Load Fake + True CSVs, label, shuffle, and build content column."""
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake = df_fake.copy()
    df_true = df_true.copy()
    df_fake["label"] = 1  # Fake news
    df_true["label"] = 0  # Real news

    df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    df["content"] = _resolve_text_column(df)
    return df


def load_prediction_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a user-uploaded CSV for inference (no labels required)."""
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["content"] = _resolve_text_column(df)
    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic ingestion / analysis metrics."""
    summary: dict[str, Any] = {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_content": int(df["content"].isna().sum()) if "content" in df.columns else 0,
    }
    if "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        summary["label_counts"] = {
            "real (0)": int(counts.get(0, 0)),
            "fake (1)": int(counts.get(1, 0)),
        }
    return summary
