"""Train baseline models from Fake.csv + True.csv and save the best one."""

from __future__ import annotations

import argparse
from pathlib import Path

from module1.ingest import load_and_combine, summarize_dataset
from module3.train import save_artifacts, train_and_select_best


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fake news detection baseline models.")
    parser.add_argument("--fake", default=str(ROOT / "data" / "Fake.csv"), help="Path to Fake.csv")
    parser.add_argument("--true", default=str(ROOT / "data" / "True.csv"), help="Path to True.csv")
    parser.add_argument("--models-dir", default=str(ROOT / "models"), help="Where to save model files")
    args = parser.parse_args()

    fake_path = Path(args.fake)
    true_path = Path(args.true)
    if not fake_path.exists() or not true_path.exists():
        raise SystemExit(
            f"Missing training CSVs.\n"
            f"  Put Fake.csv and True.csv in: {ROOT / 'data'}\n"
            f"  Or pass --fake / --true paths."
        )

    print("Module 1: Data Ingestion & Analysis...")
    df = load_and_combine(fake_path, true_path)
    summary = summarize_dataset(df)
    print(f"  Total rows: {summary['total_rows']}")
    if "label_counts" in summary:
        print(f"  Labels: {summary['label_counts']}")

    print("Module 2 + 3: Preprocess, TF-IDF, train baselines...")
    result = train_and_select_best(df)
    print(f"  Cleaned articles: {result['total_cleaned']}")
    print(f"  Train / test: {result['train_size']} / {result['test_size']}")
    print("\n=== Baseline Model Comparative Performance ===")
    print(result["metrics"].to_string(index=False))
    print(f"\nBest model: {result['best_name']}")

    path = save_artifacts(result["model"], result["vectorizer"], args.models_dir)
    print(f"Saved model artifacts to: {path.parent}")


if __name__ == "__main__":
    main()
