"""Module 3 Web UI: upload a CSV and detect fake vs real news."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from module1.ingest import load_prediction_csv, summarize_dataset
from module3.train import load_artifacts, predict_dataframe


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
MODELS_DIR = ROOT / "models"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
app.secret_key = "fake-news-detection-dev-key"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def model_ready() -> bool:
    return (MODELS_DIR / "fake_news_model.joblib").exists() and (
        MODELS_DIR / "tfidf_vectorizer.joblib"
    ).exists()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if not model_ready():
            flash("Model not trained yet. Run: make train (after placing Fake.csv and True.csv in data/).")
            return redirect(url_for("index"))

        file = request.files.get("csv_file")
        if file is None or file.filename == "":
            flash("Please choose a CSV file.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Only .csv files are allowed.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        try:
            df = load_prediction_csv(save_path)
            summary = summarize_dataset(df)
            model, vectorizer = load_artifacts(MODELS_DIR)
            predictions = predict_dataframe(df, model, vectorizer)

            out_path = UPLOAD_DIR / "predictions.csv"
            keep = [c for c in predictions.columns if c != "clean_content"]
            predictions[keep].to_csv(out_path, index=False)

            preview = predictions.head(50)
            rows = preview.to_dict(orient="records")
            columns = list(preview.columns)

            fake_count = int((predictions["prediction"] == 1).sum())
            real_count = int((predictions["prediction"] == 0).sum())

            return render_template(
                "index.html",
                model_ready=True,
                results=True,
                rows=rows,
                columns=columns,
                total=len(predictions),
                fake_count=fake_count,
                real_count=real_count,
                summary=summary,
                download_ready=True,
            )
        except Exception as exc:  # noqa: BLE001 - show user-facing error in UI
            flash(f"Could not process CSV: {exc}")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        model_ready=model_ready(),
        results=False,
        download_ready=False,
    )


@app.route("/download")
def download():
    out_path = UPLOAD_DIR / "predictions.csv"
    if not out_path.exists():
        flash("No predictions file yet. Upload a CSV first.")
        return redirect(url_for("index"))
    return send_file(out_path, as_attachment=True, download_name="predictions.csv")


def main() -> None:
    # Use 8080 — macOS often occupies 5000 (AirPlay Receiver)
    app.run(host="127.0.0.1", port=8080, debug=True)


if __name__ == "__main__":
    main()
