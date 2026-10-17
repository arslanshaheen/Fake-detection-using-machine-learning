# Fake News Detection

End-to-end pipeline from your notebook:

**Module 1** Data Ingestion & Analysis → **Module 2** Preprocess & Feature Extraction → **Module 3** Baseline Model & Web UI

## Setup

Open **Terminal.app** (not only Cursor), then:

```bash
cd ~/Desktop/fake_news_detection
make install
```

Or:

```bash
./setup_venv.sh
```

This creates `.venv` and installs dependencies.

## Train (Module 1–3)

Put the Kaggle-style files in `data/`:

- `data/Fake.csv`
- `data/True.csv`

Then:

```bash
make train
```

This cleans text, builds TF-IDF features, trains Logistic Regression / Naive Bayes / Decision Tree / Random Forest, and saves the best model to `models/`.

## Web UI — upload CSV & detect

```bash
make run
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), upload a CSV, and view FAKE / REAL labels.

Your CSV should have one of:

- `title` + `text`
- or `content` / `article` / `news`

Download `predictions.csv` from the results page.

## Project layout

```
fake_news_detection/
  module1/          # Data ingestion & analysis
  module2/          # Preprocess + TF-IDF features
  module3/          # Baseline models + predict
  templates/        # Web UI HTML
  static/           # Web UI CSS
  data/             # Put Fake.csv + True.csv here
  models/           # Saved model after make train
  train.py          # Training entrypoint
  app.py            # Flask Web UI
  Makefile
  requirements.txt
  fake_news_detecting.ipynb
```

## Manual commands (without Make)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```
