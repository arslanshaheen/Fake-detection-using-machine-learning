# Fake News Detection — Makefile
# Pipeline: Module 1 → Module 2 → Module 3 (Web UI)

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python

.PHONY: help venv install nltk train run clean

help:
	@echo "Targets:"
	@echo "  make venv     - create virtual environment (.venv)"
	@echo "  make install  - install Python dependencies into .venv"
	@echo "  make nltk     - download NLTK stopwords"
	@echo "  make train    - train model (needs data/Fake.csv + data/True.csv)"
	@echo "  make run      - start Web UI at http://127.0.0.1:8080"
	@echo "  make clean    - remove uploads and cached Python files"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual env created at $(VENV)"
	@echo "Next: make install"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@$(MAKE) nltk
	@echo "Dependencies installed."

nltk:
	$(PY) -c "import nltk; nltk.download('stopwords', quiet=True); print('NLTK stopwords ready')"

train:
	@test -f $(BIN)/python || (echo "Run make install first" && exit 1)
	$(PY) train.py --fake data/Fake.csv --true data/True.csv --models-dir models

run:
	@test -f $(BIN)/python || (echo "Run make install first" && exit 1)
	@if [ ! -f models/fake_news_model.joblib ]; then \
		echo "Note: no trained model yet. UI will open, but upload needs: make train"; \
	fi
	$(PY) app.py

clean:
	rm -rf uploads/* __pycache__ module1/__pycache__ module2/__pycache__ module3/__pycache__
	find . -name "*.pyc" -delete 2>/dev/null || true
