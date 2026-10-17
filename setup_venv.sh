#!/bin/bash
# Create virtualenv and install deps (run in Terminal.app if Cursor sandbox blocks venv)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords', quiet=True); print('NLTK stopwords ready')"
echo ""
echo "Done. Next:"
echo "  1. Put Fake.csv and True.csv into data/"
echo "  2. make train"
echo "  3. make run"
echo "  Open http://127.0.0.1:5000 and upload a CSV"
