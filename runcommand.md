Run again (simplest)
cd ~/Desktop/fake_news_detection
source .venv/bin/activate
make run
Or:

cd ~/Desktop/fake_news_detection
source .venv/bin/activate
python app.py
Then open: http://127.0.0.1:8080

Notes
make install / pip install — only once (already done).
make train / python train.py — only when you change data/model.
If you get “Address already in use” again:
lsof -ti :8080 | xargs kill -9
Then run make run again.