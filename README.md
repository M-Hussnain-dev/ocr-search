# OCR Search

Upload a photo of text. The app reads it with Tesseract OCR, searches Wikipedia, the web (DuckDuckGo) and arXiv in parallel, ranks the results and writes one cited answer with a confidence score. Python + Streamlit, 100% free tools, no login.

## Run locally
```bash
python setup.py --run
```
`setup.py` installs the Python libraries, checks Tesseract (English + Urdu) and creates the database. Later runs: `streamlit run app.py`.

## Files (all in one folder)
- `app.py` – Streamlit interface
- `ocr_engine.py` – preprocessing + OCR · `web_search.py` – sources · `ranker.py` – ranking and confidence
- `answer_writer.py` – answer writing · `database.py` – SQLite history · `pdf_export.py` – PDF export
- `setup.py` – installer · `requirements.txt` / `packages.txt` – libraries (Python / system)

## Deploy free
Upload all files to a public GitHub repo, then create the app on https://share.streamlit.io with main file `app.py`.
