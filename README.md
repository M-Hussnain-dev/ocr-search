# OCR Search

Read **printed or handwritten** text from photos and PDFs, search free sources, and get one cited, scored and explained answer. Python + Streamlit. Free tools only. No login.

## Features
- Input: images, multi-page photos, PDFs (digital PDFs use their real text), camera
- Engines: Tesseract (printed, English and Urdu), RapidOCR (neat handwriting, offline), Gemini Vision (any handwriting, free key, optional), Ollama vision (local, optional), plus a Smart auto mode that escalates only when confidence is low
- Image cleaning: denoise, contrast, deskew, adaptive threshold, ink boost for handwriting
- OCR confidence map (word boxes coloured by confidence) and editable text
- Search: Wikipedia, web (DuckDuckGo), arXiv, OpenAlex, queried in parallel
- Ranking: BM25 + query coverage + source credibility, with a "Why this answer?" explanation
- Answer: extractive summary with citations (or a local LLM), confidence score, read aloud
- Accuracy lab: CER / WER benchmark across engines and cleaning modes
- History, feedback (helpful / not helpful), analytics dashboard, export to TXT / Markdown / JSON / PDF
- Automated tests: `python test_system.py`

## Speed
Gemini is tuned for speed: thinking is switched off, the upload is shrunk before sending, one connection is reused, the last working model is remembered, and a retired or rate-limited model is skipped automatically. Choose **Fast** (Flash-Lite models, default) or **Accurate** (full Flash) in the sidebar. Answers stream word by word, slow search sources are skipped after 6 or 14 seconds, and if Gemini is unavailable the app falls back to RapidOCR or Tesseract.

## Gemini key (optional, free)
Never put the key inside the code or upload it to GitHub. Choose one:
- On your PC: `python setup.py --key` (saves it to `.streamlit/secrets.toml`; keep that file on your computer only)
- On Streamlit Cloud: app Settings > Secrets > `GEMINI_API_KEY = "your-key"`
- Or paste it into the sidebar each time (Handwriting and AI engines)

With a key you get Gemini handwriting OCR, Urdu OCR and AI-written answers. The model is chosen automatically.

## Run locally
```bash
python setup.py --run
```
Installs libraries, checks Tesseract (English + Urdu), creates the database. Later: `streamlit run app.py`.
Run `python setup.py --test` to also run the tests.

## Files (one folder)
`app.py` interface · `ocr_engine.py` OCR engines · `web_search.py` sources · `ranker.py` ranking and confidence ·
`answer_writer.py` answers · `evaluation.py` CER/WER · `database.py` history · `pdf_export.py` reports ·
`test_system.py` tests · `setup.py` installer · `requirements.txt`, `packages.txt` dependencies

## Deploy free (Streamlit Community Cloud)
Upload all files to a public GitHub repository, create the app at https://share.streamlit.io with main file `app.py`, and choose **Python 3.11** under Advanced settings.
For the Gemini engine add `GEMINI_API_KEY = "your-key"` under the app's Settings > Secrets.
