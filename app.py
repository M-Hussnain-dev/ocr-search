"""OCR Search - scan a photo of text, search free sources, get one cited answer."""
from __future__ import annotations

import hashlib
import html
import io
import time
import uuid
from dataclasses import asdict

import pandas as pd
import streamlit as st
from PIL import Image

import answer_writer as answer
import database as db
import ocr_engine as ocr
import pdf_export as pdf
import ranker as rank
import web_search as search

st.set_page_config(page_title="OCR Search", page_icon="🔎", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap');
h1,h2,h3,h4,.hero h1{font-family:'Bricolage Grotesque',sans-serif;letter-spacing:-.01em}
.block-container{padding-top:1.6rem;max-width:1180px}
.hero{position:relative;overflow:hidden;background:#12202B;color:#EAF2F5;border-radius:6px;padding:30px 34px 24px;margin-bottom:22px}
.hero h1{color:#fff;font-size:2.3rem;font-weight:800;margin:0 0 6px;padding:0}
.hero p{margin:0;max-width:640px;color:#B9CBD3;font-size:1.02rem}
.hero::after{content:"";position:absolute;top:0;bottom:0;left:-6px;width:3px;background:#38D0DC;
  box-shadow:0 0 22px 6px rgba(56,208,220,.45);animation:scan 2.2s ease-out 1 forwards}
@keyframes scan{to{left:101%}}
@media (prefers-reduced-motion:reduce){.hero::after{display:none}}
.flow{display:flex;flex-wrap:wrap;margin-top:20px}
.flow div{flex:1 1 150px;border-top:2px solid #38D0DC;padding:10px 14px 0 0;font-size:.88rem;color:#9FB6C0}
.flow b{display:block;color:#fff;font-size:1rem;margin-bottom:2px}
.src{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.3);border-left:4px solid var(--c);border-radius:4px;padding:14px 18px;margin-bottom:10px}
.src .tag{font-weight:600;color:var(--c);font-size:.82rem;margin-right:8px}
.src .ttl{font-weight:600;font-size:1.02rem}
.src p{margin:6px 0 8px;opacity:.85;font-size:.93rem}
.src a{font-size:.88rem;color:#0E7C86;text-decoration:none;font-weight:600}
.bar{height:4px;background:rgba(127,127,127,.25);border-radius:9px;margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--c)}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.3);border-radius:4px;padding:12px 16px}
.stButton>button[kind="primary"]{font-weight:600;border-radius:4px;padding:.55rem 1.2rem;background:#0E7C86;border-color:#0E7C86;color:#fff}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

COLORS = {"Wikipedia": "#0E7C86", "Web": "#C77D0A", "arXiv": "#7A4FC2"}
SOURCE_LABEL = {"Wikipedia": "Wikipedia", "Web": "Web (DuckDuckGo)", "arXiv": "arXiv research papers"}

db.init()
if "sid" not in st.query_params:  # keeps your history after a page refresh, no login needed
    st.query_params["sid"] = uuid.uuid4().hex[:16]
SID = st.query_params["sid"]
st.session_state.setdefault("result", None)


# ───────────────────────── cached helpers ─────────────────────────
@st.cache_data(show_spinner=False, max_entries=20)
def cached_ocr(raw: bytes, lang_label: str, mode: str):
    return ocr.run_ocr(Image.open(io.BytesIO(raw)), lang_label, mode)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def cached_search(query: str, kws: tuple, sources: tuple, wiki_lang: str):
    return search.search_all(query, list(kws), list(sources), wiki_lang)


# ───────────────────────── pipeline ─────────────────────────
def run_pipeline(text, query, ocr_res, lang_label, sources, max_docs, ans_mode, model):
    t0 = time.time()
    with st.status("Working on it…", expanded=True) as status:
        st.write("Searching your sources in parallel")
        wiki_lang = "ur" if rank.is_urdu(text) else "en"
        docs, report = cached_search(query, tuple(rank.keywords(query)), tuple(sources), wiki_lang)
        st.write(f"Found {len(docs)} documents")
        st.write("Ranking by relevance and source credibility")
        ranked = rank.rank_documents(query, docs)[:max_docs]
        st.write("Writing the answer")
        mode_used = "Smart summary"
        if not ranked:
            reply = ("No matching sources were found. Try editing the search query, using fewer "
                     "or different words, or enabling more sources.")
        elif ans_mode.startswith("Local") and model:
            try:
                reply = answer.ollama_answer(text, ranked, model)
                mode_used = f"Ollama · {model}"
            except Exception:
                reply = answer.extractive(query, ranked)
        else:
            reply = answer.extractive(query, ranked)
        status.update(label="Done", state="complete", expanded=False)

    result = {
        "text": text, "query": query, "answer": reply, "mode": mode_used,
        "confidence": rank.confidence(ranked), "docs": [asdict(d) for d in ranked],
        "report": report, "elapsed": round(time.time() - t0, 1),
        "lang": lang_label, "ocr_conf": ocr_res.confidence,
    }
    if ranked:
        db.add(SID, result)
    return result


# ───────────────────────── UI pieces ─────────────────────────
def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def hero():
    st.markdown("""
<div class="hero">
  <h1>Photograph text. Get the answer.</h1>
  <p>Upload a picture of any printed text. OCR Search reads it, looks it up across free sources and
  writes one answer that cites where each point came from.</p>
  <div class="flow">
    <div><b>Photo</b>Cleaned and straightened</div>
    <div><b>Text</b>Read with Tesseract OCR</div>
    <div><b>Search</b>Wikipedia, web, arXiv</div>
    <div><b>Answer</b>Ranked, cited, scored</div>
  </div>
</div>""", unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown("### Settings")
        lang = st.selectbox("Language of the text", list(ocr.LANGS))
        mode = st.selectbox("Image cleaning", ocr.MODES,
                            help="Use 'Strong' for shadows, crumpled paper or low light.")
        sources = st.multiselect("Search sources", list(SOURCE_LABEL), default=list(SOURCE_LABEL),
                                 format_func=SOURCE_LABEL.get)
        max_docs = st.slider("Sources to use", 3, 10, 6)
        models = answer.ollama_models()
        opts = ["Smart summary (free, no AI model)"] + (["Local AI (Ollama)"] if models else [])
        ans_mode = st.radio("Answer style", opts)
        model = st.selectbox("Model", models) if ans_mode.startswith("Local") else None
        st.divider()
        if ocr.tesseract_ready():
            st.success("Tesseract OCR is ready")
        else:
            st.error("Tesseract is not installed. Run `python setup.py` and follow the instructions.")
        needed = set(ocr.LANGS[lang].split("+"))
        if ocr.tesseract_ready() and not needed <= ocr.installed_langs():
            st.warning(f"Language data missing: {', '.join(sorted(needed - ocr.installed_langs()))}. Run `python setup.py`.")
    return lang, mode, sources, max_docs, ans_mode, model


def tab_scan(lang, mode, sources, max_docs, ans_mode, model):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Upload a photo of text")
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                              label_visibility="collapsed")
        with st.expander("Or take a photo with your camera"):
            cam = st.camera_input("Camera", label_visibility="collapsed")
        file = up or cam
        if not file:
            st.info("Tip: a straight, well-lit photo gives the best text. Printed text works better than handwriting.")
            return
        raw = file.getvalue()
        st.image(Image.open(io.BytesIO(raw)), caption="Your image")

    if not ocr.tesseract_ready():
        st.error("Tesseract OCR is not installed on this machine. Run `python setup.py` first.")
        return
    key = hashlib.md5(raw + f"{lang}{mode}".encode()).hexdigest()[:12]
    with st.spinner("Reading the text…"):
        res = cached_ocr(raw, lang, mode)

    with right:
        st.markdown("#### Check the extracted text")
        c1, c2, c3 = st.columns(3)
        c1.metric("OCR confidence", f"{res.confidence:.0f}%")
        c2.metric("Words read", res.words)
        c3.metric("OCR time", f"{res.seconds}s")
        if not res.text:
            st.warning("No text was found. Try 'Strong' image cleaning or a clearer photo.")
        elif res.confidence < 60:
            st.warning("Low confidence. Fix any wrong words below before searching.")
        text = st.text_area("Edit the text if OCR made mistakes", value=res.text, height=210, key=f"t_{key}")
        qkey = hashlib.md5(text.encode()).hexdigest()[:8]
        query = st.text_input("Search query (built from your text, editable)",
                              value=rank.build_query(text), key=f"q_{key}_{qkey}")
        go = st.button("Search and answer", type="primary", disabled=not (query.strip() and sources))
        if not sources:
            st.caption("Choose at least one search source in the sidebar.")
        with st.expander("See the cleaned image sent to OCR"):
            st.image(res.preview_png)

    if go:
        st.session_state.result = run_pipeline(text, query.strip(), res, lang, sources, max_docs, ans_mode, model)
    if st.session_state.result:
        render_result(st.session_state.result)


def report_text(r: dict) -> str:
    src = "\n".join(f"[{i}] {d['title']} ({d['source']}) {d['url']}" for i, d in enumerate(r["docs"], 1))
    return (f"OCR SEARCH REPORT\n\nQuery: {r['query']}\nConfidence: {r['confidence']}%\n\n"
            f"ANSWER\n{r['answer']}\n\nSCANNED TEXT\n{r['text']}\n\nSOURCES\n{src}\n")


def render_result(r: dict):
    st.divider()
    st.markdown("## Result")
    m = st.columns(4)
    m[0].metric("Answer confidence", f"{r['confidence']}%")
    m[1].metric("Sources used", len(r["docs"]))
    m[2].metric("Total time", f"{r['elapsed']}s")
    m[3].metric("Answer style", r["mode"])
    with st.container(border=True):
        st.markdown("##### Answer")
        st.markdown(r["answer"])
    if not r["docs"]:
        return
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download as TXT", report_text(r), "ocr-search-report.txt", "text/plain")
    d2.download_button("Download as Markdown", report_text(r), "ocr-search-report.md", "text/markdown")
    try:
        d3.download_button("Download as PDF", pdf.make_pdf(r["query"], r["text"], r["answer"], r["confidence"], r["docs"]),
                           "ocr-search-report.pdf", "application/pdf")
    except Exception:
        d3.caption("PDF unavailable for this text. Use TXT.")
    st.markdown("##### Sources")
    for i, d in enumerate(r["docs"], 1):
        pct = int(d["score"] * 100)
        url = d["url"] if d["url"].startswith(("http://", "https://")) else ""
        link = f'<a href="{esc(url)}" target="_blank" rel="noopener">Open source</a>' if url else ""
        snippet = d["text"][:280] + ("…" if len(d["text"]) > 280 else "")
        st.markdown(f"""<div class="src" style="--c:{COLORS.get(d['source'], '#0E7C86')}">
<span class="tag">{esc(d['source'])}</span><span class="ttl">[{i}] {esc(d['title'])}</span>
<p>{esc(snippet)}</p>{link}
<div class="bar"><i style="width:{pct}%"></i></div></div>""", unsafe_allow_html=True)
    with st.expander("Search details per source"):
        st.dataframe(pd.DataFrame(r["report"]), hide_index=True)


def tab_history():
    rows = db.history(SID)
    if not rows:
        st.info("Your scans will appear here. History belongs to this browser link only.")
        return
    for r in rows:
        with st.expander(f"{r['created']} · {r['query'][:70]} · {r['confidence']:.0f}%"):
            st.markdown(r["answer"])
            st.caption("Scanned text")
            st.code(r["ocr_text"], language=None)
            st.markdown("  \n".join(f"[{i}] [{s['title']}]({s['url']}) ({s['source']})"
                                     for i, s in enumerate(r["sources"], 1)))
    if st.button("Delete my history"):
        db.clear(SID)
        st.rerun()


def tab_analytics():
    df = db.stats()
    if df.empty:
        st.info("Charts appear after the first scan.")
        return
    m = st.columns(4)
    m[0].metric("Total scans", len(df))
    m[1].metric("Avg answer confidence", f"{df.confidence.mean():.0f}%")
    m[2].metric("Avg OCR confidence", f"{df.ocr_conf.mean():.0f}%")
    m[3].metric("Avg time", f"{df.elapsed.mean():.1f}s")
    df["day"] = pd.to_datetime(df["created"]).dt.date
    a, b = st.columns(2)
    a.markdown("##### Scans per day")
    a.bar_chart(df.groupby("day").size())
    b.markdown("##### Source usage")
    b.bar_chart(df["sources_used"].str.split(",").explode().str.strip().value_counts())
    st.markdown("##### Confidence over time")
    st.line_chart(df.set_index("created")[["confidence", "ocr_conf"]])


def tab_about():
    st.markdown("#### How it works")
    st.graphviz_chart("""digraph {rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#12202B", fontcolor="white", color="#12202B", fontname="Helvetica"];
      edge [color="#0E7C86"];
      Upload -> Preprocess -> OCR -> Query -> Search -> Rank -> Answer;
      Search -> Wikipedia; Search -> Web; Search -> arXiv;
      Wikipedia -> Rank; Web -> Rank; arXiv -> Rank;}""")
    st.markdown("""
**Preprocess.** The photo is converted to grayscale, enlarged, de-noised, contrast-enhanced and straightened.
**OCR.** Tesseract reads English, Urdu, or both. You can correct the text before searching.
**Search.** Wikipedia, DuckDuckGo and arXiv are queried at the same time; a failing source never stops the others.
**Rank.** BM25 relevance, query-word coverage and a source credibility weight (arXiv 1.0, Wikipedia 0.9, web 0.75).
**Answer.** The best non-repeating sentences are selected with citations. If Ollama is running locally, a language model can write the answer instead.
**Confidence.** A heuristic score: 55% query coverage, 25% agreement between different sources, 20% source credibility. It estimates how well the sources match your text, not whether they are true.

Limits: printed text reads far better than handwriting, and free search sources can be rate-limited. Final Year Project, BS Computer Science, University of Agriculture Faisalabad.
""")


# ───────────────────────── main ─────────────────────────
hero()
settings = sidebar()
t1, t2, t3, t4 = st.tabs(["Scan and search", "History", "Analytics", "About"])
with t1:
    tab_scan(*settings)
with t2:
    tab_history()
with t3:
    tab_analytics()
with t4:
    tab_about()
