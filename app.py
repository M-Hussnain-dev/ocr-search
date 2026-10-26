"""OCR Search - read printed or handwritten text from images and PDFs, search free sources, answer with citations."""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import platform
import re
import time
import uuid
from dataclasses import asdict

import pandas as pd
import streamlit as st
from PIL import Image

import answer_writer as answer
import database as db
import evaluation as ev
import ocr_engine as ocr
import pdf_export as pdf
import ranker as rank
import web_search as search

# ── Edit these before your presentation ──
PROJECT = {
    "title": "OCR Search: Multi-Source Answers from Printed and Handwritten Text",
    "student": "Your Name",
    "reg_no": "Your Reg. No.",
    "supervisor": "Supervisor Name",
    "department": "Department of Computer Science, University of Agriculture, Faisalabad",
}

st.set_page_config(page_title="OCR Search", page_icon="🔎", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap');
h1,h2,h3,h4,.hero h1{font-family:'Bricolage Grotesque',sans-serif;letter-spacing:-.01em}
.block-container{padding-top:1.6rem;max-width:1180px}
.hero{position:relative;overflow:hidden;background:#12202B;color:#EAF2F5;border-radius:6px;padding:30px 34px 24px;margin-bottom:22px}
.hero h1{color:#fff;font-size:2.3rem;font-weight:800;margin:0 0 6px;padding:0}
.hero p{margin:0;max-width:680px;color:#B9CBD3;font-size:1.02rem}
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
.src p{margin:6px 0 8px;opacity:.88;font-size:.93rem}
.src a{font-size:.88rem;color:#0E7C86;text-decoration:none;font-weight:600}
.bar{height:4px;background:rgba(127,127,127,.25);border-radius:9px;margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--c)}
.chip{display:inline-block;padding:2px 11px;margin:0 6px 6px 0;border:1px solid rgba(127,127,127,.4);border-radius:99px;font-size:.82rem}
mark{background:rgba(14,124,134,.28);color:inherit;padding:0 2px;border-radius:2px}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.3);border-radius:4px;padding:12px 16px}
.stButton>button[kind="primary"]{font-weight:600;border-radius:4px;padding:.55rem 1.2rem;background:#0E7C86;border-color:#0E7C86;color:#fff}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

COLORS = {"Wikipedia": "#0E7C86", "Web": "#C77D0A", "arXiv": "#7A4FC2", "OpenAlex": "#B5446E"}
SOURCE_LABEL = {"Wikipedia": "Wikipedia", "Web": "Web (DuckDuckGo)", "arXiv": "arXiv preprints",
                "OpenAlex": "OpenAlex scholarly works"}
ENGINE_LABEL = {"Auto": "Smart auto (tries fast first)", "Tesseract": "Tesseract (printed text)",
                "RapidOCR": "RapidOCR (neat handwriting, offline)", "Gemini": "Gemini Vision (best for handwriting)",
                "Ollama": "Local vision model (Ollama)"}
MAX_UPLOAD = 25 * 1024 * 1024

db.init()
if "sid" not in st.query_params:  # keeps your history after a page refresh, no login needed
    st.query_params["sid"] = uuid.uuid4().hex[:16]
SID = st.query_params["sid"]
for k, v in {"result": None, "sample": None, "bench": None, "own_bench": None}.items():
    st.session_state.setdefault(k, v)


# ───────────────────────── cached helpers ─────────────────────────
@st.cache_data(show_spinner=False, max_entries=10)
def cached_doc(files: tuple, lang: str, mode: str, eng: str, gem_key: str, gem_model: str, oll_model: str):
    return ocr.ocr_document(files, lang, mode, ocr.Engine(eng, gem_key, gem_model, oll_model))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def cached_search(query: str, kws: tuple, sources: tuple, wiki_lang: str):
    return search.search_all(query, list(kws), list(sources), wiki_lang)


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def secret(name: str) -> str:
    """Read a key from Streamlit secrets (Cloud or .streamlit/secrets.toml) or an environment variable."""
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return str(v or os.environ.get(name, ""))


# ───────────────────────── pipeline ─────────────────────────
def run_pipeline(doc, text, query, lang_label, sources, max_docs, ans_mode, model, key, gem=("", "auto")):
    timings = {"OCR": round(max(doc.seconds, 0), 2)}
    t_start = time.time()
    with st.status("Working on it…", expanded=True) as status:
        st.write("Searching your sources in parallel")
        t = time.time()
        wiki_lang = "ur" if rank.is_urdu(text) else "en"
        docs, report = cached_search(query, tuple(rank.keywords(query)), tuple(sources), wiki_lang)
        timings["Search"] = round(time.time() - t, 2)
        st.write(f"Found {len(docs)} documents")

        st.write("Ranking by relevance and source credibility")
        t = time.time()
        ranked = rank.rank_documents(query, docs)[:max_docs]
        timings["Ranking"] = round(time.time() - t, 2)

        st.write("Writing the answer")
        t = time.time()
        mode_used = "Smart summary"
        if not ranked:
            reply = ("No matching sources were found. Try editing the search query, using fewer or different "
                     "words, or enabling more sources.")
        elif ans_mode.startswith("Gemini") and gem[0]:
            try:
                reply, used = answer.gemini_answer(text, ranked, gem[0], gem[1])
                mode_used = f"Gemini · {used}"
            except Exception:
                reply = answer.extractive(query, ranked)
        elif ans_mode.startswith("Local") and model:
            try:
                reply = answer.ollama_answer(text, ranked, model)
                mode_used = f"Ollama · {model}"
            except Exception:
                reply = answer.extractive(query, ranked)
        else:
            reply = answer.extractive(query, ranked)
        timings["Answer"] = round(time.time() - t, 2)
        status.update(label="Done", state="complete", expanded=False)

    result = {
        "key": key, "text": text, "query": query, "answer": reply, "mode": mode_used,
        "confidence": rank.confidence(ranked), "parts": rank.confidence_parts(ranked),
        "docs": [asdict(d) for d in ranked], "report": report,
        "elapsed": round(timings["OCR"] + time.time() - t_start, 1), "timings": timings,
        "lang": lang_label, "ocr_conf": None if doc.confidence < 0 else doc.confidence,
        "engines": sorted({p.kind for p in doc.pages}), "analysis": rank.analyze_text(text),
        "thumb": doc.pages[0].thumb_png if doc.pages else b"",
    }
    result["scan_id"] = db.add(SID, result) if ranked else None
    return result


# ───────────────────────── sidebar ─────────────────────────
def sidebar() -> dict:
    rapid_ok = ocr.rapid_available()
    models = answer.ollama_models()
    with st.sidebar:
        st.markdown("### Settings")
        lang = st.selectbox("Language of the text", list(ocr.LANGS))
        text_type = st.radio("Text in the image", ["Printed", "Handwritten", "Not sure (auto)"],
                             help="Handwritten uses an engine built for handwriting and a special image cleaning step.")
        with st.expander("Handwriting and AI engines"):
            options = ["Automatic", "Auto", "Tesseract"] + (["RapidOCR"] if rapid_ok else []) + ["Gemini"] + \
                      (["Ollama"] if models else [])
            adv_engine = st.selectbox("OCR engine", options, format_func=lambda x: ENGINE_LABEL.get(x, x))
            adv_mode = st.selectbox("Image cleaning", ["Automatic"] + ocr.MODES)
            gem_key = st.text_input("Free Gemini API key (optional)", value=secret("GEMINI_API_KEY"), type="password",
                                    help="Get a free key at aistudio.google.com/apikey. Gives the best handwriting "
                                         "and Urdu results. The image is sent to Google when this engine is used.")
            gem_model = st.text_input("Gemini model", value="auto",
                                      help="'auto' picks the newest free Flash model your key can use. Type a model name to force one.")
            if gem_key:
                st.success("Gemini key loaded")
            oll_model = st.selectbox("Ollama vision model", models) if models else ""
        sources = st.multiselect("Search sources", list(SOURCE_LABEL), default=list(SOURCE_LABEL),
                                 format_func=SOURCE_LABEL.get)
        max_docs = st.slider("Sources to use", 3, 10, 6)
        opts = ["Smart summary (free, no AI model)"] + (["Gemini AI answer (uses your key)"] if gem_key else []) + \
               (["Local AI (Ollama)"] if models else [])
        ans_mode = st.radio("Answer style", opts)
        model = st.selectbox("Answer model", models) if ans_mode.startswith("Local") else None

        # resolve engine + cleaning mode from the simple choice
        if adv_engine != "Automatic":
            engine = adv_engine
        elif text_type == "Printed":
            engine = "Tesseract"
        elif text_type == "Handwritten":
            engine = "Gemini" if gem_key else "RapidOCR" if rapid_ok else "Tesseract"
        else:
            engine = "Auto"
        mode = adv_mode if adv_mode != "Automatic" else (ocr.HANDWRITING if text_type == "Handwritten" else ocr.MODES[0])
        st.caption(f"Using: **{engine}** with *{mode}* cleaning")
        if engine == "Gemini" and not gem_key:
            st.warning("Add a Gemini API key in 'Handwriting and AI engines'.")
        if text_type == "Handwritten" and engine == "Tesseract":
            st.warning("Tesseract is weak on handwriting. Add a free Gemini key or install RapidOCR.")
        if engine == "RapidOCR" and "urd" in ocr.LANGS[lang]:
            st.warning("RapidOCR does not read Urdu. Use Gemini or Tesseract for Urdu.")

        st.divider()
        if ocr.tesseract_ready():
            st.success("Tesseract OCR is ready")
        else:
            st.error("Tesseract is not installed. Run `python setup.py`.")
        needed = set(ocr.LANGS[lang].split("+"))
        if ocr.tesseract_ready() and not needed <= ocr.installed_langs():
            st.warning(f"Language data missing: {', '.join(sorted(needed - ocr.installed_langs()))}.")
        st.caption(f"**{PROJECT['title']}**  \nStudent: {PROJECT['student']} ({PROJECT['reg_no']})  \n"
                   f"Supervisor: {PROJECT['supervisor']}  \n{PROJECT['department']}")
    return dict(lang=lang, mode=mode, engine=engine, gem_key=gem_key, gem_model=gem_model, oll_model=oll_model,
                text_type=text_type, sources=sources, max_docs=max_docs, ans_mode=ans_mode, model=model)


def hero():
    st.markdown("""
<div class="hero">
  <h1>Photograph text. Get the answer.</h1>
  <p>Upload a photo or PDF of printed or handwritten text. OCR Search reads it, looks it up across free
  sources and writes one answer that cites where each point came from.</p>
  <div class="flow">
    <div><b>Photo or PDF</b>Cleaned, straightened, multi-page</div>
    <div><b>Text</b>Tesseract, RapidOCR or Gemini Vision</div>
    <div><b>Search</b>Wikipedia, web, arXiv, OpenAlex</div>
    <div><b>Answer</b>Ranked, cited, scored and explained</div>
  </div>
</div>""", unsafe_allow_html=True)


# ───────────────────────── scan tab ─────────────────────────
def load_sample(name: str):
    st.session_state.sample = (f"sample-{name}.png", ev.sample_png(ev.SAMPLES[name], skew=2, noise=4))
    st.session_state.result = None


def speak_button(text: str, urdu: bool):
    """Browser text-to-speech (no library, free). Text is JSON-escaped so it cannot break out of the script."""
    plain = re.sub(r"\[\d+\]|\*\*|^- ", "", text, flags=re.M)
    data = json.dumps(plain).replace("<", "\\u003c")
    btn = "padding:6px 14px;border:1px solid #888;border-radius:4px;background:transparent;cursor:pointer;color:inherit"
    page = (f"<script>const T={data};function go(){{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(T);"
            f"u.lang='{'ur-PK' if urdu else 'en-US'}';speechSynthesis.speak(u)}}</script>"
            f"<button onclick='go()' style=\"{btn}\">Read aloud</button> "
            f"<button onclick='speechSynthesis.cancel()' style=\"{btn}\">Stop</button>")
    if hasattr(st, "iframe"):  # newer Streamlit
        st.iframe(page, height=44)
    else:
        try:
                        components.html(page, height=44)
        except Exception:
            pass


def tab_scan(cfg: dict):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Upload a photo or PDF")
        up = st.file_uploader("Files", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
                              accept_multiple_files=True, label_visibility="collapsed")
        with st.expander("Or take a photo with your camera"):
            cam = st.camera_input("Camera", label_visibility="collapsed")
        files = [(f.name, f.getvalue()) for f in up or []]
        if cam:
            files.append(("camera.jpg", cam.getvalue()))
        if not files and st.session_state.sample:
            files = [st.session_state.sample]
            st.caption(f"Using built-in {files[0][0]}")
            if st.button("Remove sample"):
                st.session_state.sample = None
                st.rerun()
        if not files:
            st.info("Tip: a straight, well-lit photo works best. Choose 'Handwritten' in the sidebar for handwriting.")
            st.markdown("**No file at hand? Try a built-in sample:**")
            for col, name in zip(st.columns(len(ev.SAMPLES)), ev.SAMPLES):
                col.button(name, key=f"s_{name}", on_click=load_sample, args=(name,))
            return
        if sum(len(r) for _, r in files) > MAX_UPLOAD:
            st.error("Files are larger than 25 MB in total. Please upload smaller files.")
            return
    if not ocr.tesseract_ready() and cfg["engine"] in ("Tesseract", "Auto"):
        st.error("Tesseract OCR is not installed on this machine. Run `python setup.py` first.")
        return

    key = hashlib.md5(b"".join(r for _, r in files) + repr((cfg["lang"], cfg["mode"], cfg["engine"])).encode()).hexdigest()[:12]
    try:
        with st.spinner("Reading the text…"):
            doc = cached_doc(tuple(files), cfg["lang"], cfg["mode"], cfg["engine"], cfg["gem_key"],
                             cfg["gem_model"], cfg["oll_model"] or "")
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return

    with left:
        thumbs = [p.thumb_png for p in doc.pages[:4]]
        st.image(thumbs, caption=[f"Page {i + 1}" for i in range(len(thumbs))] if len(thumbs) > 1 else "Your file")
        if len(doc.pages) > 4:
            st.caption(f"Showing 4 of {len(doc.pages)} pages")

    with right:
        st.markdown("#### Check the extracted text")
        c = st.columns(4)
        c[0].metric("OCR confidence", "n/a" if doc.confidence < 0 else f"{doc.confidence:.0f}%")
        c[1].metric("Words read", doc.words)
        c[2].metric("Pages", len(doc.pages))
        c[3].metric("OCR time", f"{doc.seconds}s")
        st.caption("Engine: " + ", ".join(sorted({p.kind for p in doc.pages})))
        if not doc.text:
            st.warning("No text was found. Try another engine, 'Handwritten' mode, or a clearer photo.")
        elif 0 <= doc.confidence < 60:
            hint = " If this is handwriting, set 'Text in the image' to Handwritten." if cfg["text_type"] == "Printed" else ""
            st.warning("Low confidence. Fix any wrong words below before searching." + hint)
        if any(p.kind in ("Gemini", "Ollama") for p in doc.pages):
            st.caption("AI engines give no per-word confidence, so please proofread the text.")
        text = st.text_area("Edit the text if OCR made mistakes", value=doc.text, height=200, key=f"t_{key}")
        info = rank.analyze_text(text)
        st.markdown("".join(f'<span class="chip">{esc(x)}</span>' for x in
                            [info["language"], info["kind"], f"{info['words']} words", f"{info['sentences']} sentences"]
                            + [f"#{k}" for k in info["keywords"][:6]]), unsafe_allow_html=True)
        qkey = hashlib.md5(text.encode()).hexdigest()[:8]
        query = st.text_input("Search query (built from your text, editable)", value=rank.build_query(text),
                              key=f"q_{key}_{qkey}")
        go = st.button("Search and answer", type="primary", disabled=not (query.strip() and cfg["sources"]))
        if not cfg["sources"]:
            st.caption("Choose at least one search source in the sidebar.")
        with st.expander("OCR confidence map and cleaned image"):
            i = st.selectbox("Page", range(len(doc.pages)), format_func=lambda x: f"Page {x + 1} ({doc.pages[x].kind})") \
                if len(doc.pages) > 1 else 0
            pg = doc.pages[i]
            st.image(pg.overlay_png)
            if pg.kind in ("Tesseract", "RapidOCR"):
                st.caption("Green boxes: confident. Amber: please check. Red: probably wrong.")
            if pg.preview_png:
                st.image(pg.preview_png, caption="Cleaned image sent to the engine")

    if go:
        st.session_state.result = run_pipeline(doc, text, query.strip(), cfg["lang"], cfg["sources"], cfg["max_docs"],
                                               cfg["ans_mode"], cfg["model"], key,
                                               (cfg["gem_key"], cfg["gem_model"]))
    r = st.session_state.result
    if r and r["key"] == key:
        render_result(r)


def report_text(r: dict) -> str:
    refs = "\n".join(f"[{i}] {d['title']}. {d['source']}. {d['url']}" for i, d in enumerate(r["docs"], 1))
    oc = "n/a" if r["ocr_conf"] is None else f"{r['ocr_conf']:.0f}%"
    return (f"OCR SEARCH REPORT\n\nQuery: {r['query']}\nAnswer confidence: {r['confidence']}%\nOCR confidence: {oc}\n\n"
            f"ANSWER\n{r['answer']}\n\nSCANNED TEXT\n{r['text']}\n\nREFERENCES\n{refs}\n")


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
        if r["docs"]:
            speak_button(r["answer"], r["analysis"]["language"] == "Urdu")
    if not r["docs"]:
        return

    if r.get("scan_id"):
        st.caption("Was this answer helpful?")
        fb = st.feedback("thumbs", key=f"fb_{r['scan_id']}")
        if fb is not None:
            db.set_feedback(r["scan_id"], fb)

    d = st.columns(4)
    d[0].download_button("TXT", report_text(r), "ocr-search-report.txt", "text/plain")
    d[1].download_button("Markdown", report_text(r), "ocr-search-report.md", "text/markdown")
    d[2].download_button("JSON", json.dumps({k: r[k] for k in ("query", "answer", "confidence", "ocr_conf", "text",
                                                                 "docs", "timings")}, indent=2, ensure_ascii=False),
                         "ocr-search-report.json", "application/json")
    try:
        d[3].download_button("PDF", pdf.make_pdf(r), "ocr-search-report.pdf", "application/pdf")
    except Exception:
        d[3].caption("PDF unavailable")

    st.markdown("##### Sources")
    terms = rank.tokenize(r["query"])
    for i, s in enumerate(r["docs"], 1):
        pct = int(s["score"] * 100)
        url = s["url"] if s["url"].startswith(("http://", "https://")) else ""
        link = f'<a href="{esc(url)}" target="_blank" rel="noopener">Open source</a>' if url else ""
        snippet = s["text"][:300] + ("…" if len(s["text"]) > 300 else "")
        st.markdown(f"""<div class="src" style="--c:{COLORS.get(s['source'], '#0E7C86')}">
<span class="tag">{esc(s['source'])}</span><span class="ttl">[{i}] {esc(s['title'])}</span>
<p>{rank.highlight(snippet, terms)}</p>{link}
<div class="bar"><i style="width:{pct}%"></i></div></div>""", unsafe_allow_html=True)

    with st.expander("Why this answer? (scores explained)"):
        p = r["parts"]
        c = st.columns(3)
        c[0].metric("Query coverage (55%)", f"{p['coverage']:.0%}", help="Share of your query words found in the top 3 sources.")
        c[1].metric("Source agreement (25%)", f"{p['agreement']:.0%}", help="Do different kinds of sources agree?")
        c[2].metric("Credibility (20%)", f"{p['credibility']:.0%}", help="Weight of the source type.")
        st.dataframe(pd.DataFrame([{
            "#": i, "Source": s["source"], "Title": s["title"][:70], "Query words found": s["coverage"],
            "Relevance": s["relevance"], "Credibility": rank.CREDIBILITY.get(s["source"], 0.7), "Final score": s["score"]}
            for i, s in enumerate(r["docs"], 1)]), hide_index=True)
        st.caption("Final score = 0.8 × relevance + 0.2 × credibility, where relevance = 0.6 × BM25 + 0.4 × query coverage.")
    with st.expander("Speed and search details"):
        st.bar_chart(pd.Series(r["timings"], name="seconds"))
        st.dataframe(pd.DataFrame(r["report"]), hide_index=True)


# ───────────────────────── evaluation tab ─────────────────────────
def show_bench(df: pd.DataFrame):
    st.dataframe(df, hide_index=True, column_config={
        "Accuracy %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "CER %": st.column_config.NumberColumn(format="%.2f"), "WER %": st.column_config.NumberColumn(format="%.2f")})
    best = df.sort_values(["CER %", "Seconds"]).iloc[0]
    st.success(f"Best result: {best['Engine']} with '{best['Cleaning mode']}' cleaning "
               f"(CER {best['CER %']}%, accuracy {best['Accuracy %']}%).")
    st.download_button("Download results (CSV)", df.to_csv(index=False), "ocr-benchmark.csv", "text/csv")


def tab_eval(cfg: dict):
    st.markdown("#### Measure OCR accuracy")
    st.write("Accuracy is measured with **Character Error Rate (CER)** and **Word Error Rate (WER)** against known "
             "correct text. Lower error is better.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Built-in benchmark")
        st.caption("Three generated test images (clean, tilted, poor quality) across engines and cleaning modes.")
        if st.button("Run benchmark"):
            bar = st.progress(0.0, "Starting")
            st.session_state.bench = ev.benchmark_suite(lambda p, msg: bar.progress(p, msg))
            bar.empty()
    with b:
        st.markdown("##### Test your own image or handwriting")
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], key="eval_up")
        truth = st.text_area("Correct text (type exactly what the image says)", height=120)
        avail = ["Tesseract"] + (["RapidOCR"] if ocr.rapid_available() else []) + (["Gemini"] if cfg["gem_key"] else [])
        engines = st.multiselect("Engines to compare", avail, default=avail[:2])
        if st.button("Compare", disabled=not (f and truth.strip() and engines)):
            with st.spinner("Running OCR with each engine and cleaning mode…"):
                eng = [ocr.Engine(n, cfg["gem_key"], cfg["gem_model"]) for n in engines]
                st.session_state.own_bench = ev.benchmark_image(Image.open(f), truth, cfg["lang"], "Your image", eng)
    if st.session_state.bench is not None:
        st.markdown("##### Benchmark results")
        show_bench(st.session_state.bench)
    if st.session_state.own_bench is not None:
        st.markdown("##### Your image")
        show_bench(st.session_state.own_bench)


# ───────────────────────── history, analytics, about ─────────────────────────
def tab_history():
    rows = db.history(SID)
    if not rows:
        st.info("Your scans will appear here. History belongs to this browser link only.")
        return
    for r in rows:
        fb = {1: " · helpful", 0: " · not helpful"}.get(r.get("feedback"), "")
        with st.expander(f"{r['created']} · {r['query'][:70]} · {r['confidence']:.0f}%{fb}"):
            st.markdown(r["answer"])
            st.caption("Scanned text")
            st.code(r["ocr_text"], language=None)
            st.markdown("  \n".join(f"[{i}] [{s['title']}]({s['url']}) ({s['source']})" for i, s in enumerate(r["sources"], 1)))
    if st.button("Delete my history"):
        db.clear(SID)
        st.rerun()


def tab_analytics():
    df = db.stats()
    if df.empty:
        st.info("Charts appear after the first scan.")
        return
    rated = df["feedback"].dropna()
    m = st.columns(5)
    m[0].metric("Total scans", len(df))
    m[1].metric("Avg answer confidence", f"{df.confidence.mean():.0f}%")
    m[2].metric("Avg OCR confidence", "n/a" if df.ocr_conf.isna().all() else f"{df.ocr_conf.mean():.0f}%")
    m[3].metric("Avg time", f"{df.elapsed.mean():.1f}s")
    m[4].metric("Helpful rate", f"{rated.mean() * 100:.0f}%" if len(rated) else "no votes yet")
    df["day"] = pd.to_datetime(df["created"]).dt.date
    a, b = st.columns(2)
    a.markdown("##### Scans per day")
    a.bar_chart(df.groupby("day").size())
    b.markdown("##### Source usage")
    b.bar_chart(df["sources_used"].str.split(",").explode().str.strip().value_counts())
    c, d = st.columns(2)
    stages = [json.loads(t) for t in df["timings"].dropna() if t]
    if stages:
        c.markdown("##### Average time per stage (seconds)")
        c.bar_chart(pd.DataFrame(stages).mean())
    d.markdown("##### Confidence distribution")
    d.bar_chart(pd.cut(df.confidence, [-1, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81-100"]).value_counts().sort_index())
    st.markdown("##### Confidence over time")
    st.line_chart(df.set_index("created")[["confidence", "ocr_conf"]])


def tab_about():
    st.markdown("#### How it works")
    st.graphviz_chart("""digraph {rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#12202B", fontcolor="white", color="#12202B", fontname="Helvetica"];
      edge [color="#0E7C86"];
      Upload -> Preprocess -> OCR -> Query -> Search -> Rank -> Answer;
      OCR -> Tesseract; OCR -> RapidOCR; OCR -> Gemini;
      Search -> Wikipedia; Search -> Web; Search -> arXiv; Search -> OpenAlex;
      Wikipedia -> Rank; Web -> Rank; arXiv -> Rank; OpenAlex -> Rank;}""")
    st.dataframe(pd.DataFrame([
        ("Input", "Images, multi-page photos and PDFs (digital text layer is used directly)", "Pillow, PyMuPDF"),
        ("Preprocessing", "Grayscale, resize, denoise, CLAHE contrast, deskew, adaptive threshold, ink boost for handwriting", "OpenCV"),
        ("OCR", "Tesseract (printed, English/Urdu), RapidOCR (neat handwriting, offline), Gemini or Ollama vision (any handwriting)", "pytesseract, ONNX, REST"),
        ("Smart auto", "Fast engine first; escalates to a stronger engine only when confidence is low", "Python"),
        ("Search", "Four free sources queried in parallel; one failing source never stops the others", "httpx, threads"),
        ("Ranking", "BM25 + query coverage + source credibility", "rank-bm25"),
        ("Answer", "Extractive summary with citations, or a local LLM", "Ollama (optional)"),
        ("Evaluation", "CER and WER benchmark across engines and cleaning modes", "Levenshtein distance"),
    ], columns=["Stage", "Technique", "Tools"]), hide_index=True)
    st.markdown("##### Formulas")
    st.latex(r"CER=\frac{S+D+I}{N}\qquad WER=\frac{S_w+D_w+I_w}{N_w}")
    st.latex(r"score=0.8\,(0.6\,BM25_{norm}+0.4\,coverage)+0.2\,credibility")
    st.latex(r"confidence=100\,(0.55\,coverage+0.25\,agreement+0.20\,credibility)")
    st.markdown("""
**Limits.** Printed text is read far more accurately than handwriting. Cursive and messy writing need the Gemini or
Ollama vision engine. Free web sources can be rate-limited. The confidence score estimates how well the sources match
the text; it does not prove the answer is true.

**Future work.** Fine-tuning a handwriting model on local data, user accounts, more languages, and semantic (embedding) ranking.
""")
    st.markdown("##### System status")
    try:
        tv = str(ocr.pytesseract.get_tesseract_version())
    except Exception:
        tv = "not found"
    st.dataframe(pd.DataFrame([
        ("Python", platform.python_version()), ("Streamlit", st.__version__), ("Tesseract", tv),
        ("OCR languages", ", ".join(sorted(ocr.installed_langs())) or "none"),
        ("RapidOCR", "available" if ocr.rapid_available() else "not installed"),
        ("Stored scans", str(len(db.stats()))),
    ], columns=["Component", "Status"]), hide_index=True)
    st.caption(f"{PROJECT['title']} · {PROJECT['student']} · Supervisor: {PROJECT['supervisor']} · {PROJECT['department']}")


# ───────────────────────── main ─────────────────────────
hero()
settings = sidebar()
t1, t2, t3, t4, t5 = st.tabs(["Scan and search", "Accuracy lab", "History", "Analytics", "About"])
with t1:
    tab_scan(settings)
with t2:
    tab_eval(settings)
with t3:
    tab_history()
with t4:
    tab_analytics()
with t5:
    tab_about()
