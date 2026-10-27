"""Image preprocessing, Tesseract OCR, and multi-page / PDF input."""
from __future__ import annotations

import base64
import functools
import importlib.util
import io
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytesseract
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
LOCAL_TESSDATA = ROOT / "tessdata"  # filled by setup.py if the system folder is read-only

LANGS = {"English": "eng", "Urdu": "urd", "English + Urdu": "eng+urd"}
HANDWRITING = "Handwriting (ink boost)"
MODES = ["Clean (recommended)", "Strong (poor photos)", HANDWRITING, "None"]
ENGINES = ["Auto", "Tesseract", "RapidOCR", "Gemini", "Ollama"]
AI_PROMPT = ("Transcribe all the text in this image exactly as written, including handwriting. Keep the line "
             "breaks. Do not translate, summarize or comment. Write [illegible] for unreadable words. "
             "Output only the transcription.")
MAX_PAGES = 5
GOOD, FAIR = 85, 60  # word-confidence thresholds for the confidence map

if not shutil.which("tesseract"):
    _win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _win.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_win)


@dataclass
class OcrResult:
    text: str
    confidence: float    # 0-100, or -1 when the engine gives no confidence (AI models)
    seconds: float
    preview_png: bytes   # image after cleaning
    overlay_png: bytes   # word boxes coloured by confidence
    thumb_png: bytes     # small copy of the original
    words: int
    kind: str = "Tesseract"  # Tesseract | RapidOCR | Gemini | Ollama | Text layer (digital PDF)
    detail: str = ""         # e.g. which Gemini model answered, or why a fallback was used


@dataclass
class DocResult:
    pages: list
    text: str
    confidence: float
    words: int
    seconds: float


@dataclass(frozen=True)
class Engine:
    name: str = "Tesseract"
    gemini_key: str = ""
    gemini_model: str = "auto"
    gemini_speed: str = "Fast"
    ollama_model: str = ""
    ollama_host: str = "http://localhost:11434"


# ───────────── environment checks ─────────────
def tesseract_ready() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def installed_langs() -> set[str]:
    langs: set[str] = set()
    try:
        langs |= set(pytesseract.get_languages(config=""))
    except Exception:
        pass
    if LOCAL_TESSDATA.exists():
        langs |= {p.stem for p in LOCAL_TESSDATA.glob("*.traineddata")}
    return langs


def _tessdata_arg(lang: str) -> str:
    need = lang.split("+")
    if LOCAL_TESSDATA.exists() and all((LOCAL_TESSDATA / f"{l}.traineddata").exists() for l in need):
        return f'--tessdata-dir "{LOCAL_TESSDATA.as_posix()}"'
    return ""


# ───────────── preprocessing ─────────────
def clean_text(t: str) -> str:
    t = t.replace("\r", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)  # re-join hyphenated line breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _resize(gray: np.ndarray) -> np.ndarray:
    longest = max(gray.shape)
    if longest < 1600:
        f = min(1600 / longest, 3.0)
        return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    if longest > 3200:
        f = 3200 / longest
        return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return gray


def _skew_angle(gray: np.ndarray) -> float:
    """Try small rotations; straight text gives the sharpest row-sum profile."""
    f = 600 / max(gray.shape)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = bw.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-8, 8.01, 0.5):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def preprocess(img: Image.Image, mode: str = MODES[0]) -> np.ndarray:
    img = ImageOps.exif_transpose(img)
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = _resize(gray)
    if mode.startswith("None"):
        return gray
    if mode.startswith("Handwriting"):
        # estimate the paper (dilation removes ink), divide it out to flatten shadows, boost contrast
        bg = cv2.GaussianBlur(cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), (0, 0), 21)
        gray = cv2.divide(gray, bg, scale=255)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    angle = _skew_angle(gray)
    if abs(angle) >= 0.5:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if mode.startswith("Strong"):
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        gray = cv2.medianBlur(gray, 3)
    return gray


# ───────────── image encoding helpers ─────────────
def _thumb(img: Image.Image) -> bytes:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _shrink(vis: np.ndarray, side: int = 1400) -> np.ndarray:
    f = side / max(vis.shape[:2])
    return cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else vis


def _overlay(proc: np.ndarray, boxes: list) -> bytes:
    vis = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    t = max(2, round(max(vis.shape[:2]) / 700))
    for x, y, w, h, c in boxes:
        col = (76, 175, 80) if c >= GOOD else (0, 170, 245) if c >= FAIR else (60, 60, 220)
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, t)
    ok, buf = cv2.imencode(".jpg", _shrink(vis), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


# ───────────── OCR ─────────────
def run_ocr(img: Image.Image, lang_label: str = "English", mode: str = MODES[0]) -> OcrResult:
    t0 = time.time()
    lang = LANGS[lang_label]
    proc = preprocess(img, mode)
    cfg = f"--oem 1 --psm 3 {_tessdata_arg(lang)}".strip()
    data = pytesseract.image_to_data(proc, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    boxes: list[tuple] = []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
        boxes.append((data["left"][i], data["top"][i], data["width"][i], data["height"][i], conf))

    text = clean_text("\n".join(" ".join(v) for _, v in sorted(lines.items())))
    ok, buf = cv2.imencode(".png", _shrink(proc))
    return OcrResult(
        text=text,
        confidence=round(float(np.mean(confs)), 1) if confs else 0.0,
        seconds=round(time.time() - t0, 2),
        preview_png=buf.tobytes() if ok else b"",
        overlay_png=_overlay(proc, boxes),
        thumb_png=_thumb(img),
        words=len(confs),
    )


# ───────────── extra engines ─────────────
_RAPID_LOCK = threading.Lock()


def rapid_available() -> bool:
    try:
        return importlib.util.find_spec("rapidocr_onnxruntime") is not None
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _rapid_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _group_lines(items: list) -> str:
    """items: (x0, y0, x1, y1, text, score). Join boxes into reading-order lines."""
    if not items:
        return ""
    items = sorted(items, key=lambda it: (it[1] + it[3]) / 2)
    heights = sorted(it[3] - it[1] for it in items)
    med = heights[len(heights) // 2] or 1
    rows, cur = [], [items[0]]
    cy = (items[0][1] + items[0][3]) / 2
    for it in items[1:]:
        c = (it[1] + it[3]) / 2
        if abs(c - cy) <= 0.6 * med:
            cur.append(it)
            cy = sum((x[1] + x[3]) / 2 for x in cur) / len(cur)
        else:
            rows.append(cur)
            cur, cy = [it], c
    rows.append(cur)
    return "\n".join(" ".join(x[4] for x in sorted(r, key=lambda x: x[0])) for r in rows)


def run_rapid(img: Image.Image, mode: str = HANDWRITING) -> OcrResult:
    """RapidOCR (PP-OCR on ONNX): light, offline, and much better than Tesseract on neat handwriting."""
    t0 = time.time()
    proc = preprocess(img, mode)
    with _RAPID_LOCK:
        result, _ = _rapid_engine()(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
    items, boxes, wsum, csum = [], [], 0.0, 0
    for box, text, score in result or []:
        text = str(text).strip()
        if not text:
            continue
        pts = np.array(box, dtype=float)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        sc = float(score) * 100
        items.append((x0, y0, x1, y1, text, sc))
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), sc))
        wsum += sc * len(text)
        csum += len(text)
    text = clean_text(_group_lines(items))
    ok, buf = cv2.imencode(".png", _shrink(proc))
    return OcrResult(text, round(wsum / csum, 1) if csum else 0.0, round(time.time() - t0, 2),
                     buf.tobytes() if ok else b"", _overlay(proc, boxes), _thumb(img), len(text.split()), "RapidOCR")


def _jpeg_b64(img: Image.Image, side: int = 1568, quality: int = 82) -> str:
    """Smaller upload = faster Gemini. 1568 px is plenty to read handwriting."""
    im = ImageOps.exif_transpose(img).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _ai_result(text: str, img: Image.Image, t0: float, kind: str) -> OcrResult:
    text = clean_text(text)
    th = _thumb(img)
    return OcrResult(text, -1.0, round(time.time() - t0, 2), b"", th, th, len(text.split()), kind)


def run_gemini(img: Image.Image, api_key: str, model: str = "auto", speed: str = "Fast") -> OcrResult:
    """Google Gemini vision (free API key). Best accuracy on handwriting, cursive and Urdu."""
    import gemini
    t0 = time.time()
    text, used = gemini.generate(api_key, [{"text": AI_PROMPT},
                                           {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(img)}}], model, speed=speed)
    res = _ai_result(text, img, t0, "Gemini")
    res.detail = used
    return res


def _fallback(img: Image.Image, lang_label: str, why: Exception) -> OcrResult:
    """If Gemini is rate-limited or offline, keep working with a local engine."""
    if rapid_available() and "urd" not in LANGS[lang_label]:
        res = run_rapid(img, HANDWRITING)
    else:
        res = run_ocr(img, lang_label, MODES[0])
    res.detail = f"Gemini unavailable ({str(why)[:90]}); used {res.kind} instead"
    return res


def run_ollama(img: Image.Image, model: str, host: str = "http://localhost:11434") -> OcrResult:
    """Local vision model through Ollama (offline, e.g. qwen2.5vl, llama3.2-vision, minicpm-v)."""
    t0 = time.time()
    r = httpx.post(f"{host}/api/generate", timeout=300,
                   json={"model": model, "prompt": AI_PROMPT, "images": [_jpeg_b64(img, 1600)],
                         "stream": False, "options": {"temperature": 0}})
    r.raise_for_status()
    return _ai_result(r.json().get("response", ""), img, t0, "Ollama")


def read_image(img: Image.Image, lang_label: str, mode: str, eng: Engine) -> OcrResult:
    """Run the chosen engine. 'Auto' tries the fast engine first and escalates only when quality is low."""
    name = eng.name
    if name == "Tesseract":
        return run_ocr(img, lang_label, mode)
    if name == "RapidOCR":
        return run_rapid(img, mode)
    if name == "Gemini":
        try:
            return run_gemini(img, eng.gemini_key, eng.gemini_model, eng.gemini_speed)
        except Exception as e:
            return _fallback(img, lang_label, e)
    if name == "Ollama":
        return run_ollama(img, eng.ollama_model, eng.ollama_host)

    best = run_ocr(img, lang_label, mode)
    if best.confidence >= 75:
        return best
    if rapid_available() and "urd" not in LANGS[lang_label]:  # RapidOCR models are not trained for Urdu
        try:
            alt = run_rapid(img, HANDWRITING if best.confidence < 60 else mode)
            if alt.confidence > best.confidence:
                best = alt
        except Exception:
            pass
    if best.confidence < 65 and eng.gemini_key:
        try:
            best = run_gemini(img, eng.gemini_key, eng.gemini_model, eng.gemini_speed)
        except Exception:
            pass
    return best


# ───────────── documents: images, multi-page, PDF ─────────────
def _pdf_pages(raw: bytes) -> list[tuple[Image.Image, str]]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    out = []
    with fitz.open(stream=raw, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            layer = page.get_text().strip()
            pix = page.get_pixmap(dpi=170, alpha=False)
            out.append((Image.frombytes("RGB", (pix.width, pix.height), pix.samples), layer))
    return out


def ocr_document(files, lang_label: str = "English", mode: str = MODES[0], engine: Engine | None = None) -> DocResult:
    """files: iterable of (name, bytes). Images and PDFs, up to MAX_PAGES pages, read in parallel."""
    engine = engine or Engine("Tesseract")
    t0 = time.time()
    tasks: list[tuple[Image.Image, str]] = []
    for name, raw in files:
        if raw[:5] == b"%PDF-":
            tasks += _pdf_pages(raw)
        else:
            try:
                img = Image.open(io.BytesIO(raw))
                img.load()
            except Exception:
                raise ValueError(f"'{name}' is not a supported image or PDF file.")
            tasks.append((img, ""))
    tasks = tasks[:MAX_PAGES]

    def work(task):
        img, layer = task
        if len(layer) > 40:  # digital PDF: use the real text, no OCR needed
            th = _thumb(img)
            return OcrResult(clean_text(layer), 100.0, 0.0, b"", th, th, len(layer.split()), "Text layer")
        return read_image(img, lang_label, mode, engine)

    with ThreadPoolExecutor(max_workers=min(3, max(1, len(tasks)))) as ex:
        pages = list(ex.map(work, tasks))

    known = [p for p in pages if p.confidence >= 0]
    kw = sum(p.words for p in known)
    conf = sum(p.confidence * p.words for p in known) / kw if kw else -1.0
    return DocResult(
        pages=pages,
        text="\n\n".join(p.text for p in pages if p.text),
        confidence=round(conf, 1),
        words=sum(p.words for p in pages),
        seconds=round(time.time() - t0, 2),
    )
