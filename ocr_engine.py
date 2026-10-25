"""Image preprocessing and OCR (Tesseract)."""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
LOCAL_TESSDATA = ROOT / "tessdata"  # filled by setup.py if the system folder is read-only

LANGS = {"English": "eng", "Urdu": "urd", "English + Urdu": "eng+urd"}
MODES = ["Clean (recommended)", "Strong (poor photos)", "None"]

if not shutil.which("tesseract"):
    _win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _win.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_win)


@dataclass
class OcrResult:
    text: str
    confidence: float
    seconds: float
    preview_png: bytes
    words: int


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
    """Try small rotations; the straightest text gives the sharpest row profile."""
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


def run_ocr(img: Image.Image, lang_label: str = "English", mode: str = MODES[0]) -> OcrResult:
    t0 = time.time()
    lang = LANGS[lang_label]
    proc = preprocess(img, mode)
    cfg = f"--oem 1 --psm 3 {_tessdata_arg(lang)}".strip()
    data = pytesseract.image_to_data(proc, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
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

    text = clean_text("\n".join(" ".join(v) for _, v in sorted(lines.items())))
    ok, buf = cv2.imencode(".png", proc)
    return OcrResult(
        text=text,
        confidence=round(float(np.mean(confs)), 1) if confs else 0.0,
        seconds=round(time.time() - t0, 2),
        preview_png=buf.tobytes() if ok else b"",
        words=len(confs),
    )
