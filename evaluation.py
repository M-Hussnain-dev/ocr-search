"""OCR accuracy measurement (CER / WER) and built-in benchmark images."""
from __future__ import annotations

import io
import random
import re
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import ocr_engine as ocr

SAMPLES = {
    "Photosynthesis": ("Photosynthesis is the process used by plants, algae and some bacteria to convert light "
                       "energy into chemical energy. Chlorophyll in the leaves absorbs sunlight, and the plant "
                       "uses carbon dioxide and water to produce glucose and oxygen."),
    "Machine learning": ("Machine learning is a field of artificial intelligence in which computers learn patterns "
                         "from data instead of following fixed rules. It is widely used in image recognition, "
                         "speech recognition and recommendation systems."),
    "Climate change": ("Climate change refers to long term shifts in temperatures and weather patterns. Burning "
                       "fossil fuels such as coal, oil and gas releases greenhouse gases that trap heat in the "
                       "atmosphere and raise global temperatures."),
}

# image quality levels used by the benchmark: (sample, degradations)
LEVELS = {
    "Clean scan": ("Photosynthesis", dict()),
    "Tilted photo": ("Machine learning", dict(skew=3, noise=6, bg=(244, 240, 228))),
    "Poor quality": ("Climate change", dict(skew=-2, noise=14, blur=1.0, bg=(228, 228, 228))),
}


# ───────────── metrics ─────────────
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / len(ref) if ref else 0.0


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / len(ref) if ref else 0.0


# ───────────── sample images ─────────────
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def make_sample_image(text: str, skew: float = 0, noise: float = 0, blur: float = 0,
                      size: int = 34, width: int = 38, bg=(255, 255, 255)) -> Image.Image:
    f = _font(size)
    lines = textwrap.wrap(text, width)
    img = Image.new("RGB", (1150, 100 + len(lines) * (size + 22)), bg)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((50, 50 + i * (size + 22)), line, fill=(25, 25, 25), font=f)
    if skew:
        img = img.rotate(skew, expand=True, fillcolor=bg)
    if noise:
        rng = np.random.default_rng(7)
        arr = np.array(img).astype(int) + rng.normal(0, noise, (img.height, img.width, 1))
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def sample_png(text: str, **kw) -> bytes:
    buf = io.BytesIO()
    make_sample_image(text, **kw).save(buf, "PNG")
    return buf.getvalue()


# ───────────── benchmarks ─────────────
def benchmark_image(img: Image.Image, truth: str, lang_label: str = "English", quality: str = "Your image",
                    engines=None, modes=None) -> pd.DataFrame:
    engines = engines or [ocr.Engine("Tesseract")]
    modes = modes or ocr.MODES
    rows, truth = [], truth[:3000]
    for eng in engines:
        for mode in modes:
            r = ocr.read_image(img, lang_label, mode, eng)
            c, w = cer(truth, r.text), wer(truth, r.text)
            rows.append({"Image quality": quality, "Engine": eng.name, "Cleaning mode": mode,
                         "CER %": round(100 * c, 2), "WER %": round(100 * w, 2),
                         "Accuracy %": round(max(0.0, 100 - 100 * c), 2), "Seconds": r.seconds})
    return pd.DataFrame(rows)


def benchmark_suite(progress=None) -> pd.DataFrame:
    engines = [ocr.Engine("Tesseract")] + ([ocr.Engine("RapidOCR")] if ocr.rapid_available() else [])
    modes = [ocr.MODES[0], ocr.MODES[1], ocr.HANDWRITING]
    frames = []
    for i, (quality, (sample, kw)) in enumerate(LEVELS.items()):
        if progress:
            progress(i / len(LEVELS), f"Testing: {quality}")
        frames.append(benchmark_image(make_sample_image(SAMPLES[sample], **kw), SAMPLES[sample],
                                      "English", quality, engines, modes))
    if progress:
        progress(1.0, "Done")
    return pd.concat(frames, ignore_index=True)
