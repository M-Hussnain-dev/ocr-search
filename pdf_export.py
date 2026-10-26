"""Branded PDF report (built-in PDF fonts are Latin only; use the TXT/JSON download for Urdu)."""
from __future__ import annotations

import io
from datetime import datetime

from fpdf import FPDF


def _safe(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        self.set_fill_color(18, 32, 43)
        self.rect(0, 0, 210, 20, "F")
        self.set_xy(10, 6)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "OCR Search Report")
        self.set_text_color(0, 0, 0)
        self.set_y(26)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}  |  {datetime.now():%Y-%m-%d %H:%M}", align="C")


def make_pdf(r: dict) -> bytes:
    """r: the result dict produced by the app (query, answer, confidence, text, docs, ...)."""
    pdf = Report()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    def block(txt: str, size: int = 11, style: str = "", gap: float = 2):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 6, _safe(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    a = r.get("analysis", {})
    oc = r.get("ocr_conf")
    oc_txt = "n/a" if oc is None else f"{oc:.0f}%"
    block(f"Query: {r['query']}", 11, "B", 1)
    block(f"Answer confidence: {r['confidence']}%   |   OCR confidence: {oc_txt}   |   "
          f"Language: {a.get('language', '-')}   |   Sources: {len(r['docs'])}", 9, "", 4)
    if r.get("thumb"):
        try:
            pdf.image(io.BytesIO(r["thumb"]), w=60)
            pdf.ln(4)
        except Exception:
            pass
    block("Answer", 14, "B")
    block(r["answer"].replace("**", ""))
    block("Scanned text", 14, "B")
    block(r["text"][:3000])
    block("References", 14, "B")
    for i, d in enumerate(r["docs"], 1):
        block(f"[{i}] {d['title']}. {d['source']}. {d['url']}", 9, "", 1)
    return bytes(pdf.output())
