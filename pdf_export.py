"""PDF export (built-in PDF fonts support Latin text; use the TXT download for Urdu)."""
from __future__ import annotations

from fpdf import FPDF


def _safe(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


def make_pdf(query: str, text: str, answer: str, confidence: int, docs: list[dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    def block(txt: str, size: int = 11, style: str = "", gap: float = 2):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 6, _safe(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    block("OCR Search Report", 20, "B", 3)
    block(f"Query: {query}\nAnswer confidence: {confidence}%", 10)
    block("Answer", 14, "B")
    block(answer.replace("**", ""))
    block("Scanned text", 14, "B")
    block(text[:3000])
    block("Sources", 14, "B")
    for i, d in enumerate(docs, 1):
        block(f"[{i}] {d['title']} ({d['source']})\n{d['url']}", 10, "", 1)
    return bytes(pdf.output())
