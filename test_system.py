"""Automated tests. Run:  python test_system.py   (no internet needed)"""
import io
import sys
import tempfile
from pathlib import Path

import database as db
import evaluation as ev
import ocr_engine as ocr
import pdf_export
import ranker as rank
import web_search as ws
from web_search import Doc

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


class Skip(Exception):
    pass


@test
def metrics_cer_wer():
    assert ev.cer("abc", "abc") == 0
    assert abs(ev.cer("abc", "abd") - 1 / 3) < 1e-9
    assert abs(ev.wer("a b c", "a x c") - 1 / 3) < 1e-9
    assert ev.edit_distance("kitten", "sitting") == 3


@test
def query_builder():
    assert rank.build_query("What is photosynthesis?") == "What is photosynthesis?"
    long = "Photosynthesis converts light energy into chemical energy in plants " * 4
    q = rank.build_query(long)
    assert 0 < len(q.split()) <= 8


@test
def ranking_and_confidence():
    q = "photosynthesis light energy plants"
    docs = [Doc("Web", "Football", "https://a", "Football is a sport played with a ball by two teams."),
            Doc("Wikipedia", "Photosynthesis", "https://b", "Photosynthesis lets plants turn light energy into sugar."),
            Doc("arXiv", "Photosynthesis study", "https://c", "We study light energy capture in plants.")]
    ranked = rank.rank_documents(q, docs)
    assert ranked[0].title != "Football" and ranked[-1].title == "Football"
    assert 0 <= rank.confidence(ranked) <= 100 and rank.confidence([]) == 0


@test
def duplicate_urls_removed():
    d = [Doc("Web", "A", "https://x", "text about plants"), Doc("Web", "B", "https://x", "text about plants")]
    assert len(rank.rank_documents("plants", d)) == 1


@test
def highlight_is_safe():
    out = rank.highlight("<script>alert(1)</script> plants grow", ["plants", "amp"])
    assert "<script>" not in out and "<mark>plants</mark>" in out


@test
def text_analysis():
    a = rank.analyze_text("What is machine learning?")
    assert a["kind"] == "Question" and a["language"] == "English"
    assert rank.analyze_text("یہ ایک اردو جملہ ہے جو آزمائش کے لیے ہے")["language"] == "Urdu"


@test
def openalex_abstract_rebuild():
    assert ws._abstract({"Plants": [0], "make": [1], "sugar": [2]}) == "Plants make sugar"
    assert ws._abstract(None) == ""


@test
def database_roundtrip():
    db.DB = Path(tempfile.mkdtemp()) / "t.db"
    db.init()
    rec = dict(lang="English", ocr_conf=None, query="q", text="t", answer="a", confidence=80, elapsed=1.0,
               docs=[{"title": "T", "url": "u", "source": "Web"}], timings={"OCR": 1})
    sid = db.add("s1", rec)
    db.set_feedback(sid, 1)
    h = db.history("s1")
    assert len(h) == 1 and h[0]["feedback"] == 1 and len(db.stats()) == 1
    db.clear("s1")
    assert db.history("s1") == []


@test
def pdf_report():
    r = dict(query="q", answer="**In short:** x [1]", confidence=70, ocr_conf=None, text="hello", thumb=b"",
             docs=[{"title": "T", "url": "https://x/" + "a" * 150, "source": "Web"}], analysis={"language": "English"})
    assert pdf_export.make_pdf(r)[:4] == b"%PDF"


@test
def ocr_printed_accuracy():
    if not ocr.tesseract_ready():
        raise Skip("Tesseract not installed")
    text = ev.SAMPLES["Photosynthesis"]
    r = ocr.run_ocr(ev.make_sample_image(text, skew=3, noise=6), "English", ocr.MODES[0])
    assert ev.cer(text, r.text) < 0.05, f"CER too high: {ev.cer(text, r.text):.3f}"


@test
def ocr_handwriting_engine():
    if not ocr.rapid_available():
        raise Skip("RapidOCR not installed")
    text = ev.SAMPLES["Machine learning"]
    r = ocr.run_rapid(ev.make_sample_image(text, skew=2, noise=6), ocr.HANDWRITING)
    assert ev.cer(text, r.text) < 0.10, f"CER too high: {ev.cer(text, r.text):.3f}"


@test
def multi_page_pdf_input():
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        raise Skip("PyMuPDF not installed")
    from fpdf import FPDF
    p = FPDF()
    p.add_page()
    p.set_font("Helvetica", size=14)
    p.multi_cell(0, 8, "This digital PDF has a real text layer about photosynthesis in plants and sunlight.")
    doc = ocr.ocr_document([("a.pdf", bytes(p.output()))], "English", ocr.MODES[0])
    assert doc.pages[0].kind == "Text layer" and "photosynthesis" in doc.text.lower()


@test
def bad_file_rejected():
    try:
        ocr.ocr_document([("x.txt", b"not an image")])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


class MockGemini:
    """Local stand-in for the Gemini API: lists models, retires one, answers with another."""
    @staticmethod
    def start():
        import json, threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                models = [("gemini-3.6-flash", ["generateContent"]), ("gemini-3-flash-preview", ["generateContent"]),
                          ("gemini-3.6-flash-tts", ["generateContent"]), ("gemini-embedding-001", ["embedContent"]),
                          ("gemini-3.1-flash-lite", ["generateContent"])]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"models": [{"name": f"models/{n}", "supportedGenerationMethods": m} for n, m in models]}).encode())

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                assert self.headers.get("x-goog-api-key") == "test-key"
                if "gemini-3.6-flash:" in self.path:  # pretend the newest model is retired
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"model not found"}}')
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"candidates": [{"content": {"parts": [{"text": f"answer from {self.path.split('/')[-1].split(':')[0]}"}]}}]}).encode())

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


@test
def gemini_auto_model_and_fallback():
    import gemini
    srv = MockGemini.start()
    old = gemini.BASE
    gemini.BASE = f"http://127.0.0.1:{srv.server_port}/v1beta"
    gemini._cache.clear()
    try:
        assert gemini.candidates("test-key") == ("gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview")
        text, used = gemini.generate("test-key", [{"text": "hi"}])
        assert used == "gemini-3.1-flash-lite" and "answer from" in text   # skipped the retired model
        from web_search import Doc
        ans, used = __import__("answer_writer").gemini_answer("scan text", [Doc("Web", "T", "u", "body")], "test-key")
        assert ans.startswith("answer from")
        try:
            gemini.generate("", [{"text": "hi"}])
        except gemini.GeminiError:
            pass
        else:
            raise AssertionError("empty key should fail")
    finally:
        gemini.BASE = old
        srv.shutdown()


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Skip as s:
            print(f"SKIP  {fn.__name__} ({s})")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed} of {len(TESTS)} tests passed or skipped")
    sys.exit(1 if failed else 0)
