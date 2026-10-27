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
    """Local stand-in for the Gemini API. Records every request so tests can inspect it."""
    log: list = []

    @classmethod
    def start(cls, models=None):
        import json, threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        models = models or [("gemini-3.6-flash", ["generateContent"]), ("gemini-3-flash-preview", ["generateContent"]),
                            ("gemini-3.1-flash-lite", ["generateContent"]), ("gemini-3.6-flash-tts", ["generateContent"]),
                            ("gemini-embedding-001", ["embedContent"])]
        cls.log = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"models": [{"name": f"models/{n}", "supportedGenerationMethods": m} for n, m in models]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                name = self.path.split("/")[-1].split(":")[0]
                cfg = body.get("generationConfig", {})
                cls.log.append((name, cfg.get("thinkingConfig")))
                if name == "gemini-3.6-flash":  # pretend this model was retired
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"model not found"}}')
                    return
                if name == "gemini-2.5-flash" and "thinkingConfig" in cfg:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"Thinking config is not supported for this model"}}')
                    return
                self.send_response(200)
                self.end_headers()
                if "stream" in self.path:
                    for piece in ("Hel", "lo"):
                        self.wfile.write(f'data: {json.dumps({"candidates": [{"content": {"parts": [{"text": piece}]}}]})}\n\n'.encode())
                else:
                    self.wfile.write(json.dumps({"candidates": [{"content": {"parts": [{"text": f"answer from {name}"}]}}]}).encode())

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv


def _mock_gemini(models=None):
    import gemini
    srv = MockGemini.start(models)
    old = gemini.BASE
    gemini.BASE = f"http://127.0.0.1:{srv.server_port}/v1beta"
    gemini._models.clear()
    gemini._good.clear()
    gemini._no_think.clear()
    return gemini, srv, old


@test
def gemini_model_order_fast_vs_accurate():
    gemini, srv, old = _mock_gemini()
    try:
        assert gemini.candidates("test-key", "Fast") == ("gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3-flash-preview")
        assert gemini.candidates("test-key", "Accurate") == ("gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview")
    finally:
        gemini.BASE = old
        srv.shutdown()


@test
def gemini_skips_retired_model_remembers_winner_and_turns_thinking_off():
    gemini, srv, old = _mock_gemini()
    try:
        text, used = gemini.generate("test-key", [{"text": "hi"}], speed="Accurate")
        assert used == "gemini-3.1-flash-lite" and text.startswith("answer from")
        assert MockGemini.log[0][0] == "gemini-3.6-flash" and MockGemini.log[0][1] == {"thinkingLevel": "minimal"}
        MockGemini.log.clear()
        gemini.generate("test-key", [{"text": "again"}], speed="Accurate")
        assert [n for n, _ in MockGemini.log] == ["gemini-3.1-flash-lite"]   # remembered: no wasted attempt
        from web_search import Doc
        ans, used = __import__("answer_writer").gemini_answer("scan text", [Doc("Web", "T", "u", "body")], "test-key")
        assert ans.startswith("answer from")
    finally:
        gemini.BASE = old
        srv.shutdown()


@test
def gemini_retries_without_thinking_when_rejected():
    gemini, srv, old = _mock_gemini()
    try:
        text, used = gemini.generate("test-key", [{"text": "hi"}], model="gemini-2.5-flash")
        assert used == "gemini-2.5-flash"
        assert MockGemini.log == [("gemini-2.5-flash", {"thinkingBudget": 0}), ("gemini-2.5-flash", None)]
        MockGemini.log.clear()
        gemini.generate("test-key", [{"text": "again"}], model="gemini-2.5-flash")
        assert MockGemini.log == [("gemini-2.5-flash", None)]   # rejected setting is remembered
    finally:
        gemini.BASE = old
        srv.shutdown()


@test
def gemini_streaming_reports_model():
    gemini, srv, old = _mock_gemini()
    try:
        info = {}
        pieces = list(__import__("answer_writer").gemini_stream("scan", [__import__("web_search").Doc("Web", "T", "u", "b")], "test-key", info=info))
        assert "".join(pieces) == "Hello" and info["model"] == "gemini-3.1-flash-lite"
        try:
            list(gemini.stream("", [{"text": "x"}]))
        except gemini.GeminiError:
            pass
        else:
            raise AssertionError("empty key should fail")
    finally:
        gemini.BASE = old
        srv.shutdown()


@test
def gemini_failure_falls_back_to_local_ocr():
    if not ocr.tesseract_ready():
        raise Skip("Tesseract not installed")
    import gemini
    old = gemini.BASE
    gemini.BASE = "http://127.0.0.1:9/v1beta"   # nothing listens here
    try:
        img = ev.make_sample_image(ev.SAMPLES["Photosynthesis"])
        r = ocr.read_image(img, "English", ocr.MODES[0], ocr.Engine("Gemini", "k", "gemini-3.1-flash-lite"))
        assert r.detail.startswith("Gemini unavailable") and ev.cer(ev.SAMPLES["Photosynthesis"], r.text) < 0.1
    finally:
        gemini.BASE = old


@test
def slow_search_source_is_skipped():
    import time
    docs_fast = [Doc("Wikipedia", "A", "https://a", "plants")]
    old_w, old_web = ws.wikipedia, ws.web
    ws.wikipedia = lambda q, lang="en", n=3: docs_fast
    ws.web = lambda q, n=6: (time.sleep(3), [])[1]
    try:
        t0 = time.time()
        docs, report = ws.search_all("plants", ["plants"], ["Wikipedia", "Web"], "en", deadline=0.5)
        assert time.time() - t0 < 2 and len(docs) == 1
        assert any(r["source"] == "Web" and r["error"].startswith("skipped") for r in report)
    finally:
        ws.wikipedia, ws.web = old_w, old_web


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
