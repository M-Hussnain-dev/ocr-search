"""Free multi-source search (no API keys): Wikipedia, DuckDuckGo web, arXiv, OpenAlex."""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import httpx

HEADERS = {"User-Agent": "OCRSearchFYP/1.0 (educational project; set CONTACT in the app) python-httpx"}
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
CONTACT = ""


def set_contact(contact: str) -> None:
    """Wikimedia requires every client to identify itself with real contact details (an email or your GitHub link)
    or it answers '403 Please respect our robot policy'."""
    global CONTACT
    CONTACT = (contact or "").strip()
    who = CONTACT or "educational project; set CONTACT in the app"
    HEADERS["User-Agent"] = f"OCRSearchFYP/1.0 ({who}) python-httpx"
TIMEOUT = httpx.Timeout(12.0)
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class Doc:
    source: str  # "Wikipedia" | "Web" | "arXiv" | "OpenAlex"
    title: str
    url: str
    text: str
    coverage: float = 0.0   # share of query words found in the document
    relevance: float = 0.0  # BM25 + coverage
    score: float = 0.0      # relevance + source credibility


def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def _wikipedia_api(query: str, lang: str, n: int) -> list[Doc]:
    api = WIKI_API.format(lang=lang)
    with httpx.Client(headers=HEADERS, timeout=TIMEOUT) as c:
        r = c.get(api, params={"action": "query", "list": "search", "srsearch": query[:300],
                               "srlimit": n, "format": "json"})
        r.raise_for_status()
        titles = [h["title"] for h in r.json().get("query", {}).get("search", [])]
        if not titles:
            return []
        r = c.get(api, params={"action": "query", "prop": "extracts|info", "exintro": 1,
                               "explaintext": 1, "exsentences": 10, "exlimit": "max",
                               "inprop": "url", "redirects": 1, "titles": "|".join(titles),
                               "format": "json"})
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {}).values()
    return [Doc("Wikipedia", p["title"], p.get("fullurl", ""), _clean(p.get("extract")))
            for p in pages if p.get("extract")]


def wikipedia(query: str, lang: str = "en", n: int = 3) -> list[Doc]:
    """Wikipedia API. If Wikimedia refuses (403/429/503), fall back to a web search limited to Wikipedia pages."""
    try:
        return _wikipedia_api(query, lang, n)
    except httpx.HTTPStatusError as e:
        if e.response.status_code not in (403, 429, 503):
            raise
        try:
            docs = web(f"{query[:200]} site:{lang}.wikipedia.org", n=n + 2)
        except Exception:
            raise e
        found = [Doc("Wikipedia", re.sub(r"\s*[-|]\s*Wikipedia.*$", "", d.title), d.url, d.text)
                 for d in docs if "wikipedia.org" in d.url]
        if not found:
            raise e
        return found[:n]


def web(query: str, n: int = 6) -> list[Doc]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    with DDGS() as d:
        hits = list(d.text(query[:300], max_results=n))
    return [Doc("Web", _clean(h.get("title")), h.get("href", ""), _clean(h.get("body")))
            for h in hits if h.get("body")]


def arxiv(keywords: list[str], n: int = 3) -> list[Doc]:
    words = [w for w in keywords if w.isascii()][:4]
    if not words:
        return []
    docs: list[Doc] = []
    with httpx.Client(headers=HEADERS, timeout=TIMEOUT) as c:
        for joiner in (" AND ", " OR "):  # strict first, then lenient
            q = joiner.join(f"all:{w}" for w in words)
            r = c.get("https://export.arxiv.org/api/query", params={"search_query": q, "max_results": n})
            r.raise_for_status()
            for e in ET.fromstring(r.text).findall(f"{ATOM}entry"):
                docs.append(Doc("arXiv", _clean(e.findtext(f"{ATOM}title")),
                                _clean(e.findtext(f"{ATOM}id")), _clean(e.findtext(f"{ATOM}summary"))))
            if docs:
                break
    return docs


def _abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the sentence."""
    if not inverted:
        return ""
    pos = {i: w for w, idxs in inverted.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def openalex(query: str, n: int = 3) -> list[Doc]:
    with httpx.Client(headers=HEADERS, timeout=TIMEOUT) as c:
        r = c.get("https://api.openalex.org/works",
                  params={"search": query[:300], "per-page": n,
                          "select": "id,title,abstract_inverted_index,doi,publication_year",
                          **({"mailto": CONTACT} if "@" in CONTACT else {})})
        r.raise_for_status()
        results = r.json().get("results", [])
    docs = []
    for w in results:
        text = _abstract(w.get("abstract_inverted_index"))
        if text:
            title = f"{w.get('title') or 'Untitled'} ({w.get('publication_year') or 'n.d.'})"
            docs.append(Doc("OpenAlex", _clean(title), w.get("doi") or w.get("id") or "", _clean(text)))
    return docs


def search_all(query: str, keywords: list[str], sources: list[str], wiki_lang: str = "en", deadline: float = 10.0):
    """Query all chosen sources in parallel and stop waiting after `deadline` seconds.
    A slow source is skipped instead of holding everything up. Returns (documents, per-source report)."""
    jobs = {}
    if "Wikipedia" in sources:
        jobs["Wikipedia"] = lambda: wikipedia(query, wiki_lang)
    if "Web" in sources:
        jobs["Web"] = lambda: web(query)
    if "arXiv" in sources and wiki_lang == "en":
        jobs["arXiv"] = lambda: arxiv(keywords)
    if "OpenAlex" in sources and wiki_lang == "en":
        jobs["OpenAlex"] = lambda: openalex(query)

    def timed(name, fn):
        t0 = time.time()
        try:
            out, err = fn(), ""
        except Exception as e:  # one failing source must never break the others
            out, err = [], f"{type(e).__name__}: {str(e)[:120]}"
        return name, out, err, round(time.time() - t0, 2)

    docs: list[Doc] = []
    report = []
    ex = ThreadPoolExecutor(max_workers=max(1, len(jobs)))
    futures = {ex.submit(timed, n, f): n for n, f in jobs.items()}
    done, pending = wait(futures, timeout=deadline)
    for fut in done:
        name, out, err, secs = fut.result()
        docs += out
        report.append({"source": name, "found": len(out), "seconds": secs, "error": err})
    for fut in pending:
        report.append({"source": futures[fut], "found": 0, "seconds": deadline, "error": f"skipped: no answer within {deadline:.0f}s"})
    ex.shutdown(wait=False, cancel_futures=True)
    return docs, report
