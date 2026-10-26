"""Query building, ranking, sentence selection and confidence score."""
from __future__ import annotations

import html
import re
from collections import Counter

from rank_bm25 import BM25Okapi

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this
that these those it its i you he she we they them his her their our your my me not no do does did have has had
will would can could should may might about into over than then so such also there here which who whom what
when where why how""".split())

TOKEN = re.compile(r"\w+", re.UNICODE)
SENT = re.compile(r"(?<=[.!?۔؟])\s+")
CREDIBILITY = {"arXiv": 1.0, "OpenAlex": 0.95, "Wikipedia": 0.9, "Web": 0.75}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOP]


def is_urdu(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    arabic = sum("\u0600" <= c <= "\u06ff" for c in letters)
    return bool(letters) and arabic / len(letters) > 0.4


def keywords(text: str, k: int = 8) -> list[str]:
    toks = tokenize(text)
    freq = Counter(toks)
    first: dict[str, int] = {}
    for i, t in enumerate(toks):
        first.setdefault(t, i)
    top = sorted(freq, key=lambda t: (-freq[t], first[t]))[:k]
    return sorted(top, key=lambda t: first[t])


def build_query(text: str, max_words: int = 14) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat.split()) <= max_words:
        return flat
    return " ".join(keywords(flat, 8))


def rank_documents(query: str, docs: list) -> list:
    seen, unique = set(), []
    for d in docs:  # drop duplicate URLs
        if d.url and d.url in seen:
            continue
        seen.add(d.url)
        unique.append(d)
    if not unique:
        return []
    q = tokenize(query)
    qset = set(q)
    corpus = [tokenize(f"{d.title} {d.title} {d.text}") for d in unique]
    bm = [0.0] * len(unique)
    if q and any(corpus):
        bm = [max(float(s), 0.0) for s in BM25Okapi([c or ["_"] for c in corpus]).get_scores(q)]
    top = max(bm) or 1.0
    for d, toks, b in zip(unique, corpus, bm):
        d.coverage = round(len(qset & set(toks)) / len(qset), 3) if qset else 0.0
        d.relevance = round(0.6 * b / top + 0.4 * d.coverage, 3)
        d.score = round(0.8 * d.relevance + 0.2 * CREDIBILITY.get(d.source, 0.7), 3)
    return sorted(unique, key=lambda d: d.score, reverse=True)


def top_sentences(query: str, docs: list, n: int = 5, per_doc: int = 2) -> list[tuple[int, str]]:
    """Pick the best-matching, non-repeating sentences. Returns [(doc_index, sentence)]."""
    qset = set(tokenize(query))
    cands = []
    for di, d in enumerate(docs):
        for pos, s in enumerate(x.strip() for x in SENT.split(d.text)):
            toks = set(tokenize(s))
            if len(s) < 30 or not toks or not qset:
                continue
            overlap = len(qset & toks) / len(qset)
            if overlap > 0:
                cands.append((overlap * (0.6 + 0.4 * d.score), di, pos, s, toks))
    cands.sort(key=lambda c: c[0], reverse=True)
    chosen, used = [], Counter()
    for score, di, pos, s, toks in cands:
        if used[di] >= per_doc:
            continue
        if any(len(toks & c[4]) / len(toks | c[4]) > 0.6 for c in chosen):
            continue
        chosen.append((score, di, pos, s, toks))
        used[di] += 1
        if len(chosen) == n:
            break
    chosen.sort(key=lambda c: (c[1], c[2]))  # reading order: best document first
    return [(c[1], c[3]) for c in chosen]


def confidence_parts(docs: list) -> dict:
    """The three ingredients of the confidence score, each between 0 and 1."""
    if not docs:
        return {"coverage": 0.0, "agreement": 0.0, "credibility": 0.0}
    top3 = docs[:3]
    return {
        "coverage": sum(d.coverage for d in top3) / len(top3),
        "agreement": min(len({d.source for d in docs[:5]}) / 2, 1.0),
        "credibility": sum(CREDIBILITY.get(d.source, 0.7) for d in top3) / len(top3),
    }


def confidence(docs: list) -> int:
    """0-100 heuristic: 55% query coverage + 25% agreement between sources + 20% credibility."""
    p = confidence_parts(docs)
    return round(100 * (0.55 * p["coverage"] + 0.25 * p["agreement"] + 0.20 * p["credibility"]))


def analyze_text(text: str) -> dict:
    flat = re.sub(r"\s+", " ", text).strip()
    words = flat.split()
    sentences = [s for s in SENT.split(flat) if s.strip()]
    kind = "Question" if flat.endswith(("?", "؟")) else "Short phrase" if len(words) < 6 else "Passage"
    return {
        "words": len(words), "sentences": len(sentences), "chars": len(flat),
        "language": "Urdu" if is_urdu(flat) else "English", "kind": kind,
        "keywords": keywords(flat, 8), "reading_min": max(1, round(len(words) / 200)),
    }


def highlight(text: str, terms) -> str:
    """HTML-escape text and wrap words that start with a query term in <mark>."""
    out = html.escape(text)
    terms = sorted({t for t in terms if len(t) > 2 and t not in {"amp", "quot"}}, key=len, reverse=True)
    if not terms:
        return out
    pat = re.compile(r"(?<![&\w])((?:" + "|".join(re.escape(t) for t in terms) + r")\w*)", re.I | re.UNICODE)
    return pat.sub(r"<mark>\1</mark>", out)
