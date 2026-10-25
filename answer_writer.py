"""Answer synthesis: free extractive summary, or a local Ollama model if available."""
from __future__ import annotations

import httpx

import ranker as rank


def extractive(query: str, docs: list, n: int = 5) -> str:
    picks = rank.top_sentences(query, docs, n)
    if not picks:
        return f"{docs[0].text[:400]}… [1]"
    return "\n\n".join(f"- {s} [{i + 1}]" for i, s in picks)


def ollama_models(host: str = "http://localhost:11434") -> list[str]:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=1.5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_answer(text: str, docs: list, model: str, host: str = "http://localhost:11434", k: int = 4) -> str:
    context = "\n\n".join(f"[{i + 1}] {d.title} ({d.source})\n{d.text[:1200]}" for i, d in enumerate(docs[:k]))
    prompt = (
        "A user scanned the text below with OCR. Using ONLY the numbered sources, explain what it is "
        "about, or answer it if it is a question. Be concise and accurate, cite sources like [1], and "
        "say so if the sources are not enough.\n\n"
        f"SCANNED TEXT:\n{text[:1500]}\n\nSOURCES:\n{context}\n\nANSWER:"
    )
    r = httpx.post(f"{host}/api/generate", timeout=180,
                   json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}})
    r.raise_for_status()
    return r.json().get("response", "").strip()
