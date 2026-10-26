"""Google Gemini helper. Model names change often, so by default it asks the API which Flash models your key can use
and falls back to the next one if a model is retired or its quota is used up."""
from __future__ import annotations

import re
import time

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta"
FALLBACK = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash")
SKIP = ("tts", "image", "live", "audio", "embedding", "robotics", "computer", "native", "learnlm", "gemma")
RETRY_CODES = (404, 429, 500, 503)
_cache: dict[str, tuple[float, tuple]] = {}


class GeminiError(RuntimeError):
    def __init__(self, msg: str, code: int = 0):
        super().__init__(msg)
        self.code = code


def _message(r: httpx.Response) -> str:
    try:
        msg = r.json()["error"]["message"]
    except Exception:
        msg = r.text[:200]
    hint = {400: "Check the API key.", 401: "Check the API key.", 403: "The key may not have access to this model.",
            404: "Model not found.", 429: "Rate limit reached."}.get(r.status_code, "")
    return f"Gemini error {r.status_code}: {msg[:220]} {hint}".strip()


def candidates(key: str) -> tuple[str, ...]:
    """Flash models available to this key: stable before preview, full before lite, newest first."""
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    try:
        r = httpx.get(f"{BASE}/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=15)
    except httpx.HTTPError:
        return FALLBACK
    if r.status_code in (400, 401, 403):
        raise GeminiError(_message(r), r.status_code)
    if r.status_code != 200:
        return FALLBACK
    names = []
    for m in r.json().get("models", []):
        n = m.get("name", "").removeprefix("models/")
        if "generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini") \
                and "flash" in n and not any(s in n for s in SKIP):
            names.append(n)

    def rank(n: str):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (bool(re.search(r"preview|exp", n)), "lite" in n, -(float(v.group(1)) if v else 0.0), n)

    out = tuple(sorted(set(names), key=rank)[:5]) or FALLBACK
    _cache[key] = (time.time(), out)
    return out


def generate(key: str, parts: list, model: str = "auto", system: str = "", timeout: float = 90) -> tuple[str, str]:
    """Returns (text, model_used). parts: [{'text': ...}, {'inline_data': {...}}]."""
    if not key:
        raise GeminiError("Add your Gemini API key.")
    names = [model] if model and model != "auto" else list(candidates(key))
    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    last: Exception = GeminiError("No Gemini model could answer.")
    for name in names:
        try:
            r = httpx.post(f"{BASE}/models/{name}:generateContent", headers={"x-goog-api-key": key}, json=payload, timeout=timeout)
        except httpx.HTTPError as e:
            last = GeminiError(f"Could not reach Gemini ({type(e).__name__}).")
            continue
        if r.status_code == 200:
            cands = r.json().get("candidates") or []
            text = "".join(p.get("text", "") for p in ((cands[0].get("content") or {}).get("parts", []) if cands else []))
            if text.strip():
                return text, name
            last = GeminiError("Gemini returned no text (the content may have been blocked).")
            continue
        last = GeminiError(_message(r), r.status_code)
        if r.status_code not in RETRY_CODES:
            raise last
    raise last
