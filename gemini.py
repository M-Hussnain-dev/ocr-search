"""Google Gemini helper, tuned for speed.
- thinking is switched off (the biggest cause of slow answers),
- one shared connection is reused,
- the model that worked last time is tried first,
- model names are discovered from your key, with automatic fallback when one is retired or rate-limited."""
from __future__ import annotations

import json
import re
import time
from typing import Iterator

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta"
FALLBACK = {"Fast": ("gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.6-flash", "gemini-2.5-flash"),
            "Accurate": ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash")}
SPEEDS = ["Fast", "Accurate"]
SKIP = ("tts", "image", "live", "audio", "embedding", "robotics", "computer", "native", "learnlm", "gemma")
RETRY_CODES = (404, 429, 500, 503)
_client = httpx.Client(timeout=httpx.Timeout(60, connect=10), limits=httpx.Limits(max_keepalive_connections=8))
_models: dict[str, tuple[float, list]] = {}
_good: dict[tuple, str] = {}
_no_think: set[str] = set()   # models that rejected the thinking setting: do not ask again


class GeminiError(RuntimeError):
    def __init__(self, msg: str, code: int = 0):
        super().__init__(msg)
        self.code = code


def _message(code: int, body: str) -> str:
    try:
        msg = json.loads(body)["error"]["message"]
    except Exception:
        msg = body[:200]
    hint = {400: "Check the API key.", 401: "Check the API key.", 403: "The key may not have access to this model.",
            404: "Model not found.", 429: "Rate limit reached: wait a moment."}.get(code, "")
    return f"Gemini error {code}: {msg[:220]} {hint}".strip()


def thinking_config(name: str) -> dict | None:
    """Ask the model not to 'think' first: much faster, and plenty for OCR and grounded answers."""
    if "2.5" in name:
        return {"thinkingBudget": 0}
    if name.startswith("gemini-3"):
        return {"thinkingLevel": "low" if "pro" in name else "minimal"}
    return None


def _list_models(key: str) -> list[str]:
    hit = _models.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    try:
        r = _client.get(f"{BASE}/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=15)
    except httpx.HTTPError:
        return []
    if r.status_code in (400, 401, 403):
        raise GeminiError(_message(r.status_code, r.text), r.status_code)
    if r.status_code != 200:
        return []
    names = []
    for m in r.json().get("models", []):
        n = m.get("name", "").removeprefix("models/")
        if "generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini") and "flash" in n \
                and not any(s in n for s in SKIP):
            names.append(n)
    _models[key] = (time.time(), names)
    return names


def candidates(key: str, speed: str = "Fast") -> tuple[str, ...]:
    """Models to try, best first. Fast prefers Flash-Lite; Accurate prefers the full Flash. Stable before preview."""
    def rank(n: str):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (bool(re.search(r"preview|exp", n)), ("lite" in n) == (speed == "Accurate"), -(float(v.group(1)) if v else 0.0), n)

    names = sorted(set(_list_models(key)), key=rank)[:5] or list(FALLBACK.get(speed, FALLBACK["Fast"]))
    good = _good.get((key, speed))
    if good in names:  # remember what worked last time: saves the failed attempts
        names.remove(good)
        names.insert(0, good)
    return tuple(names)


def _payload(parts: list, system: str, name: str, thinking: bool) -> dict:
    cfg: dict = {"temperature": 0}
    tc = thinking_config(name) if thinking else None
    if tc:
        cfg["thinkingConfig"] = tc
    p = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
    if system:
        p["systemInstruction"] = {"parts": [{"text": system}]}
    return p


def _order(key: str, model: str, speed: str) -> list[str]:
    if not key:
        raise GeminiError("Add your Gemini API key.")
    return [model] if model and model != "auto" else list(candidates(key, speed))


def generate(key: str, parts: list, model: str = "auto", system: str = "", speed: str = "Fast",
             timeout: float = 60) -> tuple[str, str]:
    """Returns (text, model_used)."""
    last: Exception = GeminiError("No Gemini model could answer.")
    for name in _order(key, model, speed):
        for thinking in ((True, False) if thinking_config(name) and name not in _no_think else (False,)):
            try:
                r = _client.post(f"{BASE}/models/{name}:generateContent", headers={"x-goog-api-key": key},
                                 json=_payload(parts, system, name, thinking), timeout=timeout)
            except httpx.HTTPError as e:
                last = GeminiError(f"Could not reach Gemini ({type(e).__name__}).")
                break
            if r.status_code == 400 and thinking and "thinking" in r.text.lower():
                _no_think.add(name)
                continue  # this model rejects the setting: retry without it
            if r.status_code == 200:
                cands = r.json().get("candidates") or []
                text = "".join(p.get("text", "") for p in ((cands[0].get("content") or {}).get("parts", []) if cands else []))
                if text.strip():
                    _good[(key, speed)] = name
                    return text, name
                last = GeminiError("Gemini returned no text (the content may have been blocked).")
            else:
                last = GeminiError(_message(r.status_code, r.text), r.status_code)
                if r.status_code not in RETRY_CODES:
                    raise last
            break
    raise last


def stream(key: str, parts: list, model: str = "auto", system: str = "", speed: str = "Fast",
           info: dict | None = None) -> Iterator[str]:
    """Yields text pieces as they arrive. info['model'] is filled with the model that answered."""
    last: Exception = GeminiError("No Gemini model could answer.")
    for name in _order(key, model, speed):
        for thinking in ((True, False) if thinking_config(name) and name not in _no_think else (False,)):
            started = False
            try:
                with _client.stream("POST", f"{BASE}/models/{name}:streamGenerateContent?alt=sse",
                                    headers={"x-goog-api-key": key},
                                    json=_payload(parts, system, name, thinking)) as r:
                    if r.status_code != 200:
                        body = r.read().decode("utf-8", "replace")
                        if r.status_code == 400 and thinking and "thinking" in body.lower():
                            _no_think.add(name)
                            continue
                        last = GeminiError(_message(r.status_code, body), r.status_code)
                        if r.status_code not in RETRY_CODES:
                            raise last
                        break
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            j = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        c = (j.get("candidates") or [{}])[0]
                        piece = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []))
                        if piece:
                            if not started and info is not None:
                                info["model"] = name
                            started = True
                            _good[(key, speed)] = name
                            yield piece
                if started:
                    return
                last = GeminiError("Gemini returned no text (the content may have been blocked).")
                break
            except httpx.HTTPError as e:
                if started:
                    raise GeminiError(f"The connection to Gemini was interrupted ({type(e).__name__}).")
                last = GeminiError(f"Could not reach Gemini ({type(e).__name__}).")
                break
    raise last
