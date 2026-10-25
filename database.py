"""SQLite storage for scan history and analytics (no login: history is keyed by a session id)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parent / "history.db"


def _conn():
    return sqlite3.connect(DB)


def init() -> None:
    with closing(_conn()) as c, c:
        c.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT, created TEXT, lang TEXT,
            ocr_conf REAL, query TEXT, ocr_text TEXT, answer TEXT, confidence REAL,
            elapsed REAL, sources TEXT, sources_used TEXT)""")


def add(sid: str, rec: dict) -> None:
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO scans (sid, created, lang, ocr_conf, query, ocr_text, answer, confidence, elapsed, sources, sources_used)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, datetime.now().strftime("%Y-%m-%d %H:%M"), rec["lang"], rec["ocr_conf"], rec["query"],
             rec["text"], rec["answer"], rec["confidence"], rec["elapsed"],
             json.dumps([{"title": d["title"], "url": d["url"], "source": d["source"]} for d in rec["docs"]]),
             ",".join(sorted({d["source"] for d in rec["docs"]}))))


def history(sid: str, limit: int = 50) -> list[dict]:
    with closing(_conn()) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM scans WHERE sid=? ORDER BY id DESC LIMIT ?", (sid, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"] or "[]")
        out.append(d)
    return out


def clear(sid: str) -> None:
    with closing(_conn()) as c, c:
        c.execute("DELETE FROM scans WHERE sid=?", (sid,))


def stats() -> pd.DataFrame:
    """Anonymous aggregate data only (no scanned text)."""
    with closing(_conn()) as c:
        return pd.read_sql_query(
            "SELECT created, lang, ocr_conf, confidence, elapsed, sources_used FROM scans ORDER BY id", c)
