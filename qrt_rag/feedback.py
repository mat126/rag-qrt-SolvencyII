"""Scarica il feedback raccolto su Supabase dalla versione web.

Usa la chiave SEGRETA di Supabase (service_role o secret), che puo' leggere la
tabella: tienila in un secret (es. Kaggle Secrets) e non metterla mai nel browser
ne' nel repository. Il file prodotto ha lo stesso formato del feedback Gradio, quindi
qrt_rag.evaluation.load_feedback_cases() lo legge senza modifiche.
"""
import json
import urllib.request
from pathlib import Path


def _headers(key):
    h = {"apikey": key, "Accept": "application/json"}
    if key.startswith("eyJ"):            # chiavi "legacy" in formato JWT
        h["Authorization"] = f"Bearer {key}"
    return h


def fetch_supabase_feedback(url, key, out_path, table="feedback", page_size=1000):
    """Scarica tutte le righe (in ordine di arrivo) e le salva come JSONL."""
    rows, start = [], 0
    while True:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/rest/v1/{table}?select=*&order=created_at.asc",
            headers={**_headers(key), "Range-Unit": "items",
                     "Range": f"{start}-{start + page_size - 1}"},
        )
        with urllib.request.urlopen(req) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        rows += batch
        if len(batch) < page_size:
            break
        start += page_size

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "question": r["question"], "annex": r.get("annex"), "k": r.get("k"),
                "mode": r.get("mode"), "answer": None, "lang": r.get("lang"),
                "hits": r.get("hits") or [], "rating": r["rating"],
                "expected": r.get("expected"), "note": r.get("note"),
                "time": r.get("created_at"), "index_version": r.get("index_version"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Feedback scaricato: {len(rows)} righe -> {out_path}")
    return rows
