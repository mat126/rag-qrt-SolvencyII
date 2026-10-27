"""Esportazione per la versione statica (ricerca nel browser, web/).

Produce in out_dir:
  chunks.json          i chunk con i soli campi usati dal browser
  embeddings.f16.bin   embeddings dei chunk in float16 (little-endian, n x dim)
  config.json          modello, parametri di ricerca e risorse di lingua (stopword,
                       parole di servizio, regex) identiche a quelle di Python
  eval_set.json        il set di valutazione nella lingua scelta
  parity.json          per ogni domanda: token, vettori della domanda e primi 20
                       risultati di Python, per il test di parita' di web/parity.html
"""
import datetime
import json
from pathlib import Path

import numpy as np

from .codes import RE_CELL, RE_TEMPLATE, query_codes
from .evaluation import eval_retrieval, summarize
from .lang import LABELS, QUERY_META, SCOPE_RE, STOPWORDS, SYSTEM_PROMPTS, UI, check_lang
from .retrieval import Retriever, query_scope, tokenize_query

WEB_FIELDS = ("id", "kind", "page", "annex", "article", "template", "template_full",
              "section", "cells", "text")

# Stessa regex di retrieval.TOKEN_RE, scritta per JavaScript: in Python \w (Unicode)
# corrisponde a lettere, cifre e trattino basso, cioe' [\p{L}\p{N}_] con il flag "u".
JS_TOKEN_PATTERN = r"(?:sr|se|s|e)\.\d{2}\.\d{2}(?:\.\d{2})?|[rcz]\d{4}|[\p{L}\p{N}_]+"


class CachedEncoder:
    """Restituisce vettori gia' calcolati: Python e browser usano cosi' esattamente
    gli stessi vettori delle domande nel test di parita'."""

    def __init__(self, cache):
        self.cache = cache

    def encode(self, texts, **kwargs):
        return np.stack([self.cache[t] for t in texts])


def query_texts(query, lang):
    """Testi codificati per una domanda, come in Retriever.first_stage()."""
    clean_q = " ".join(tokenize_query(query, lang))
    return [query] + ([clean_q] if clean_q and clean_q != query.lower() else [])


def export_web(out_dir, chunks, E, encoder, model_info, cases, lang="en",
               n_cand=40, depth=20):
    """encoder: oggetto con encode() per le domande (es. PrefixedEncoder con "query: ").
    model_info: dict con "hf" (nome Hugging Face), "web" (nome per transformers.js),
    "q_prefix", "p_prefix", "pooling" ("mean" per e5)."""
    check_lang(lang)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Chunk
    light = [{k: c[k] for k in WEB_FIELDS} for c in chunks]
    (out / "chunks.json").write_text(json.dumps(light, ensure_ascii=False), encoding="utf-8")

    # 2. Embeddings in float16: il browser lavora con questi, quindi anche i
    #    riferimenti Python qui sotto usano la stessa versione arrotondata
    E16 = E.astype("<f2")
    (out / "embeddings.f16.bin").write_bytes(E16.tobytes())
    E_ref = E16.astype("float32")

    # 3. Vettori delle domande, calcolati una volta sola
    cache = {}
    for c in cases:
        for t in query_texts(c["q"], lang):
            if t not in cache:
                cache[t] = np.asarray(encoder.encode([t], normalize_embeddings=True,
                                                     convert_to_numpy=True)[0], dtype="float32")
    r_hyb = Retriever(chunks, E_ref, CachedEncoder(cache), lang=lang)
    r_bm = Retriever(chunks, lang=lang)

    # 4. Riferimenti per il test di parita'
    parity = []
    for c in cases:
        texts = query_texts(c["q"], lang)
        tmpls, cells = query_codes(c["q"])
        parity.append({
            "id": c["id"], "tipo": c["tipo"], "q": c["q"],
            "expected": {k: c.get(k) for k in ("annex", "template", "cells", "article", "negative")},
            "tokens": tokenize_query(c["q"], lang),
            "scope": query_scope(c["q"], lang),
            "templates": sorted(tmpls), "cells": sorted(cells),
            "query_texts": texts,
            "query_vectors": [[float(v) for v in cache[t]] for t in texts],
            "bm25_top": [h["id"] for h in r_bm.search(c["q"], k=depth, n_cand=n_cand)],
            "hybrid_top": [h["id"] for h in r_hyb.search(c["q"], k=depth, n_cand=n_cand)],
        })
    (out / "parity.json").write_text(json.dumps(parity, ensure_ascii=False), encoding="utf-8")
    (out / "eval_set.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    # 5. Metriche Python con gli embeddings float16 (devono restare quelle della fase 2)
    metrics = {
        "solo BM25": summarize(eval_retrieval(r_bm, cases, depth=depth), "solo BM25"),
        "BM25 + embeddings": summarize(eval_retrieval(r_hyb, cases, depth=depth), "BM25 + embeddings"),
    }

    # 6. Configurazione per il browser
    re_group, re_solo = SCOPE_RE[lang]
    config = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "lang": lang,
        "n_chunks": len(chunks), "dim": int(E.shape[1]), "dtype": "float16",
        "model": model_info,
        "search": {"n_cand": n_cand, "rrf_k": 60, "code_boost": 0.05,
                   "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25}},
        "patterns": {
            "token": {"source": JS_TOKEN_PATTERN, "flags": "gu"},
            "template": {"source": RE_TEMPLATE.pattern, "flags": "g"},
            "cell": {"source": RE_CELL.pattern, "flags": "g"},
            "group": {"source": re_group.pattern, "flags": "i"},
            "solo": {"source": re_solo.pattern, "flags": "i"},
        },
        "stopwords": sorted(STOPWORDS[lang]),
        "query_meta": sorted(QUERY_META[lang]),
        "labels": LABELS[lang], "ui": UI[lang], "system_prompt": SYSTEM_PROMPTS[lang],
        "python_metrics": {k: {m: (round(float(v), 4) if isinstance(v, float) else v)
                               for m, v in row.items()} for k, row in metrics.items()},
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in sorted(out.iterdir())}
    print("File esportati (MB):", sizes)
    print("Metriche Python con embeddings float16:")
    for k, row in metrics.items():
        print(f"  {k:<20} hit@5 {row['hit@5']:.3f}  hit@10 {row['hit@10']:.3f}  MRR {row['MRR']:.3f}")
    return config
