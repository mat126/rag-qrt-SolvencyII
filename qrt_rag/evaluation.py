"""Valutazione del recupero e delle risposte, e diagnostica per stadio.

Ogni caso del set dice DOVE sta la risposta giusta (allegato, template, celle
oppure articolo), non il testo della risposta. Richiede gli extra [eval].
"""
import glob
import json
import re
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .codes import RE_CELL, RE_TEMPLATE, norm_tmpl, template_base

DEFAULT_FEEDBACK_GLOBS = ("/kaggle/working/feedback*.jsonl",
                          "/kaggle/input/**/feedback*.jsonl")

try:                                   # tabelle formattate nei notebook
    from IPython.display import display as _display
except ImportError:                    # fuori dai notebook: stampa semplice
    _display = print


def load_eval_set(path=None):
    """Set di sviluppo (31 casi + 2 negativi). Di default quello incluso nel pacchetto."""
    if path is None:
        text = resources.files("qrt_rag").joinpath("eval_set.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def load_feedback_cases(patterns=DEFAULT_FEEDBACK_GLOBS):
    """Casi dal feedback Gradio: ogni riga con "fonte attesa" diventa un caso.
    Formati accettati: "S.23.01 R0540" (template e/o celle) oppure "Articolo 7"."""
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    cases = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                r = json.loads(line)
                exp = (r.get("expected") or "").upper()
                if not exp:
                    continue
                case = {"id": f"fb-{Path(fp).stem}-{n:03d}", "tipo": "feedback", "q": r["question"]}
                art = re.search(r"ART(?:ICOLO)?\.?\s*(\d+)", exp)
                m = RE_TEMPLATE.search(exp)
                if art:
                    case["article"] = art.group(1)
                elif m or RE_CELL.search(exp):
                    if m:
                        case["template"] = template_base(m)
                    case["cells"] = RE_CELL.findall(exp)
                    if r.get("annex"):
                        case["annex"] = r["annex"]
                else:
                    continue   # formato non riconosciuto
                cases.append(case)
    print(f"Casi dal feedback: {len(cases)} (da {len(files)} file)")
    return cases


def is_match(h, case):
    """Un chunk (o un risultato di ricerca) soddisfa il caso?"""
    if case.get("article"):
        return h["article"] == case["article"]
    if case.get("annex") and h["annex"] != case["annex"]:
        return False
    if case.get("template") and norm_tmpl(h["template"]) != norm_tmpl(case["template"]):
        return False
    return set(case.get("cells", [])) <= set(h["cells"])


def eval_retrieval(retriever, cases, depth=20, **search_kwargs):
    """Posizione della prima fonte corretta per ogni caso (None = oltre `depth`)."""
    rows = []
    for c in cases:
        if c.get("negative"):
            continue
        hits = retriever.search(c["q"], k=depth, **search_kwargs)
        rank = next((h["rank"] for h in hits if is_match(h, c)), None)
        rows.append({"id": c["id"], "tipo": c["tipo"], "rank": rank, "domanda": c["q"][:80]})
    return pd.DataFrame(rows)


def summarize(df, label):
    r = df["rank"].astype(float)
    return {"configurazione": label, "n": len(df),
            "hit@1": (r <= 1).mean(), "hit@5": (r <= 5).mean(),
            "hit@10": (r <= 10).mean(), "MRR": (1 / r).fillna(0).mean()}


def report_retrieval(retriever, cases, out_dir=None):
    """Confronto con e senza reranker, hit@5 per tipo, casi falliti; salva un CSV."""
    t0 = time.time()
    df_rr = eval_retrieval(retriever, cases)                 # pipeline attuale
    df_no = eval_retrieval(retriever, cases, rerank=False)   # senza reranker
    print(f"Valutazione in {time.time() - t0:.0f}s\n")

    _display(pd.DataFrame([summarize(df_rr, "fusione + reranker"),
                           summarize(df_no, "solo fusione")]).round(3))
    print("\nhit@5 per tipo di domanda (con reranker):")
    _display(df_rr.assign(hit5=df_rr["rank"].astype(float) <= 5)
                  .groupby("tipo")["hit5"].agg(["mean", "count"]).round(2))
    print("\nDomande in cui la fonte giusta NON e' nei primi 5:")
    misses = df_rr[df_rr["rank"].isna() | (df_rr["rank"].astype(float) > 5)]
    _display(misses if len(misses) else "nessuna")

    if out_dir is not None:
        stamp = time.strftime("%Y%m%d_%H%M")
        df_rr.to_csv(Path(out_dir) / f"eval_recupero_{stamp}.csv", index=False)
    return df_rr, df_no


def eval_llm(cases, answer_fn, cache=None, out_dir=None):
    """Fa rispondere il modello a ogni caso. answer_fn(domanda) -> testo.

    Le risposte restano in `cache` (un dict): rilanciando, si riparte dalla
    prima domanda non ancora fatta. Il controllo automatico verifica solo che la
    risposta citi le celle attese o "Articolo N": il contenuto va giudicato a mano."""
    cache = {} if cache is None else cache
    todo = [c for c in cases if c["id"] not in cache]
    print(f"Domande da fare: {len(todo)} (gia' fatte: {len(cases) - len(todo)})")
    for c in todo:
        print("=" * 80, "\n", c["id"], c["q"])
        try:
            cache[c["id"]] = answer_fn(c["q"])
        except RuntimeError as ex:          # torch.cuda.OutOfMemoryError e' un RuntimeError
            if "out of memory" not in str(ex).lower():
                raise
            print("  saltata: memoria GPU esaurita anche con meno estratti")

    rows = []
    for c in cases:
        ans = cache.get(c["id"])
        if ans is None or c.get("negative"):
            ok = None   # negative: da controllare a mano, deve dire che non c'e' risposta
        elif c.get("article"):
            ok = f"ARTICOLO {c['article']}" in ans.upper()
        else:
            ok = all(cell in ans.upper() for cell in c["cells"]) if c.get("cells") else None
        rows.append({"id": c["id"], "tipo": c["tipo"], "cita_fonte_attesa": ok, "risposta": ans})
    df = pd.DataFrame(rows)
    _display(df[["id", "tipo", "cita_fonte_attesa"]])
    if out_dir is not None:
        stamp = time.strftime("%Y%m%d_%H%M")
        df.to_csv(Path(out_dir) / f"eval_llm_{stamp}.csv", index=False)
    return df


def diagnose(retriever, cases, case_id, show=10):
    """In quale stadio la fonte giusta perde posizioni? BM25, embeddings,
    fusione senza reranker, pipeline completa; poi i primi risultati."""
    from .retrieval import query_scope, tokenize_query

    c = next(x for x in cases if x["id"] == case_id)
    chunks = retriever.chunks
    target = {i for i, ch in enumerate(chunks) if is_match(ch, c)}
    print(c["id"], "-", c["q"], f"| chunk corretti nell'indice: {len(target)}")
    if not target:
        print("Nessun chunk corrisponde alla fonte attesa: controlla il caso.")
        return
    bm, dense = retriever.first_stage(c["q"])
    print("Parole usate da BM25:", tokenize_query(c["q"]),
          "| allegato implicito:", query_scope(c["q"]))

    def best(scores):
        pos = np.empty(len(scores), dtype=int)
        pos[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
        return int(min(pos[i] for i in target))

    def pos_in(hits):
        return next((h["rank"] for h in hits if h["id"] in target), None)

    print(f"Solo BM25: {best(bm)} | Solo embeddings: {best(dense)} | "
          f"Fusione: {pos_in(retriever.search(c['q'], k=40, rerank=False))} | "
          f"Finale: {pos_in(retriever.search(c['q'], k=40))}")
    print("\nPrimi risultati finali (<< = fonte attesa):")
    for h in retriever.search(c["q"], k=show):
        mark = "  <<" if h["id"] in target else ""
        body = h["text"].split("\n", 1)[-1][:90].replace("\n", " ")
        print(f"{h['rank']:>2} {h['annex'] or '-':<4} {h['template_full'] or '-':<11} {body}{mark}")
