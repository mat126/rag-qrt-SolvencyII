"""Ricerca ibrida: BM25 + embeddings, fusione RRF, boost sui codici,
reranker come voto aggiuntivo e allegato implicito nella domanda."""
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from .codes import norm_tmpl, query_codes
from .lang import QUERY_META, SCOPE_RE, STOPWORDS, check_lang

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

TOKEN_RE = re.compile(r"(?:sr|se|s|e)\.\d{2}\.\d{2}(?:\.\d{2})?|[rcz]\d{4}|\w+")

def tokenize(text, lang="it"):
    """Token per BM25: tiene interi i codici (s.02.01, r0010) e toglie le stopword."""
    stop = STOPWORDS[lang]
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in stop]


def tokenize_query(q, lang="it"):
    meta = QUERY_META[lang]
    return [t for t in tokenize(q, lang) if t not in meta]


def query_scope(q, lang="it"):
    """Allegato implicito nella domanda: III se parla di gruppi, II se di
    singole imprese, None se nessuno dei due o entrambi."""
    re_group, re_solo = SCOPE_RE[lang]
    g, s = bool(re_group.search(q)), bool(re_solo.search(q))
    if g and not s:
        return "III"
    if s and not g:
        return "II"
    return None


def load_reranker(model_id=RERANKER_MODEL, device=None, max_length=1024):
    from sentence_transformers import CrossEncoder
    from .index import default_device
    return CrossEncoder(model_id, device=device or default_device(), max_length=max_length)


class Retriever:
    """Indice di ricerca su chunk ed embeddings gia' calcolati.

    emb_model serve a codificare le domande (stesso modello degli embeddings);
    con embeddings=None ed emb_model=None la ricerca usa solo BM25.
    reranker e' facoltativo: senza, la ricerca usa solo la fusione.
    """

    def __init__(self, chunks, embeddings=None, emb_model=None, reranker=None, lang="it"):
        self.lang = check_lang(lang)
        self.chunks = chunks
        self.E = embeddings
        self.emb_model = emb_model
        self.reranker = reranker
        self.bm25 = BM25Okapi([tokenize(c["text"], lang) for c in chunks])
        self.chunk_tmpl = [norm_tmpl(c["template"]) for c in chunks]
        self.chunk_cells = [set(c["cells"]) for c in chunks]
        self.annexes = list(dict.fromkeys(c["annex"] for c in chunks if c["annex"]))

    def first_stage(self, query):
        """Punteggi BM25 ed embeddings su tutti i chunk. Per gli embeddings usa sia la
        domanda originale sia la versione "pulita" (solo parole di contenuto) e tiene
        il massimo: le formule di cortesia della domanda non devono dominare.
        Senza modello di embedding restituisce dense=None (solo BM25)."""
        q_tokens = tokenize_query(query, self.lang)
        bm = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.chunks))
        if self.emb_model is None or self.E is None:
            return bm, None
        clean_q = " ".join(q_tokens)
        texts = [query] + ([clean_q] if clean_q and clean_q != query.lower() else [])
        qvs = self.emb_model.encode(texts, normalize_embeddings=True,
                                    convert_to_numpy=True).astype("float32")
        dense = (self.E @ qvs.T).max(axis=1)
        return bm, dense

    def search(self, query, k=8, n_cand=40, rerank=True, annex=None):
        """Primi k chunk per la domanda, ciascuno con il campo "rank".
        annex: es. "II" per cercare solo in quell'allegato (None = tutto)."""
        chunks = self.chunks
        tmpls, cells = query_codes(query)
        bm, dense = self.first_stage(query)
        keep = np.array([annex is None or c["annex"] == annex for c in chunks])
        bm = np.where(keep, bm, -np.inf)

        # Reciprocal Rank Fusion delle classifiche disponibili
        rrf = defaultdict(float)
        for r, i in enumerate(np.argsort(-bm)[:n_cand]):
            rrf[int(i)] += 1 / (60 + r)
        if dense is not None:
            dense = np.where(keep, dense, -np.inf)
            for r, i in enumerate(np.argsort(-dense)[:n_cand]):
                rrf[int(i)] += 1 / (60 + r)

        # Se la domanda contiene template o codici cella, promuove i chunk esatti
        if tmpls or cells:
            for i in range(len(chunks)):
                if not keep[i]:
                    continue
                t_ok = self.chunk_tmpl[i] in tmpls if tmpls else True
                c_ok = bool(self.chunk_cells[i] & cells) if cells else True
                if t_ok and c_ok:
                    rrf[i] += 0.05

        cand = sorted(rrf, key=rrf.get, reverse=True)[:n_cand]
        if not cand:
            return []
        if rerank and self.reranker is not None:
            rr = np.asarray(self.reranker.predict(
                [(query, chunks[i]["text"][:4000]) for i in cand],
                batch_size=16, show_progress_bar=False))
            rr_rank = np.empty(len(cand), dtype=int)
            rr_rank[np.argsort(-rr)] = np.arange(1, len(cand) + 1)
            # Il reranker e' un voto in piu', non il giudice unico: la sua classifica
            # si combina (RRF) con quella della fusione BM25 + embeddings.
            # Motivo: non distingue termini come "disponibili" vs "ammissibili".
            fusion_rank = np.arange(1, len(cand) + 1)   # cand e' gia' ordinato per rrf
            scores = 1 / (10 + fusion_rank) + 1 / (10 + rr_rank)
        else:
            scores = np.array([rrf[i] for i in cand])

        # Il reranker non "capisce" i codici: chi corrisponde a template e/o cella
        # citati nella domanda resta in testa, a parita' decide il punteggio
        def match_level(i):
            return int(bool(tmpls) and self.chunk_tmpl[i] in tmpls) + \
                   int(bool(cells) and bool(self.chunk_cells[i] & cells))

        # Se la domanda parla di gruppi (o di singole imprese) e non c'e' un filtro
        # esplicito, l'allegato "opposto" scende in classifica (articoli e
        # Allegato I non vengono toccati)
        scope = query_scope(query, self.lang) if annex is None else None
        other = {"II": "III", "III": "II"}.get(scope)

        def scope_ok(i):
            return int(chunks[i]["annex"] != other) if other else 1

        order = sorted(range(len(cand)),
                       key=lambda j: (match_level(cand[j]), scope_ok(cand[j]), scores[j]),
                       reverse=True)
        cand = [cand[j] for j in order]
        return [{**chunks[i], "rank": r + 1} for r, i in enumerate(cand[:k])]

    def rank_of(self, query, cell, template=None, annex=None, depth=50):
        """Posizione del primo chunk che contiene la cella attesa (None = non trovato)."""
        for h in self.search(query, k=depth, annex=annex):
            if cell in h["cells"] and (template is None or h["template"] == template):
                return h["rank"]
        return None


def show(hits, n_chars=300):
    """Stampa compatta dei risultati di Retriever.search()."""
    for h in hits:
        tm = h["template_full"] or "-"
        print(f"#{h['rank']}  pag.{h['page']}  {h['kind']}  {tm}  celle={h['cells'][:6]}")
        print("   ", h["text"][:n_chars].replace("\n", " "), "\n")
