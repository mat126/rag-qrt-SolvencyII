"""Abbinamento delle pagine tra due versioni linguistiche del regolamento.

Le due versioni (es. inglese e italiana) non hanno le stesse pagine: le lingue
occupano spazi diversi e lo scarto cresce lungo il documento. Le istruzioni LOG
pero' hanno una chiave indipendente dalla lingua (allegato + template + codice
della cella), quindi la pagina corrispondente si ricava cosi':

  exact     istruzione con la stessa chiave nell'altra versione
  article   stesso articolo: prima pagina dell'articolo + scarto interno
  template  stesso template completo (es. tabelle dell'Allegato I): stesso metodo
  offset    scarto dell'istruzione abbinata piu' vicina (stima)
"""
import bisect
import re
from collections import Counter


def code_field(chunk):
    """Prima colonna di una riga LOG, normalizzata: "C0010-C0020/ R0110" -> "C0010-C0020/R0110"."""
    body = chunk["text"].split("\n", 1)[-1]
    return re.sub(r"\s+", "", body.split(" | ", 1)[0]).upper()


def _first_pages(chunks, key_fn):
    first = {}
    for c in chunks:
        k = key_fn(c)
        if k is not None:
            first.setdefault(k, c["page"])
    return first


def align_pages(src_chunks, ref_chunks):
    """Per ogni chunk di src_chunks, la pagina stimata nella versione di ref_chunks.
    Restituisce una lista di dict {"page", "method", "ref_id"} nello stesso ordine:
    ref_id e' l'id del chunk corrispondente quando l'abbinamento e' esatto, altrimenti None."""
    def instr_key(c):
        return ((c["annex"], c["template"], code_field(c))
                if c["kind"] == "istruzione" and c["template"] else None)

    def article_key(c):
        return c["article"]

    def tmpl_key(c):
        return (c["annex"], c["template_full"]) if c["template_full"] else None

    # La stessa chiave puo' comparire piu' volte (es. stessa cella in sezioni diverse
    # dello stesso template): la k-esima occorrenza si abbina alla k-esima dell'altra versione
    ref_instr = {}
    for j, c in enumerate(ref_chunks):
        k = instr_key(c)
        if k is not None:
            ref_instr.setdefault(k, []).append((c["page"], j))
    ref_art, src_art = _first_pages(ref_chunks, article_key), _first_pages(src_chunks, article_key)
    ref_tmpl, src_tmpl = _first_pages(ref_chunks, tmpl_key), _first_pages(src_chunks, tmpl_key)
    max_ref = max(c["page"] for c in ref_chunks)

    out = [None] * len(src_chunks)
    anchors = []                                   # (pagina src, pagina ref) delle istruzioni abbinate
    seen = Counter()
    for i, c in enumerate(src_chunks):
        k = instr_key(c)
        if k is not None and k in ref_instr:
            matches = ref_instr[k]
            ref_page, ref_idx = matches[min(seen[k], len(matches) - 1)]
            seen[k] += 1
            out[i] = {"page": ref_page, "method": "exact", "ref_id": ref_chunks[ref_idx]["id"]}
            anchors.append((c["page"], ref_page))
    anchors.sort()
    anchor_pages = [a[0] for a in anchors]

    for i, c in enumerate(src_chunks):
        if out[i] is not None:
            continue
        a, t = article_key(c), tmpl_key(c)
        if a is not None and a in ref_art and a in src_art:
            page, method = ref_art[a] + (c["page"] - src_art[a]), "article"
        elif t is not None and t in ref_tmpl and t in src_tmpl:
            page, method = ref_tmpl[t] + (c["page"] - src_tmpl[t]), "template"
        elif anchors:
            j = bisect.bisect_right(anchor_pages, c["page"]) - 1   # ancora precedente...
            sp, rp = anchors[max(j, 0)]                             # ...o la prima se non c'e'
            page, method = c["page"] + (rp - sp), "offset"
        else:
            page, method = c["page"], "offset"
        out[i] = {"page": int(min(max(page, 1), max_ref)), "method": method, "ref_id": None}
    return out


def alignment_report(src_chunks, aligned):
    """Riepilogo: quanti chunk per metodo e com'e' distribuito lo scarto di pagina."""
    methods = Counter(a["method"] for a in aligned)
    shifts = Counter(a["page"] - c["page"] for c, a in zip(src_chunks, aligned) if a["method"] == "exact")
    print("Pagine abbinate per metodo:", dict(methods))
    print("Scarto di pagina nelle istruzioni abbinate (ref - src: numero di chunk):",
          dict(sorted(shifts.items())))
    return methods
