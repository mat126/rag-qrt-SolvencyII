"""Chunking strutturale del Markdown prodotto da Docling.

Una macchina a stati legge le pagine in ordine e tiene traccia di allegato,
articolo, template e sezione correnti. Ogni riga di una tabella di istruzioni
(LOG) con codice cella diventa un chunk a se'; il resto e' raggruppato in chunk
di testo o di tabella fino a `max_chars` caratteri.
"""
import re
from collections import Counter
from statistics import median

from .codes import RE_CELL, RE_TEMPLATE, template_base, template_full

RE_ANNEX = re.compile(r"^(?:#+\s*)?\**\s*(ALLEGATO|ANNEX)\s+([IVXLC]+)\b", re.I)
RE_ARTICLE = re.compile(r"^(?:#+\s*)?\**\s*(Articolo|Article)\s+(\d+)\b", re.I)
RE_TMPL_HEADING = re.compile(r"^(?:#+\s*)?\**\s*(SR|SE|S|E)\.\s?\d{2}\.\d{2}")
RE_SEP = re.compile(r"^\|?\s*:?-{3,}")
GENERIC_SECTIONS = {"osservazioni generali", "general comments"}


def clean_row(line):
    """Riga di tabella markdown -> 'a | b | c', togliendo celle vuote e
    duplicati adiacenti (Docling ripete il testo delle celle unite)."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    out = []
    for c in cells:
        if c and (not out or c != out[-1]):
            out.append(c)
    return " | ".join(out)


class Chunker:
    def __init__(self, max_chars=1500):
        self.max_chars = max_chars
        self.chunks = []
        self.annex = self.article = self.section = None
        self.annex_title, self.await_title = None, False
        self.template = self.template_full = self.template_title = None
        self.text_buf, self.text_page = [], None
        self.tbl_header, self.tbl_buf, self.tbl_page = None, [], None
        self.in_table = False

    def prefix(self, page):
        # Prefisso breve: un prefisso lungo e uguale per centinaia di chunk
        # "diluisce" gli embeddings delle istruzioni corte
        parts = []
        if self.annex:
            label = f"Allegato {self.annex}"
            if self.annex_title:
                short = re.sub(r"^.*segnalazione\s+(?:delle|dei|degli)\s+", "",
                               self.annex_title)
                label += f" ({short[:60]})"
            parts.append(label)
        elif self.article:
            parts.append(f"Articolo {self.article}")
        if self.template_title:
            parts.append(self.template_title[:120])
        if (self.section and self.section != self.template_title
                and self.section.lower() not in GENERIC_SECTIONS):
            parts.append(self.section[:120])
        if self.template_full and not any(self.template_full in p for p in parts):
            parts.append(self.template_full)
        parts.append(f"pag. {page}")
        return "[" + " · ".join(parts) + "]"

    def add(self, body, page, kind):
        body = body.strip()
        if len(body) < 3:
            return
        self.chunks.append({
            "id": len(self.chunks),
            "kind": kind,                      # testo | tabella | istruzione
            "page": page,
            "annex": self.annex,
            "article": None if self.annex else self.article,
            "template": self.template,         # es. S.02.01
            "template_full": self.template_full,
            "section": self.section,
            "cells": sorted(set(RE_CELL.findall(body))),
            "text": self.prefix(page) + "\n" + body,
        })

    def flush_text(self):
        body = "\n".join(self.text_buf)
        if len(body) >= 60:   # i soli titoli finiscono comunque nel prefisso
            self.add(body, self.text_page, "testo")
        self.text_buf, self.text_page = [], None

    def flush_table(self):
        if self.tbl_buf:
            head = (self.tbl_header + "\n") if self.tbl_header else ""
            self.add(head + "\n".join(self.tbl_buf), self.tbl_page, "tabella")
        self.tbl_buf, self.tbl_page = [], None

    def text_line(self, line, page):
        if self.text_page is None:
            self.text_page = page
        self.text_buf.append(line)
        if sum(len(l) for l in self.text_buf) > self.max_chars:
            self.flush_text()

    def table_row(self, row, page):
        head = (self.tbl_header or "").upper()
        is_log = "ISTRUZIONI" in head or "INSTRUCTIONS" in head
        # Tabelle di istruzioni (LOG): ogni riga con codice e' un chunk a se'
        if is_log and RE_CELL.search(row):
            self.flush_table()
            self.add(row, page, "istruzione")   # intestazione inutile, la tolgo
            return
        # Riga LOG senza codice = continuazione dell'istruzione precedente
        if is_log and self.chunks and self.chunks[-1]["kind"] == "istruzione":
            last = self.chunks[-1]
            last["text"] += " " + row
            last["cells"] = sorted(set(last["cells"]) | set(RE_CELL.findall(row)))
            return
        # Altre tabelle (es. modelli dell'Allegato I)
        if RE_CELL.search(row) and len(row) > 120:
            self.flush_table()
            head_line = (self.tbl_header + "\n") if self.tbl_header else ""
            self.add(head_line + row, page, "istruzione")
            return
        # Righe brevi (struttura dei template) -> raggruppate
        if self.tbl_page is None:
            self.tbl_page = page
        self.tbl_buf.append(row)
        if sum(len(r) for r in self.tbl_buf) > self.max_chars:
            self.flush_table()

    def handle_text(self, line, page):
        clean = line.lstrip("#").strip().strip("*").strip()
        is_heading = line.startswith("#")
        m_annex = RE_ANNEX.match(line) if len(clean) < 80 else None
        m_art = RE_ARTICLE.match(line) if len(clean) < 60 else None

        if m_annex:
            self.flush_text()
            self.annex = m_annex.group(2).upper()
            self.annex_title, self.await_title = None, True
            self.template = self.template_full = self.template_title = None
            self.section = clean
        elif m_art and not self.annex:
            self.flush_text()
            self.article = m_art.group(2)
            self.section = clean
        elif RE_TMPL_HEADING.match(line) and len(clean) < 200:
            m = RE_TEMPLATE.search(clean)
            self.flush_text()
            self.template, self.template_full = template_base(m), template_full(m)
            self.template_title = clean
            self.section = clean
            self.await_title = False
        elif is_heading:
            self.flush_text()
            self.section = clean
            if self.await_title:   # primo titolo dopo "ALLEGATO X"
                self.annex_title, self.await_title = clean, False
        self.text_line(clean, page)

    def process_page(self, md, page):
        self.in_table = False
        lines = md.splitlines()
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line or line.startswith("<!--"):
                continue
            if line.startswith("|"):
                if RE_SEP.match(line):
                    continue
                row = clean_row(line)
                if not row:
                    continue
                if not self.in_table:
                    self.flush_text()
                    self.in_table = True
                    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    # Prima riga = intestazione, a meno che non sia la
                    # continuazione di una tabella dalla pagina precedente
                    if RE_SEP.match(nxt) and not RE_CELL.search(row):
                        self.flush_table()
                        self.tbl_header = row
                        continue
                self.table_row(row, page)
                continue
            if self.in_table:
                self.flush_table()
                self.in_table = False
            self.handle_text(line, page)
        self.flush_table()

    def finish(self):
        self.flush_text()
        self.flush_table()
        return self.chunks


def build_chunks(pages, max_chars=1500):
    """Chunk di tutte le pagine ({numero_pagina: markdown}), in ordine di pagina."""
    chunker = Chunker(max_chars=max_chars)
    for pg in sorted(pages):
        chunker.process_page(pages[pg], pg)
    return chunker.finish()


def print_chunk_stats(chunks, n_pages=None, examples=5):
    """Riepilogo dei chunk e controllo sulle istruzioni dell'Allegato II."""
    lens = [len(c["text"]) for c in chunks]
    head = f"Pagine: {n_pages} | " if n_pages is not None else ""
    print(f"{head}chunk: {len(chunks)}")
    print("Per tipo:", Counter(c["kind"] for c in chunks))
    print("Template piu' frequenti:",
          Counter(c["template"] for c in chunks if c["template"]).most_common(10))
    print(f"Lunghezza mediana {int(median(lens))} car., massima {max(lens)}")
    for c in chunks[:: max(1, len(chunks) // examples)][:examples]:
        print("-" * 80)
        print(c["text"][:500])

    # Nell'Allegato II dovrebbero essere quasi tutte "istruzione"
    log = [c for c in chunks if c["annex"] == "II"]
    print("\nAllegato II:", Counter(c["kind"] for c in log))
