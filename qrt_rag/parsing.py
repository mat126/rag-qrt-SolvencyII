"""Parsing del PDF in Markdown, pagina per pagina, con Docling.

Richiede gli extra [parsing] (docling, pypdf); su GPU T4 circa 25 minuti
per le 1597 pagine del Regolamento (UE) 2023/894.
"""
import fnmatch
import gc
import json
import os
import re
import time
from pathlib import Path


def run_parsing(pdf_path, pages_dir, work_dir, batch=50, page_range=None, show=3):
    """Converte il PDF a batch di `batch` pagine, con checkpoint su disco.

    Ogni batch diventa pages_dir/pages_SSSS_EEEE.json, una lista di record
    {"page": numero assoluto, "md": markdown}. Un batch gia' salvato viene
    saltato, quindi una sessione interrotta riparte dall'ultimo batch.

    page_range: (inizio, fine) con indici 0-based e fine esclusa; None = tutto.
    show: quante pagine dell'intervallo stampare alla fine per un controllo visivo.
    """
    import torch
    from pypdf import PdfReader, PdfWriter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pages_dir, work_dir = Path(pages_dir), Path(work_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)

    opts = PdfPipelineOptions()
    opts.do_ocr = False                      # il PDF di EUR-Lex ha gia' il testo
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )  # usa la GPU in automatico se disponibile

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    start_all, end_all = page_range if page_range else (0, n_pages)
    tmp_dir = work_dir / "tmp_batches"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pagine totali: {n_pages} | da elaborare: {start_all + 1}-{end_all}")

    for s in range(start_all, end_all, batch):
        e = min(s + batch, end_all)
        out = pages_dir / f"pages_{s + 1:04d}_{e:04d}.json"
        if out.exists():
            print("gia' fatto:", out.name)
            continue

        writer = PdfWriter()
        for i in range(s, e):
            writer.add_page(reader.pages[i])
        bpath = tmp_dir / f"batch_{s:04d}.pdf"
        with open(bpath, "wb") as f:
            writer.write(f)

        t0 = time.time()
        doc = converter.convert(str(bpath)).document
        records = []
        for p in range(1, e - s + 1):
            try:
                md = doc.export_to_markdown(page_no=p)
            except Exception as ex:
                print(f"  errore pagina {s + p}: {ex}")
                md = ""
            records.append({"page": s + p, "md": md})  # numero di pagina assoluto
        out.write_text(json.dumps(records, ensure_ascii=False))

        bpath.unlink()
        del doc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dt = time.time() - t0
        print(f"{out.name}: {dt:.0f}s ({dt / (e - s):.1f}s/pagina)")

    # Controllo visivo: prime pagine dell'intervallo appena elaborato
    if show:
        pages = load_pages(pages_dir, normalize=False, verbose=False)
        for pg in range(start_all + 1, min(start_all + 1 + show, end_all + 1)):
            print("=" * 30, "pagina", pg)
            print(pages.get(pg, "(non elaborata)")[:2500])


def load_pages(src_dir, normalize=True, verbose=True):
    """Legge tutti i pages_*.json sotto src_dir, anche in sottocartelle
    (Kaggle puo' aggiungere un livello di cartella quando estrae un Dataset).

    Restituisce {numero_pagina: markdown}. Con normalize=True rimuove i
    trattini di sillabazione invisibili (U+00AD) e lo spazio che li segue.
    """
    src_dir = Path(src_dir)
    # os.walk con followlinks=True: su Kaggle le cartelle dei Dataset possono essere
    # collegamenti simbolici, che Path.rglob (Python 3.13) non attraversa
    page_files = sorted(
        Path(root) / name
        for root, _, files in os.walk(src_dir, followlinks=True)
        for name in files
        if fnmatch.fnmatch(name, "pages_*.json")
    )
    if not page_files:
        raise FileNotFoundError(f"Nessun file pages_*.json in {src_dir} (cercato anche nelle sottocartelle)")
    if verbose:
        print(f"Leggo {len(page_files)} file da {src_dir}")
    pages = {}
    for fp in page_files:
        for rec in json.loads(fp.read_text()):
            md = rec["md"]
            pages[rec["page"]] = re.sub(r"\u00ad\s*", "", md) if normalize else md
    return pages
