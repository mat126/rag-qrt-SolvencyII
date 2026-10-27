"""RAG sul Regolamento di esecuzione (UE) 2023/894 (modelli QRT Solvency II).

Moduli:
- codes       riconoscimento dei codici di template e cella
- parsing     PDF -> Markdown per pagina con Docling          (extra [parsing])
- chunking    chunking strutturale per allegato/template/cella
- index       embeddings bge-m3 e salvataggio dell'indice
- retrieval   ricerca ibrida BM25 + embeddings + reranker
- generation  risposte con LLM locale                          (extra [llm])
- ui          interfaccia Gradio con feedback                  (extra [ui])
- evaluation  set di valutazione, metriche, diagnostica        (extra [eval])

I moduli non si importano a vicenda piu' del necessario: importare
qrt_rag.retrieval non richiede Docling, Gradio o transformers.
"""
__version__ = "0.1.0"
