# RAG QRT

Question answering over **Commission Implementing Regulation (EU) 2023/894**, which lays down the Solvency II quantitative reporting templates (QRTs). Every answer cites the annex, template, cell codes and page of the excerpts it is based on.

Pipeline: PDF parsing with Docling → structural chunking (one chunk per LOG instruction) → hybrid index of BM25 and `BAAI/bge-m3` embeddings → Reciprocal Rank Fusion, code-aware boosting and `BAAI/bge-reranker-v2-m3` used as an extra vote → answer generation with `Qwen/Qwen2.5-7B-Instruct` → Gradio interface with feedback collection.

The pipeline supports both the English and the Italian version of the regulation (`lang="en"` or `lang="it"`): stopwords, implicit-annex detection, prompts, interface and evaluation questions follow the chosen language, while template and cell codes are language-independent. Code comments are in Italian.

## Repository layout

```
qrt_rag/
  codes.py        recognition of template (S.02.01) and cell (R0010) codes
  lang.py         language resources (en/it): stopwords, labels, prompts, UI
  parsing.py      PDF -> Markdown page by page, in checkpointed batches
  chunking.py     state machine tracking annex, article, template and LOG rows
  index.py        bge-m3 embeddings, index save/load
  retrieval.py    Retriever: BM25, embeddings, RRF, reranker, implicit annex
  generation.py   LocalLLM (Qwen) and system prompt
  ui.py           Gradio interface with answer rating
  evaluation.py   evaluation set, hit@k and MRR, per-stage diagnostics
  eval_set*.json  development set (it, en): 31 questions with expected source + 2 negatives
notebooks/
  qrt_rag_kaggle.ipynb     end-to-end run on Kaggle (2 x T4 GPU)
  fase2_browser_eval.ipynb  can retrieval run in the browser? small embedding models vs bge-m3
```

## Installation

```bash
pip install "qrt-rag[all] @ git+https://github.com/mat126/rag-qrt.git"
```

Optional extras: `parsing` (Docling), `llm` (transformers), `ui` (Gradio), `eval` (pandas). The retrieval core only needs `numpy`, `rank-bm25` and `sentence-transformers`.

## Running on Kaggle

1. Import `notebooks/qrt_rag_kaggle.ipynb` into Kaggle (File → Import Notebook).
2. Settings: GPU T4 x2, Internet on, secrets `GRADIO_USER` and `GRADIO_PASS`.
3. Attach the regulation PDF and, if available, the dataset with the saved parsing (`PARSED_INPUT`).
4. Run All: after about 10 minutes the Gradio interface is available at the `*.gradio.live` link printed by the interface cell.

## Using it as a library

```python
from qrt_rag.index import load_index, load_embedder
from qrt_rag.retrieval import Retriever, load_reranker, show

chunks, E = load_index("index/")
retriever = Retriever(chunks, E, load_embedder(), reranker=load_reranker(), lang="en")
show(retriever.search("Instructions for S.23.01 R0540", k=5))
```

## Results

On the development set (Italian version, 31 questions, no annex filter) the correct source is in the top 5 results for every question:

| Configuration | hit@1 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- |
| Fusion + reranker as a vote | 0.516 | 1.000 | 1.000 | 0.737 |
| Fusion only | 0.452 | 0.935 | 0.968 | 0.667 |

This is an optimistic estimate, since the same set was used to tune the pipeline. Generated answers should always be checked against the cited excerpts.

## Data

The regulation PDF, the parsed pages and the index are not included in the repository. The regulation is published on EUR-Lex.
