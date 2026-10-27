---
title: RAG QRT
emoji: 📑
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Hybrid search over the Solvency II QRT reporting regulation
---

# RAG QRT – search interface

Hybrid search (BM25 + `BAAI/bge-m3` embeddings + `BAAI/bge-reranker-v2-m3`) over Commission Implementing Regulation (EU) 2023/894, the Solvency II quantitative reporting templates. Questions are in Italian; results show the annex, template, cell codes and page of each excerpt.

Code and documentation: https://github.com/mat126/rag-qrt

This Space runs in search-only mode on a free CPU. It wakes up from sleep on the first visit, which can take a few minutes.
