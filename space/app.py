"""App Gradio per Hugging Face Spaces: ricerca sul Regolamento (UE) 2023/894 su CPU.

Scarica l'indice (chunks.jsonl, embeddings.npy) da un Dataset su Hugging Face,
carica bge-m3 e, se attivo, il reranker, e avvia l'interfaccia in sola ricerca.

Variabili e secret dello Space:
  INDEX_REPO     Dataset con l'indice, es. "mat126/qrt-rag-index"
  USE_RERANKER   "1" (default) oppure "0" se su CPU risulta troppo lento
  HF_TOKEN       solo se il Dataset e' privato
  GRADIO_USER / GRADIO_PASS   facoltativi: se presenti, l'app chiede il login
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from qrt_rag.index import load_embedder, load_index
from qrt_rag.retrieval import Retriever, load_reranker
from qrt_rag.ui import build_demo

INDEX_REPO = os.environ.get("INDEX_REPO", "mat126/qrt-rag-index")
USE_RERANKER = os.environ.get("USE_RERANKER", "1") == "1"
DATA_DIR = Path("data")


def download_index():
    for name in ("chunks.jsonl", "embeddings.npy"):
        hf_hub_download(repo_id=INDEX_REPO, filename=name, repo_type="dataset",
                        local_dir=DATA_DIR, token=os.environ.get("HF_TOKEN"))
    return DATA_DIR


chunks, E = load_index(download_index())
emb_model = load_embedder(device="cpu")
reranker = load_reranker(device="cpu") if USE_RERANKER else None
retriever = Retriever(chunks, E, emb_model, reranker=reranker)
print(f"Indice caricato: {len(chunks)} chunk | reranker: {'si' if reranker else 'no'}")

# Nota: sul piano gratuito il disco si azzera a ogni riavvio, quindi questo
# feedback e' temporaneo finche' non viene salvato su un Dataset.
demo = build_demo(retriever, llm=None, feedback_path=DATA_DIR / "feedback.jsonl")

user, pwd = os.environ.get("GRADIO_USER"), os.environ.get("GRADIO_PASS")
auth = (user, pwd) if user and pwd else None
demo.launch(auth=auth, auth_message="RAG QRT - accesso riservato" if auth else None)
