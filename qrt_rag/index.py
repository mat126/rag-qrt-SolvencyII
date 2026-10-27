"""Embeddings dei chunk (BAAI/bge-m3) e salvataggio/caricamento dell'indice.

L'indice sono due file: chunks.jsonl (un chunk per riga) ed embeddings.npy
(matrice float32 di vettori normalizzati, una riga per chunk).
"""
import json
import time
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-m3"


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(model_id=EMBEDDING_MODEL, device=None, max_seq_length=2048):
    """Carica il modello di embedding; in fp16 solo su GPU."""
    from sentence_transformers import SentenceTransformer
    device = device or default_device()
    model = SentenceTransformer(model_id, device=device)
    if device.startswith("cuda"):
        model.half()                     # piu' veloce e meno memoria sulla T4
    model.max_seq_length = max_seq_length   # alcuni chunk superano i 7000 caratteri
    return model


def embed_chunks(model, chunks, batch_size=16):
    """Matrice (n_chunk x 1024) di embeddings normalizzati in float32.
    Con sequenze lunghe, se la GPU va in out-of-memory, scendi a batch_size=8."""
    t0 = time.time()
    E = model.encode(
        [c["text"] for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    print(f"Embeddings {E.shape} in {time.time() - t0:.0f}s")
    return E


def save_index(index_dir, chunks, E):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", E)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("Indice salvato in", index_dir)


def load_index(index_dir):
    """Ricarica chunk ed embeddings salvati da save_index()."""
    index_dir = Path(index_dir)
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    E = np.load(index_dir / "embeddings.npy")
    if len(chunks) != len(E):
        raise ValueError(f"{len(chunks)} chunk ma {len(E)} embeddings: indice incoerente")
    return chunks, E
