"""Interfaccia Gradio: domanda, filtri, risposta, estratti e raccolta del feedback.

build_demo() restituisce un gr.Blocks; chi lo usa decide come lanciarlo
(su Kaggle: demo.launch(share=True, auth=...)). Richiede gli extra [ui].
"""
import datetime
import json
from pathlib import Path

import gradio as gr

SEARCH_ONLY = "Solo ricerca"


def format_sources(hits):
    blocks = []
    for n, h in enumerate(hits, 1):
        header, _, body = h["text"].partition("\n")
        blocks.append(
            f"**[{n}]** pag. {h['page']} · `{h['template_full'] or '-'}` · {h['kind']}  \n"
            f"<small>{header}</small>\n\n{body[:1200]}"
        )
    return "\n\n---\n\n".join(blocks)


def build_demo(retriever, llm=None, feedback_path="feedback.jsonl",
               title="RAG sul Regolamento di esecuzione (UE) 2023/894"):
    """retriever: un qrt_rag.retrieval.Retriever.
    llm: oggetto con .name e .generate(question, hits) -> (risposta, estratti),
    ad esempio qrt_rag.generation.LocalLLM; None = solo ricerca."""
    feedback_path = Path(feedback_path)
    modes = [llm.name, SEARCH_ONLY] if llm is not None else [SEARCH_ONLY]

    def run(question, annex, k, mode):
        if not question.strip():
            return "Scrivi una domanda.", "", None
        annex_val = None if annex == "Tutti" else annex
        hits = retriever.search(question, k=int(k), annex=annex_val)
        if llm is not None and mode == llm.name:
            answer, _ = llm.generate(question, hits)
        else:
            answer = "_Modalita' solo ricerca: guarda gli estratti a destra._"
        state = {
            "question": question, "annex": annex_val, "k": int(k), "mode": mode,
            "answer": answer,
            "hits": [{"id": h["id"], "page": h["page"], "template": h["template_full"],
                      "cells": h["cells"]} for h in hits],
        }
        return answer, format_sources(hits), state

    def save_feedback(state, rating, expected, note):
        if not state:
            return "Prima fai una domanda.", expected, note
        rec = {
            **state,
            "rating": rating,
            "expected": expected.strip().upper() or None,   # es. "S.23.01 R0540"
            "note": note.strip() or None,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(feedback_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        # svuota "fonte attesa" e "nota", cosi' non restano attaccate alla domanda dopo
        return f"Salvato ({'👍' if rating == 'up' else '👎'}). Feedback totali: {n}", "", ""

    with gr.Blocks(title="RAG QRT") as demo:
        gr.Markdown(f"## {title}")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                q = gr.Textbox(label="Domanda", lines=3,
                               placeholder="Es. Cosa va riportato in S.23.01 R0540?")
                with gr.Row():
                    annex = gr.Dropdown(["Tutti"] + retriever.annexes, value="Tutti",
                                        label="Allegato")
                    k = gr.Slider(3, 15, value=10, step=1, label="Estratti (k)")
                mode = gr.Radio(modes, value=modes[0], label="Modalita'")
                btn = gr.Button("Chiedi", variant="primary")
                gr.Markdown("### Risposta")
                answer = gr.Markdown()
                gr.Markdown("### Valuta")
                expected = gr.Textbox(
                    label="Fonte attesa (facoltativa): es. S.23.01 R0540 oppure Articolo 2")
                note = gr.Textbox(label="Nota (facoltativa)")
                with gr.Row():
                    up = gr.Button("👍 Corretta")
                    down = gr.Button("👎 Sbagliata")
                fb_status = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("### Estratti recuperati")
                sources = gr.Markdown()

        btn.click(run, [q, annex, k, mode], [answer, sources, state])
        up.click(lambda s, e, n: save_feedback(s, "up", e, n),
                 [state, expected, note], [fb_status, expected, note])
        down.click(lambda s, e, n: save_feedback(s, "down", e, n),
                   [state, expected, note], [fb_status, expected, note])
    return demo
