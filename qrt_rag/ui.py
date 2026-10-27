"""Interfaccia Gradio: domanda, filtri, risposta, estratti e raccolta del feedback.

build_demo() restituisce un gr.Blocks; chi lo usa decide come lanciarlo
(su Kaggle: demo.launch(share=True, auth=...)). Richiede gli extra [ui].
"""
import datetime
import json
from pathlib import Path

import gradio as gr

from .lang import LABELS, UI


def format_sources(hits, lang="it"):
    page = LABELS[lang]["page"]
    blocks = []
    for n, h in enumerate(hits, 1):
        header, _, body = h["text"].partition("\n")
        blocks.append(
            f"**[{n}]** {page} {h['page']} · `{h['template_full'] or '-'}` · {h['kind']}  \n"
            f"<small>{header}</small>\n\n{body[:1200]}"
        )
    return "\n\n---\n\n".join(blocks)


def build_demo(retriever, llm=None, feedback_path="feedback.jsonl", title=None, lang=None):
    """retriever: un qrt_rag.retrieval.Retriever.
    llm: oggetto con .name e .generate(question, hits) -> (risposta, estratti),
    ad esempio qrt_rag.generation.LocalLLM; None = solo ricerca.
    lang: lingua dell'interfaccia; di default quella del retriever."""
    lang = lang or retriever.lang
    T = UI[lang]
    feedback_path = Path(feedback_path)
    modes = [llm.name, T["search_only"]] if llm is not None else [T["search_only"]]

    def run(question, annex, k, mode):
        if not question.strip():
            return T["empty"], "", None
        annex_val = None if annex == T["all"] else annex
        hits = retriever.search(question, k=int(k), annex=annex_val)
        if llm is not None and mode == llm.name:
            answer, _ = llm.generate(question, hits)
        else:
            answer = T["search_only_msg"]
        state = {
            "question": question, "annex": annex_val, "k": int(k), "mode": mode,
            "answer": answer, "lang": lang,
            "hits": [{"id": h["id"], "page": h["page"], "template": h["template_full"],
                      "cells": h["cells"]} for h in hits],
        }
        return answer, format_sources(hits, lang), state

    def save_feedback(state, rating, expected, note):
        if not state:
            return T["no_state"], expected, note
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
        return T["saved"].format(emoji="👍" if rating == "up" else "👎", n=n), "", ""

    with gr.Blocks(title="RAG QRT") as demo:
        gr.Markdown(f"## {title or T['title']}")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                q = gr.Textbox(label=T["question"], lines=3, placeholder=T["placeholder"])
                with gr.Row():
                    annex = gr.Dropdown([T["all"]] + retriever.annexes, value=T["all"],
                                        label=T["annex"])
                    k = gr.Slider(3, 15, value=10, step=1, label=T["k"])
                mode = gr.Radio(modes, value=modes[0], label=T["mode"])
                btn = gr.Button(T["ask"], variant="primary")
                gr.Markdown(f"### {T['answer_h']}")
                answer = gr.Markdown()
                gr.Markdown(f"### {T['rate_h']}")
                expected = gr.Textbox(label=T["expected"])
                note = gr.Textbox(label=T["note"])
                with gr.Row():
                    up = gr.Button(T["up"])
                    down = gr.Button(T["down"])
                fb_status = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown(f"### {T['sources_h']}")
                sources = gr.Markdown()

        btn.click(run, [q, annex, k, mode], [answer, sources, state])
        up.click(lambda s, e, n: save_feedback(s, "up", e, n),
                 [state, expected, note], [fb_status, expected, note])
        down.click(lambda s, e, n: save_feedback(s, "down", e, n),
                   [state, expected, note], [fb_status, expected, note])
    return demo
