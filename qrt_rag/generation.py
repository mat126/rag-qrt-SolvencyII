"""Generazione delle risposte a partire dagli estratti recuperati.

LocalLLM usa Qwen2.5-7B-Instruct in fp16 su GPU (circa 15 GB, divisi sulle
GPU disponibili). Richiede gli extra [llm].
"""
import gc
import time

from .lang import LABELS, SYSTEM_PROMPTS, UI, check_lang

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"   # licenza Apache 2.0, buon italiano e inglese

SYSTEM_PROMPT = SYSTEM_PROMPTS["it"]   # compatibilita'


def build_context(hits, max_chars_chunk=2000, max_chars_total=9000, lang="it"):
    """Contesto per l'LLM con un tetto di lunghezza: alcuni chunk (articoli,
    tabelle dell'Allegato I) superano i 7000 caratteri e, su una T4, un prompt
    troppo lungo esaurisce la memoria nel calcolo dell'attenzione."""
    parts, total = [], 0
    for n, h in enumerate(hits, 1):
        text = h["text"]
        if len(text) > max_chars_chunk:
            text = text[:max_chars_chunk] + " [...]"
        block = (f"[{n}] ({LABELS[lang]['page']} {h['page']}, "
                 f"{h['template_full'] or h['section'] or ''})\n{text}")
        if total + len(block) > max_chars_total and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def user_message(question, hits, lang="it", **context_kwargs):
    L = LABELS[lang]
    tag = L["excerpts"]
    return (f"<{tag}>\n{build_context(hits, lang=lang, **context_kwargs)}\n</{tag}>"
            f"\n\n{L['question']}: {question}")


class LocalLLM:
    """LLM locale. Il prompt si cambia senza ricaricare il modello:
    llm.system_prompt = NUOVO_PROMPT."""

    def __init__(self, model_id=DEFAULT_LLM, system_prompt=None, lang="it",
                 max_chars_chunk=2000, max_chars_total=9000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.lang = check_lang(lang)
        self.name = UI[lang]["llm_name"]
        self.system_prompt = system_prompt or SYSTEM_PROMPTS[lang]
        self.context_kwargs = {"max_chars_chunk": max_chars_chunk,
                               "max_chars_total": max_chars_total}
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"   # divide sulle GPU
        )
        self.model.eval()

    def _generate(self, question, hits, max_new_tokens):
        import torch
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",
             "content": user_message(question, hits, lang=self.lang, **self.context_kwargs)},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, repetition_penalty=1.05)
            return self.tok.decode(out[0, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
        finally:
            del inputs
            gc.collect()
            torch.cuda.empty_cache()   # libera la memoria tra una domanda e l'altra

    def generate(self, question, hits, max_new_tokens=700):
        """Restituisce (risposta, estratti effettivamente usati). Se la GPU esaurisce
        la memoria, riprova con meta' degli estratti invece di bloccarsi."""
        import torch
        try:
            return self._generate(question, hits, max_new_tokens), hits
        except torch.cuda.OutOfMemoryError:
            print("  (memoria GPU esaurita: riprovo con meno estratti)")
            hits = hits[: max(2, len(hits) // 2)]
            return self._generate(question, hits, max_new_tokens), hits


def ask(retriever, llm, question, k=6, annex=None, show_sources=True, max_new_tokens=700):
    """Ricerca + risposta, con stampa della risposta e delle fonti."""
    hits = retriever.search(question, k=k, annex=annex)
    t0 = time.time()
    answer, hits = llm.generate(question, hits, max_new_tokens)
    print(answer)
    print(f"\n(generata in {time.time() - t0:.0f}s)")
    if show_sources:
        print("\nFONTI:")
        for n, h in enumerate(hits, 1):
            print(f"  [{n}] pag. {h['page']}  {h['template_full'] or '-'}  {h['kind']}")
    return answer, hits
