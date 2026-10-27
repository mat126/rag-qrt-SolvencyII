"""Risorse che dipendono dalla lingua del documento e delle domande: "it" o "en".

Tutto il resto della pipeline (codici, parsing, fusione, metriche) e' indipendente
dalla lingua. Per aggiungere una lingua basta aggiungere una voce a ogni dizionario.
"""
import re

LANGS = ("it", "en")


def check_lang(lang):
    if lang not in LANGS:
        raise ValueError(f"Lingua non supportata: {lang!r} (usa una tra {LANGS})")
    return lang


# Parole vuote per BM25: con rank_bm25 anche le parole frequentissime ricevono un
# peso minimo non trascurabile e, sommate, coprono le parole utili.
# Le negazioni ("non", "senza", "not", "without") restano: cambiano il significato.
STOPWORDS = {
    "it": set("""
il lo la i gli le l un uno una di a da in con su per tra fra
del dello della dei degli delle dell al allo alla ai agli alle all
dal dallo dalla dai dagli dalle dall nel nello nella nei negli nelle nell
sul sullo sulla sui sugli sulle sull d e ed o od che se ma anche
come dove quando quale quali qual cosa chi cui si ci ne è sono essere
viene vengono va vanno deve devono può possono questo questa questi queste
""".split()),
    "en": set("""
a an the of for to in on at by with from as into onto over under about
and or nor but if then than so that this these those it its itself
is are be been being was were am do does did done has have had having
shall should must may might can could would will
which what where when how who whom whose why
such any each every both either other another same also only very s
""".split()),
}

# Parole "di servizio" tipiche delle domande, tolte solo dalla query
QUERY_META = {
    "it": {"istruzioni", "istruzione", "riportato", "riportata", "riportare",
           "riporta", "segnala", "segnalare", "segnalato", "compilare", "cella"},
    "en": {"instructions", "instruction", "reported", "report", "reports",
           "fill", "filled", "cell"},
}

# Allegato implicito: (regex "gruppi", regex "singole imprese")
SCOPE_RE = {
    "it": (re.compile(r"\bgrupp[oi]\b", re.I),
           re.compile(r"\bsingol[ae]\s+impres[ae]\b", re.I)),
    "en": (re.compile(r"\bgroups?\b", re.I),
           re.compile(r"\b(?:individual|solo)\s+undertakings?\b", re.I)),
}

# Abbreviazione del titolo dell'allegato nel prefisso dei chunk:
# "Istruzioni relative ai modelli per la segnalazione delle singole imprese" -> "singole imprese"
# Per l'inglese la formula va verificata sul testo reale (vedi print_chunk_stats).
ANNEX_TITLE_RE = {
    "it": re.compile(r"^.*segnalazione\s+(?:delle|dei|degli)\s+", re.I),
    "en": re.compile(r"^.*reporting\s+(?:for|of|by)\s+(?:the\s+)?", re.I),
}

LABELS = {
    "it": {"annex": "Allegato", "article": "Articolo", "page": "pag.",
           "excerpts": "estratti", "question": "Domanda"},
    "en": {"annex": "Annex", "article": "Article", "page": "p.",
           "excerpts": "excerpts", "question": "Question"},
}

SYSTEM_PROMPTS = {
    "it": """Sei un assistente esperto di reporting di vigilanza Solvency II.
Rispondi SOLO sulla base degli estratti del regolamento forniti.
- Cita ogni affermazione con il numero dell'estratto, es. [2].
- Riporta codici di template e celle (S.xx.xx, R0000, C0000) esattamente come nel testo.
- Lo stesso template puo' avere istruzioni diverse in allegati diversi (es. imprese
  singole e gruppi): indica sempre l'allegato e, se la domanda non lo specifica,
  distingui i casi invece di mescolarli.
- Riporta gli elenchi chiusi esattamente come nel testo, con i loro codici
  numerici originali: non rinumerarli e non aggiungere o togliere opzioni.
- Se un'istruzione e' identica per singole imprese e gruppi, dillo una volta sola
  invece di ripeterla.
- Se gli estratti non contengono la risposta, dillo chiaramente invece di supporre.
- Rispondi in italiano, in modo conciso.""",
    "en": """You are an expert assistant on Solvency II supervisory reporting.
Answer ONLY on the basis of the excerpts from the regulation provided.
- Cite every statement with the number of the excerpt, e.g. [2].
- Report template and cell codes (S.xx.xx, R0000, C0000) exactly as in the text.
- The same template may have different instructions in different annexes (e.g.
  individual undertakings and groups): always state the annex and, if the question
  does not specify it, distinguish the cases instead of mixing them.
- Report closed lists exactly as in the text, with their original numeric codes:
  do not renumber them and do not add or remove options.
- If an instruction is identical for individual undertakings and groups, state it
  only once instead of repeating it.
- If the excerpts do not contain the answer, say so clearly instead of guessing.
- Answer in English, concisely.""",
}

UI = {
    "it": {
        "title": "RAG sul Regolamento di esecuzione (UE) 2023/894",
        "question": "Domanda", "placeholder": "Es. Cosa va riportato in S.23.01 R0540?",
        "annex": "Allegato", "all": "Tutti", "k": "Estratti (k)", "mode": "Modalita'",
        "ask": "Chiedi", "answer_h": "Risposta", "rate_h": "Valuta",
        "expected": "Fonte attesa (facoltativa): es. S.23.01 R0540 oppure Articolo 2",
        "note": "Nota (facoltativa)", "up": "👍 Corretta", "down": "👎 Sbagliata",
        "sources_h": "Estratti recuperati", "search_only": "Solo ricerca",
        "search_only_msg": "_Modalita' solo ricerca: guarda gli estratti a destra._",
        "empty": "Scrivi una domanda.", "no_state": "Prima fai una domanda.",
        "saved": "Salvato ({emoji}). Feedback totali: {n}", "llm_name": "Qwen (locale)",
        "auth": "RAG QRT - accesso riservato",
    },
    "en": {
        "title": "RAG on Commission Implementing Regulation (EU) 2023/894",
        "question": "Question", "placeholder": "E.g. What must be reported in S.23.01 R0540?",
        "annex": "Annex", "all": "All", "k": "Excerpts (k)", "mode": "Mode",
        "ask": "Ask", "answer_h": "Answer", "rate_h": "Rate",
        "expected": "Expected source (optional): e.g. S.23.01 R0540 or Article 2",
        "note": "Note (optional)", "up": "👍 Correct", "down": "👎 Wrong",
        "sources_h": "Retrieved excerpts", "search_only": "Search only",
        "search_only_msg": "_Search-only mode: see the excerpts on the right._",
        "empty": "Please type a question.", "no_state": "Ask a question first.",
        "saved": "Saved ({emoji}). Total feedback: {n}", "llm_name": "Qwen (local)",
        "auth": "RAG QRT - restricted access",
    },
}
