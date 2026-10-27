"""Riconoscimento dei codici QRT.

- Template: S.02.01, S.02.01.02, SR.25.01.01 (fondi separati/MAP), SE., E.
- Celle: R0010 (righe), C0020 (colonne), Z0010 (dimensioni aperte).
"""
import re

RE_TEMPLATE = re.compile(r"\b(SR|SE|S|E)\.\s?(\d{2})\.(\d{2})(?:\.(\d{2}))?\b")
RE_CELL = re.compile(r"\b[RCZ]\d{4}\b")


def template_base(m):
    """Codice base a tre livelli da un match di RE_TEMPLATE, es. S.02.01."""
    return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"


def template_full(m):
    """Codice completo, es. S.02.01.02 (o il base se manca il quarto livello)."""
    return template_base(m) + (f".{m.group(4)}" if m.group(4) else "")


def norm_tmpl(t):
    """SR.25.01 (fondi separati/MAP) va trattato come variante di S.25.01."""
    return re.sub(r"^SR\.", "S.", t or "")


def query_codes(q):
    """Template (normalizzati) e celle citati in una domanda."""
    up = q.upper()
    return ({norm_tmpl(template_base(m)) for m in RE_TEMPLATE.finditer(up)},
            set(RE_CELL.findall(up)))
