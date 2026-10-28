// Ricerca ibrida nel browser: stessa logica di qrt_rag.retrieval.Retriever (senza reranker).
// BM25 replica rank_bm25.BM25Okapi; fusione RRF, boost sui codici e allegato implicito
// seguono riga per riga il codice Python. I file in assets/ sono prodotti da
// qrt_rag.web_export.export_web().

// ---------------------------------------------------------------- utilita'

function f16ToF32(u16) {
  // Conversione esatta float16 -> float32 (ogni float16 e' rappresentabile in float32)
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) out[i] = sign * 2 ** -14 * (frac / 1024);
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return out;
}

const normTmpl = (t) => (t || "").replace(/^SR\./, "S.");

// Indici dei primi n punteggi in ordine decrescente; a parita' vince l'indice minore
// (come np.argsort(-x, kind="stable")).
function topIndices(scores, n, keep, positiveOnly) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    if (!keep(i) || !Number.isFinite(s)) continue;
    if (positiveOnly && !(s > 0)) continue;
    idx.push(i);
  }
  idx.sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, n);
}

// ---------------------------------------------------------------- retriever

export class WebRetriever {
  constructor(chunks, embeddings, config) {
    this.chunks = chunks;
    this.E = embeddings;              // Float32Array n_chunks * dim, oppure null (solo BM25)
    this.dim = config.dim;
    this.config = config;
    this.lang = config.lang;

    const p = config.patterns;
    this.tokenRe = new RegExp(p.token.source, p.token.flags);
    this.templateRe = new RegExp(p.template.source, p.template.flags);
    this.cellRe = new RegExp(p.cell.source, p.cell.flags);
    this.groupRe = new RegExp(p.group.source, p.group.flags);
    this.soloRe = new RegExp(p.solo.source, p.solo.flags);
    this.stop = new Set(config.stopwords);
    this.meta = new Set(config.query_meta);

    this.chunkTmpl = chunks.map((c) => normTmpl(c.template));
    this.chunkCells = chunks.map((c) => new Set(c.cells));
    this.annexes = [...new Set(chunks.map((c) => c.annex).filter(Boolean))];
    this.buildBM25();
  }

  // suffix: "" per la lingua principale, "_it" per i file italiani (chunks_it.json, ...)
  static async load(base = "assets/", suffix = "") {
    const get = async (name, kind) => {
      const r = await fetch(base + name);
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return kind === "json" ? r.json() : r.arrayBuffer();
    };
    const [config, chunks, buf] = await Promise.all([
      get(`config${suffix}.json`, "json"),
      get(`chunks${suffix}.json`, "json"),
      get(`embeddings${suffix}.f16.bin`, "bin"),
    ]);
    const E = f16ToF32(new Uint16Array(buf));   // little-endian, come scritto da Python
    if (E.length !== config.n_chunks * config.dim) {
      throw new Error(`Embeddings incoerenti: ${E.length} valori per ${config.n_chunks} x ${config.dim}`);
    }
    return new WebRetriever(chunks, E, config);
  }

  // ------------------------------------------------------------ testo

  tokenize(text) {
    const out = [];
    for (const m of text.toLowerCase().matchAll(this.tokenRe)) {
      if (!this.stop.has(m[0])) out.push(m[0]);
    }
    return out;
  }

  tokenizeQuery(q) {
    return this.tokenize(q).filter((t) => !this.meta.has(t));
  }

  // Testi da codificare per una domanda, come Retriever.first_stage()
  queryTexts(q) {
    const clean = this.tokenizeQuery(q).join(" ");
    return clean && clean !== q.toLowerCase() ? [q, clean] : [q];
  }

  queryScope(q) {
    const g = this.groupRe.test(q);
    const s = this.soloRe.test(q);
    if (g && !s) return "III";
    if (s && !g) return "II";
    return null;
  }

  queryCodes(q) {
    const up = q.toUpperCase();
    const tmpls = new Set();
    for (const m of up.matchAll(this.templateRe)) tmpls.add(normTmpl(`${m[1]}.${m[2]}.${m[3]}`));
    const cells = new Set(up.match(this.cellRe) || []);
    return { tmpls, cells };
  }

  // ------------------------------------------------------------ BM25 (rank_bm25.BM25Okapi)

  buildBM25() {
    const { k1, b, epsilon } = this.config.search.bm25;
    const N = this.chunks.length;
    const postings = new Map();          // termine -> [[doc, tf], ...]
    this.docLen = new Float64Array(N);
    let total = 0;
    this.chunks.forEach((c, d) => {
      const toks = this.tokenize(c.text);
      this.docLen[d] = toks.length;
      total += toks.length;
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
      for (const [t, f] of tf) {
        if (!postings.has(t)) postings.set(t, []);
        postings.get(t).push([d, f]);
      }
    });
    this.avgdl = total / N;
    this.k1 = k1;
    this.b = b;
    this.postings = postings;

    // IDF come rank_bm25: i valori negativi diventano epsilon * IDF medio
    this.idf = new Map();
    let idfSum = 0;
    const negative = [];
    for (const [t, list] of postings) {
      const n = list.length;
      const idf = Math.log(N - n + 0.5) - Math.log(n + 0.5);
      this.idf.set(t, idf);
      idfSum += idf;
      if (idf < 0) negative.push(t);
    }
    const eps = epsilon * (idfSum / this.idf.size);
    for (const t of negative) this.idf.set(t, eps);
  }

  bm25Scores(tokens) {
    const scores = new Float64Array(this.chunks.length);
    for (const q of tokens) {                       // le ripetizioni contano, come in Python
      const idf = this.idf.get(q) || 0;
      const list = this.postings.get(q);
      if (!list) continue;
      for (const [d, f] of list) {
        const denom = f + this.k1 * (1 - this.b + (this.b * this.docLen[d]) / this.avgdl);
        scores[d] += idf * ((f * (this.k1 + 1)) / denom);
      }
    }
    return scores;
  }

  // ------------------------------------------------------------ denso

  denseScores(queryVectors) {
    // Similarita' massima tra il chunk e le versioni della domanda (vettori normalizzati)
    const n = this.chunks.length;
    const d = this.dim;
    const scores = new Float64Array(n).fill(-Infinity);
    for (const v of queryVectors) {
      for (let i = 0; i < n; i++) {
        let s = 0;
        const off = i * d;
        for (let j = 0; j < d; j++) s += this.E[off + j] * v[j];
        if (s > scores[i]) scores[i] = s;
      }
    }
    return scores;
  }

  // ------------------------------------------------------------ ricerca

  // queryVectors: vettori di queryTexts(query), gia' calcolati dal modello di embedding;
  // null = solo BM25.
  search(query, { k = 8, nCand = null, annex = null, queryVectors = null } = {}) {
    const S = this.config.search;
    nCand = nCand || S.n_cand;
    const chunks = this.chunks;
    const { tmpls, cells } = this.queryCodes(query);
    const keep = annex ? (i) => chunks[i].annex === annex : () => true;

    // Reciprocal Rank Fusion; la Map conserva l'ordine di inserimento come il dict di Python
    const rrf = new Map();
    const add = (i, v) => rrf.set(i, (rrf.get(i) || 0) + v);
    const bm = this.bm25Scores(this.tokenizeQuery(query));
    topIndices(bm, nCand, keep, true).forEach((i, r) => add(i, 1 / (S.rrf_k + r)));
    if (queryVectors && this.E) {
      const dense = this.denseScores(queryVectors);
      topIndices(dense, nCand, keep, false).forEach((i, r) => add(i, 1 / (S.rrf_k + r)));
    }

    // Boost ai chunk che corrispondono a template e/o celle citati
    if (tmpls.size || cells.size) {
      for (let i = 0; i < chunks.length; i++) {
        if (!keep(i)) continue;
        const tOk = tmpls.size ? tmpls.has(this.chunkTmpl[i]) : true;
        const cOk = cells.size ? [...cells].some((c) => this.chunkCells[i].has(c)) : true;
        if (tOk && cOk) add(i, S.code_boost);
      }
    }

    let cand = [...rrf.keys()];
    cand.sort((a, b) => rrf.get(b) - rrf.get(a));   // stabile, come sorted(..., reverse=True)
    cand = cand.slice(0, nCand);
    if (!cand.length) return [];

    const matchLevel = (i) =>
      (tmpls.size && tmpls.has(this.chunkTmpl[i]) ? 1 : 0) +
      (cells.size && [...cells].some((c) => this.chunkCells[i].has(c)) ? 1 : 0);
    const scope = annex ? null : this.queryScope(query);
    const other = { II: "III", III: "II" }[scope];
    const scopeOk = (i) => (other ? (chunks[i].annex !== other ? 1 : 0) : 1);

    const keyed = cand.map((i) => ({ i, m: matchLevel(i), s: scopeOk(i), sc: rrf.get(i) }));
    keyed.sort((x, y) => y.m - x.m || y.s - x.s || y.sc - x.sc);   // stabile
    return keyed.slice(0, k).map(({ i }, r) => ({ ...chunks[i], rank: r + 1 }));
  }
}

// ---------------------------------------------------------------- valutazione

// Stessa regola di qrt_rag.evaluation.is_match
export function isMatch(h, expected) {
  if (expected.article) return h.article === expected.article;
  if (expected.annex && h.annex !== expected.annex) return false;
  if (expected.template && normTmpl(h.template) !== normTmpl(expected.template)) return false;
  return (expected.cells || []).every((c) => h.cells.includes(c));
}
