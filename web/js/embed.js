// Embedding delle domande nel browser con transformers.js (ONNX Runtime Web, WebAssembly).
// Il modello si scarica una volta (circa 120 MB per multilingual-e5-small quantizzato)
// e poi resta nella cache del browser.
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

env.allowLocalModels = false;   // i modelli arrivano dall'Hub di Hugging Face

// model: config.model di assets/config.json ({web, q_prefix, pooling})
// Restituisce una funzione async: testi -> array di Float32Array normalizzati.
export async function loadEmbedder(model, { dtype = "q8", onProgress = null } = {}) {
  const extractor = await pipeline("feature-extraction", model.web, {
    dtype,
    progress_callback: onProgress || undefined,
  });
  return async (texts) => {
    const out = await extractor(texts.map((t) => model.q_prefix + t), {
      pooling: model.pooling || "mean",
      normalize: true,
    });
    const [n, d] = out.dims;
    return Array.from({ length: n }, (_, i) => out.data.slice(i * d, (i + 1) * d));
  };
}
