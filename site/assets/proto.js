/* Prototypical-network maths, identical to src/model.py:
 *   prototype = plain mean of the support embeddings (embeddings are already L2-normalised; the mean is not)
 *   score     = squared Euclidean distance; prediction = nearest prototype
 *   softmax over -scale * d^2 gives the model's own class probabilities (scale = learned temperature).
 * Shared by index.html and scripts/check_showcase.js so the page is tested with the code it runs. */
(function (root) {
  function decodeF16(b64) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const n = bin.length >> 1, out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
             : e === 31 ? (f ? NaN : s * Infinity)
             : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  }

  function row(Z, dim, i) { return Z.subarray(i * dim, (i + 1) * dim); }

  function prototypes(Z, dim, support /* [nWay][k] image indices */) {
    return support.map(ids => {
      const p = new Float32Array(dim);
      for (const i of ids) { const z = row(Z, dim, i); for (let d = 0; d < dim; d++) p[d] += z[d]; }
      for (let d = 0; d < dim; d++) p[d] /= ids.length;
      return p;
    });
  }

  function sqDist(a, b) { let s = 0; for (let d = 0; d < a.length; d++) { const t = a[d] - b[d]; s += t * t; } return s; }

  function classify(Z, dim, protos, q, scale) {
    const z = row(Z, dim, q);
    const d2 = protos.map(p => sqDist(z, p));
    const logits = d2.map(v => -scale * v), m = Math.max(...logits);
    const ex = logits.map(v => Math.exp(v - m)), sum = ex.reduce((a, b) => a + b, 0);
    let pred = 0; for (let c = 1; c < d2.length; c++) if (d2[c] < d2[pred]) pred = c;
    return { d2, prob: ex.map(v => v / sum), pred };
  }

  /* An episode row is [classes(nWay), support(nWay*k), query(nWay*nQuery)], class-major, as in src/data.py. */
  function unpack(ep, nWay, k, nQuery) {
    const classes = ep.slice(0, nWay), sup = [], qry = [];
    for (let c = 0; c < nWay; c++) {
      sup.push(ep.slice(nWay + c * k, nWay + (c + 1) * k));
      qry.push(ep.slice(nWay + nWay * k + c * nQuery, nWay + nWay * k + (c + 1) * nQuery));
    }
    return { classes, sup, qry };
  }

  function episodeAccuracy(Z, dim, ep, nWay, k, nQuery, scale) {
    const { sup, qry } = unpack(ep, nWay, k, nQuery);
    const P = prototypes(Z, dim, sup);
    let right = 0;
    qry.forEach((ids, c) => ids.forEach(q => { if (classify(Z, dim, P, q, scale).pred === c) right++; }));
    return right / (nWay * nQuery);
  }

  const api = { decodeF16, prototypes, classify, unpack, episodeAccuracy, sqDist, row };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Proto = api;
})(typeof window !== "undefined" ? window : globalThis);
