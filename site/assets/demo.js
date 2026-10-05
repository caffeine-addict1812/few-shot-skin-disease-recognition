/* Demo page: Story (auto-play), Explore (free orbit + morph) and Few-shot lab (precomputed or live model).
 * Classification always uses the real 512-d embeddings (assets/proto.js, same maths as src/model.py); the 3-D galaxy
 * only shows where things sit. Prototype rings in 3-D are drawn at the mean position of their example photos. */
(function () {
  "use strict";
  const S = window.SHOWCASE, A = S.atlas, P = window.Proto;
  const DIM = S.dim, NW = 5, NQ = 5, L = "ABCDE";
  const Z = { finetuned: P.decodeF16(S.emb.finetuned), frozen: P.decodeF16(S.emb.frozen) };
  const OFF = A.test_class_offset, COLS = S.thumb.cols, ROWS = Math.ceil(S.labels.length / COLS);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = id => document.getElementById(id);
  const ease = window.Galaxy.ease;

  // test-photo index -> galaxy point index, and members of every disease
  const testGlobal = new Int32Array(S.labels.length);
  const members = new Map();
  for (let gi = 0; gi < A.split.length; gi++) {
    if (A.split[gi] === 2) testGlobal[A.test_index[gi]] = gi;
    if (!members.has(A.cls[gi])) members.set(A.cls[gi], []);
    members.get(A.cls[gi]).push(gi);
  }

  // ---------------------------------------------------------------- photos
  function setPhoto(el, gi) {
    if (A.split[gi] === 2) {
      const i = A.test_index[gi], c = i % COLS, r = Math.floor(i / COLS);
      el.style.backgroundImage = "url(assets/thumbs.jpg)";
      el.style.backgroundSize = `${COLS * 100}% ${ROWS * 100}%`;
      el.style.backgroundPosition = `${(c / (COLS - 1)) * 100}% ${(r / (ROWS - 1)) * 100}%`;
    } else {
      el.style.backgroundImage = `url("/img/${encodeURI(A.paths[gi])}")`;  // served by scripts/serve_site.py
      el.style.backgroundSize = "cover"; el.style.backgroundPosition = "center";
    }
  }
  const kindText = gi => (A.split[gi] === 2 ? "Never seen in training" : A.split[gi] === 1 ? "Validation disease (model selection only)" : "Training disease");

  // ---------------------------------------------------------------- galaxy
  const tip = $("tip");
  const g = new window.Galaxy($("stage"), A, {
    interactive: true,
    offsetX: w => (w > 760 ? 200 : 0),
    onHover(gi, e) {
      if (gi < 0 || !e) { tip.hidden = true; return; }
      $("tipName").textContent = A.names[A.cls[gi]];
      const k = $("tipKind"); k.textContent = kindText(gi); k.className = A.split[gi] === 2 ? "u" : "";
      setPhoto($("tipThumb"), gi);
      tip.hidden = false;
      const w = tip.offsetWidth, h = tip.offsetHeight;
      tip.style.left = Math.min(window.innerWidth - w - 8, e.clientX + 16) + "px";
      tip.style.top = (e.clientY + 16 + h > window.innerHeight ? e.clientY - h - 12 : e.clientY + 16) + "px";
    },
    onClick(gi) { if (gi >= 0) { if (mode !== "explore") selectTab("explore"); focusClass(A.cls[gi]); } },
  });
  g.onT = t => {
    $("morphBadge").innerHTML = t > 0.98 ? "Layout: <i>trained model</i>" : t < 0.02 ? "Layout: <i>ImageNet features</i>" : "Layout: <i>changing&hellip;</i>";
    $("morph").value = Math.round(t * 1000);
    $("endA").classList.toggle("on", t < 0.5); $("endB").classList.toggle("on", t >= 0.5);
  };

  // ---------------------------------------------------------------- overlay state + drawing
  const ov = { focusCls: -1, ep: null, showRings: false, query: null, flyers: [] };
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const INK = css("--field-ink"), GLOW = css("--field-glow"), FIELD = css("--field");

  function protoScreen() {
    return ov.ep.sup.map(ids => g.project(g.centroid(ids.map(i => testGlobal[i]))));
  }
  g.onFrame((ctx, now) => {
    // constellation of a focused disease
    if (ov.focusCls >= 0) {
      const ids = members.get(ov.focusCls), c = g.project(g.centroid(ids));
      ctx.strokeStyle = INK; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
      ids.forEach(gi => { const p = g.project(g.worldPos(gi)); ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(p.x, p.y); ctx.stroke(); });
      ctx.globalAlpha = 1; ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(c.x, c.y, 4, 0, 7); ctx.fill();
    }
    // episode: spokes from example photos to their prototype ring
    let protos = null;
    if (ov.ep && ov.showRings) {
      protos = protoScreen();
      ctx.lineWidth = 1; ctx.strokeStyle = "rgba(233,237,246,0.28)";
      ov.ep.sup.forEach((ids, c) => ids.forEach(i => {
        const p = g.project(g.worldPos(testGlobal[i])); ctx.beginPath(); ctx.moveTo(protos[c].x, protos[c].y); ctx.lineTo(p.x, p.y); ctx.stroke();
      }));
    }
    // query: lines to every prototype, weight = model probability, winner in amber
    if (ov.ep && ov.query && ov.query.linesFrom <= now) {
      protos = protos || protoScreen();
      const q = g.project(ov.query.pos());
      const k = Math.min(1, (now - ov.query.linesFrom) / 500);
      for (let c = 0; c < NW; c++) {
        const win = c === ov.query.pred, pr = ov.query.prob[c];
        ctx.strokeStyle = win ? GLOW : INK;
        ctx.globalAlpha = (win ? 1 : 0.2 + 0.6 * pr) * k;
        ctx.lineWidth = 1 + 5 * pr;
        ctx.setLineDash(win ? [] : [4, 5]);
        const x = q.x + (protos[c].x - q.x) * ease(k), y = q.y + (protos[c].y - q.y) * ease(k);
        ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(x, y); ctx.stroke();
      }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    // prototype rings with letters (drawn last so they sit on top)
    if (protos) {
      ctx.font = `700 14px ${css("--mono")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const back = x => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
      protos.forEach((p, c) => {
        const gk = reduceMotion ? 1 : Math.min(1, Math.max(0, (now - (ov.ringT0 || 0) - c * 120) / 650));
        if (gk <= 0) return;
        const win = ov.query && ov.query.pred === c && ov.query.linesFrom <= now;
        const r = 15 * back(gk) + (win && !reduceMotion ? 2.5 * Math.sin(now / 160) : 0);
        if (win) { ctx.fillStyle = GLOW; ctx.globalAlpha = 0.18; ctx.beginPath(); ctx.arc(p.x, p.y, r + 9, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
        ctx.fillStyle = FIELD; ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 7); ctx.fill();
        ctx.strokeStyle = win ? GLOW : INK; ctx.lineWidth = win ? 3 : 2; ctx.stroke();
        ctx.globalAlpha = gk; ctx.fillStyle = INK; ctx.fillText(L[c], p.x, p.y + 0.5); ctx.globalAlpha = 1;
      });
    }
    // flying / pinned photos
    for (const f of ov.flyers) {
      const tgt = g.project(f.pos()), raw = f.dur ? Math.min(1, Math.max(0, (now - f.t0) / f.dur)) : 1, k = ease(raw);
      // quadratic Bezier with a sideways bow, so photos arc in rather than sliding in a straight line
      const dx = tgt.x - f.from.x, dy = tgt.y - f.from.y, cx = f.from.x + dx / 2 - dy * f.bow, cy = f.from.y + dy / 2 + dx * f.bow;
      const x = (1 - k) * (1 - k) * f.from.x + 2 * (1 - k) * k * cx + k * k * tgt.x;
      const y = (1 - k) * (1 - k) * f.from.y + 2 * (1 - k) * k * cy + k * k * tgt.y;
      const s = f.s0 + (f.s1 - f.s0) * k, rot = (1 - k) * f.spin;
      f.el.style.width = f.el.style.height = s + "px";
      f.el.style.transform = `translate(${x - s / 2}px, ${y - s / 2}px) rotate(${rot}deg)`;
      f.el.style.opacity = now < f.t0 ? 0 : tgt.visible ? 1 : 0;
      if (f.trail && raw > 0 && raw < 1) { f.trail.push([x, y]); if (f.trail.length > 26) f.trail.shift(); }
      if (f.trail && raw >= 1 && f.trail.length) f.trail.shift();
      if (f.trail && f.trail.length > 1) {
        for (let n = 1; n < f.trail.length; n++) {
          ctx.strokeStyle = GLOW; ctx.globalAlpha = (n / f.trail.length) * 0.55; ctx.lineWidth = 1 + 4 * (n / f.trail.length);
          ctx.beginPath(); ctx.moveTo(...f.trail[n - 1]); ctx.lineTo(...f.trail[n]); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      if (raw >= 1 && !f.landed && now >= f.t0) { f.landed = true; f.el.classList.add("landed"); }
    }
  });
  function clearFlyers() { ov.flyers.forEach(f => f.el.remove()); ov.flyers = []; }
  function addFlyer(posFn, opts) {
    const el = document.createElement("div");
    el.className = "fly" + (opts.cls ? " " + opts.cls : "");
    if (opts.gi !== undefined) setPhoto(el, opts.gi);
    if (opts.url) { el.style.backgroundImage = `url("${opts.url}")`; el.style.backgroundSize = "cover"; }
    if (opts.tag) { const t = document.createElement("span"); t.className = "tag"; t.textContent = opts.tag; el.appendChild(t); }
    g.html.appendChild(el);
    const f = { el, pos: posFn, from: opts.from, t0: performance.now() + (opts.delay || 0), dur: reduceMotion ? 0 : opts.dur || 900,
                s0: opts.s0 || 120, s1: opts.s1 || 34, bow: opts.bow !== undefined ? opts.bow : (Math.random() - 0.5) * 0.5,
                spin: reduceMotion ? 0 : (Math.random() - 0.5) * 24, trail: opts.trail && !reduceMotion ? [] : null };
    ov.flyers.push(f);
    return f;
  }

  // ---------------------------------------------------------------- episodes (shared by story and lab)
  const lab = { k: 5, model: "finetuned", ep: { 1: 0, 5: 0 }, qi: 0, src: "test", right: 0, done: 0, live: null };

  function loadEpisode(k, idx) {
    const u = P.unpack(S.episodes[`5way_${k}shot`][idx], NW, k, NQ);
    const order = []; for (let j = 0; j < NQ; j++) for (let c = 0; c < NW; c++) order.push([u.qry[c][j], c]);
    return { ...u, k, idx, order };
  }
  function protosFor(ep, model) { return P.prototypes(Z[model], DIM, ep.sup); }
  function classifyVec(protos, z, scale) {
    const d2 = protos.map(p => P.sqDist(z, p)), logits = d2.map(v => -scale * v), m = Math.max(...logits);
    const ex = logits.map(v => Math.exp(v - m)), s = ex.reduce((a, b) => a + b, 0);
    let pred = 0; for (let c = 1; c < d2.length; c++) if (d2[c] < d2[pred]) pred = c;
    return { d2, prob: ex.map(v => v / s), pred };
  }
  function showEpisode(ep, animate) {
    clearFlyers(); ov.query = null; ov.focusCls = -1; ov.ep = ep; ov.showRings = false;
    const map = new Map();
    ep.classes.forEach(c => members.get(OFF + c).forEach(gi => map.set(gi, 1)));
    ep.sup.flat().forEach(i => map.set(testGlobal[i], 2));
    g.setFocus(map, 0.1);
    const ids = ep.sup.flat().map(i => testGlobal[i]), c = g.centroid(ids);
    g.flyTo(c, g.radiusFor(ids, c), animate ? 1600 : 0);
    const origin = { x: window.innerWidth < 760 ? window.innerWidth / 2 : 420, y: window.innerHeight * 0.55 };
    ep.sup.forEach((sids, cc) => sids.forEach((i, j) => {
      const gi = testGlobal[i];
      addFlyer(() => g.worldPos(gi), { gi, from: origin, delay: animate ? 900 + (cc * ep.k + j) * (ep.k === 1 ? 160 : 45) : 0,
                                       dur: 900, s0: 96, s1: ep.k === 1 ? 46 : 30, tag: j === 0 ? L[cc] : "" });
    }));
    setTimeout(() => { if (ov.ep === ep) { ov.showRings = true; ov.ringT0 = performance.now(); } }, animate && !reduceMotion ? 900 + NW * ep.k * (ep.k === 1 ? 160 : 45) + 700 : 0);
  }
  function showQuery(ep, gi, res, opts) {
    opts = opts || {};
    ov.flyers.filter(f => f.isQuery).forEach(f => f.el.remove());
    ov.flyers = ov.flyers.filter(f => !f.isQuery);
    const posFn = opts.posFn || (() => g.worldPos(gi));
    const from = { x: window.innerWidth - 120, y: 140 };
    const f = addFlyer(posFn, { gi: opts.url ? undefined : gi, url: opts.url, cls: "q" + (opts.live ? " live" : ""), from, dur: 1300, s0: 150, s1: 54,
                                tag: opts.tag || "query", bow: 0.28, trail: true });
    f.isQuery = true;
    ov.query = { pos: posFn, pred: res.pred, prob: res.prob, linesFrom: performance.now() + (reduceMotion ? 0 : 1300) };
  }

  // ---------------------------------------------------------------- tabs
  let mode = "story";
  function selectTab(m) {
    mode = m;
    for (const [id, pid, name] of [["tabStory", "pStory", "story"], ["tabExplore", "pExplore", "explore"], ["tabLab", "pLab", "lab"]]) {
      $(id).setAttribute("aria-selected", m === name); $(pid).hidden = m !== name;
    }
    $("caption").hidden = m !== "story";
    if (m !== "story") story.pause();
    if (m === "explore") { clearFlyers(); ov.ep = null; ov.query = null; g.setFocus(null); applyLayers(); g.flyHome(1200); g.rotSpeed = 0.06; g.autoRotate = !reduceMotion; }
    if (m === "lab") { g.setLayers(1, 1); g.autoRotate = false; labDeal(false, true); }
    if (m === "story") { g.autoRotate = !reduceMotion; }
  }
  $("tabStory").onclick = () => { selectTab("story"); story.play(); };
  $("tabExplore").onclick = () => selectTab("explore");
  $("tabLab").onclick = () => selectTab("lab");

  // ---------------------------------------------------------------- explore
  function applyLayers() { g.setLayers($("lSeen").checked ? 1 : 0, $("lUnseen").checked ? 1 : 0); }
  $("lSeen").onchange = applyLayers; $("lUnseen").onchange = applyLayers;
  $("morph").oninput = () => { g.tweens = g.tweens.filter(t => t.kind !== "morph"); g.setT($("morph").value / 1000); };
  $("morphPlay").onclick = () => { g.setT(0); g.animateT(1, 3200); };
  $("nnStat").innerHTML = `Never-seen photos whose nearest other never-seen photo shows the same disease: <strong>${S.nn_same.frozen.toFixed(0)}%</strong> with ImageNet features, <strong>${S.nn_same.finetuned.toFixed(0)}%</strong> after training. ${A.spread_summary.tighter} of ${A.spread_summary.of} never-seen diseases group more tightly (median ${A.spread_summary.median_change.toFixed(0)}% relative spread).`;
  const dl = $("diseases");
  [...new Set(A.names)].sort().forEach(n => { const o = document.createElement("option"); o.value = n; dl.appendChild(o); });
  $("find").onchange = () => { const c = A.names.indexOf($("find").value); if (c >= 0) focusClass(c); };
  function focusClass(c) {
    clearFlyers(); ov.ep = null; ov.query = null; ov.focusCls = c;
    const ids = members.get(c), map = new Map(ids.map(gi => [gi, 1]));
    g.setFocus(map, 0.12);
    const ctr = g.centroid(ids); g.flyTo(ctr, g.radiusFor(ids, ctr), 1400); g.autoRotate = false;
    $("focusCard").hidden = false;
    $("fName").textContent = A.names[c];
    $("fKind").textContent = kindText(ids[0]) + ` · ${ids.length} photos`;
    const s0 = A.spread.frozen[c], s1 = A.spread.finetuned[c], ch = 100 * (s1 - s0) / s0;
    $("fSpread").innerHTML = `Relative spread in 512-d: <b>${s0.toFixed(2)}</b> → <b>${s1.toFixed(2)}</b> (${ch < 0 ? "" : "+"}${ch.toFixed(0)}% after training; lower is tighter)`;
    const strip = $("fStrip"); strip.innerHTML = "";
    ids.slice(0, 12).forEach(gi => { const d = document.createElement("div"); d.className = "th"; setPhoto(d, gi); strip.appendChild(d); });
  }
  $("fClear").onclick = () => { ov.focusCls = -1; $("focusCard").hidden = true; g.setFocus(null); g.flyHome(1200); g.autoRotate = !reduceMotion; $("find").value = ""; };

  // ---------------------------------------------------------------- lab
  const ICON_OK = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M6.2 11.6 2.8 8.2l1.1-1.1 2.3 2.3 5.9-5.9 1.1 1.1z"/></svg>';
  const ICON_NO = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="m4.1 3 3.9 3.9L11.9 3 13 4.1 9.1 8l3.9 3.9-1.1 1.1L8 9.1 4.1 13 3 11.9 6.9 8 3 4.1z"/></svg>';
  let LE = null, LP = null;
  function press(a, b, on) { $(a).setAttribute("aria-pressed", on); $(b).setAttribute("aria-pressed", !on); }
  function renderSupport() {
    const box = $("sup"); box.className = "sup" + (LE.k === 1 ? " k1" : ""); box.innerHTML = "";
    LE.classes.forEach((cid, c) => {
      const r = document.createElement("div"); r.className = "r";
      r.innerHTML = `<div class="letter" data-c="${c}">${L[c]}</div><div style="min-width:0"><div class="nm">${S.classes[cid]}</div><div class="ths"></div></div>`;
      LE.sup[c].forEach(i => { const d = document.createElement("div"); d.className = "th"; setPhoto(d, testGlobal[i]); r.querySelector(".ths").appendChild(d); });
      box.appendChild(r);
    });
    $("epLabel").textContent = `Test episode ${LE.idx + 1} of 600`;
  }
  function renderResult(res, truth, thumbFn, truthText) {
    thumbFn($("qThumb"));
    $("qTruth").textContent = truthText;
    const v = $("verdict");
    if (truth === null) { v.className = "verdict na"; v.textContent = `Nearest prototype: ${L[res.pred]}`; }
    else {
      const ok = res.pred === truth;
      v.className = "verdict " + (ok ? "ok" : "no");
      v.innerHTML = ok ? `${ICON_OK}Correct: ${L[res.pred]}` : `${ICON_NO}Wrong: picked ${L[res.pred]}`;
    }
    document.querySelectorAll("#sup .letter").forEach(el => el.classList.toggle("hit", +el.dataset.c === res.pred));
    const max = Math.max(...res.d2);
    $("bars").innerHTML = res.d2.map((d, c) => `<div class="bar${c === res.pred ? " win" : ""}"><span class="l">${L[c]}</span>
      <div class="track"><div class="fill" style="transform:scaleX(${(d / max).toFixed(4)})"></div></div><span class="v">${d.toFixed(3)} · ${(100 * res.prob[c]).toFixed(0)}%</span></div>`).join("");
  }
  function labQuery() {
    const [q, truth] = LE.order[lab.qi];
    const res = P.classify(Z[lab.model], DIM, LP, q, S.scale[lab.model]);
    renderResult(res, truth, el => setPhoto(el, testGlobal[q]), `${L[truth]} · ${S.classes[LE.classes[truth]]}`);
    showQuery(LE, testGlobal[q], res, { tag: `query ${lab.qi + 1}/25` });
    lab.cur = { res, truth };
    renderTally();
  }
  function renderTally() { $("tally").innerHTML = `This model and setting: <b>${lab.right} / ${lab.done}</b> correct${lab.done ? ` (${(100 * lab.right / lab.done).toFixed(0)}%)` : ""}`; }
  function resetTally() { lab.right = 0; lab.done = 0; $("runAllOut").textContent = ""; renderTally(); }
  function labDeal(advance, animate) {
    if (advance) lab.ep[lab.k] = (lab.ep[lab.k] + 1) % 600;
    LE = loadEpisode(lab.k, lab.ep[lab.k]); LP = protosFor(LE, lab.model); lab.qi = 0;
    renderSupport(); showEpisode(LE, animate !== false);
    setTimeout(() => { if (mode === "lab" && lab.src === "test") labQuery(); }, reduceMotion ? 0 : 2600);
    if (lab.src === "live") { $("verdict").textContent = ""; $("bars").innerHTML = ""; $("qTruth").textContent = "Drop a photo or embed a test photo"; }
  }
  $("k1").onclick = () => { if (lab.k === 1) return; lab.k = 1; press("k1", "k5", true); resetTally(); labDeal(false); };
  $("k5").onclick = () => { if (lab.k === 5) return; lab.k = 5; press("k1", "k5", false); resetTally(); labDeal(false); };
  function setModel(m) {
    lab.model = m; press("mFt", "mFz", m === "finetuned"); resetTally(); LP = protosFor(LE, m);
    g.animateT(m === "finetuned" ? 1 : 0, 1600);   // the galaxy follows the model being used
    if (lab.src === "test") labQuery(); else if (lab.live) liveShow(lab.live);
  }
  $("mFt").onclick = () => setModel("finetuned");
  $("mFz").onclick = () => setModel("frozen");
  $("deal").onclick = () => labDeal(true);
  $("next").onclick = () => {
    if (lab.src === "test" && lab.cur) { lab.done++; if (lab.cur.res.pred === lab.cur.truth) lab.right++; }
    if (lab.src !== "test") { setSource("test"); return; }
    if (lab.qi + 1 >= LE.order.length) { labDeal(true); return; }
    lab.qi++; labQuery();
  };
  $("runAll").onclick = () => {
    const key = `5way_${lab.k}shot`;
    const acc = 100 * S.episodes[key].reduce((a, ep) => a + P.episodeAccuracy(Z[lab.model], DIM, ep, NW, lab.k, NQ, S.scale[lab.model]), 0) / 600;
    $("runAllOut").innerHTML = `${lab.model === "finetuned" ? "Trained model" : "ImageNet only"}, ${lab.k} photo${lab.k > 1 ? "s" : ""} each: <b>${acc.toFixed(1)}%</b> over 15,000 query predictions (report: ${S.stored_acc[lab.model][key].toFixed(1)}%${lab.model === "finetuned" ? ", seed 42" : ""}).`;
  };

  // ---- live model (scripts/serve_site.py)
  let liveOK = false;
  fetch("/api/health").then(r => (r.ok ? r.json() : Promise.reject())).then(j => {
    liveOK = !!j.live; $("liveHealth").innerHTML = liveOK ? "Live model: <b>running on this computer</b>" : "";
  }).catch(() => {
    $("srcLive").disabled = true;
    $("liveHealth").textContent = "Live model needs the local server: python scripts/serve_site.py";
  });
  function setSource(src) {
    lab.src = src; press("srcTest", "srcLive", src === "test"); $("liveBox").hidden = src !== "live";
    if (src === "test") labQuery();
    else { $("verdict").textContent = ""; $("bars").innerHTML = ""; $("qTruth").textContent = "Drop a photo or embed a test photo"; $("qThumb").style.backgroundImage = ""; }
  }
  $("srcTest").onclick = () => setSource("test");
  $("srcLive").onclick = () => { if (liveOK) setSource("live"); };
  async function liveEmbed(body, query, label) {
    $("liveStatus").textContent = "Running the model…";
    try {
      const r = await fetch("/api/embed" + query, { method: "POST", body });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "The server could not embed that photo.");
      lab.live = { j, label };
      liveShow(lab.live);
    } catch (err) { $("liveStatus").textContent = err.message; }
  }
  function liveShow({ j, label }) {
    const m = lab.model, z = Float32Array.from(j[m].embedding);
    const res = classifyVec(LP, z, S.scale[m]);
    const xyzA = j.frozen.xyz, xyzB = j.finetuned.xyz;
    let truth = null, truthText = "Your photo (no label)", gi;
    if (label.gi !== undefined) {
      gi = label.gi; const c = LE.classes.indexOf(A.cls[gi] - OFF);
      truth = c >= 0 ? c : null; truthText = `${c >= 0 ? L[c] + " · " : ""}${A.names[A.cls[gi]]}`;
    }
    renderResult(res, truth, el => { if (gi !== undefined) setPhoto(el, gi); else { el.style.backgroundImage = `url("${label.url}")`; el.style.backgroundSize = "cover"; } }, truthText);
    showQuery(LE, gi, res, { posFn: () => g.mixXYZ(xyzA, xyzB), url: label.url, live: true, tag: "live" });
    const near = j[m].neighbours[0];
    $("liveStatus").innerHTML = `Embedded live in <b>${j.ms} ms</b> on CPU. ` +
      (j.max_abs_diff_vs_precomputed !== undefined ? `Matches the precomputed embedding to <b>${j.max_abs_diff_vs_precomputed.toExponential(1)}</b>. ` : "") +
      `Closest photo in the whole dataset: <b>${near.name}</b>. Placed in the galaxy by its 10 nearest photos in 512-d.`;
  }
  $("file").onchange = e => { const f = e.target.files[0]; if (f) liveEmbed(f, "", { url: URL.createObjectURL(f) }); };
  const drop = $("drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) liveEmbed(f, "", { url: URL.createObjectURL(f) }); });
  $("liveRandom").onclick = () => {
    const [q] = LE.order[Math.floor(Math.random() * LE.order.length)], gi = testGlobal[q];
    liveEmbed(null, "?path=" + encodeURIComponent(A.paths[gi]), { gi });
  };

  // ---------------------------------------------------------------- story
  const H = S.headline;
  const pct = v => v.toFixed(1) + "%";
  const SE = loadEpisode(5, 0);
  const SEP = protosFor(SE, "finetuned");
  const storyQuery = j => () => {
    const [q, truth] = SE.order[j], res = P.classify(Z.finetuned, DIM, SEP, q, S.scale.finetuned);
    showQuery(SE, testGlobal[q], res, { tag: `query ${j + 1}` });
    const ok = res.pred === truth;
    return { h: ok ? `It picks ${L[res.pred]}: ${S.classes[SE.classes[res.pred]]}. Correct.` : `It picks ${L[res.pred]}, but the photo is ${L[truth]}.`,
             p: `Squared distance to the winning prototype: ${res.d2[res.pred].toFixed(3)}, next closest ${[...res.d2].sort((a, b) => a - b)[1].toFixed(3)}. The model's probability for its choice is ${(100 * res.prob[res.pred]).toFixed(0)}%.` };
  };
  const steps = [
    { name: "6,584 photos", k: "SD-198 · 198 skin diseases", h: "Every dot is a clinical photo.", p: "The model turns each photo into 512 numbers. Photos it considers similar end up close together; this is a 3-D map of that space.", dur: 7000,
      enter() { clearFlyers(); ov.ep = null; ov.query = null; ov.focusCls = -1; g.setFocus(null); g.setLayers(1, 1, 0); g.setT(1); g.flyHome(0);
                g.rotSpeed = 0.06; g.autoRotate = !reduceMotion; g.playIntro(3200); } },
    { name: "Diseases it learned from", k: "Training", h: "It learns from 101 diseases.", p: "Grey dots are the training diseases (plus 27 used only to pick the best checkpoint). The model saw labels for these.", dur: 6500,
      enter() { g.setLayers(1, 0.06); } },
    { name: "Diseases it never saw", k: "Held out", h: "70 rare diseases never appear in training.", p: "Amber dots: every disease with fewer than 20 photos. No photo of them was used to train the model or choose its settings.", dur: 7000,
      enter() { g.setLayers(0.25, 1); } },
    { name: "Before training", k: "ImageNet features only", h: "Without training, the space is muddled.", p: `With plain ImageNet features, ${S.nn_same.frozen.toFixed(0)}% of never-seen photos have a nearest neighbour of the same disease.`, dur: 6500,
      enter() { g.setLayers(0.35, 1); g.animateT(0, 3800); } },
    { name: "After training", k: "Episodic training", h: "Training pulls each disease together.", p: `After episodic training that rises to ${S.nn_same.finetuned.toFixed(0)}%, and ${A.spread_summary.tighter} of 70 unseen diseases group more tightly, though none of them was trained on.`, dur: 7000,
      enter() { g.animateT(1, 3800); } },
    { name: "A new test", k: `Test episode 1 of 600 · 5 diseases × 5 photos`, h: "Show it five diseases it has never seen.", p: "Five example photos of each. Their average position in the 512-d space becomes the prototype for that disease (the lettered rings).", dur: 8500,
      enter() { g.setLayers(1, 1); g.autoRotate = !reduceMotion; g.rotSpeed = 0.022; showEpisode(SE, true); } },
    { name: "Query 1", k: "A held-out photo", dur: 7000, enter: storyQuery(0) },
    { name: "Query 2", k: "Another held-out photo", dur: 7000, enter: storyQuery(1) },
    { name: "Query 3", k: "And another", dur: 7000, enter: storyQuery(2) },
    { name: "600 tests", k: "Across all 600 test episodes", h: `${pct(H.trained["5way_5shot"])} from five photos, ${pct(H.trained["5way_1shot"])} from one.`,
      p: `Plain ImageNet features reach ${pct(H.frozen["5way_5shot"])} and ${pct(H.frozen["5way_1shot"])}; chance is 20%. Averages over 3 training runs.`, dur: 9000,
      enter() { clearFlyers(); ov.ep = null; ov.query = null; g.setFocus(null); g.flyHome(2400); g.rotSpeed = 0.06; g.autoRotate = !reduceMotion; } },
  ];
  const story = (() => {
    let i = -1, playing = false, t0 = 0, raf = 0, elapsed = 0;
    const list = $("steps");
    steps.forEach((s, j) => { const li = document.createElement("li"); li.innerHTML = `<span class="i">${String(j + 1).padStart(2, "0")}</span><span>${s.name}</span>`; li.onclick = () => go(j, true); list.appendChild(li); });
    function show(j) {
      const s = steps[j];
      const extra = s.enter() || {};
      $("capK").textContent = s.k; $("capH").textContent = extra.h || s.h; $("capP").textContent = extra.p || s.p;
      const cap = $("caption"); cap.classList.remove("enter"); void cap.offsetWidth; cap.classList.add("enter");
      [...list.children].forEach((li, n) => { li.className = n === j ? "on" : n < j ? "done" : ""; });
    }
    function go(j, user) { i = (j + steps.length) % steps.length; elapsed = 0; t0 = performance.now(); show(i); if (user && !playing) $("sBar").style.width = "0%"; }
    function tick(now) {
      if (playing) {
        const d = steps[i].dur * (reduceMotion ? 1.3 : 1);
        const k = (elapsed + now - t0) / d;
        $("sBar").style.width = Math.min(100, ((i + Math.min(1, k)) / steps.length) * 100) + "%";
        if (k >= 1) { if (i + 1 >= steps.length) { pause(); } else go(i + 1); }
      }
      raf = requestAnimationFrame(tick);
    }
    function play() { if (i < 0 || (i === steps.length - 1 && !playing)) go(0); playing = true; t0 = performance.now(); label(); }
    function pause() { if (playing) elapsed += performance.now() - t0; playing = false; label(); }
    function label() { $("sPlayTxt").textContent = playing ? "Pause" : "Play"; $("sPlayIcon").innerHTML = playing ? '<path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/>' : '<path d="M4 2.5v11l9.5-5.5z"/>'; }
    $("sPlay").onclick = () => (playing ? pause() : play());
    $("sNext").onclick = () => go(i + 1, true);
    $("sPrev").onclick = () => go(i - 1, true);
    $("sRestart").onclick = () => { go(0, true); play(); };
    raf = requestAnimationFrame(tick);
    return { play, pause, go };
  })();

  // present mode: hides the controls and the nav for a clean full-screen recording; keyboard drives the story
  function togglePresent(on) {
    const want = on === undefined ? !document.body.classList.contains("present") : on;
    document.body.classList.toggle("present", want);
    if (want) { if (mode !== "story") selectTab("story"); story.play(); $("presentHint").classList.remove("show"); void $("presentHint").offsetWidth; $("presentHint").classList.add("show"); }
    setTimeout(() => g.resize(), 50);
  }
  $("sPresent").onclick = () => togglePresent(true);
  document.addEventListener("keydown", e => {
    if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.key === "p" || e.key === "P") { togglePresent(); e.preventDefault(); }
    else if (e.key === "Escape" && document.body.classList.contains("present")) togglePresent(false);
    else if (mode === "story" && (e.key === " " || e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      if (e.key === " ") $("sPlay").click(); else if (e.key === "ArrowRight") $("sNext").click(); else $("sPrev").click();
      e.preventDefault();
    }
  });

  // start: the story plays by itself, unless a link asked for a tab (demo.html#lab, #explore)
  story.go(0);
  const want = location.hash.replace("#", "");
  if (want === "lab" || want === "explore") selectTab(want);
  else setTimeout(() => { if (mode === "story") story.play(); }, 600);
})();
