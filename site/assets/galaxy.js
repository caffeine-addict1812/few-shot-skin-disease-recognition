/* The embedding galaxy: all 6,584 SD-198 photos as points in a 3-D t-SNE of the model's 512-d embeddings.
 * WebGL (three.js) draws the points and morphs them on the GPU between the frozen-ImageNet layout (t = 0) and the
 * trained-model layout (t = 1). A screen-space overlay (2-D canvas + HTML) carries lines, rings, labels and photos.
 *   const g = new Galaxy(container, window.SHOWCASE.atlas, { autoRotate: true, interactive: true });
 */
(function () {
  "use strict";
  const SCALE = 10;                                   // world units for the [-1, 1] layout
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = x => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);

  const VERT = `
    attribute vec3 posA; attribute vec3 posB; attribute float kind; attribute float state; attribute float delay; attribute float phase;
    uniform float uT, uPx, uSeen, uUnseen, uDim, uFocus, uTime, uIntro;
    varying vec4 vColor;
    float easeIO(float x) { return x < 0.5 ? 4.0 * x * x * x : 1.0 - pow(-2.0 * x + 2.0, 3.0) / 2.0; }
    void main() {
      float k = easeIO(clamp((uT - delay * 0.4) / 0.6, 0.0, 1.0));          // the morph sweeps across the cloud
      float ki = clamp((uIntro - delay * 0.5) / 0.5, 0.0, 1.0); ki = 1.0 - pow(1.0 - ki, 3.0);
      vec3 p = mix(posA, posB, k) * mix(0.02, 1.0, ki);                      // intro: burst out from the centre
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_Position = projectionMatrix * mv;
      bool unseen = kind > 0.5;
      float a = unseen ? uUnseen : uSeen;
      vec3 c = unseen ? vec3(1.0, 0.66, 0.30) : vec3(0.58, 0.64, 0.82);
      float size = unseen ? 3.4 : 2.3;
      if (uFocus > 0.5) {
        if (state > 0.5) { a = 1.0; size *= (state > 1.5 ? 2.6 : 1.9); c = mix(c, vec3(1.0), state > 1.5 ? 0.15 : 0.0); }
        else { a *= uDim; }
      }
      size *= 1.0 + 0.16 * sin(uTime * 1.3 + phase);                          // gentle twinkle
      vColor = vec4(c, a * (unseen ? 1.0 : 0.62) * ki);
      gl_PointSize = size * uPx * (44.0 / -mv.z);
    }`;
  const FRAG = `
    varying vec4 vColor;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      float core = smoothstep(0.5, 0.0, d) * 0.55 + smoothstep(0.32, 0.12, d) * 0.6;
      gl_FragColor = vec4(vColor.rgb, vColor.a * core);
    }`;

  function Galaxy(container, A, opts) {
    opts = opts || {};
    const self = this;
    this.A = A;
    this.n = A.split.length;
    this.t = 1;
    this.interactive = !!opts.interactive;
    this.autoRotate = opts.autoRotate !== false && !reduceMotion;
    this.frameHooks = [];
    this.cinematic = opts.cinematic !== false;
    this.offsetX = opts.offsetX || null;          // shift the view centre right (px) to clear a docked panel

    // ---- renderer, scene, camera
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0e1322, 1);
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
    this.renderer = renderer; this.camera = camera; this.scene = scene;

    // overlay: 2-D canvas for lines + an HTML layer for rings, labels and photos
    const over = document.createElement("canvas");
    over.className = "g-overlay";
    container.appendChild(over);
    const html = document.createElement("div");
    html.className = "g-html";
    container.appendChild(html);
    this.overlay = over; this.octx = over.getContext("2d"); this.html = html;

    // ---- points
    const posA = new Float32Array(this.n * 3), posB = new Float32Array(this.n * 3);
    for (let i = 0; i < this.n * 3; i++) { posA[i] = A.frozen[i] / 1000 * SCALE; posB[i] = A.finetuned[i] / 1000 * SCALE; }
    this.posA = posA; this.posB = posB;
    const kind = new Float32Array(this.n), state = new Float32Array(this.n), delay = new Float32Array(this.n), phase = new Float32Array(this.n);
    let rs = 12345; const rnd = () => ((rs = (rs * 16807) % 2147483647) / 2147483647);   // deterministic jitter
    for (let i = 0; i < this.n; i++) {
      kind[i] = A.split[i] === 2 ? 1 : 0;
      delay[i] = Math.min(1, Math.max(0, 0.85 * (A.finetuned[3 * i] / 1000 + 1) / 2 + 0.15 * rnd()));
      phase[i] = rnd() * 6.2832;
    }
    this.delay = delay;
    this.state = state;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(posB.slice(), 3));   // only used for bounds
    geo.setAttribute("posA", new THREE.BufferAttribute(posA, 3));
    geo.setAttribute("posB", new THREE.BufferAttribute(posB, 3));
    geo.setAttribute("kind", new THREE.BufferAttribute(kind, 1));
    geo.setAttribute("state", new THREE.BufferAttribute(state, 1));
    geo.setAttribute("delay", new THREE.BufferAttribute(delay, 1));
    geo.setAttribute("phase", new THREE.BufferAttribute(phase, 1));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), SCALE * 2);
    this.uniforms = { uT: { value: 1 }, uPx: { value: renderer.getPixelRatio() }, uSeen: { value: 1 }, uUnseen: { value: 1 },
                      uDim: { value: 0.22 }, uFocus: { value: 0 }, uTime: { value: 0 }, uIntro: { value: 1 } };
    const mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
                                           transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    this.points = new THREE.Points(geo, mat);
    scene.add(this.points);
    this.geo = geo;

    // ---- orbit camera (spherical around a target), with fly-to tweens
    this.orbit = { theta: 0.7, phi: 1.15, r: 27, target: new THREE.Vector3() };
    this.home = { theta: 0.7, phi: 1.15, r: 27, target: new THREE.Vector3() };
    this.tweens = [];
    this._place();

    // ---- input
    if (this.interactive) {
      const el = renderer.domElement, ptrs = new Map();
      let lastPinch = 0, moved = 0;
      el.style.touchAction = "none";
      el.addEventListener("pointerdown", e => { ptrs.set(e.pointerId, [e.clientX, e.clientY]); moved = 0; el.setPointerCapture(e.pointerId); self.userActive(); });
      el.addEventListener("pointerup", e => {
        ptrs.delete(e.pointerId);
        if (moved < 6 && opts.onClick) { const r = el.getBoundingClientRect(); opts.onClick(self.pick(e.clientX - r.left, e.clientY - r.top), e); }
      });
      el.addEventListener("pointercancel", e => ptrs.delete(e.pointerId));
      el.addEventListener("pointermove", e => {
        const r = el.getBoundingClientRect();
        if (!ptrs.has(e.pointerId)) { if (opts.onHover) opts.onHover(self.pick(e.clientX - r.left, e.clientY - r.top), e); return; }
        const prev = ptrs.get(e.pointerId), dx = e.clientX - prev[0], dy = e.clientY - prev[1];
        ptrs.set(e.pointerId, [e.clientX, e.clientY]);
        moved += Math.abs(dx) + Math.abs(dy);
        if (ptrs.size === 1) {
          self.orbit.theta -= dx * 0.006;
          self.orbit.phi = Math.min(Math.PI - 0.12, Math.max(0.12, self.orbit.phi - dy * 0.006));
        } else if (ptrs.size === 2) {
          const [a, b] = [...ptrs.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (lastPinch) self.orbit.r = Math.min(70, Math.max(3, self.orbit.r * lastPinch / d));
          lastPinch = d;
        }
        self.tweens = [];
        if (opts.onHover) opts.onHover(-1, e);
      });
      el.addEventListener("pointerup", () => { lastPinch = 0; });
      el.addEventListener("wheel", e => { e.preventDefault(); self.userActive(); self.tweens = [];
        self.orbit.r = Math.min(70, Math.max(3, self.orbit.r * Math.exp(e.deltaY * 0.0012))); }, { passive: false });
      el.addEventListener("pointerleave", () => { if (opts.onHover) opts.onHover(-1); });
    }

    // ---- size + loop
    const ro = new ResizeObserver(() => self.resize());
    ro.observe(container);
    this.container = container;
    this.resize();
    let last = performance.now();
    const loop = now => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      self._tick(now, dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  Galaxy.prototype.userActive = function () { this.idleSince = performance.now(); this.autoRotatePaused = true; };
  Galaxy.prototype.resize = function () {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + "px"; this.renderer.domElement.style.height = h + "px";
    this.camera.aspect = w / Math.max(1, h);
    const off = this.offsetX ? this.offsetX(w) : 0;
    if (off) this.camera.setViewOffset(w, h, -off, 0, w, h); else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.overlay.width = Math.round(w * dpr); this.overlay.height = Math.round(h * dpr);
    this.overlay.style.width = w + "px"; this.overlay.style.height = h + "px";
    this.odpr = dpr;
  };
  Galaxy.prototype._place = function (now) {
    const o = this.orbit, c = this.camera;
    // cinematic drift: a slow vertical sway and a breathing dolly on top of the orbit (never accumulates)
    const cine = this.cinematic && !reduceMotion && !this.autoRotatePaused ? 1 : 0, ts = (now || 0) / 1000;
    const phi = Math.min(Math.PI - 0.12, Math.max(0.12, o.phi + cine * 0.07 * Math.sin(ts * 0.17)));
    const r = o.r * (1 + cine * 0.045 * Math.sin(ts * 0.11));
    c.position.set(o.target.x + r * Math.sin(phi) * Math.sin(o.theta), o.target.y + r * Math.cos(phi),
                   o.target.z + r * Math.sin(phi) * Math.cos(o.theta));
    c.lookAt(o.target);
    c.updateMatrixWorld();
  };
  Galaxy.prototype._tick = function (now, dt) {
    // tweens
    this.tweens = this.tweens.filter(tw => {
      const k = Math.min(1, (now - tw.start) / tw.dur), e = ease(k);
      tw.step(e);
      if (k >= 1 && tw.done) tw.done();
      return k < 1;
    });
    if (this.autoRotatePaused && now - (this.idleSince || 0) > 6000) this.autoRotatePaused = false;
    if (this.autoRotate && !this.autoRotatePaused) this.orbit.theta += dt * (this.rotSpeed || 0.06);
    this.uniforms.uT.value = this.t;          // the shader eases per point; worldPos() repeats the same maths
    this.uniforms.uTime.value = reduceMotion ? 0 : now / 1000;
    this._place(now);
    this.renderer.render(this.scene, this.camera);
    const ctx = this.octx;
    ctx.setTransform(this.odpr, 0, 0, this.odpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    for (const f of this.frameHooks) f(ctx, now);
  };

  // ---- morph
  Galaxy.prototype.setT = function (t) { this.t = Math.max(0, Math.min(1, t)); if (this.onT) this.onT(this.t); };
  Galaxy.prototype.animateT = function (to, ms, done) {
    const from = this.t, self = this;
    if (reduceMotion || ms <= 0) { this.setT(to); if (done) done(); return; }
    this.tweens = this.tweens.filter(t => t.kind !== "morph");
    this.tweens.push({ kind: "morph", start: performance.now(), dur: ms, step: e => self.setT(from + (to - from) * e), done });
  };

  // ---- camera
  Galaxy.prototype.flyTo = function (target, r, ms, done) {
    const o = this.orbit, t0 = o.target.clone(), r0 = o.r, self = this;
    if (reduceMotion || !ms) { o.target.copy(target); o.r = r; if (done) done(); return; }
    this.tweens = this.tweens.filter(t => t.kind !== "fly");
    this.tweens.push({ kind: "fly", start: performance.now(), dur: ms, done,
                       step: e => { o.target.lerpVectors(t0, target, e); o.r = r0 + (r - r0) * e; } });
    void self;
  };
  Galaxy.prototype.flyHome = function (ms, done) { this.flyTo(this.home.target.clone(), this.home.r, ms, done); };

  // ---- geometry helpers
  Galaxy.prototype.worldPos = function (i, out) {
    const d = this.delay[i], t = ease(Math.min(1, Math.max(0, (this.t - d * 0.4) / 0.6)));
    let ki = Math.min(1, Math.max(0, (this.uniforms.uIntro.value - d * 0.5) / 0.5)); ki = 1 - Math.pow(1 - ki, 3);
    const s = 0.02 + 0.98 * ki, a = this.posA, b = this.posB, j = 3 * i;
    out = out || new THREE.Vector3();
    return out.set((a[j] + (b[j] - a[j]) * t) * s, (a[j + 1] + (b[j + 1] - a[j + 1]) * t) * s, (a[j + 2] + (b[j + 2] - a[j + 2]) * t) * s);
  };
  Galaxy.prototype.mixXYZ = function (xyzA, xyzB, out) {
    const t = ease(this.t); out = out || new THREE.Vector3();
    return out.set((xyzA[0] + (xyzB[0] - xyzA[0]) * t) * SCALE, (xyzA[1] + (xyzB[1] - xyzA[1]) * t) * SCALE, (xyzA[2] + (xyzB[2] - xyzA[2]) * t) * SCALE);
  };
  Galaxy.prototype.project = function (v) {
    const p = v.clone().project(this.camera);
    return { x: (p.x + 1) / 2 * this.w, y: (1 - p.y) / 2 * this.h, z: p.z, visible: p.z < 1 && p.z > -1 };
  };
  Galaxy.prototype.centroid = function (ids) {
    const c = new THREE.Vector3(), v = new THREE.Vector3();
    ids.forEach(i => c.add(this.worldPos(i, v)));
    return c.multiplyScalar(1 / Math.max(1, ids.length));
  };
  Galaxy.prototype.radiusFor = function (ids, c) {
    let r = 0; const v = new THREE.Vector3();
    ids.forEach(i => { r = Math.max(r, this.worldPos(i, v).distanceTo(c)); });
    return Math.max(6, r * 2.6);
  };
  Galaxy.prototype.pick = function (mx, my) {
    let best = -1, bd = 12 * 12, bz = 2;
    const v = new THREE.Vector3(), s = this.uniforms;
    for (let i = 0; i < this.n; i++) {
      const unseen = this.A.split[i] === 2;
      if ((unseen ? s.uUnseen.value : s.uSeen.value) < 0.15) continue;
      if (s.uFocus.value > 0.5 && this.state[i] < 0.5 && s.uDim.value < 0.3) continue;
      this.worldPos(i, v).project(this.camera);
      if (v.z > 1) continue;
      const x = (v.x + 1) / 2 * this.w, y = (1 - v.y) / 2 * this.h, d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd || (d < 36 && v.z < bz)) { bd = d; best = i; bz = v.z; }
    }
    return best;
  };

  // ---- emphasis (all transitions fade)
  Galaxy.prototype.tweenUniform = function (name, to, ms) {
    const u = this.uniforms[name], from = u.value;
    this.tweens = this.tweens.filter(t => t.kind !== "u:" + name);
    if (reduceMotion || !ms) { u.value = to; return; }
    this.tweens.push({ kind: "u:" + name, start: performance.now(), dur: ms, step: e => { u.value = from + (to - from) * e; } });
  };
  Galaxy.prototype.setLayers = function (seen, unseen, ms) {
    this.tweenUniform("uSeen", seen, ms === undefined ? 900 : ms); this.tweenUniform("uUnseen", unseen, ms === undefined ? 900 : ms);
  };
  Galaxy.prototype.playIntro = function (ms) { this.uniforms.uIntro.value = 0; this.tweenUniform("uIntro", 1, ms || 2800); };
  Galaxy.prototype.setFocus = function (map /* {index: 1|2} or null */, dim) {
    this.state.fill(0);
    if (map) for (const [i, v] of map) this.state[i] = v;
    this.geo.attributes.state.needsUpdate = true;
    this.uniforms.uFocus.value = map ? 1 : 0;
    if (map) { this.uniforms.uDim.value = 1; this.tweenUniform("uDim", dim === undefined ? 0.22 : dim, 900); }   // others fade back
  };
  Galaxy.prototype.onFrame = function (f) { this.frameHooks.push(f); };

  window.Galaxy = Galaxy;
  window.Galaxy.SCALE = SCALE;
  window.Galaxy.ease = ease;
})();
