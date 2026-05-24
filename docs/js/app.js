/**
 * music2vec — three.js scene + audio playback.
 *
 * 179 works rendered as glowing composer-colored spheres in 3D embedding
 * space. The scene swaps between three encoders (MERT-95M / MERT-330M /
 * MuQ-large) and three projections (PCA / t-SNE / UMAP); each combination
 * gets its own animated transition. Click a sphere → fly the camera, open
 * the detail panel, autoplay the 30 s audio preview.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ─── DOM refs ─────────────────────────────────────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const canvas      = $("#scene");
const boot        = $("#boot");
const tooltip     = $("#tooltip");
const tipSwatch   = $("#tip-swatch");
const tipName     = $("#tip-name");
const tipMeta     = $("#tip-meta");
const detail      = $("#detail");
const detailBadge = $("#detail-badge");
const detailComposer = $("#detail-composer");
const detailTitle = $("#detail-title");
const detailKvs   = $("#detail-kvs");
const detailSource= $("#detail-source");
const detailClose = $("#detail-close");
const filterCount = $("#filter-count");
const legend      = $("#legend");
const resetBtn    = $("#reset-view");
const fsBtn       = $("#fullscreen");
const clearBtn    = $("#clear-filters");
const searchInp   = $("#search");
const audioEl     = $("#audio");
const playBtn     = $("#play-btn");
const playIcon    = $("#play-icon");
const playerBar   = $("#player-bar");
const playerFill  = $("#player-fill");
const playerHandle= $("#player-handle");
const timeCur     = $("#time-cur");
const timeTot     = $("#time-tot");

// ─── Color tokens (mirror style.css) ──────────────────────────────────────
const COMPOSER_COLORS = {
  "Bach":      "#d4a256",
  "Beethoven": "#b73e3e",
  "Chopin":    "#d088c0",
  "Mozart":    "#5b8def",
  "Dvořák":    "#4a9d4a",
  "Vivaldi":   "#e07a2f",
  "Unknown":   "#8892a6",
};
const ERA_COLORS = {
  baroque:   "#d4a256",
  classical: "#5b8def",
  romantic:  "#d088c0",
};
const INSTRUMENT_COLORS = {
  solo_keyboard_harpsichord: "#d4a256",
  solo_keyboard_piano:       "#d088c0",
  unaccompanied_string:      "#6dd5ed",
  chamber_winds_strings:     "#7ad879",
  chamber_strings:           "#5b8def",
  orchestra:                 "#ff6a3d",
  string_quartet:            "#4a9d4a",
  choral:                    "#e7b417",
};
const SCHOOL_COLORS = {
  german_contrapuntal: "#d4a256",
  polish_romantic:    "#d088c0",
  czech_nationalist:  "#4a9d4a",
  italian_operatic:   "#e07a2f",
};
const DEVICE_COLORS = {
  fugue:       "#ffd166",
  passacaglia: "#d088c0",
};
const FALLBACK = "#aaaaaa";

const colorFor = (mode, w) => {
  if (mode === "composer") return COMPOSER_COLORS[w.composer] || FALLBACK;
  if (mode === "era") return ERA_COLORS[w.era] || FALLBACK;
  if (mode === "instrumentation") return INSTRUMENT_COLORS[w.tax?.instrumentation] || FALLBACK;
  if (mode === "national_school") return SCHOOL_COLORS[w.tax?.national_school] || FALLBACK;
  if (mode === "compositional_device") return DEVICE_COLORS[w.tax?.compositional_device] || FALLBACK;
  return FALLBACK;
};

const formatLabel = (v) => (v || "—").replaceAll("_", " ");
const initialsOf = (composer) => (composer || "?").charAt(0).toUpperCase();

// Format seconds → "M:SS"
const fmtTime = (s) => {
  if (!isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
};

// ─── State ────────────────────────────────────────────────────────────────
const DEFAULT_FILTERS = () => ({
  composer:             new Set(),
  era:                  new Set(),
  instrumentation:      new Set(),
  national_school:      new Set(),
  opus_cycle:           new Set(),
  compositional_device: new Set(),
  dance_type:           new Set(),
});

const state = {
  encoder: "audio_mert95",
  projection: "tsne3",
  colorBy: "composer",
  filters: DEFAULT_FILTERS(),
  query: "",
  data: null,
  works: [],
  byId: new Map(),
  visibleMask: [],
  hoveredIdx: -1,
  selectedIdx: -1,
  scene: null, camera: null, renderer: null, controls: null,
  raycaster: new THREE.Raycaster(),
  mouseNDC: new THREE.Vector2(),
  spheres: null,        // InstancedMesh
  haloSprites: [],      // per-sphere sprite for the glow
  sphereCount: 0,
  pulse: { idx: -1, phase: 0 },  // sphere being pulsed to indicate "playing"
};

// Sphere size in world units. We make the radius small so the screen-space
// footprint isn't crowded — but clickable via OffsetSphere bbox below.
const SPHERE_RADIUS = 0.07;
const SPREAD = 4.5;

// ─── Boot ─────────────────────────────────────────────────────────────────
async function boot_app(){
  const resp = await fetch("data/works.json");
  state.data = await resp.json();
  state.works = state.data.works;
  state.works.forEach((w) => state.byId.set(w.id, w));
  $("#hero-count").textContent = state.works.length;

  initScene();
  buildSpheres();
  buildFilters();
  buildLegend();
  wireToolbar();
  wireFilters();
  wireTour();
  wireSearch();
  wirePlayer();
  applyFilters();
  applyColors();

  requestAnimationFrame(() => boot.classList.add("gone"));
  setTimeout(() => boot.remove(), 700);
}

// ─── Scene ────────────────────────────────────────────────────────────────
function initScene(){
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth, H = wrap.clientHeight;

  state.scene = new THREE.Scene();
  state.scene.fog = new THREE.Fog(0x06070d, 20, 70);

  state.camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 200);
  state.camera.position.set(7, 5, 10);

  state.renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: false,
    powerPreference: "high-performance",
  });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.renderer.setSize(W, H, false);
  state.renderer.setClearColor(0x06070d, 1);
  state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.renderer.toneMappingExposure = 1.15;

  // Lights — sphere material needs lighting for the depth cue.
  state.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const key = new THREE.PointLight(0xffd166, 1.6, 60);
  key.position.set(8, 8, 6);
  state.scene.add(key);
  const fill = new THREE.PointLight(0xd088c0, 0.9, 60);
  fill.position.set(-6, -4, -3);
  state.scene.add(fill);

  // Star background — gives spatial depth cue.
  addStarfield();

  // Controls
  state.controls = new OrbitControls(state.camera, canvas);
  state.controls.enableDamping = true;
  state.controls.dampingFactor = 0.07;
  state.controls.rotateSpeed = 0.7;
  state.controls.minDistance = 1.5;
  state.controls.maxDistance = 50;
  state.controls.autoRotate = true;
  state.controls.autoRotateSpeed = 0.35;
  canvas.addEventListener("pointerdown", () => { state.controls.autoRotate = false; }, { once: true });
  canvas.addEventListener("wheel", () => { state.controls.autoRotate = false; }, { once: true, passive: true });

  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerleave", () => { hideTooltip(); state.hoveredIdx = -1; });
  canvas.addEventListener("click", onPointerClick);

  // Resize
  const ro = new ResizeObserver(() => {
    const W = wrap.clientWidth, H = wrap.clientHeight;
    state.renderer.setSize(W, H, false);
    state.camera.aspect = W/H;
    state.camera.updateProjectionMatrix();
  });
  ro.observe(wrap);

  // Buttons
  resetBtn.addEventListener("click", () => {
    state.filters = DEFAULT_FILTERS();
    syncChipsFromState();
    applyFilters();
    animateCamera(new THREE.Vector3(7, 5, 10), new THREE.Vector3(0, 0, 0), 1.2);
    state.controls.autoRotate = true;
  });
  detailClose.addEventListener("click", closeDetail);

  fsBtn.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else wrap.requestFullscreen().catch((e) => console.warn("fullscreen denied", e));
  });
  document.addEventListener("fullscreenchange", () => {
    fsBtn.classList.toggle("on", !!document.fullscreenElement);
  });

  // Scroll/blur kills the tooltip so it doesn't get pinned mid-scroll.
  const onScrollOrBlur = () => { if (!tooltip.hidden) hideTooltip(); state.hoveredIdx = -1; };
  window.addEventListener("scroll", onScrollOrBlur, { passive: true });
  window.addEventListener("blur", onScrollOrBlur);
  document.addEventListener("visibilitychange", () => { if (document.hidden) onScrollOrBlur(); });

  state.renderer.setAnimationLoop(tick);
}

function addStarfield(){
  // 1200 little point-stars in a sphere around the scene. Atmosphere only.
  const N = 1200;
  const positions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++){
    const r = 30 + Math.random() * 25;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i*3]   = r * Math.sin(phi) * Math.cos(theta);
    positions[i*3+1] = r * Math.cos(phi);
    positions[i*3+2] = r * Math.sin(phi) * Math.sin(theta);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xffffff, size: 0.04, sizeAttenuation: true,
    transparent: true, opacity: 0.55, depthWrite: false,
  });
  state.scene.add(new THREE.Points(geom, mat));
}

// ─── Build the InstancedMesh of spheres + per-instance halo sprites ──────
function buildSpheres(){
  const N = state.works.length;
  state.sphereCount = N;
  state.visibleMask = new Array(N).fill(true);

  const geom = new THREE.IcosahedronGeometry(SPHERE_RADIUS, 2);
  // PhongMaterial gives a soft specular highlight; flatShading=false for round.
  const mat = new THREE.MeshPhongMaterial({
    color: 0xffffff, shininess: 60, specular: 0x444444,
    transparent: true,
  });
  state.spheres = new THREE.InstancedMesh(geom, mat, N);
  state.spheres.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  state.spheres.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
  state.scene.add(state.spheres);

  // Glow halo: one Sprite per sphere, scaled & tinted to match. Single texture,
  // a radial-gradient sprite created procedurally so we don't need an external
  // asset.
  const haloTex = makeHaloTexture();
  state.haloSprites = [];
  const dummy = new THREE.Object3D();
  for (let i = 0; i < N; i++){
    const w = state.works[i];
    const c = (w.coords[state.encoder] || {})[state.projection] || [0,0,0];
    dummy.position.set(c[0]*SPREAD, c[1]*SPREAD, c[2]*SPREAD);
    dummy.updateMatrix();
    state.spheres.setMatrixAt(i, dummy.matrix);

    const mat2 = new THREE.SpriteMaterial({
      map: haloTex, color: 0xffffff,
      blending: THREE.AdditiveBlending,
      transparent: true, depthWrite: false, depthTest: true,
      opacity: 0.55,
    });
    const sprite = new THREE.Sprite(mat2);
    sprite.userData.index = i;
    sprite.position.copy(dummy.position);
    sprite.scale.set(0.45, 0.45, 1);
    state.scene.add(sprite);
    state.haloSprites.push(sprite);
  }
  state.spheres.instanceMatrix.needsUpdate = true;
}

function makeHaloTexture(){
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const grd = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.3, "rgba(255,255,255,0.5)");
  grd.addColorStop(0.7, "rgba(255,255,255,0.08)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ─── Animation loop ───────────────────────────────────────────────────────
function tick(){
  state.controls.update();
  updatePulse();
  updatePlayerProgress();
  state.renderer.render(state.scene, state.camera);
}

function updatePulse(){
  if (state.pulse.idx < 0) return;
  state.pulse.phase += 0.05;
  const halo = state.haloSprites[state.pulse.idx];
  if (halo){
    const s = 0.55 + Math.sin(state.pulse.phase) * 0.15;
    halo.scale.set(s, s, 1);
    halo.material.opacity = 0.7 + Math.sin(state.pulse.phase) * 0.25;
  }
}

// ─── Picking ──────────────────────────────────────────────────────────────
function onPointerMove(ev){
  const rect = canvas.getBoundingClientRect();
  state.mouseNDC.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  state.mouseNDC.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  state.raycaster.setFromCamera(state.mouseNDC, state.camera);

  // InstancedMesh raycast gives intersection.instanceId
  const hits = state.raycaster.intersectObject(state.spheres, false)
    .filter(h => state.visibleMask[h.instanceId]);

  if (hits.length){
    const idx = hits[0].instanceId;
    state.hoveredIdx = idx;
    showTooltip(idx, ev.clientX, ev.clientY);
  } else {
    hideTooltip();
    state.hoveredIdx = -1;
  }
}

function onPointerClick(){
  if (state.hoveredIdx >= 0){
    selectWork(state.hoveredIdx);
  }
}

function showTooltip(idx, x, y){
  const w = state.works[idx];
  const wrapRect = canvas.parentElement.getBoundingClientRect();
  const dx = x - wrapRect.left + 14;
  const dy = y - wrapRect.top + 14;
  tooltip.style.left = `${Math.min(dx, wrapRect.width - 300)}px`;
  tooltip.style.top  = `${Math.min(dy, wrapRect.height - 90)}px`;
  const swatch = COMPOSER_COLORS[w.composer] || FALLBACK;
  tipSwatch.style.background = swatch;
  tipSwatch.style.color = swatch;
  tipName.textContent = w.title || w.id;
  const meta = [];
  if (w.composer) meta.push(w.composer);
  if (w.year) meta.push(w.year);
  if (w.tax?.instrumentation) meta.push(formatLabel(w.tax.instrumentation));
  tipMeta.textContent = meta.join(" · ");
  tooltip.hidden = false;
}
function hideTooltip(){ tooltip.hidden = true; }

// ─── Selection / audio playback ───────────────────────────────────────────
function selectWork(idx){
  state.selectedIdx = idx;
  const w = state.works[idx];
  detail.hidden = false;

  const cColor = COMPOSER_COLORS[w.composer] || FALLBACK;
  detailBadge.textContent = initialsOf(w.composer);
  detailBadge.style.background = cColor;
  detailBadge.style.setProperty("--shadow-color", cColor);
  detailComposer.textContent = w.composer || "Unknown";

  detailTitle.textContent = w.title || w.id;

  const rows = [];
  if (w.year) rows.push(["year", w.year]);
  if (w.era) rows.push(["era", w.era]);
  if (w.instrumentation) rows.push(["instr.", w.instrumentation]);
  const taxKeys = ["instrumentation","national_school","opus_cycle","compositional_device","dance_type","sacred_function"];
  const tagged = [];
  for (const t of taxKeys){
    if (w.tax?.[t]) tagged.push(`<span class="tag">${formatLabel(w.tax[t])}</span>`);
  }
  if (tagged.length) rows.push(["labels", tagged.join("")]);
  detailKvs.innerHTML = rows.map(([k,v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");

  detailSource.href = w.source_url || "https://github.com/Rome-1/music2vec";

  // Audio: load + auto-play
  loadAndPlay(w);

  // Visual feedback
  state.pulse = { idx, phase: 0 };
  flyToWork(idx);
}

function closeDetail(){
  detail.hidden = true;
  state.selectedIdx = -1;
  state.pulse = { idx: -1, phase: 0 };
  // Reset all halo scales/opacities in case one was pulsing
  state.haloSprites.forEach((s) => { s.scale.set(0.45, 0.45, 1); s.material.opacity = 0.55; });
  if (!audioEl.paused) audioEl.pause();
  setPlayIcon(false);
}

function flyToWork(idx){
  const halo = state.haloSprites[idx];
  const target = halo.position.clone();
  const direction = state.camera.position.clone().sub(state.controls.target).normalize();
  const newCam = target.clone().add(direction.multiplyScalar(3.0));
  animateCamera(newCam, target, 0.9);
}

function animateCamera(camTo, targetTo, duration){
  const camFrom = state.camera.position.clone();
  const tgtFrom = state.controls.target.clone();
  const t0 = performance.now();
  function step(){
    const t = Math.min(1, (performance.now() - t0) / (duration * 1000));
    const e = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;
    state.camera.position.lerpVectors(camFrom, camTo, e);
    state.controls.target.lerpVectors(tgtFrom, targetTo, e);
    state.controls.update();
    if (t < 1) requestAnimationFrame(step);
  }
  step();
}

// ─── Player ───────────────────────────────────────────────────────────────
function wirePlayer(){
  playBtn.addEventListener("click", () => {
    if (audioEl.src && !audioEl.paused){
      audioEl.pause();
    } else if (audioEl.src){
      audioEl.play().catch(()=>{});
    }
  });
  audioEl.addEventListener("play",  () => setPlayIcon(true));
  audioEl.addEventListener("pause", () => setPlayIcon(false));
  audioEl.addEventListener("ended", () => setPlayIcon(false));
  audioEl.addEventListener("loadedmetadata", () => {
    timeTot.textContent = fmtTime(audioEl.duration);
  });

  // Click-to-seek on the bar.
  playerBar.addEventListener("click", (ev) => {
    if (!audioEl.duration) return;
    const r = playerBar.getBoundingClientRect();
    const ratio = (ev.clientX - r.left) / r.width;
    audioEl.currentTime = Math.max(0, Math.min(audioEl.duration, ratio * audioEl.duration));
  });
}

function loadAndPlay(w){
  playBtn.dataset.state = "loading";
  audioEl.src = w.audio;
  audioEl.currentTime = 0;
  audioEl.play().then(() => {
    playBtn.dataset.state = "playing";
  }).catch((e) => {
    // Autoplay may be blocked until user interacts. The play button still works.
    playBtn.dataset.state = "";
    console.warn("autoplay blocked, click play", e);
  });
}

function setPlayIcon(playing){
  if (playing) {
    // pause icon ⏸
    playIcon.setAttribute("d", "M6 4h4v16H6zM14 4h4v16h-4z");
    playBtn.setAttribute("aria-label", "pause");
  } else {
    // play icon ▶
    playIcon.setAttribute("d", "M8 5v14l11-7z");
    playBtn.setAttribute("aria-label", "play");
  }
}

function updatePlayerProgress(){
  if (!audioEl.duration || isNaN(audioEl.duration)) return;
  const ratio = audioEl.currentTime / audioEl.duration;
  const pct = (ratio * 100).toFixed(2) + "%";
  playerFill.style.width = pct;
  playerHandle.style.left = pct;
  timeCur.textContent = fmtTime(audioEl.currentTime);
}

// ─── Filters ──────────────────────────────────────────────────────────────
function buildFilters(){
  const m = state.data.meta;
  buildChips("composer",            m.composers,             (v) => COMPOSER_COLORS[v]   || FALLBACK);
  buildChips("era",                 m.eras,                  (v) => ERA_COLORS[v]        || FALLBACK);
  buildChips("instrumentation",     m.taxonomies.instrumentation || [],     (v) => INSTRUMENT_COLORS[v] || FALLBACK, formatLabel);
  buildChips("national_school",     m.taxonomies.national_school || [],     (v) => SCHOOL_COLORS[v]     || FALLBACK, formatLabel);
  buildChips("opus_cycle",          m.taxonomies.opus_cycle || [],          (_) => FALLBACK,           formatLabel);
  buildChips("compositional_device",m.taxonomies.compositional_device || [],(v) => DEVICE_COLORS[v]    || FALLBACK, formatLabel);
  buildChips("dance_type",          m.taxonomies.dance_type || [],          (_) => FALLBACK,           formatLabel);
}

function buildChips(filterKey, values, colorFn, labelFn = (v) => v){
  const container = document.querySelector(`.chips[data-filter="${filterKey}"]`);
  if (!container) return;
  container.innerHTML = "";
  const active = state.filters[filterKey];
  values.forEach(v => {
    const chip = document.createElement("button");
    chip.className = "chip" + (active.has(v) ? " on" : "");
    chip.dataset.filter = filterKey;
    chip.dataset.value = v;
    chip.innerHTML = `<span class="dot-color" style="background:${colorFn(v)}"></span>${labelFn(v)}`;
    chip.addEventListener("click", () => toggleFilter(filterKey, v, chip));
    container.appendChild(chip);
  });
  const cnt = document.getElementById("cnt-" + filterKey);
  if (cnt) cnt.textContent = values.length;
}

function toggleFilter(key, value, chipEl){
  const set = state.filters[key];
  if (set.has(value)){ set.delete(value); chipEl.classList.remove("on"); }
  else { set.add(value); chipEl.classList.add("on"); }
  applyFilters();
}

function applyFilters(){
  const f = state.filters;
  const q = state.query.trim().toLowerCase();
  let n = 0;
  state.works.forEach((w, i) => {
    let ok = true;
    if (f.composer.size && !f.composer.has(w.composer)) ok = false;
    if (ok && f.era.size && !f.era.has(w.era)) ok = false;
    for (const tax of ["instrumentation","national_school","opus_cycle","compositional_device","dance_type"]){
      if (ok && f[tax].size && !f[tax].has(w.tax?.[tax])) ok = false;
    }
    if (ok && q){
      const hay = `${w.title} ${w.composer} ${w.id}`.toLowerCase();
      if (!hay.includes(q)) ok = false;
    }
    state.visibleMask[i] = ok;
    if (ok) n++;
  });
  filterCount.textContent = n;
  refreshVisibility();
}

function refreshVisibility(){
  // Hide non-matching spheres by setting their instance scale to 0; halos
  // mirror visibility.
  const dummy = new THREE.Object3D();
  for (let i = 0; i < state.sphereCount; i++){
    const w = state.works[i];
    const c = (w.coords[state.encoder] || {})[state.projection] || [0,0,0];
    const visible = state.visibleMask[i];
    dummy.position.set(c[0]*SPREAD, c[1]*SPREAD, c[2]*SPREAD);
    const scale = visible ? (i === state.selectedIdx ? 1.6 : 1.0) : 0;
    dummy.scale.set(scale, scale, scale);
    dummy.updateMatrix();
    state.spheres.setMatrixAt(i, dummy.matrix);

    const halo = state.haloSprites[i];
    halo.position.copy(dummy.position);
    const hs = visible ? (i === state.selectedIdx ? 0.7 : 0.45) : 0;
    halo.scale.set(hs, hs, 1);
    halo.material.opacity = visible ? (state.selectedIdx < 0 ? 0.55 : (i === state.selectedIdx ? 0.85 : 0.25)) : 0;
  }
  state.spheres.instanceMatrix.needsUpdate = true;
}

function wireFilters(){
  clearBtn.addEventListener("click", () => {
    Object.values(state.filters).forEach(s => s.clear());
    $$(".chip.on").forEach(c => c.classList.remove("on"));
    searchInp.value = "";
    state.query = "";
    applyFilters();
  });
}

function syncChipsFromState(){
  $$(".chip").forEach(chip => {
    const k = chip.dataset.filter;
    const v = chip.dataset.value;
    if (!k) return;
    chip.classList.toggle("on", state.filters[k]?.has(v));
  });
  searchInp.value = state.query || "";
}

function wireSearch(){
  searchInp.addEventListener("input", (e) => {
    state.query = e.target.value;
    applyFilters();
  });
}

// ─── Toolbar (encoder + projection + color) ───────────────────────────────
function wireToolbar(){
  $$(".toolbar .seg").forEach(seg => {
    const ctrl = seg.dataset.control;
    seg.querySelectorAll(".seg-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        seg.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        if (ctrl === "encoder")    setEncoder(btn.dataset.value);
        if (ctrl === "projection") setProjection(btn.dataset.value);
        if (ctrl === "color")      setColorMode(btn.dataset.value);
      });
    });
  });
}

function setEncoder(enc){
  if (enc === state.encoder) return;
  state.encoder = enc;
  animatePositions();
}

function setProjection(p){
  if (p === state.projection) return;
  state.projection = p;
  animatePositions();
}

// Animate all spheres + halos to their new positions over ~900ms.
function animatePositions(){
  const N = state.sphereCount;
  const from = new Float32Array(N * 3);
  const to   = new Float32Array(N * 3);
  for (let i = 0; i < N; i++){
    const halo = state.haloSprites[i];
    from[i*3]   = halo.position.x;
    from[i*3+1] = halo.position.y;
    from[i*3+2] = halo.position.z;
    const c = (state.works[i].coords[state.encoder] || {})[state.projection] || [0,0,0];
    to[i*3]   = c[0] * SPREAD;
    to[i*3+1] = c[1] * SPREAD;
    to[i*3+2] = c[2] * SPREAD;
  }
  const t0 = performance.now();
  const D = 900;
  const dummy = new THREE.Object3D();
  function step(){
    const t = Math.min(1, (performance.now() - t0) / D);
    const e = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;
    for (let i = 0; i < N; i++){
      const x = from[i*3]   + (to[i*3]   - from[i*3]  ) * e;
      const y = from[i*3+1] + (to[i*3+1] - from[i*3+1]) * e;
      const z = from[i*3+2] + (to[i*3+2] - from[i*3+2]) * e;
      const halo = state.haloSprites[i];
      halo.position.set(x, y, z);
      const visible = state.visibleMask[i];
      const scale = visible ? (i === state.selectedIdx ? 1.6 : 1.0) : 0;
      dummy.position.set(x, y, z);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      state.spheres.setMatrixAt(i, dummy.matrix);
    }
    state.spheres.instanceMatrix.needsUpdate = true;
    if (t < 1) requestAnimationFrame(step);
  }
  step();
}

function setColorMode(mode){
  state.colorBy = mode;
  applyColors();
  buildLegend();
}

function applyColors(){
  const tmp = new THREE.Color();
  for (let i = 0; i < state.sphereCount; i++){
    const w = state.works[i];
    tmp.set(colorFor(state.colorBy, w));
    state.spheres.setColorAt(i, tmp);
    state.haloSprites[i].material.color.copy(tmp);
  }
  if (state.spheres.instanceColor) state.spheres.instanceColor.needsUpdate = true;
}

function buildLegend(){
  const items = legendItems(state.colorBy);
  legend.innerHTML = items.map(([label, color]) =>
    `<div class="row"><span class="swatch" style="background:${color}"></span>${label}</div>`
  ).join("");
}

function legendItems(mode){
  const m = state.data.meta;
  if (mode === "composer") return m.composers.map(c => [c, COMPOSER_COLORS[c] || FALLBACK]);
  if (mode === "era")      return m.eras.map(e => [e, ERA_COLORS[e] || FALLBACK]);
  if (mode === "instrumentation") return (m.taxonomies.instrumentation || []).map(v => [formatLabel(v), INSTRUMENT_COLORS[v] || FALLBACK]);
  if (mode === "national_school") return (m.taxonomies.national_school || []).map(v => [formatLabel(v), SCHOOL_COLORS[v] || FALLBACK]);
  if (mode === "compositional_device") return (m.taxonomies.compositional_device || []).map(v => [formatLabel(v), DEVICE_COLORS[v] || FALLBACK]);
  return [];
}

// ─── Tour ─────────────────────────────────────────────────────────────────
const TOURS = {
  fugue:      { filter: { compositional_device: ["fugue"] } },
  chopin:     { filter: { composer: ["Chopin"] } },
  instrument: { filter: {}, color: "instrumentation" },
  cycles:     { filter: { opus_cycle: ["bach_wtc_1", "bach_wtc_2"] } },
  encoder:    { filter: { compositional_device: ["fugue"] }, encoder: "audio_muq" },
};

function wireTour(){
  $$(".tour-card .tour-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.parentElement.dataset.tour;
      runTour(key);
    });
  });
}

function runTour(key){
  const t = TOURS[key];
  if (!t) return;
  Object.values(state.filters).forEach(s => s.clear());

  for (const [k, vals] of Object.entries(t.filter || {})){
    for (const v of vals){
      if (state.filters[k]) state.filters[k].add(v);
    }
  }
  syncChipsFromState();

  if (t.color){
    state.colorBy = t.color;
    $$(".seg[data-control=color] .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.value === t.color));
    applyColors();
    buildLegend();
  }
  if (t.encoder){
    state.encoder = t.encoder;
    $$(".seg[data-control=encoder] .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.value === t.encoder));
  }

  applyFilters();
  if (t.encoder) animatePositions();

  document.getElementById("viz").scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => {
    const idxs = state.visibleMask.map((b,i) => b ? i : -1).filter(i => i >= 0);
    if (!idxs.length) return;
    const centroid = new THREE.Vector3();
    idxs.forEach(i => centroid.add(state.haloSprites[i].position));
    centroid.divideScalar(idxs.length);
    let radius = 0;
    idxs.forEach(i => { radius = Math.max(radius, state.haloSprites[i].position.distanceTo(centroid)); });
    const dist = Math.max(radius * 2.4, 3.5);
    const dir = new THREE.Vector3(0.6, 0.6, 1).normalize();
    const camTo = centroid.clone().add(dir.multiplyScalar(dist));
    state.controls.autoRotate = false;
    animateCamera(camTo, centroid, 1.4);
  }, 450);
}

// ─── Go ────────────────────────────────────────────────────────────────────
boot_app().catch(err => {
  console.error(err);
  boot.querySelector(".boot-text").textContent = "failed to load — " + err.message;
});
