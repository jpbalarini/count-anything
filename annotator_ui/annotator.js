'use strict';

// Box colors, assigned to classes in order. Saturated hues that are
// rare in real footage come first (magenta, cyan), so the first classes
// stand out on most videos; neighbours differ in hue.
const PALETTE = [
  '#D946EF', '#06B6D4', '#F59E0B', '#10B981', '#F43F5E', '#6366F1',
  '#84CC16', '#F97316', '#0EA5E9', '#A855F7', '#14B8A6', '#EC4899',
];
const HANDLE = 5;        // resize handle half-size, screen px
const HIT_PAD = 4;       // extra grab margin around boxes, screen px
const DRAG_START = 4;    // pointer travel before a press becomes a drag
const SAVE_DELAY = 400;  // ms of quiet before a change is written
const FILL_IOU = 0.3;    // "fill from previous": overlap that counts as already there
const MAX_UNDO = 300;

const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const norm = (s) => s.trim().toLowerCase().replace(/[_-]/g, ' ');
const icon = (name) => `<svg class="ic"><use href="#i-${name}"/></svg>`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const S = {
  project: null,
  W: 0, H: 0,
  indices: [],            // frame indices in the file, ascending
  counts: new Map(),      // index -> box count, -1 = detection failed
  edited: new Set(),
  pos: -1,                // position of the current frame in `indices`
  cache: new Map(),       // index -> plain boxes (or null = failed)
  boxes: [],              // current frame: {id, label, confidence, box}
  failed: false,
  loaded: false,
  selected: null,
  hover: null,
  tool: 'box',
  classes: [],
  activeClass: null,
  hiddenClasses: new Set(),
  hideAll: false,
  showLabels: false,
  onion: false,
  minConf: 0,
  view: { scale: 1, tx: 0, ty: 0 },
  lockView: true,
  brightness: 1,
  contrast: 1,
  img: null,
  undo: [],
  redo: [],
  clipboard: null,
  filter: 'all',
  playing: false,
};
let nextId = 1;

// ------------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------------

const curIndex = () => S.indices[S.pos];
const withId = (b) => ({ id: nextId++, label: b.label, confidence: b.confidence ?? 1, box: [...b.box] });
const plain = (b) => ({ label: b.label, confidence: b.confidence, box: b.box.map((v) => Math.round(v * 100) / 100) });
const snapshot = () => S.boxes.filter((b) => b.label).map(plain);
const findBox = (id) => S.boxes.find((b) => b.id === id) || null;
const selectedBox = () => (S.selected == null ? null : findBox(S.selected));
const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);

function colorOf(label) {
  const i = S.classes.findIndex((c) => norm(c) === norm(label));
  return PALETTE[(i < 0 ? S.classes.length : i) % PALETTE.length];
}

function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / (area(a) + area(b) - inter);
}

function fmtTime(index) {
  const t = index / (S.project?.fps || 30);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
}

function isVisible(b) {
  if (b.id === S.selected) return true;
  if (S.hideAll) return false;
  if (S.hiddenClasses.has(norm(b.label))) return false;
  return (b.confidence ?? 1) >= S.minConf;
}

function pref(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(`annot.${key}`));
    localStorage.setItem(`annot.${key}`, JSON.stringify(value));
  } catch { /* storage unavailable */ }
  return null;
}

let toastTimer = null;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
}

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = await res.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch { /* not JSON */ }
    const err = new Error(String(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// Every request names the files this page was loaded for (the session).
const sq = () => `s=${S.project?.id ?? ''}`;

// ------------------------------------------------------------------
// Data: annotations, images, saving
// ------------------------------------------------------------------

async function getAnnotations(index) {
  if (S.cache.has(index)) return S.cache.get(index);
  const data = await api(`/api/frames/${index}?${sq()}`);
  if (!S.cache.has(index)) S.cache.set(index, data.boxes);
  return S.cache.get(index);
}

const imgCache = new Map();
function getImage(index) {
  let p = imgCache.get(index);
  if (p) {
    imgCache.delete(index);  // move to the end (LRU)
    imgCache.set(index, p);
    return p;
  }
  p = new Promise((resolve, reject) => {
    const im = new Image();
    im.decoding = 'async';
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`cannot load frame ${index}`));
    im.src = `/api/frames/${index}/image?${sq()}`;
  });
  p.catch(() => imgCache.delete(index));
  imgCache.set(index, p);
  while (imgCache.size > 40) imgCache.delete(imgCache.keys().next().value);
  return p;
}

let saveTimer = null;
let dirtyIndex = null;
let saveChain = Promise.resolve();
let inflight = 0;
let saveFailed = false;

function setStatus() {
  const el = $('#saveStatus');
  let state = 'saved', text = 'Saved';
  if (stale) { state = 'error'; text = 'Not saved — reload'; }
  else if (saveFailed) { state = 'error'; text = 'Not saved — retry'; }
  else if (inflight) { state = 'saving'; text = 'Saving…'; }
  else if (dirtyIndex !== null) { state = 'dirty'; text = 'Unsaved'; }
  el.dataset.state = state;
  el.querySelector('.txt').textContent = text;
}

// The current frame changed: update the cache and counters, and write
// it to the server shortly.
function markDirty() {
  const index = curIndex();
  const boxes = snapshot();
  S.cache.set(index, boxes);
  S.counts.set(index, boxes.length);
  S.failed = false;
  S.edited.add(index);
  if (dirtyIndex !== null && dirtyIndex !== index) flushSave();
  dirtyIndex = index;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSave, SAVE_DELAY);
  setStatus();
  updateThumb(index);
  updateBanner();
  drawTimeline();
  renderSidebar();
  requestDraw();
}

function flushSave() {
  clearTimeout(saveTimer);
  saveTimer = null;
  if (dirtyIndex === null) return saveChain;
  const index = dirtyIndex;
  dirtyIndex = null;
  inflight++;
  setStatus();
  saveChain = saveChain
    .then(() => api(`/api/frames/${index}?${sq()}`, { method: 'PUT', body: { boxes: S.cache.get(index) } }))
    .then(() => { saveFailed = false; })
    .catch((err) => {
      console.error(err);
      saveFailed = true;
      if (err.status === 409) { staleSession(); return; }
      if (dirtyIndex === null) dirtyIndex = index;
      else if (dirtyIndex !== index) setTimeout(() => retrySave(index), 0);
      clearTimeout(saveTimer);
      saveTimer = setTimeout(flushSave, 3000);
      toast(`Could not save frame #${index}, retrying…`);
    })
    .finally(() => { inflight--; setStatus(); });
  return saveChain;
}

// Another tab opened other files: this page can't save any more.
let stale = false;
function staleSession() {
  stale = true;
  clearTimeout(saveTimer);
  saveTimer = null;
  setStatus();
  const el = $('#banner');
  el.hidden = false;
  el.textContent = 'Other files were opened in another tab, so changes here can no longer be saved. Reload the page.';
}

function retrySave(index) {
  inflight++;
  saveChain = saveChain
    .then(() => api(`/api/frames/${index}?${sq()}`, { method: 'PUT', body: { boxes: S.cache.get(index) } }))
    .catch(() => toast(`Frame #${index} still not saved`))
    .finally(() => { inflight--; setStatus(); });
}

function saveClasses() {
  api(`/api/classes?${sq()}`, { method: 'PUT', body: { classes: S.classes } })
    .catch(() => toast('Could not save the class list'));
}

window.addEventListener('beforeunload', (e) => {
  if (dirtyIndex !== null || inflight) {
    flushSave();
    e.preventDefault();
  }
});

// ------------------------------------------------------------------
// Edits & history
// ------------------------------------------------------------------

// Record an edit of the current frame. `before` is a snapshot taken
// before it; consecutive edits with the same `tag` within a second are
// merged into one undo step (e.g. arrow-key nudges).
function commit(before, tag = null) {
  if (!S.loaded) {
    // The frame is still loading: drop the edit.
    S.boxes = before.map(withId);
    return;
  }
  const index = curIndex();
  const after = snapshot();
  const last = S.undo[S.undo.length - 1];
  const now = Date.now();
  if (tag && last && last.tag === tag && last.index === index && now - last.t < 1000) {
    last.after = after;
    last.t = now;
  } else {
    S.undo.push({ index, before, after, tag, t: now });
    if (S.undo.length > MAX_UNDO) S.undo.shift();
  }
  S.redo = [];
  markDirty();
  updateHistoryButtons();
}

async function stepHistory(from, to, key) {
  const entry = from.pop();
  if (!entry) return;
  if (entry.index !== curIndex()) {
    const pos = S.indices.indexOf(entry.index);
    await goTo(pos);
    if (!S.loaded || curIndex() !== entry.index) { from.push(entry); return; }
  }
  S.boxes = entry[key].map(withId);
  S.selected = null;
  to.push(entry);
  markDirty();
  updateHistoryButtons();
}
const undo = () => stepHistory(S.undo, S.redo, 'before');
const redo = () => stepHistory(S.redo, S.undo, 'after');

function updateHistoryButtons() {
  $('#toolUndo').disabled = !S.undo.length;
  $('#toolRedo').disabled = !S.redo.length;
}

function select(id) {
  S.selected = id;
  renderSidebar();
  requestDraw();
}

function deleteSelected() {
  const b = selectedBox();
  if (!b) return;
  const before = snapshot();
  S.boxes = S.boxes.filter((x) => x !== b);
  S.selected = null;
  closeEditor();
  if (b.label) commit(before);
}

function setLabel(b, label) {
  label = ensureClass(label);
  S.activeClass = label;
  if (b.label === label) { renderSidebar(); return; }
  const before = snapshot();
  b.label = label;
  commit(before);
}

function nudge(dx, dy) {
  const b = selectedBox();
  if (!b) return;
  const before = snapshot();
  moveBox(b, b.box, dx, dy);
  commit(before, `nudge-${b.id}`);
}

function moveBox(b, orig, dx, dy) {
  dx = clamp(dx, -orig[0], S.W - orig[2]);
  dy = clamp(dy, -orig[1], S.H - orig[3]);
  b.box = [orig[0] + dx, orig[1] + dy, orig[2] + dx, orig[3] + dy];
}

function addBoxes(list, { offset = 0 } = {}) {
  const before = snapshot();
  const added = list.map((src) => {
    const b = withId(src);
    if (offset) moveBox(b, b.box, offset, offset);
    return b;
  });
  S.boxes.push(...added);
  for (const b of added) ensureClass(b.label, false);
  if (added.length === 1) S.selected = added[0].id;
  commit(before);
  return added.length;
}

function clearFrame() {
  if (!S.boxes.length) { toast('No boxes on this frame'); return; }
  const before = snapshot();
  const n = S.boxes.length;
  S.boxes = [];
  S.selected = null;
  closeEditor();
  commit(before);
  toast(`Deleted ${n} box${n === 1 ? '' : 'es'} (⌘Z to undo)`);
}

async function previousBoxes() {
  if (S.pos <= 0) { toast('This is the first frame'); return null; }
  const prev = S.indices[S.pos - 1];
  try {
    return { index: prev, boxes: (await getAnnotations(prev)) || [] };
  } catch {
    toast('Could not load the previous frame');
    return null;
  }
}

// Add the previous frame's boxes that have no match here (same class,
// overlapping): the objects the detector missed on this frame.
async function fillFromPrevious() {
  const index = curIndex();
  const prev = await previousBoxes();
  if (!prev || curIndex() !== index) return;
  const missing = prev.boxes.filter((p) => !S.boxes.some(
    (b) => norm(b.label) === norm(p.label) && iou(b.box, p.box) >= FILL_IOU,
  ));
  if (!missing.length) { toast(`Nothing missing compared to frame #${prev.index}`); return; }
  addBoxes(missing.map((b) => ({ ...b, confidence: 1 })));
  S.selected = null;
  renderSidebar();
  toast(`Added ${missing.length} box${missing.length === 1 ? '' : 'es'} from frame #${prev.index}`);
}

async function replaceWithPrevious() {
  const index = curIndex();
  const prev = await previousBoxes();
  if (!prev || curIndex() !== index) return;
  const before = snapshot();
  S.boxes = prev.boxes.map(withId);
  S.selected = null;
  commit(before);
  toast(`Copied ${prev.boxes.length} boxes from frame #${prev.index}`);
}

function copyBoxes() {
  const b = selectedBox();
  const list = b ? [b] : S.boxes.filter(isVisible);
  if (!list.length) return;
  S.clipboard = { index: curIndex(), boxes: list.map(plain) };
  toast(`Copied ${list.length} box${list.length === 1 ? '' : 'es'}`);
}

function pasteBoxes() {
  if (!S.clipboard) return;
  const sameFrame = S.clipboard.index === curIndex();
  const n = addBoxes(S.clipboard.boxes, { offset: sameFrame ? 10 : 0 });
  toast(`Pasted ${n} box${n === 1 ? '' : 'es'}`);
}

function duplicateSelected() {
  const b = selectedBox();
  if (!b) { toast('Select a box first'); return; }
  addBoxes([plain(b)], { offset: 10 });
}

// ------------------------------------------------------------------
// Classes
// ------------------------------------------------------------------

// Return the existing spelling of `name` (classes that only differ in
// case or `_`/`-` are the same class), adding it if it's new.
function ensureClass(name, persist = true) {
  name = name.trim();
  const found = S.classes.find((c) => norm(c) === norm(name));
  if (found) return found;
  S.classes.push(name);
  if (persist) saveClasses();
  renderSidebar();
  return name;
}

function removeClass(name) {
  S.classes = S.classes.filter((c) => c !== name);
  if (S.activeClass === name) S.activeClass = S.classes[0] || null;
  api(`/api/classes?${sq()}`, { method: 'PUT', body: { classes: S.classes } })
    .then((r) => {
      // The server keeps classes that are still used on other frames.
      if (r.classes.some((c) => c === name)) toast(`"${name}" is still used on other frames`);
      S.classes = r.classes;
      renderSidebar();
    })
    .catch(() => toast('Could not save the class list'));
  renderSidebar();
}

// ------------------------------------------------------------------
// Navigation
// ------------------------------------------------------------------

let loadToken = 0;

async function goTo(pos) {
  if (!S.indices.length) return;
  pos = clamp(pos, 0, S.indices.length - 1);
  if (pos === S.pos && S.loaded) return;
  flushSave();
  closeEditor(true);
  cancelDrag();
  S.loaded = false;
  S.pos = pos;
  S.selected = null;
  S.hover = null;
  const index = curIndex();
  const token = ++loadToken;
  updateFrameChrome();
  scrollThumbIntoView();

  const spin = setTimeout(() => { $('#banner').hidden = false; $('#banner').textContent = 'Loading frame…'; }, 250);
  try {
    const [boxes, img] = await Promise.all([getAnnotations(index), getImage(index)]);
    if (token !== loadToken) return;
    S.failed = boxes === null;
    S.boxes = (boxes || []).map(withId);
    S.img = img;
    S.loaded = true;
    if (S.onion) await loadOnion();
    if (token !== loadToken) return;
  } catch (err) {
    if (token !== loadToken) return;
    console.error(err);
    toast(`Could not load frame #${index}`);
  } finally {
    clearTimeout(spin);
  }
  if (!S.lockView) fitView();
  updateBanner();
  renderSidebar();
  requestDraw();
  try { history.replaceState(null, '', `#f=${index}`); } catch { /* ignore */ }
  pref(`pos:${S.project.detections_path}`, index);
  for (const d of [1, -1, 2]) {
    const i = S.indices[pos + d];
    if (i !== undefined) { getImage(i).catch(() => {}); getAnnotations(i).catch(() => {}); }
  }
}

function navigable(pos) {
  return S.filter === 'all' || !thumbEls[pos]?.hidden || pos === S.pos;
}

function step(delta) {
  const dir = Math.sign(delta);
  let left = Math.abs(delta);
  let pos = S.pos;
  let target = S.pos;
  while (left > 0) {
    pos += dir;
    if (pos < 0 || pos >= S.indices.length) break;
    if (navigable(pos)) { target = pos; left--; }
  }
  if (target !== S.pos) goTo(target);
  return target !== S.pos;
}

let onionBoxes = null;
async function loadOnion() {
  onionBoxes = null;
  if (S.pos <= 0) return;
  try { onionBoxes = (await getAnnotations(S.indices[S.pos - 1])) || []; } catch { /* ignore */ }
}

async function togglePlay() {
  S.playing = !S.playing;
  $('#playBtn').innerHTML = icon(S.playing ? 'pause' : 'play');
  if (!S.playing) return;
  const meta = S.project.meta;
  const sampleFps = (S.project.fps || 30) / (meta.sample_step || 1);
  const delay = 1000 / clamp(sampleFps, 1, 12);
  while (S.playing) {
    const t0 = performance.now();
    const moved = step(1);
    if (!moved) break;
    await new Promise((r) => setTimeout(r, Math.max(0, delay - (performance.now() - t0))));
    while (S.playing && !S.loaded) await new Promise((r) => setTimeout(r, 20));
  }
  S.playing = false;
  $('#playBtn').innerHTML = icon('play');
}

// ------------------------------------------------------------------
// Canvas: view, drawing
// ------------------------------------------------------------------

const stage = $('#stage');
const canvas = $('#canvas');
const ctx = canvas.getContext('2d');
let dpr = window.devicePixelRatio || 1;
let drawQueued = false;

function resizeCanvas() {
  dpr = window.devicePixelRatio || 1;
  const r = stage.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  requestDraw();
  drawTimeline();
}

function fitScale() {
  const r = stage.getBoundingClientRect();
  const pad = { l: 24, r: 84, t: 74, b: 80 };
  return Math.max(0.01, Math.min((r.width - pad.l - pad.r) / S.W, (r.height - pad.t - pad.b) / S.H));
}

function fitView() {
  if (!S.W) return;
  const r = stage.getBoundingClientRect();
  const pad = { l: 24, r: 84, t: 74, b: 80 };
  const s = fitScale();
  S.view.scale = s;
  S.view.tx = pad.l + (r.width - pad.l - pad.r - S.W * s) / 2;
  S.view.ty = pad.t + (r.height - pad.t - pad.b - S.H * s) / 2;
  S.view.fitted = true;
  updateZoomLabel();
  requestDraw();
}

function zoomAt(factor, sp) {
  const v = S.view;
  const s = clamp(v.scale * factor, fitScale() * 0.25, 40);
  const f = s / v.scale;
  v.tx = sp.x - (sp.x - v.tx) * f;
  v.ty = sp.y - (sp.y - v.ty) * f;
  v.scale = s;
  v.fitted = false;
  updateZoomLabel();
  requestDraw();
}

function zoomCenter(factor) {
  const r = stage.getBoundingClientRect();
  zoomAt(factor, { x: r.width / 2, y: r.height / 2 });
}

function updateZoomLabel() {
  $('#zoomVal').textContent = `${Math.round(S.view.scale * 100)}%`;
}

const toImg = (sp) => ({ x: (sp.x - S.view.tx) / S.view.scale, y: (sp.y - S.view.ty) / S.view.scale });
const toScreen = (x, y) => ({ x: x * S.view.scale + S.view.tx, y: y * S.view.scale + S.view.ty });
const clampImg = (p) => ({ x: clamp(p.x, 0, S.W), y: clamp(p.y, 0, S.H) });

function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function textColor(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lum = 0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  return lum > 150 ? '#111827' : '#ffffff';
}

function handlePoints(box) {
  const [x1, y1, x2, y2] = box;
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  return {
    nw: [x1, y1], n: [mx, y1], ne: [x2, y1], e: [x2, my],
    se: [x2, y2], s: [mx, y2], sw: [x1, y2], w: [x1, my],
  };
}

function draw() {
  const r = stage.getBoundingClientRect();
  const v = S.view;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!S.W) return;

  // Image
  ctx.save();
  ctx.translate(v.tx, v.ty);
  ctx.scale(v.scale, v.scale);
  ctx.shadowColor = 'rgba(17,24,39,.18)';
  ctx.shadowBlur = 18 / v.scale;
  ctx.fillStyle = '#e5e7eb';
  ctx.fillRect(0, 0, S.W, S.H);
  ctx.shadowColor = 'transparent';
  if (S.img) {
    if (S.brightness !== 1 || S.contrast !== 1) ctx.filter = `brightness(${S.brightness}) contrast(${S.contrast})`;
    ctx.imageSmoothingEnabled = v.scale < 2;
    ctx.drawImage(S.img, 0, 0, S.W, S.H);
    ctx.filter = 'none';
  }
  ctx.restore();

  const sel = selectedBox();
  const editing = !$('#editor').hidden;
  const focus = sel != null;

  // Previous frame's boxes (onion skin)
  if (S.onion && onionBoxes) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    for (const b of onionBoxes) {
      const a = toScreen(b.box[0], b.box[1]);
      const c = toScreen(b.box[2], b.box[3]);
      ctx.strokeStyle = 'rgba(255,255,255,.95)';
      ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
      ctx.lineDashOffset = 4.5;
      ctx.strokeStyle = 'rgba(17,24,39,.75)';
      ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
      ctx.lineDashOffset = 0;
    }
    ctx.restore();
  }

  // Boxes, biggest first so small ones stay on top
  const visible = S.boxes.filter(isVisible).sort((a, b) => area(b.box) - area(a.box));
  for (const b of visible) {
    const color = colorOf(b.label || S.activeClass || '');
    const a = toScreen(b.box[0], b.box[1]);
    const c = toScreen(b.box[2], b.box[3]);
    const isSel = b.id === S.selected;
    const isHover = b.id === S.hover;
    const dim = focus && !isSel && !isHover;
    ctx.globalAlpha = dim ? (editing ? 0.25 : 0.45) : 1;
    ctx.fillStyle = hexA(color, isSel ? 0.22 : isHover ? 0.3 : 0.14);
    ctx.fillRect(a.x, a.y, c.x - a.x, c.y - a.y);
    ctx.lineWidth = isSel || isHover ? 2.5 : 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
  }
  ctx.globalAlpha = 1;

  // Labels: all of them (L), or the hovered / selected one
  ctx.font = '600 11px Inter, system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (const b of visible) {
    if (!(S.showLabels || b.id === S.hover || b.id === S.selected) || !b.label) continue;
    if (focus && S.showLabels && b.id !== S.selected && b.id !== S.hover) ctx.globalAlpha = 0.45;
    const color = colorOf(b.label);
    const a = toScreen(b.box[0], b.box[1]);
    const conf = (b.confidence ?? 1) < 1 ? ` ${b.confidence.toFixed(2)}` : '';
    const text = b.label + conf;
    const w = ctx.measureText(text).width + 10;
    const y = a.y >= 18 ? a.y - 18 : a.y;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(a.x - 1, y, w, 18, a.y >= 18 ? [4, 4, 4, 0] : [0, 0, 4, 4]);
    ctx.fill();
    ctx.fillStyle = textColor(color);
    ctx.fillText(text, a.x + 4, y + 9.5);
    ctx.globalAlpha = 1;
  }

  // Resize handles
  if (sel) {
    const color = colorOf(sel.label || S.activeClass || '');
    for (const [x, y] of Object.values(handlePoints(sel.box))) {
      const p = toScreen(x, y);
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fillRect(p.x - HANDLE, p.y - HANDLE, HANDLE * 2, HANDLE * 2);
      ctx.strokeRect(p.x - HANDLE, p.y - HANDLE, HANDLE * 2, HANDLE * 2);
    }
  }

  // Box being drawn
  if (drag?.mode === 'draw' && drag.p1) {
    const color = colorOf(S.activeClass || '');
    const a = toScreen(Math.min(drag.p0.x, drag.p1.x), Math.min(drag.p0.y, drag.p1.y));
    const c = toScreen(Math.max(drag.p0.x, drag.p1.x), Math.max(drag.p0.y, drag.p1.y));
    ctx.fillStyle = hexA(color, 0.18);
    ctx.fillRect(a.x, a.y, c.x - a.x, c.y - a.y);
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y);
    ctx.setLineDash([]);
    const sz = `${Math.round((c.x - a.x) / v.scale)} × ${Math.round((c.y - a.y) / v.scale)}`;
    ctx.font = '500 11px ui-monospace, Menlo, monospace';
    const w = ctx.measureText(sz).width + 10;
    ctx.fillStyle = 'rgba(17,24,39,.8)';
    ctx.beginPath();
    ctx.roundRect(c.x - w, c.y + 4, w, 18, 4);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(sz, c.x - w + 5, c.y + 13.5);
  }

  // Crosshair guides for the box tool
  if (S.tool === 'box' && mouse && !spaceDown && (!drag || drag.mode === 'draw') && !overBox) {
    const p = toImg(mouse);
    if (p.x >= 0 && p.y >= 0 && p.x <= S.W && p.y <= S.H) {
      const a = toScreen(0, 0), c = toScreen(S.W, S.H);
      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,.9)';
      ctx.beginPath();
      ctx.moveTo(mouse.x + 0.5, a.y); ctx.lineTo(mouse.x + 0.5, c.y);
      ctx.moveTo(a.x, mouse.y + 0.5); ctx.lineTo(c.x, mouse.y + 0.5);
      ctx.stroke();
      ctx.lineDashOffset = 5.5;
      ctx.strokeStyle = 'rgba(17,24,39,.55)';
      ctx.stroke();
      ctx.restore();
    }
  }

  if (editing) positionEditor();
}

// ------------------------------------------------------------------
// Canvas: hit testing & pointer
// ------------------------------------------------------------------

let drag = null;
let mouse = null;
let overBox = false;
let spaceDown = false;

function screenPt(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function handleAt(sp, b) {
  for (const [name, [x, y]] of Object.entries(handlePoints(b.box))) {
    const p = toScreen(x, y);
    if (Math.abs(p.x - sp.x) <= HANDLE + 3 && Math.abs(p.y - sp.y) <= HANDLE + 3) return name;
  }
  return null;
}

// Visible boxes under a screen point, smallest first.
function boxesAt(sp) {
  const p = toImg(sp);
  const pad = HIT_PAD / S.view.scale;
  return S.boxes
    .filter((b) => isVisible(b)
      && p.x >= b.box[0] - pad && p.x <= b.box[2] + pad
      && p.y >= b.box[1] - pad && p.y <= b.box[3] + pad)
    .sort((a, b) => area(a.box) - area(b.box));
}

const CURSORS = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' };

function updateHover(sp) {
  const sel = selectedBox();
  const h = sel && handleAt(sp, sel);
  const hits = boxesAt(sp);
  const top = hits[0] || null;
  const onSel = sel && hits.includes(sel);
  overBox = Boolean(h || top);
  const hoverId = h ? sel.id : onSel ? sel.id : top ? top.id : null;
  if (hoverId !== S.hover) { S.hover = hoverId; highlightLayer(); }
  let cursor = S.tool === 'box' ? 'crosshair' : 'grab';
  if (spaceDown) cursor = 'grab';
  else if (h) cursor = CURSORS[h];
  else if (onSel) cursor = 'move';
  else if (top) cursor = 'pointer';
  canvas.style.cursor = cursor;
  requestDraw();
}

function cancelDrag() {
  if (drag && (drag.mode === 'move' || drag.mode === 'resize')) {
    S.boxes = drag.before.map(withId);
    S.selected = null;
  }
  drag = null;
}

canvas.addEventListener('pointerdown', (e) => {
  if (e.button === 2 || !S.loaded) return;
  canvas.setPointerCapture(e.pointerId);
  closePopups();
  const sp = screenPt(e);
  const v = S.view;
  if (e.button === 1 || spaceDown) {
    drag = { mode: 'pan', sp, tx: v.tx, ty: v.ty, moved: false, clickBox: undefined };
    canvas.style.cursor = 'grabbing';
    return;
  }
  if (!$('#editor').hidden) closeEditor();
  const sel = selectedBox();
  const h = sel && handleAt(sp, sel);
  if (h) {
    drag = { mode: 'resize', handle: h, id: sel.id, sp, orig: [...sel.box], before: snapshot(), moved: false };
    return;
  }
  const hits = boxesAt(sp);
  if (sel && hits.includes(sel)) {
    drag = { mode: 'move', id: sel.id, sp, orig: [...sel.box], before: snapshot(), moved: false, hits };
    return;
  }
  const clickBox = hits[0]?.id ?? null;
  if (S.tool === 'pan') {
    drag = { mode: 'pan', sp, tx: v.tx, ty: v.ty, moved: false, clickBox };
    canvas.style.cursor = 'grabbing';
    return;
  }
  drag = { mode: 'pending', sp, p0: clampImg(toImg(sp)), clickBox };
});

canvas.addEventListener('pointermove', (e) => {
  const sp = screenPt(e);
  mouse = sp;
  if (!drag) { updateHover(sp); return; }
  const dx = sp.x - drag.sp.x;
  const dy = sp.y - drag.sp.y;
  const far = Math.hypot(dx, dy) > DRAG_START;
  const s = S.view.scale;
  switch (drag.mode) {
    case 'pan':
      if (far) drag.moved = true;
      if (drag.moved) {
        S.view.tx = drag.tx + dx;
        S.view.ty = drag.ty + dy;
        S.view.fitted = false;
        requestDraw();
      }
      break;
    case 'pending':
      if (!far) break;
      drag.mode = 'draw';
      // fall through
    case 'draw':
      drag.p1 = clampImg(toImg(sp));
      requestDraw();
      break;
    case 'move': {
      if (far) drag.moved = true;
      if (!drag.moved) break;
      const b = findBox(drag.id);
      if (b) moveBox(b, drag.orig, dx / s, dy / s);
      requestDraw();
      break;
    }
    case 'resize': {
      drag.moved = true;
      const b = findBox(drag.id);
      if (!b) break;
      let [x1, y1, x2, y2] = drag.orig;
      const h = drag.handle;
      const px = clamp(toImg(sp).x, 0, S.W), py = clamp(toImg(sp).y, 0, S.H);
      if (h.includes('w')) x1 = px;
      if (h.includes('e')) x2 = px;
      if (h.includes('n')) y1 = py;
      if (h.includes('s')) y2 = py;
      b.box = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
      requestDraw();
      break;
    }
  }
});

function endDrag() {
  const d = drag;
  drag = null;
  if (!d) return;
  switch (d.mode) {
    case 'pan':
      if (!d.moved && d.clickBox !== undefined) select(d.clickBox);
      break;
    case 'pending':
      select(d.clickBox);
      break;
    case 'draw':
      finishDraw(d);
      break;
    case 'move':
      if (d.moved) {
        commit(d.before);
      } else if (d.hits.length > 1) {
        // Click on the selected box: cycle through the boxes under the cursor.
        const i = d.hits.findIndex((b) => b.id === d.id);
        select(d.hits[(i + 1) % d.hits.length].id);
      }
      break;
    case 'resize': {
      const b = findBox(d.id);
      if (!d.moved || !b) break;
      if ((b.box[2] - b.box[0]) * S.view.scale < 2 || (b.box[3] - b.box[1]) * S.view.scale < 2) {
        b.box = d.orig;  // collapsed to nothing: undo the resize
        break;
      }
      commit(d.before);
      break;
    }
  }
  if (mouse) updateHover(mouse);
  requestDraw();
}

canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', () => { cancelDrag(); requestDraw(); });
canvas.addEventListener('pointerleave', () => {
  mouse = null;
  if (!drag && S.hover !== null) { S.hover = null; highlightLayer(); }
  requestDraw();
});

function finishDraw(d) {
  const x1 = Math.min(d.p0.x, d.p1.x), x2 = Math.max(d.p0.x, d.p1.x);
  const y1 = Math.min(d.p0.y, d.p1.y), y2 = Math.max(d.p0.y, d.p1.y);
  if ((x2 - x1) * S.view.scale < 3 || (y2 - y1) * S.view.scale < 3) {
    select(d.clickBox);
    return;
  }
  const before = snapshot();
  const label = S.activeClass || '';
  const b = { id: nextId++, label, confidence: 1, box: [x1, y1, x2, y2] };
  S.boxes.push(b);
  S.selected = b.id;
  if (label) {
    commit(before);
    if (S.classes.length > 1) openEditor(b.id);
  } else {
    // No class yet: the box only exists once it gets a name.
    renderSidebar();
    openEditor(b.id, before);
  }
}

canvas.addEventListener('dblclick', (e) => {
  const hit = boxesAt(screenPt(e))[0];
  if (hit) { select(hit.id); openEditor(hit.id); }
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const sp = screenPt(e);
  const k = e.ctrlKey ? 0.01 : 0.0015;  // pinch gestures report small deltas
  const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
  zoomAt(Math.exp(-dy * k), sp);
}, { passive: false });

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// ------------------------------------------------------------------
// Annotation editor
// ------------------------------------------------------------------

const editor = { id: null, before: null, items: [], hi: 0, fresh: true };

function openEditor(id, before = null) {
  const b = findBox(id);
  if (!b) return;
  editor.id = id;
  editor.before = before;
  editor.fresh = true;
  $('#editor').hidden = false;
  const input = $('#editorInput');
  input.value = b.label;
  renderEditorList();
  positionEditor();
  input.focus();
  input.select();
  requestDraw();
}

// `discard`: the frame is changing, drop an unnamed box silently.
function closeEditor(discard = false) {
  if ($('#editor').hidden) return;
  $('#editor').hidden = true;
  const b = findBox(editor.id);
  if (b && !b.label) {
    S.boxes = S.boxes.filter((x) => x !== b);
    if (S.selected === b.id) S.selected = null;
  }
  editor.id = null;
  editor.before = null;
  if (document.activeElement === $('#editorInput')) $('#editorInput').blur();
  if (!discard) { renderSidebar(); requestDraw(); }
}

function positionEditor() {
  const b = findBox(editor.id);
  const el = $('#editor');
  if (!b) return;
  const r = stage.getBoundingClientRect();
  const w = el.offsetWidth, h = el.offsetHeight;
  const a = toScreen(b.box[0], b.box[1]);
  const c = toScreen(b.box[2], b.box[3]);
  let x = c.x + 14;
  if (x + w > r.width - 76) x = a.x - w - 14;
  if (x < 12) x = clamp(c.x + 14, 12, r.width - w - 76);
  const y = clamp(a.y, 12, Math.max(12, r.height - h - 12));
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
}

function renderEditorList() {
  const q = $('#editorInput').value.trim();
  const nq = norm(q);
  const b = findBox(editor.id);
  const filter = editor.fresh || !q;
  const matches = S.classes.filter((c) => filter || norm(c).includes(nq));
  const known = new Set(S.classes.map(norm));
  const extra = (!filter && q)
    ? (S.project.suggestions || []).filter((c) => !known.has(norm(c)) && norm(c).includes(nq)).slice(0, 8)
    : [];
  // Existing classes first, then COCO names, then creating what was typed.
  editor.items = matches.map((name) => ({ name }));
  const firstExtra = editor.items.length;
  editor.items.push(...extra.map((name) => ({ name, suggestion: true })));
  if (q && !known.has(nq) && !extra.some((c) => norm(c) === nq)) editor.items.push({ name: q, create: true });

  const cur = b ? norm(b.label) : '';
  const exact = editor.items.findIndex((it) => !it.create && norm(it.name) === (filter ? cur : nq));
  editor.hi = exact >= 0 ? exact : 0;

  const ul = $('#editorList');
  ul.innerHTML = '';
  editor.items.forEach((it, i) => {
    if (i === firstExtra && extra.length) {
      const sec = document.createElement('li');
      sec.className = 'section';
      sec.textContent = 'COCO classes';
      ul.append(sec);
    }
    const li = document.createElement('li');
    li.dataset.i = i;
    const n = S.classes.indexOf(it.name);
    const key = !it.create && n >= 0 && n < 9 ? `<span class="num">${n + 1}</span>` : '<span class="num">+</span>';
    const label = it.create ? `Create “${esc(it.name)}”` : esc(it.name);
    li.innerHTML = `${key}<span class="cname">${label}</span><span class="dot" style="background:${colorOf(it.name)}"></span>`;
    li.addEventListener('mousedown', (e) => { e.preventDefault(); applyEditor(it.name); });
    li.addEventListener('mousemove', () => { if (editor.hi !== i) { editor.hi = i; highlightEditorItem(); } });
    ul.append(li);
  });
  highlightEditorItem();
}

function highlightEditorItem() {
  for (const li of $('#editorList').querySelectorAll('li[data-i]')) {
    const on = Number(li.dataset.i) === editor.hi;
    li.classList.toggle('hi', on);
    const dot = li.querySelector('.dot');
    if (dot) dot.classList.toggle('ring', on);
    if (on) li.scrollIntoView({ block: 'nearest' });
  }
}

function applyEditor(name) {
  const b = findBox(editor.id);
  name = (name ?? '').trim();
  if (!b || !name) return;
  if (!b.label) {
    // New box that had no class: add it and the class in one undo step.
    b.label = ensureClass(name);
    S.activeClass = b.label;
    commit(editor.before || []);
  } else {
    setLabel(b, name);
  }
  closeEditor();
}

$('#editorInput').addEventListener('input', () => { editor.fresh = false; renderEditorList(); });
$('#editorInput').addEventListener('keydown', (e) => {
  const input = e.target;
  const allSelected = input.selectionStart === 0 && input.selectionEnd === input.value.length;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!editor.items.length) return;
    editor.hi = (editor.hi + (e.key === 'ArrowDown' ? 1 : -1) + editor.items.length) % editor.items.length;
    highlightEditorItem();
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const it = editor.items[editor.hi];
    applyEditor(it ? it.name : input.value);
  } else if (e.key === 'Escape') {
    e.preventDefault();
    closeEditor();
  } else if (/^[1-9]$/.test(e.key) && (allSelected || !input.value) && S.classes[Number(e.key) - 1]) {
    e.preventDefault();
    applyEditor(S.classes[Number(e.key) - 1]);
  }
  e.stopPropagation();
});
$('#editorSave').addEventListener('click', () => {
  const it = editor.items[editor.hi];
  applyEditor(it ? it.name : $('#editorInput').value);
});
$('#editorDelete').addEventListener('click', () => {
  const b = findBox(editor.id);
  if (!b) return;
  S.selected = b.id;
  deleteSelected();
});
$('#editorClose').addEventListener('click', () => closeEditor());

// ------------------------------------------------------------------
// Sidebars
// ------------------------------------------------------------------

function renderSidebar() {
  if (!S.project) return;
  const counts = new Map();
  for (const b of S.boxes) if (b.label) counts.set(norm(b.label), (counts.get(norm(b.label)) || 0) + 1);
  const total = S.boxes.filter((b) => b.label).length;
  $('#annCount').textContent = total;

  const classList = $('#classList');
  const unusedList = $('#unusedList');
  classList.innerHTML = '';
  unusedList.innerHTML = '';
  S.classes.forEach((name, i) => {
    const n = counts.get(norm(name)) || 0;
    const hidden = S.hiddenClasses.has(norm(name));
    const li = document.createElement('li');
    li.className = 'class-row';
    li.classList.toggle('active', name === S.activeClass);
    li.classList.toggle('hidden-class', hidden);
    li.title = name === S.activeClass ? 'New boxes get this class' : `Use for new boxes${i < 9 ? ` (${i + 1})` : ''}`;
    const tag = name === S.activeClass ? '<span class="active-tag">new boxes</span>' : '';
    if (n) {
      li.innerHTML = `<span class="dot" style="background:${colorOf(name)}"></span><span class="cname">${esc(name)}</span>${tag}`
        + `<span class="row-actions"><button class="icon-btn" data-act="eye" title="${hidden ? 'Show' : 'Hide'}">${icon(hidden ? 'eye-off' : 'eye')}</button>`
        + `<button class="icon-btn" data-act="del" title="Delete all ${esc(name)} boxes on this frame">${icon('trash')}</button></span>`
        + `<span class="num hide-on-hover">${n}</span>`;
      classList.append(li);
    } else {
      li.innerHTML = `<span class="dot" style="background:${colorOf(name)}"></span><span class="cname">${esc(name)}</span>${tag}`
        + `<span class="row-actions"><button class="icon-btn" data-act="remove" title="Remove class">${icon('x')}</button></span>`;
      unusedList.append(li);
    }
    li.addEventListener('click', (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'eye') {
        if (hidden) S.hiddenClasses.delete(norm(name)); else S.hiddenClasses.add(norm(name));
        const sel = selectedBox();
        if (sel && !hidden && norm(sel.label) === norm(name)) S.selected = null;
        requestDraw();
      } else if (act === 'del') {
        const before = snapshot();
        S.boxes = S.boxes.filter((b) => norm(b.label) !== norm(name));
        S.selected = null;
        commit(before);
        toast(`Deleted ${n} ${name} box${n === 1 ? '' : 'es'} (⌘Z to undo)`);
        return;
      } else if (act === 'remove') {
        removeClass(name);
        return;
      } else {
        S.activeClass = name;
      }
      renderSidebar();
    });
  });

  renderLayers();
  renderConf();

  // Frame facts
  const index = curIndex();
  $('#factIndex').textContent = index ?? '–';
  $('#factTime').textContent = index != null ? fmtTime(index) : '–';
  $('#factSample').textContent = `${S.pos + 1} of ${S.indices.length}`;
  const st = S.failed ? ['failed', 'Detection failed'] : S.edited.has(index) ? ['edited', 'Edited'] : ['original', 'As detected'];
  $('#factStatus').innerHTML = `<span class="st ${st[0]}">${st[1]}</span>`;
  $('#fillPrevBtn').disabled = S.pos <= 0;
  $('#toolFill').disabled = S.pos <= 0;
  $('#toolDup').disabled = !selectedBox();
}

function renderLayers() {
  const ul = $('#layerList');
  ul.innerHTML = '';
  const list = S.boxes.filter((b) => b.label);
  list.forEach((b, i) => {
    const li = document.createElement('li');
    li.className = 'layer-row';
    li.dataset.id = b.id;
    li.classList.toggle('selected', b.id === S.selected);
    li.classList.toggle('dim', !isVisible(b));
    const w = Math.round(b.box[2] - b.box[0]), h = Math.round(b.box[3] - b.box[1]);
    const conf = (b.confidence ?? 1) < 1 ? ` · ${b.confidence.toFixed(2)}` : '';
    li.innerHTML = `<span class="dot" style="background:${colorOf(b.label)}"></span>`
      + `<span class="cname">${esc(b.label)} <span class="meta">#${i + 1} · ${w}×${h}${conf}</span></span>`
      + `<span class="row-actions"><button class="icon-btn" data-act="del" title="Delete">${icon('trash')}</button></span>`;
    li.addEventListener('click', (e) => {
      if (e.target.closest('[data-act="del"]')) { S.selected = b.id; deleteSelected(); return; }
      select(b.id);
    });
    li.addEventListener('mouseenter', () => { S.hover = b.id; requestDraw(); });
    li.addEventListener('mouseleave', () => { if (S.hover === b.id) { S.hover = null; requestDraw(); } });
    ul.append(li);
  });
  if (!list.length) ul.innerHTML = '<li class="layer-row" style="cursor:default;color:var(--faint)">No boxes on this frame</li>';
}

function highlightLayer() {
  for (const li of $('#layerList').querySelectorAll('.layer-row[data-id]')) {
    li.style.background = Number(li.dataset.id) === S.hover && Number(li.dataset.id) !== S.selected ? 'var(--line-2)' : '';
  }
}

function renderConf() {
  const scored = S.project.meta.detector === 'rfdetr' || S.boxes.some((b) => (b.confidence ?? 1) < 1);
  $('#confPanel').hidden = !scored;
  if (!scored) return;
  const below = S.boxes.filter((b) => b.label && (b.confidence ?? 1) < S.minConf).length;
  const btn = $('#confDelete');
  btn.disabled = !below;
  btn.textContent = below ? `Delete ${below} hidden box${below === 1 ? '' : 'es'} on this frame` : 'Delete hidden boxes on this frame';
}

function updateBanner() {
  if (stale) return;
  const el = $('#banner');
  el.hidden = !S.failed;
  el.textContent = 'Detection failed on this frame. Boxes you add here are saved like any other frame.';
}

// ------------------------------------------------------------------
// Frames panel (thumbnails), pager, timeline
// ------------------------------------------------------------------

let thumbEls = [];

function buildThumbs() {
  const wrap = $('#thumbs');
  wrap.innerHTML = '';
  const ratio = `${S.W} / ${S.H}`;
  thumbEls = S.indices.map((index, pos) => {
    const el = document.createElement('button');
    el.className = 'thumb';
    el.dataset.pos = pos;
    el.innerHTML = `<span class="thumb-img" style="aspect-ratio:${ratio}">`
      + `<img loading="lazy" decoding="async" alt="" src="/api/frames/${index}/thumbnail?${sq()}">`
      + `<span class="thumb-badge"></span><span class="thumb-check">${icon('check')}</span></span>`
      + `<span class="thumb-label">#${index} · ${fmtTime(index)}</span>`;
    el.addEventListener('click', () => goTo(pos));
    wrap.append(el);
    return el;
  });
  S.indices.forEach(updateThumb);
}

function updateThumb(index) {
  const pos = S.indices.indexOf(index);
  const el = thumbEls[pos];
  if (!el) return;
  const n = S.counts.get(index);
  const badge = el.querySelector('.thumb-badge');
  badge.textContent = n < 0 ? 'failed' : `${n} box${n === 1 ? '' : 'es'}`;
  badge.classList.toggle('failed', n < 0);
  el.classList.toggle('edited', S.edited.has(index));
}

function applyFilter() {
  S.filter = $('#filterSelect').value;
  let shown = 0;
  S.indices.forEach((index, pos) => {
    const n = S.counts.get(index);
    const ok = {
      all: true,
      edited: S.edited.has(index),
      unedited: !S.edited.has(index),
      empty: n === 0,
      failed: n < 0,
    }[S.filter];
    thumbEls[pos].hidden = !ok;
    if (ok) shown++;
  });
  if (!shown) toast('No frames match this filter');
  drawTimeline();
  scrollThumbIntoView();
}

function updateFrameChrome() {
  thumbEls.forEach((el, pos) => el.classList.toggle('current', pos === S.pos));
  $('#pageInput').value = S.pos + 1;
  $('#pageTotal').textContent = S.indices.length;
  $('#prevBtn').disabled = S.pos <= 0;
  $('#nextBtn').disabled = S.pos >= S.indices.length - 1;
  drawTimeline();
}

function scrollThumbIntoView() {
  const el = thumbEls[S.pos];
  if (el && !el.hidden) el.scrollIntoView({ block: 'nearest' });
}

const cssVars = {};
const cssVar = (name) => (cssVars[name] ??= getComputedStyle(document.documentElement).getPropertyValue(name).trim());

const tl = $('#timeline');
const tctx = tl.getContext('2d');

function drawTimeline() {
  if (!S.indices.length) return;
  const r = tl.getBoundingClientRect();
  if (!r.width) return;
  const d = window.devicePixelRatio || 1;
  if (tl.width !== Math.round(r.width * d)) { tl.width = Math.round(r.width * d); tl.height = Math.round(r.height * d); }
  tctx.setTransform(d, 0, 0, d, 0, 0);
  tctx.clearRect(0, 0, r.width, r.height);
  const n = S.indices.length;
  const max = Math.max(1, ...S.counts.values());
  const bw = r.width / n;
  const base = r.height - 4;
  tctx.fillStyle = '#f3f4f6';
  tctx.fillRect(0, base, r.width, 2);
  S.indices.forEach((index, pos) => {
    const c = S.counts.get(index);
    const x = pos * bw;
    const dim = S.filter !== 'all' && thumbEls[pos]?.hidden;
    tctx.globalAlpha = dim ? 0.25 : 1;
    if (c < 0) {
      tctx.fillStyle = '#e11d48';
      tctx.fillRect(x, base - 10, Math.max(1, bw - (bw > 3 ? 1 : 0)), 10);
    } else {
      const h = Math.max(c ? 2 : 0, (c / max) * (base - 6));
      tctx.fillStyle = S.edited.has(index) ? cssVar('--accent') : cssVar('--accent-line');
      tctx.fillRect(x, base - h, Math.max(1, bw - (bw > 3 ? 1 : 0)), h);
    }
  });
  tctx.globalAlpha = 1;
  if (S.pos >= 0) {
    const x = (S.pos + 0.5) * bw;
    tctx.fillStyle = '#111827';
    tctx.fillRect(Math.round(x) - 1, 0, 2, r.height);
    tctx.beginPath();
    tctx.moveTo(x - 5, 0); tctx.lineTo(x + 5, 0); tctx.lineTo(x, 6);
    tctx.fill();
  }
}

function tlPos(e) {
  const r = tl.getBoundingClientRect();
  return clamp(Math.floor(((e.clientX - r.left) / r.width) * S.indices.length), 0, S.indices.length - 1);
}
let tlDrag = false;
tl.addEventListener('pointerdown', (e) => { tlDrag = true; tl.setPointerCapture(e.pointerId); goTo(tlPos(e)); });
tl.addEventListener('pointermove', (e) => {
  const pos = tlPos(e);
  const index = S.indices[pos];
  const c = S.counts.get(index);
  const tip = $('#tlTip');
  tip.hidden = false;
  tip.textContent = `#${index} · ${fmtTime(index)} · ${c < 0 ? 'failed' : `${c} boxes`}${S.edited.has(index) ? ' · edited' : ''}`;
  const r = tl.getBoundingClientRect();
  tip.style.left = `${clamp(e.clientX - r.left, 60, r.width - 60)}px`;
  if (tlDrag && pos !== S.pos) goTo(pos);
});
tl.addEventListener('pointerup', () => { tlDrag = false; });
tl.addEventListener('pointerleave', () => { $('#tlTip').hidden = true; });

// ------------------------------------------------------------------
// Toolbar, menus, toggles
// ------------------------------------------------------------------

function setTool(tool) {
  S.tool = tool;
  for (const b of document.querySelectorAll('.tool[data-tool]')) b.classList.toggle('active', b.dataset.tool === tool);
  pref('tool', tool);
  if (mouse) updateHover(mouse);
}

function syncToggles() {
  $('#eyeBtn').innerHTML = icon(S.hideAll ? 'eye-off' : 'eye');
  $('#eyeBtn').classList.toggle('on', S.hideAll);
  $('#onionBtn').classList.toggle('on', S.onion);
  $('#labelsBtn').classList.toggle('on', S.showLabels);
  $('#lockBtn').innerHTML = icon(S.lockView ? 'lock' : 'unlock');
  $('#lockBtn').classList.toggle('on', S.lockView);
  $('#lockBtn').title = S.lockView ? 'Zoom is kept when changing frames (click to fit every frame)' : 'Every frame is fitted to the screen (click to keep zoom)';
}

function toggle(key) {
  S[key] = !S[key];
  if (key === 'hideAll' && S.hideAll) S.selected = null;
  pref(key, S[key]);
  syncToggles();
  if (key === 'onion' && S.onion) loadOnion().then(requestDraw);
  renderSidebar();
  requestDraw();
}

function closePopups() {
  $('#menu').hidden = true;
  $('#lightPop').hidden = true;
}

function applyLight() {
  S.brightness = Number($('#bright').value);
  S.contrast = Number($('#contrast').value);
  $('#brightVal').textContent = `${Math.round(S.brightness * 100)}%`;
  $('#contrastVal').textContent = `${Math.round(S.contrast * 100)}%`;
  $('#lightBtn').classList.toggle('primary', S.brightness !== 1 || S.contrast !== 1);
  pref('light', [S.brightness, S.contrast]);
  requestDraw();
}

function wireUi() {
  for (const b of document.querySelectorAll('.tool[data-tool]')) b.addEventListener('click', () => setTool(b.dataset.tool));
  $('#toolFill').addEventListener('click', fillFromPrevious);
  $('#fillPrevBtn').addEventListener('click', fillFromPrevious);
  $('#toolDup').addEventListener('click', duplicateSelected);
  $('#toolUndo').addEventListener('click', undo);
  $('#toolRedo').addEventListener('click', redo);
  $('#toolClear').addEventListener('click', clearFrame);

  $('#eyeBtn').addEventListener('click', () => toggle('hideAll'));
  $('#onionBtn').addEventListener('click', () => toggle('onion'));
  $('#labelsBtn').addEventListener('click', () => toggle('showLabels'));
  $('#lockBtn').addEventListener('click', () => { toggle('lockView'); if (!S.lockView) fitView(); });
  $('#panelBtn').addEventListener('click', () => {
    document.body.classList.toggle('no-right');
    pref('noRight', document.body.classList.contains('no-right'));
    resizeCanvas();
  });
  $('#saveStatus').addEventListener('click', () => {
    if (stale) location.reload();
    else if (saveFailed) { saveFailed = false; flushSave(); }
  });

  $('#menuBtn').addEventListener('click', (e) => { e.stopPropagation(); const m = $('#menu'); const was = m.hidden; closePopups(); m.hidden = !was; });
  $('#menu').addEventListener('click', (e) => {
    const act = e.target.closest('[data-action]')?.dataset.action;
    closePopups();
    if (act === 'fill') fillFromPrevious();
    else if (act === 'replace') replaceWithPrevious();
    else if (act === 'copyall') { S.selected = null; copyBoxes(); }
    else if (act === 'clear') clearFrame();
  });

  $('#zoomIn').addEventListener('click', () => zoomCenter(1.25));
  $('#zoomOut').addEventListener('click', () => zoomCenter(0.8));
  $('#zoomReset').addEventListener('click', fitView);
  $('#lightBtn').addEventListener('click', (e) => { e.stopPropagation(); const p = $('#lightPop'); const was = p.hidden; closePopups(); p.hidden = !was; });
  $('#lightPop').addEventListener('click', (e) => e.stopPropagation());
  $('#bright').addEventListener('input', applyLight);
  $('#contrast').addEventListener('input', applyLight);
  $('#lightReset').addEventListener('click', () => { $('#bright').value = 1; $('#contrast').value = 1; applyLight(); });
  $('#keysBtn').addEventListener('click', () => { $('#keysModal').hidden = false; });
  $('#keysClose').addEventListener('click', () => { $('#keysModal').hidden = true; });
  $('#keysModal').addEventListener('click', (e) => { if (e.target.id === 'keysModal') e.target.hidden = true; });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.menu-wrap, .popover-wrap')) closePopups();
    // Keep Space / Enter for the canvas instead of re-clicking the button.
    e.target.closest('button')?.blur();
  });

  for (const t of document.querySelectorAll('.tab')) {
    t.addEventListener('click', () => {
      for (const o of document.querySelectorAll('.tab')) o.classList.toggle('active', o === t);
      $('#classesTab').hidden = t.dataset.tab !== 'classes';
      $('#layersTab').hidden = t.dataset.tab !== 'layers';
    });
  }

  $('#confSlider').addEventListener('input', (e) => {
    S.minConf = Number(e.target.value);
    $('#confValue').textContent = S.minConf.toFixed(2);
    const sel = selectedBox();
    if (sel && (sel.confidence ?? 1) < S.minConf) S.selected = null;
    renderSidebar();
    requestDraw();
  });
  $('#confDelete').addEventListener('click', () => {
    const before = snapshot();
    const n = S.boxes.length;
    S.boxes = S.boxes.filter((b) => (b.confidence ?? 1) >= S.minConf);
    S.selected = null;
    commit(before);
    toast(`Deleted ${n - S.boxes.length} boxes (⌘Z to undo)`);
  });

  $('#addClassForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('#addClassInput');
    const name = input.value.trim();
    if (!name) return;
    S.activeClass = ensureClass(name);
    input.value = '';
    input.blur();
    renderSidebar();
    toast(`New boxes will be "${S.activeClass}"`);
  });

  $('#prevBtn').addEventListener('click', () => step(-1));
  $('#nextBtn').addEventListener('click', () => step(1));
  $('#pageInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const n = parseInt(e.target.value, 10);
      if (Number.isFinite(n)) goTo(n - 1);
      e.target.blur();
    } else if (e.key === 'Escape') {
      e.target.value = S.pos + 1;
      e.target.blur();
    }
    e.stopPropagation();
  });
  $('#filterSelect').addEventListener('change', (e) => { applyFilter(); e.target.blur(); });
  $('#playBtn').addEventListener('click', togglePlay);

  new ResizeObserver(() => { resizeCanvas(); if (S.view.fitted) fitView(); }).observe(stage);
  new ResizeObserver(() => drawTimeline()).observe(tl);
}

// ------------------------------------------------------------------
// Keyboard
// ------------------------------------------------------------------

window.addEventListener('keydown', (e) => {
  if (e.target.closest('input, textarea, select')) return;
  const mod = e.metaKey || e.ctrlKey;
  const k = e.key.toLowerCase();

  if (!$('#openModal').hidden) return;
  if (mod && k === 'o') { e.preventDefault(); openPicker(); return; }
  if (mod) {
    if (k === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if (k === 'y') { e.preventDefault(); redo(); }
    else if (k === 'c') { e.preventDefault(); copyBoxes(); }
    else if (k === 'v') { e.preventDefault(); pasteBoxes(); }
    else if (k === 'd') { e.preventDefault(); duplicateSelected(); }
    else if (k === 's') { e.preventDefault(); flushSave().then(() => toast(saveFailed ? 'Save failed' : 'Saved')); }
    return;
  }
  if (!$('#keysModal').hidden) {
    if (e.key === 'Escape' || e.key === '?') $('#keysModal').hidden = true;
    return;
  }

  const sel = selectedBox();
  const big = e.shiftKey ? 10 : 1;
  switch (e.key) {
    case ' ':
      e.preventDefault();
      if (!spaceDown) { spaceDown = true; canvas.style.cursor = 'grab'; requestDraw(); }
      return;
    case 'ArrowLeft': e.preventDefault(); if (sel) nudge(-big, 0); else step(-big); return;
    case 'ArrowRight': e.preventDefault(); if (sel) nudge(big, 0); else step(big); return;
    case 'ArrowUp': e.preventDefault(); if (sel) nudge(0, -big); return;
    case 'ArrowDown': e.preventDefault(); if (sel) nudge(0, big); return;
    case 'Delete':
    case 'Backspace': e.preventDefault(); deleteSelected(); return;
    case 'Escape':
      if (!$('#menu').hidden || !$('#lightPop').hidden) { closePopups(); return; }
      if (drag) { cancelDrag(); requestDraw(); return; }
      select(null);
      return;
    case 'Tab': {
      e.preventDefault();
      const list = S.boxes.filter(isVisible).sort((a, b) => a.box[0] - b.box[0] || a.box[1] - b.box[1]);
      if (!list.length) return;
      const i = list.findIndex((b) => b.id === S.selected);
      const next = i < 0 ? (e.shiftKey ? list.length - 1 : 0) : (i + (e.shiftKey ? -1 : 1) + list.length) % list.length;
      select(list[next].id);
      return;
    }
    case 'Enter': if (sel) { e.preventDefault(); openEditor(sel.id); } return;
    case '?': $('#keysModal').hidden = false; return;
    case '+': case '=': zoomCenter(1.25); return;
    case '-': case '_': zoomCenter(0.8); return;
    case '0': fitView(); return;
  }
  if (/^[1-9]$/.test(e.key)) {
    const name = S.classes[Number(e.key) - 1];
    if (!name) return;
    if (sel) setLabel(sel, name);
    else { S.activeClass = name; renderSidebar(); toast(`New boxes will be "${name}"`); }
    return;
  }
  switch (k) {
    case 'a': step(-big); break;
    case 'd': step(big); break;
    case 'b': setTool('box'); break;
    case 'h': setTool('pan'); break;
    case 'r': fillFromPrevious(); break;
    case 'o': toggle('onion'); break;
    case 'l': toggle('showLabels'); break;
    case 'e': toggle('hideAll'); break;
    case 'f': fitView(); break;
    case 'p': togglePlay(); break;
  }
});

window.addEventListener('keyup', (e) => {
  if (e.key === ' ') {
    spaceDown = false;
    if (mouse) updateHover(mouse);
  }
});
window.addEventListener('blur', () => { spaceDown = false; });

// ------------------------------------------------------------------
// Open files
// ------------------------------------------------------------------

const picker = { files: [], shown: [], sel: -1, required: false };

function timeAgo(seconds) {
  const d = Date.now() / 1000 - seconds;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  if (d < 86400 * 30) return `${Math.floor(d / 86400)} d ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

async function openPicker({ required = false } = {}) {
  picker.required = required;
  document.body.classList.toggle('required', required);
  closePopups();
  closeEditor();
  $('#openModal').hidden = false;
  $('#openError').hidden = true;
  $('#openSearch').value = '';
  $('#openDets').value = S.project?.detections_path ?? '';
  $('#openVideo').value = S.project?.video_path ?? '';
  $('#openSearch').focus();
  await loadPickerFiles();
}

function closePicker() {
  if (picker.required) return;
  $('#openModal').hidden = true;
}

async function loadPickerFiles() {
  $('#openRoot').textContent = '…';
  try {
    const r = await api('/api/files');
    picker.files = r.detections;
    $('#openRoot').textContent = r.root;
    $('#openRoot').title = r.root;
    $('#openDetsList').innerHTML = r.detections.map((f) => `<option value="${esc(f.path)}">`).join('');
    $('#openVideoList').innerHTML = r.videos.map((v) => `<option value="${esc(v)}">`).join('');
  } catch (err) {
    picker.files = [];
    showPickerError(`Could not list the files: ${err.message}`);
  }
  renderPicker();
}

function renderPicker() {
  const q = $('#openSearch').value.trim().toLowerCase();
  picker.shown = picker.files.filter((f) => !q
    || f.path.toLowerCase().includes(q) || (f.video || '').toLowerCase().includes(q)
    || (f.detector || '').toLowerCase().includes(q));
  const current = S.project?.detections_path;
  picker.sel = picker.shown.findIndex((f) => f.path === $('#openDets').value);
  const ul = $('#openList');
  ul.innerHTML = '';
  picker.shown.forEach((f, i) => {
    const slash = f.path.lastIndexOf('/');
    const dir = slash >= 0 ? f.path.slice(0, slash + 1) : '';
    const name = f.path.slice(slash + 1);
    const li = document.createElement('li');
    li.className = 'pick';
    li.classList.toggle('sel', i === picker.sel);
    const video = f.video
      ? `<div class="pick-video">${icon('film')}<span title="${esc(f.video)}">${esc(f.video)}</span></div>`
      : `<div class="pick-video missing">${icon('film')}<span>video ${esc(f.source || '(not recorded)')} not found, choose it below</span></div>`;
    const tags = [
      f.path === current ? '<span class="chip open">open</span>' : '',
      f.edited ? `<span class="chip edited">${f.edited} edited</span>` : '',
      `<span class="chip">${esc(f.detector || '?')}</span>`,
    ].join('');
    li.innerHTML = `<div class="pick-name" title="${esc(f.path)}"><span class="dir">${esc(dir)}</span>${esc(name)}</div>`
      + `<div class="pick-tags">${tags}</div>`
      + video
      + `<div class="pick-meta">${f.samples ?? '?'} frames · ${timeAgo(f.modified)}</div>`;
    li.addEventListener('click', () => choosePick(i));
    li.addEventListener('dblclick', () => { choosePick(i); openSelected(); });
    ul.append(li);
  });
}

function choosePick(i) {
  const f = picker.shown[i];
  if (!f) return;
  picker.sel = i;
  $('#openDets').value = f.path;
  $('#openVideo').value = f.video || '';
  $('#openError').hidden = true;
  for (const [j, li] of [...$('#openList').children].entries()) li.classList.toggle('sel', j === i);
  $('#openList').children[i]?.scrollIntoView({ block: 'nearest' });
  if (!f.video) $('#openVideo').focus();
}

function showPickerError(msg) {
  const el = $('#openError');
  el.textContent = msg;
  el.hidden = false;
}

async function openSelected() {
  const detections = $('#openDets').value.trim();
  const video = $('#openVideo').value.trim();
  if (!detections) { showPickerError('Choose a detections file.'); return; }
  const btn = $('#openGo');
  btn.disabled = true;
  btn.textContent = 'Opening…';
  try {
    if (S.project && !stale) await flushSave();
    await api('/api/open', { method: 'POST', body: { detections, video: video || null } });
    // Start the page over on the new files.
    history.replaceState(null, '', '/');
    location.reload();
  } catch (err) {
    showPickerError(err.message);
    btn.disabled = false;
    btn.textContent = 'Open';
  }
}

function wirePicker() {
  $('#openBtn').addEventListener('click', () => openPicker());
  $('#emptyOpen').addEventListener('click', () => openPicker({ required: true }));
  $('#openClose').addEventListener('click', closePicker);
  $('#openCancel').addEventListener('click', closePicker);
  $('#openRefresh').addEventListener('click', loadPickerFiles);
  $('#openGo').addEventListener('click', openSelected);
  $('#openModal').addEventListener('click', (e) => { if (e.target.id === 'openModal') closePicker(); });
  $('#openSearch').addEventListener('input', renderPicker);
  $('#openDets').addEventListener('change', (e) => {
    // A known file: fill in its video.
    const f = picker.files.find((x) => x.path === e.target.value.trim());
    if (f && f.video) $('#openVideo').value = f.video;
    renderPicker();
  });
  $('#openModal').addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); closePicker(); return; }
    const inList = e.target.id === 'openSearch';
    if (inList && (e.key === 'ArrowDown' || e.key === 'ArrowUp') && picker.shown.length) {
      e.preventDefault();
      const n = picker.shown.length;
      choosePick(picker.sel < 0 ? 0 : (picker.sel + (e.key === 'ArrowDown' ? 1 : -1) + n) % n);
      $('#openSearch').focus();
    } else if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
      e.preventDefault();
      if (inList && picker.sel < 0 && picker.shown.length) choosePick(0);
      openSelected();
    }
  });
}

// ------------------------------------------------------------------
// Start
// ------------------------------------------------------------------

async function init() {
  wireUi();
  wirePicker();
  let p;
  try {
    p = await api('/api/project');
  } catch (err) {
    if (err.status === 409) {
      // Nothing open yet: choose the files first.
      $('#fileName').textContent = 'No files open';
      $('#fileSub').textContent = '';
      $('#emptyState').hidden = false;
      document.querySelector('#timelineWrap').style.visibility = 'hidden';
      openPicker({ required: true });
    } else {
      $('#fileName').textContent = 'Could not reach the server';
      $('#fileSub').textContent = String(err.message || err);
    }
    return;
  }
  S.project = p;
  S.W = p.width;
  S.H = p.height;
  S.indices = p.frames.map((f) => f[0]);
  for (const [i, c] of p.frames) S.counts.set(i, c);
  S.edited = new Set(p.edited);
  S.classes = [...p.classes];
  S.activeClass = S.classes[0] || null;

  document.title = `${p.detections} · Annotator`;
  $('#fileName').textContent = p.detections;
  $('#fileName').title = p.detections_path;
  const sampleStep = p.meta.sample_step || 1;
  const rate = sampleStep > 1 ? `${+(p.fps / sampleStep).toFixed(2)} samples/s` : 'every frame';
  $('#fileSub').textContent = `${p.video} · ${p.width}×${p.height} · ${rate}`;
  $('#fileSub').title = p.video_path;
  if (p.warnings.length) setTimeout(() => toast(`The detections file doesn't match this video: ${p.warnings.join(', ')}`), 600);
  $('#detectorName').textContent = p.meta.detector || 'unknown';
  $('#suggestionList').innerHTML = (p.suggestions || []).map((c) => `<option value="${esc(c)}">`).join('');

  S.showLabels = pref('showLabels') ?? false;
  S.onion = pref('onion') ?? false;
  S.lockView = pref('lockView') ?? true;
  const light = pref('light');
  if (Array.isArray(light)) { $('#bright').value = light[0]; $('#contrast').value = light[1]; }
  if (pref('noRight')) document.body.classList.add('no-right');
  applyLight();
  syncToggles();
  setTool(pref('tool') || 'box');
  updateHistoryButtons();
  setStatus();

  buildThumbs();
  resizeCanvas();
  fitView();

  if (!S.indices.length) {
    $('#banner').hidden = false;
    $('#banner').textContent = 'The detections file has no frames.';
    return;
  }
  const m = location.hash.match(/f=(\d+)/);
  const wanted = m ? Number(m[1]) : pref(`pos:${p.detections_path}`);
  const start = Math.max(0, S.indices.indexOf(wanted));
  await goTo(start);
}

init();
