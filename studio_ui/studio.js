'use strict';

// AI Studio: folders and projects (home), and a project's pages:
// upload, annotate (batches), dataset, classes, train, models. Images
// are labeled in the annotator (/annotate?project=ID&batch=ID).

// Same as the annotator, so a class has the same color in both.
const PALETTE = [
  '#D946EF', '#06B6D4', '#F59E0B', '#10B981', '#F43F5E', '#6366F1',
  '#84CC16', '#F97316', '#0EA5E9', '#A855F7', '#14B8A6', '#EC4899',
];
const STATUS_NAMES = { unlabeled: 'Unlabeled', predicted: 'Pre-labeled', labeled: 'Labeled' };
const RATES = [
  [0.5, '1 frame every 2 s'],
  [1, '1 frame / second'],
  [2, '2 frames / second'],
  [5, '5 frames / second'],
  [10, '10 frames / second'],
  [0, 'Every frame'],
];
const IMAGE_EXT = /\.(jpe?g|png|bmp|webp|tiff?)$/i;
const VIDEO_EXT = /\.(mp4|mov|m4v|avi|mkv|webm)$/i;
const UPLOAD_CONCURRENCY = 4;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name, cls = '') => `<svg class="ic ${cls}"><use href="#i-${name}"/></svg>`;
const fmtInt = (n) => Number(n ?? 0).toLocaleString();
const plural = (n, word, many = `${word}s`) => `${fmtInt(n)} ${n === 1 ? word : many}`;
const fmtNum = (v, d = 3) => (v == null ? '–' : Number(v).toFixed(d));
const norm = (s) => s.trim().toLowerCase().replace(/[_-]/g, ' ');
// Long file names keep their end (frame numbers): "7RswKdW…upscale_000123.jpg".
const midTrim = (s, n = 30) => (s.length <= n ? s : `${s.slice(0, Math.ceil(n * 0.35))}…${s.slice(-Math.floor(n * 0.65))}`);

function timeAgo(when) {
  const t = typeof when === 'number' ? when * 1000 : Date.parse(when);
  if (!Number.isFinite(t)) return '';
  const d = (Date.now() - t) / 1000;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  if (d < 86400 * 2) return 'yesterday';
  if (d < 86400 * 30) return `${Math.floor(d / 86400)} days ago`;
  if (d < 86400 * 365) return `${Math.floor(d / (86400 * 30))} months ago`;
  return new Date(t).toLocaleDateString();
}

function fmtDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${s % 60} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

function pref(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(`studio.${key}`));
    localStorage.setItem(`studio.${key}`, JSON.stringify(value));
  } catch { /* storage unavailable */ }
  return null;
}

let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

const OUTDATED = 'annotate_server.py changed since it was started: restart it to use the new version.';
let warnedOutdated = false;

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (res.headers.get('X-Server-Outdated') && !warnedOutdated) {
    warnedOutdated = true;
    toast(OUTDATED, 10000);
  }
  if (!res.ok) {
    let detail = await res.text();
    try { detail = JSON.parse(detail).detail ?? detail; } catch { /* not JSON */ }
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join('; ');
    if (res.status === 405 || (res.status === 404 && detail === 'Not Found')) detail = OUTDATED;
    const err = new Error(String(detail));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const thumbUrl = (projectId, imageId) => `/api/ws/projects/${projectId}/images/${imageId}/thumbnail`;

function annotateUrl(projectId, batchId = null, imageId = null) {
  const q = new URLSearchParams({ project: projectId });
  if (batchId) q.set('batch', batchId);
  return `/annotate?${q}${imageId ? `#f=${imageId}` : ''}`;
}

function colorOf(label, classes) {
  const i = classes.findIndex((c) => norm(c) === norm(label));
  return PALETTE[(i < 0 ? classes.length : i) % PALETTE.length];
}

function stackBar(s, big = false) {
  const total = Math.max(1, s.images || 0);
  return `<div class="stack${big ? ' big' : ''}" title="${fmtInt(s.labeled)} labeled · ${fmtInt(s.predicted)} pre-labeled · ${fmtInt(s.unlabeled)} unlabeled">`
    + `<i class="s-labeled" style="width:${(100 * (s.labeled || 0)) / total}%"></i>`
    + `<i class="s-predicted" style="width:${(100 * (s.predicted || 0)) / total}%"></i></div>`;
}

function legend(s) {
  return `<div class="legend">`
    + `<span><i class="sw labeled"></i><b>${fmtInt(s.labeled)}</b> labeled</span>`
    + `<span><i class="sw predicted"></i><b>${fmtInt(s.predicted)}</b> pre-labeled</span>`
    + `<span><i class="sw unlabeled"></i><b>${fmtInt(s.unlabeled)}</b> unlabeled</span></div>`;
}

// A line chart of per-epoch values: series [{name, color, values:
// [[epoch, value], ...]}], over epochs 1..`epochs` (the planned ones,
// so a run in progress shows how far it is).
function lineChart(title, series, { epochs = 0, min = null, max = null } = {}) {
  const all = series.flatMap((s) => s.values.map(([, v]) => v));
  if (!all.length) return '';
  let lo = min ?? Math.min(...all);
  let hi = max ?? Math.max(...all);
  if (min == null || max == null) {
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 0.1;
    if (min == null) lo -= pad;
    if (max == null) hi += pad;
  }
  const last = Math.max(epochs, ...series.flatMap((s) => s.values.map(([e]) => e)));
  const W = 480, H = 150, L = 36, R = 10, T = 8, B = 22;
  const x = (e) => L + (last <= 1 ? (W - L - R) / 2 : ((e - 1) / (last - 1)) * (W - L - R));
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const label = (v) => (Math.abs(hi - lo) >= 10 ? v.toFixed(0) : Math.abs(hi - lo) >= 1 ? v.toFixed(1) : v.toFixed(2));
  let svg = '';
  for (const v of [lo, (lo + hi) / 2, hi]) {
    svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 3.5}" text-anchor="end">${label(v)}</text>`;
  }
  svg += `<text x="${L}" y="${H - 6}" text-anchor="start">1</text><text x="${W - R}" y="${H - 6}" text-anchor="end">${last}</text><text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">epoch</text>`;
  for (const s of series) {
    if (!s.values.length) continue;
    const d = s.values.map(([e, v], i) => `${i ? 'L' : 'M'}${x(e).toFixed(1)},${y(v).toFixed(1)}`).join('');
    svg += `<path d="${d}" stroke="${s.color}"/>`;
    svg += s.values.map(([e, v], i) => `<circle cx="${x(e).toFixed(1)}" cy="${y(v).toFixed(1)}" r="${i === s.values.length - 1 ? 3.2 : 2}" fill="${s.color}"><title>Epoch ${e}: ${s.name} ${fmtNum(v)}</title></circle>`).join('');
  }
  const keys = series.filter((s) => s.values.length).map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)} <b>${fmtNum(s.values[s.values.length - 1][1])}</b></span>`).join('');
  return `<div class="chart"><div class="chart-head"><b>${esc(title)}</b><span class="chart-keys">${keys}</span></div>`
    + `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)} per epoch">${svg}</svg></div>`;
}

// mAP and loss per epoch, from a training history.
function trainingCharts(history, epochs = 0) {
  const pick = (key) => history.filter((e) => e[key] != null).map((e) => [e.epoch, e[key]]);
  if (!history.length) return '';
  return `<div class="charts">`
    + lineChart('mAP', [
      { name: 'mAP50', color: '#2563eb', values: pick('map50') },
      { name: 'mAP50-95', color: '#10b981', values: pick('map') },
    ], { epochs, min: 0, max: 1 })
    + lineChart('Loss', [{ name: 'train loss', color: '#f59e0b', values: pick('loss') }], { epochs })
    + `</div>`;
}

function sparkline(values) {
  const pts = values.filter((v) => v != null);
  if (pts.length < 2) return '';
  let lo = Math.min(...pts), hi = Math.max(...pts);
  if (hi - lo < 0.02) { lo -= 0.01; hi += 0.01; }
  const w = 200, h = 40, pad = 3;
  const xy = pts.map((v, i) => [pad + (i * (w - 2 * pad)) / (pts.length - 1), pad + (1 - (v - lo) / (hi - lo)) * (h - 2 * pad)]);
  const d = xy.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
  const [lx, ly] = xy[xy.length - 1];
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><path class="base" d="M${pad},${h - pad}H${w - pad}"/><path d="${d}"/><circle cx="${lx}" cy="${ly}" r="2.5"/></svg>`;
}

// ------------------------------------------------------------------
// Menus and dialogs
// ------------------------------------------------------------------

function closeMenus() {
  for (const m of $$('.menu')) m.remove();
}

// items: [{label, icon, danger, action}] or '-' for a separator.
function openMenu(anchor, items) {
  closeMenus();
  const menu = document.createElement('div');
  menu.className = 'menu';
  for (const item of items) {
    if (item === '-') { menu.append(document.createElement('hr')); continue; }
    const b = document.createElement('button');
    b.className = item.danger ? 'danger' : '';
    b.innerHTML = `${item.icon ? icon(item.icon, 'sm') : ''}<span>${esc(item.label)}</span>`;
    b.addEventListener('click', (e) => { e.stopPropagation(); closeMenus(); item.action(); });
    menu.append(b);
  }
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  menu.style.left = `${Math.max(8, Math.min(r.right - mw, innerWidth - mw - 8))}px`;
  menu.style.top = `${r.bottom + 4 + mh > innerHeight ? r.top - mh - 4 : r.bottom + 4}px`;
}
document.addEventListener('click', (e) => { if (!e.target.closest('.menu')) closeMenus(); });
window.addEventListener('resize', closeMenus);
document.addEventListener('scroll', closeMenus, true);

// A modal. `render(body)` fills the body; `onOk()` may throw to keep it
// open (the message is shown). Resolves with onOk's result, or null.
function dialog({ title, okText = 'Save', danger = false, render, onOk, wide = false }) {
  return new Promise((resolve) => {
    const back = document.createElement('div');
    back.className = 'modal-back';
    back.innerHTML = `<form class="modal" style="${wide ? 'width:min(620px,100%)' : ''}" novalidate>`
      + `<div class="modal-head"><h3>${esc(title)}</h3><button type="button" class="icon-btn" data-close title="Close (Esc)">${icon('x')}</button></div>`
      + `<div class="modal-body"></div><div class="form-error" style="padding:0 18px" hidden></div>`
      + `<div class="modal-foot"><button type="button" class="btn btn-ghost" data-close>Cancel</button>`
      + `<button type="submit" class="btn ${danger ? 'btn-danger' : 'btn-primary'}">${esc(okText)}</button></div></form>`;
    const form = back.querySelector('form');
    const body = back.querySelector('.modal-body');
    const err = back.querySelector('.form-error');
    render(body);
    const close = (value) => {
      back.remove();
      document.removeEventListener('keydown', onKey, true);
      resolve(value);
    };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(null); } };
    document.addEventListener('keydown', onKey, true);
    for (const b of back.querySelectorAll('[data-close]')) b.addEventListener('click', () => close(null));
    back.addEventListener('mousedown', (e) => { if (e.target === back) close(null); });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const ok = form.querySelector('[type=submit]');
      ok.disabled = true;
      err.hidden = true;
      try {
        close(await onOk(body));
      } catch (ex) {
        err.textContent = ex.message;
        err.hidden = false;
        ok.disabled = false;
      }
    });
    $('#layer').append(back);
    (body.querySelector('[autofocus]') || form.querySelector('[type=submit]')).focus();
    body.querySelector('[autofocus]')?.select?.();
  });
}

function askName(title, value, okText = 'Save', label = 'Name') {
  return dialog({
    title,
    okText,
    render: (body) => {
      body.innerHTML = `<label class="field"><span>${esc(label)}</span><input class="input" name="name" value="${esc(value)}" autofocus autocomplete="off" spellcheck="false"></label>`;
    },
    onOk: (body) => {
      const name = body.querySelector('[name=name]').value.trim();
      if (!name) throw new Error('Enter a name.');
      return name;
    },
  });
}

function confirmDialog(title, html, okText = 'Delete', danger = true) {
  return dialog({
    title,
    okText,
    danger,
    render: (body) => { body.innerHTML = html; },
    onOk: () => true,
  });
}

// Pick a folder; `exclude` (a folder id) and its sub-folders are greyed.
async function pickFolder(title, current, exclude = null) {
  let folders;
  try { folders = (await api('/api/ws/folders')).folders; } catch (err) { toast(err.message); return undefined; }
  const blocked = new Set();
  if (exclude) {
    blocked.add(exclude);
    let grew = true;
    while (grew) {
      grew = false;
      for (const f of folders) if (f.parent && blocked.has(f.parent) && !blocked.has(f.id)) { blocked.add(f.id); grew = true; }
    }
  }
  let chosen = current || '';
  const result = await dialog({
    title,
    okText: 'Move here',
    render: (body) => {
      const depth = (f) => f.path.split(' / ').length - 1;
      body.innerHTML = `<div class="folder-pick">`
        + `<button type="button" data-id="">${icon('images', 'sm')} Projects (top level)</button>`
        + folders.map((f) => `<button type="button" data-id="${f.id}" ${blocked.has(f.id) ? 'disabled' : ''} style="padding-left:${12 + 18 * (depth(f) + 1)}px">${icon('folder', 'sm')} ${esc(f.name)}</button>`).join('')
        + `</div>`;
      const sync = () => { for (const b of body.querySelectorAll('[data-id]')) b.classList.toggle('on', b.dataset.id === chosen); };
      for (const b of body.querySelectorAll('[data-id]')) b.addEventListener('click', () => { chosen = b.dataset.id; sync(); });
      sync();
    },
    onOk: () => chosen,
  });
  return result === null ? undefined : result;
}

// ------------------------------------------------------------------
// Workspace actions (shared by home cards and the project nav)
// ------------------------------------------------------------------

async function renameNode(node, after) {
  const name = await askName(`Rename ${node.kind}`, node.name);
  if (!name || name === node.name) return;
  try {
    await api(`/api/ws/nodes/${node.id}`, { method: 'PATCH', body: { name } });
    toast(`Renamed to “${name}”`);
    after?.();
  } catch (err) { toast(err.message); }
}

async function moveNode(node, after) {
  const parent = await pickFolder(`Move “${node.name}” to…`, node.parent || '', node.kind === 'folder' ? node.id : null);
  if (parent === undefined || (parent || null) === (node.parent || null)) return;
  await moveTo(node, parent, after);
}

async function moveTo(node, parent, after) {
  try {
    await api(`/api/ws/nodes/${node.id}`, { method: 'PATCH', body: { parent: parent || '' } });
    toast(`Moved “${node.name}”`);
    after?.();
  } catch (err) { toast(err.message); }
}

async function deleteNode(node, after) {
  let html;
  if (node.kind === 'folder') {
    const what = [node.folders && plural(node.folders, 'folder'), node.projects && plural(node.projects, 'project')].filter(Boolean).join(' and ');
    html = `<p>Delete the folder <b>${esc(node.name)}</b>${what ? ` and everything in it (${what}, with their images, labels and models)` : ''}?</p><p class="muted">This can't be undone.</p>`;
  } else {
    const s = node.stats || {};
    html = `<p>Delete the project <b>${esc(node.name)}</b> with its ${plural(s.images || 0, 'image')}, their labels${node.models ? ` and ${plural(node.models, 'model')}` : ''}?</p><p class="muted">This can't be undone.</p>`;
  }
  if (!await confirmDialog(`Delete ${node.kind}?`, html)) return;
  try {
    await api(`/api/ws/nodes/${node.id}`, { method: 'DELETE' });
    toast(`Deleted “${node.name}”`);
    after?.();
  } catch (err) { toast(err.message); }
}

async function newFolder(parent) {
  const name = await askName('New folder', '', 'Create folder');
  if (!name) return;
  try {
    await api('/api/ws/folders', { method: 'POST', body: { name, parent: parent || null } });
    toast(`Created the folder “${name}”`);
    showHome(parent);
  } catch (err) { toast(err.message); }
}

async function newProject(parent) {
  let folders = [];
  try { folders = (await api('/api/ws/folders')).folders; } catch { /* top level only */ }
  const created = await dialog({
    title: 'New project',
    okText: 'Create project',
    render: (body) => {
      body.innerHTML = `<label class="field"><span>Project name</span><input class="input" name="name" placeholder="e.g. Hand detection" autofocus autocomplete="off" spellcheck="false"></label>`
        + `<label class="field"><span>Folder</span><select class="select" name="parent"><option value="">Projects (top level)</option>`
        + folders.map((f) => `<option value="${f.id}" ${f.id === parent ? 'selected' : ''}>${esc(f.path)}</option>`).join('')
        + `</select></label>`
        + `<div class="field"><span>Type</span><div class="type-option on">${icon('box')}<div><b>Object detection</b><small>Draw boxes around objects; train RF-DETR to find them.</small></div></div></div>`;
    },
    onOk: async (body) => {
      const name = body.querySelector('[name=name]').value.trim();
      if (!name) throw new Error('Enter a name.');
      const p = body.querySelector('[name=parent]').value || null;
      return api('/api/ws/projects', { method: 'POST', body: { name, parent: p, type: 'object-detection' } });
    },
  });
  if (created) location.hash = `#/p/${created.id}/upload`;
}

// ------------------------------------------------------------------
// Home: folders and projects
// ------------------------------------------------------------------

const H = { folder: null, data: null, q: '', sort: pref('sort') || 'updated', seq: 0, loc: null };

const SORTS = {
  updated: ['Date edited', (a, b) => (b.updated || '').localeCompare(a.updated || '')],
  name: ['Name', (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })],
  created: ['Date created', (a, b) => (b.created || '').localeCompare(a.created || '')],
  images: ['Images', (a, b) => (b.stats?.images || 0) - (a.stats?.images || 0)],
};

async function showHome(folderId) {
  stopPolling();
  P.id = null;
  P.section = null;
  $('#subnav').hidden = true;
  pollNavJobs();
  const main = $('#main');
  if (H.folder !== folderId || !$('#homePage')) {
    H.folder = folderId;
    main.innerHTML = `<div class="page" id="homePage">
      <header class="page-head">
        <div><nav class="crumbs" id="crumbs"></nav><h1 id="homeTitle">Projects</h1></div>
        <div class="actions">
          <button class="btn" id="newFolderBtn">${icon('folder-plus')} New folder</button>
          <button class="btn btn-primary" id="newProjectBtn">${icon('plus')} New project</button>
        </div>
      </header>
      <div class="toolbar">
        <label class="search">${icon('search')}<input id="homeSearch" placeholder="Search all projects and folders" autocomplete="off" spellcheck="false"></label>
        <label class="select-label">Sort <select class="select" id="homeSort">${Object.entries(SORTS).map(([k, [label]]) => `<option value="${k}">${label}</option>`).join('')}</select></label>
        <button class="btn btn-ghost btn-sm loc-btn" id="locBtn" title="Where projects are kept on disk" hidden>${icon('folder')}<span class="mono" id="locPath"></span></button>
      </div>
      <div id="locNotice"></div>
      <div id="homeBody"><div class="grid"><div class="skeleton" style="height:140px"></div><div class="skeleton" style="height:140px"></div><div class="skeleton" style="height:140px"></div></div></div>
    </div>`;
    $('#homeSearch').value = H.q;
    $('#homeSort').value = H.sort;
    $('#newFolderBtn').addEventListener('click', () => newFolder(H.folder));
    $('#newProjectBtn').addEventListener('click', () => newProject(H.folder));
    let timer = null;
    $('#homeSearch').addEventListener('input', (e) => {
      clearTimeout(timer);
      timer = setTimeout(() => { H.q = e.target.value; loadHome(); }, 180);
    });
    $('#homeSort').addEventListener('change', (e) => { H.sort = e.target.value; pref('sort', H.sort); renderHomeBody(); });
    $('#locBtn').addEventListener('click', changeLocation);
    loadLocation();
  }
  await loadHome();
}

// -- Projects location (where the workspace is on disk)

const tilde = (path) => {
  const home = H.loc?.home;
  return home && (path === home || path.startsWith(`${home}/`)) ? `~${path.slice(home.length)}` : path;
};

async function loadLocation() {
  try { H.loc = await api('/api/ws/location'); } catch { return; /* older server */ }
  renderLocation();
}

function renderLocation() {
  const loc = H.loc;
  if (!loc || !$('#locBtn')) return;
  $('#locBtn').hidden = false;
  $('#locPath').textContent = tilde(loc.path);
  $('#locNotice').innerHTML = loc.source === 'legacy'
    ? `<div class="callout warn" style="margin:-8px 0 20px">${icon('alert')}<div style="flex:1">Your projects are kept inside the app's own folder (<span class="mono">${esc(tilde(loc.path))}</span>). Pick a folder of your own, so they stay safe when the app is updated or moved.</div><button class="btn btn-sm" id="locFix">Choose location</button></div>`
    : '';
  $('#locFix')?.addEventListener('click', changeLocation);
}

async function changeLocation() {
  const loc = H.loc;
  if (!loc) return;
  if (loc.source === 'flag') {
    await dialog({
      title: 'Projects location',
      okText: 'OK',
      render: (body) => { body.innerHTML = `<p style="margin:0">Projects are kept in <span class="mono">${esc(tilde(loc.path))}</span>, set by <span class="mono">--workspace</span> when the server was started. Restart it without that option to pick the location here.</p>`; },
      onOk: () => true,
    });
    return;
  }
  const hasProjects = !!(H.data && (H.data.folders.length || H.data.projects.length || H.folder));
  const suggested = loc.source === 'legacy' ? tilde(loc.default) : tilde(loc.path);
  const result = await dialog({
    title: 'Projects location',
    okText: 'Use this folder',
    render: (body) => {
      body.innerHTML = `<p style="margin:0 0 14px;color:var(--muted)">Now: <span class="mono">${esc(tilde(loc.path))}</span></p>`
        + `<label class="field"><span>Folder</span><input class="input mono" name="path" value="${esc(suggested)}" autofocus autocomplete="off" spellcheck="false" placeholder="~/Documents/Projects"></label>`
        + `<div class="hint" style="color:var(--muted);font-size:12.5px;margin:6px 0 16px">A full path; <span class="mono">~</span> is your home folder. It is created if it doesn't exist.</div>`
        + `<div class="radio-cards">`
        + `<label class="radio-card${hasProjects ? ' on' : ''}"><input type="radio" name="mode" value="move" ${hasProjects ? 'checked' : ''}><div><b>Move my projects there</b><small>Every folder and project goes along, with its images and models.</small></div></label>`
        + `<label class="radio-card${hasProjects ? '' : ' on'}"><input type="radio" name="mode" value="switch" ${hasProjects ? '' : 'checked'}><div><b>Only switch to that folder</b><small>Shows the projects already there. The current ones stay where they are.</small></div></label></div>`;
      for (const r of body.querySelectorAll('[name=mode]')) {
        r.addEventListener('change', () => {
          for (const c of body.querySelectorAll('.radio-card')) c.classList.toggle('on', c.querySelector('input').checked);
        });
      }
    },
    onOk: (body) => {
      let path = body.querySelector('[name=path]').value.trim();
      if (!path) throw new Error('Enter a folder.');
      if (path === '~' || path.startsWith('~/')) path = loc.home + path.slice(1);
      const move = body.querySelector('[name=mode]:checked').value === 'move';
      return api('/api/ws/location', { method: 'PUT', body: { path, move } });
    },
  });
  if (!result) return;
  H.loc = result;
  renderLocation();
  toast(result.moved ? `Moved ${plural(result.moved, 'folder or project', 'folders and projects')} to ${tilde(result.path)}` : `Projects are now kept in ${tilde(result.path)}`);
  if (H.folder && !result.moved) location.hash = '#/';
  else await loadHome();
}

async function loadHome() {
  const seq = ++H.seq;
  try {
    const q = new URLSearchParams({ folder: H.folder || '', q: H.q.trim() });
    const data = await api(`/api/ws?${q}`);
    if (seq !== H.seq) return;
    H.data = data;
  } catch (err) {
    if (seq !== H.seq) return;
    if (err.status === 404) { location.hash = '#/'; return; }
    $('#homeBody').innerHTML = `<div class="empty">${icon('alert', 'big')}<h3>Could not load the projects</h3><p>${esc(err.message)}</p></div>`;
    return;
  }
  renderCrumbs();
  renderHomeBody();
}

function renderCrumbs() {
  const d = H.data;
  const trail = d.trail || [];
  $('#homeTitle').textContent = d.folder ? d.folder.name : 'Projects';
  document.title = `${d.folder ? `${d.folder.name} · ` : ''}AI Studio`;
  const crumbs = $('#crumbs');
  crumbs.hidden = !trail.length;
  crumbs.innerHTML = `<a href="#/" data-drop="">Projects</a>`
    + trail.slice(0, -1).map((f) => `${icon('right')}<a href="#/f/${f.id}" data-drop="${f.id}">${esc(f.name)}</a>`).join('')
    + (trail.length ? `${icon('right')}<span>${esc(trail[trail.length - 1].name)}</span>` : '');
  for (const a of crumbs.querySelectorAll('[data-drop]')) dropTarget(a, a.dataset.drop);
}

function renderHomeBody() {
  const d = H.data;
  const body = $('#homeBody');
  const sort = SORTS[H.sort][1];
  const folders = [...d.folders].sort(SORTS[H.sort === 'images' ? 'name' : H.sort][1]);
  const projects = [...d.projects].sort(sort);
  const searching = !!H.q.trim();
  if (!folders.length && !projects.length) {
    body.innerHTML = searching
      ? `<div class="empty">${icon('search', 'big')}<h3>Nothing matches “${esc(H.q.trim())}”</h3><p>Searched every folder and project by name.</p></div>`
      : `<div class="empty">${icon(d.folder ? 'folder' : 'images', 'big')}<h3>${d.folder ? 'This folder is empty' : 'No projects yet'}</h3>`
        + `<p>A project holds the images (and video frames) you label, and the models trained on them. Folders keep related projects together.</p>`
        + `<div class="btns"><button class="btn" data-act="folder">${icon('folder-plus')} New folder</button><button class="btn btn-primary" data-act="project">${icon('plus')} New project</button></div></div>`;
    body.querySelector('[data-act=folder]')?.addEventListener('click', () => newFolder(H.folder));
    body.querySelector('[data-act=project]')?.addEventListener('click', () => newProject(H.folder));
    return;
  }
  body.innerHTML = (folders.length ? `${searching ? '<h4 class="section-title">Folders</h4>' : ''}<div class="grid" id="folderGrid"></div>` : '')
    + (projects.length ? `${searching || folders.length ? '<h4 class="section-title">Projects</h4>' : ''}<div class="grid" id="projectGrid"></div>` : '');
  for (const f of folders) $('#folderGrid').append(folderCard(f, searching));
  for (const p of projects) $('#projectGrid').append(projectCard(p, searching));
}

function whereLine(node, searching) {
  if (!searching) return '';
  return `<div class="where">${icon('folder')} ${esc(node.where || 'Projects')}</div>`;
}

function folderCard(f, searching) {
  const a = document.createElement('a');
  a.className = 'card fcard';
  a.href = `#/f/${f.id}`;
  const parts = [f.projects && plural(f.projects, 'project'), f.folders && plural(f.folders, 'folder')].filter(Boolean);
  a.innerHTML = `<span class="ficon">${icon('folder')}</span>`
    + `<div style="min-width:0"><h3>${esc(f.name)}</h3><div class="meta">${parts.join(' · ') || 'Empty'}</div>${whereLine(f, searching)}</div>`
    + `<button class="icon-btn kebab" title="More">${icon('more')}</button>`;
  wireCard(a, f);
  dropTarget(a, f.id);
  return a;
}

function projectCard(p, searching) {
  const s = p.stats || {};
  const a = document.createElement('a');
  a.className = 'card pcard';
  a.href = `#/p/${p.id}`;
  const cover = s.cover != null
    ? `<img loading="lazy" alt="" src="${thumbUrl(p.id, s.cover)}">`
    : icon('images');
  const done = s.images ? Math.round((100 * (s.labeled || 0)) / s.images) : 0;
  a.innerHTML = `<div class="cover-img">${cover}</div>`
    + `<div class="pcard-body"><span class="type-chip">${icon('box')} ${esc(p.type_name)}</span>`
    + `<h3 title="${esc(p.name)}">${esc(p.name)}</h3>`
    + `<div class="meta">Edited ${esc(timeAgo(p.updated))}</div>`
    + `<div class="meta">${plural(s.images || 0, 'image')} · ${plural(p.models || 0, 'model')}${s.images ? ` · ${done}% labeled` : ''}</div>`
    + (s.images ? stackBar(s) : '')
    + whereLine(p, searching)
    + `</div><button class="icon-btn kebab" title="More">${icon('more')}</button>`;
  wireCard(a, p);
  return a;
}

function wireCard(el, node) {
  const refresh = () => loadHome();
  el.querySelector('.kebab').addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    openMenu(e.currentTarget, [
      { label: 'Open', icon: node.kind === 'folder' ? 'folder' : 'right', action: () => { location.hash = el.getAttribute('href'); } },
      { label: 'Rename…', icon: 'edit', action: () => renameNode(node, refresh) },
      { label: 'Move to…', icon: 'folder-move', action: () => moveNode(node, refresh) },
      '-',
      { label: `Delete ${node.kind}…`, icon: 'trash', danger: true, action: () => deleteNode(node, refresh) },
    ]);
  });
  // Drag a card onto a folder (or a breadcrumb) to move it.
  el.draggable = true;
  el.addEventListener('dragstart', (e) => {
    e.dataTransfer.setData('text/x-node', JSON.stringify(node));
    e.dataTransfer.effectAllowed = 'move';
    H.dragging = node;
    el.classList.add('dragging');
  });
  el.addEventListener('dragend', () => { el.classList.remove('dragging'); H.dragging = null; });
}

function dropTarget(el, folderId) {
  const ok = () => H.dragging && H.dragging.id !== folderId && (H.dragging.parent || '') !== (folderId || '');
  el.addEventListener('dragover', (e) => {
    if (!ok()) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    el.classList.add('drop-over');
  });
  el.addEventListener('dragleave', () => el.classList.remove('drop-over'));
  el.addEventListener('drop', (e) => {
    el.classList.remove('drop-over');
    if (!ok()) return;
    e.preventDefault();
    const node = H.dragging;
    H.dragging = null;
    moveTo(node, folderId, () => loadHome());
  });
}

// ------------------------------------------------------------------
// Project pages
// ------------------------------------------------------------------

const P = { id: null, data: null, section: null, jobs: [], allJobs: [], defaults: null };
const SECTIONS = {
  upload: ['Upload', 'upload'],
  annotate: ['Annotate', 'pen'],
  dataset: ['Dataset', 'images'],
  classes: ['Classes', 'tag'],
  train: ['Train', 'cpu'],
  models: ['Models', 'model'],
  detections: ['Detections', 'film'],
};

async function showProject(id, section, query) {
  closeMenus();
  if (P.id !== id) {
    P.id = id;
    P.data = null;
    DS.reset();
  }
  $('#subnav').hidden = false;
  if (!P.data) {
    $('#subnav').innerHTML = '';
    $('#main').innerHTML = `<div class="page"><div class="skeleton" style="height:36px;width:260px;margin-bottom:22px"></div><div class="skeleton" style="height:180px"></div></div>`;
  }
  if (!await loadProject()) return;
  if (!section || !SECTIONS[section]) {
    const s = P.data.stats || {};
    section = !s.images ? 'upload' : (s.unlabeled || s.predicted) ? 'annotate' : 'dataset';
    history.replaceState(null, '', `#/p/${id}/${section}`);
  }
  P.section = section;
  P.query = query;
  renderSubnav();
  renderSection();
  schedulePoll();
  updateNavJobs(P.allJobs);
}

async function loadProject() {
  const id = P.id;
  try {
    const [data, jobs] = await Promise.all([api(`/api/ws/projects/${id}`), api('/api/jobs')]);
    if (id !== P.id) return false;
    P.data = data;
    P.allJobs = jobs.jobs;
    P.jobs = jobs.jobs.filter((j) => j.result?.project === id);
    updateNavJobs(jobs.jobs);
  } catch (err) {
    if (err.status === 404) {
      toast('That project no longer exists');
      location.hash = '#/';
    } else {
      $('#main').innerHTML = `<div class="page"><div class="empty">${icon('alert', 'big')}<h3>Could not load the project</h3><p>${esc(err.message)}</p></div></div>`;
    }
    return false;
  }
  document.title = `${P.data.name} · AI Studio`;
  return true;
}

function renderSubnav() {
  const d = P.data;
  const s = d.stats || {};
  const parent = d.trail?.[d.trail.length - 1];
  const todo = (s.unlabeled || 0) + (s.predicted || 0);
  const counts = {
    upload: '',
    annotate: todo ? fmtInt(todo) : '',
    dataset: fmtInt(s.images || 0),
    classes: fmtInt(d.classes.length),
    train: '',
    models: fmtInt(d.models.length),
    detections: '',
  };
  const item = (key) => {
    const [label, ic] = SECTIONS[key];
    const kind = { models: 'train', detections: 'detect' }[key];
    const running = kind && P.jobs.some((j) => j.kind === kind && j.status === 'running');
    return `<a class="sub-item${P.section === key ? ' active' : ''}" href="#/p/${d.id}/${key}">${icon(ic)} ${label}`
      + (running ? '<span class="pulse" style="margin-left:auto" title="Training"></span>' : `<span class="count">${counts[key]}</span>`) + `</a>`;
  };
  const sub = $('#subnav');
  sub.innerHTML = `<a class="back-link" href="${parent ? `#/f/${parent.id}` : '#/'}">${icon('arrow-left', 'sm')} ${esc(parent ? parent.name : 'Projects')}</a>`
    + `<div class="sub-cover"><div class="cover-img" style="width:100%;aspect-ratio:16/9;border-radius:0">${s.cover != null ? `<img alt="" src="${thumbUrl(d.id, s.cover)}">` : icon('images')}</div>`
    + `<span class="type-chip">${icon('box')} ${esc(d.type_name)}</span></div>`
    + `<div class="sub-title"><h2 title="${esc(d.name)}">${esc(d.name)}</h2><button class="icon-btn" id="projMenu" title="Project options">${icon('more')}</button></div>`
    + `<div class="sub-section">Data</div>${['upload', 'annotate', 'dataset', 'classes'].map(item).join('')}`
    + `<div class="sub-section">Models</div>${['train', 'models', 'detections'].map(item).join('')}`;
  $('#projMenu').addEventListener('click', (e) => {
    e.stopPropagation();
    const node = { ...d, kind: 'project' };
    openMenu(e.currentTarget, [
      { label: 'Rename…', icon: 'edit', action: () => renameNode(node, refreshProject) },
      { label: 'Move to…', icon: 'folder-move', action: () => moveNode(node, refreshProject) },
      '-',
      { label: 'Delete project…', icon: 'trash', danger: true, action: () => deleteNode(node, () => { location.hash = parent ? `#/f/${parent.id}` : '#/'; }) },
    ]);
  });
}

// Reload the project and redraw. `poll`: a periodic refresh, which
// leaves the pages with forms or their own lists alone.
async function refreshProject(opts = {}) {
  if (!P.id) return;
  if (!await loadProject()) return;
  renderSubnav();
  if (opts.poll && !['annotate', 'models', 'detections'].includes(P.section)) return;
  renderSection(opts);
}

function renderSection(opts = {}) {
  const fn = {
    upload: renderUpload,
    annotate: renderAnnotate,
    dataset: renderDataset,
    classes: renderClasses,
    train: renderTrain,
    models: renderModels,
    detections: renderDetections,
  }[P.section];
  fn(opts);
}

function pageHead(title, sub, actions = '') {
  return `<header class="page-head"><div><h1>${esc(title)}</h1>${sub ? `<div class="sub">${sub}</div>` : ''}</div><div class="actions">${actions}</div></header>`;
}

// Live data (frames being sampled, training) is polled while shown.
let pollTimer = null;
function stopPolling() { clearTimeout(pollTimer); pollTimer = null; }
function schedulePoll() {
  stopPolling();
  const busy = P.data?.batches.some((b) => b.state === 'processing' || b.autolabeling) || P.jobs.some((j) => j.status === 'running');
  if (!busy) return;
  pollTimer = setTimeout(async () => {
    if (!P.id) return;
    const was = P.jobs.filter((j) => j.status === 'running').map((j) => j.id);
    const labeling = P.data.batches.filter((b) => b.autolabeling).map((b) => b.id);
    await refreshProject({ poll: true });
    for (const b of P.data.batches) {
      if (!labeling.includes(b.id) || b.autolabeling) continue;
      const r = b.autolabel || {};
      toast(r.error ? `Auto label of “${b.name}” stopped: ${r.error}`
        : `Auto-labeled ${plural(r.images || 0, 'image')} of “${b.name}”: ${plural(r.boxes || 0, 'box', 'boxes')} to review`, 5000);
    }
    for (const j of P.jobs) {
      if (was.includes(j.id) && j.status !== 'running') toast(j.kind === 'train' ? 'Training finished' : 'Detection finished');
    }
    schedulePoll();
  }, 1500);
}

// ------------------------------------------------------------------
// Upload
// ------------------------------------------------------------------

// Files waiting to be (or being) uploaded, per project; kept while
// moving between pages.
const U = { projectId: null, items: [], imagesName: '', busy: false, done: null, seq: 0 };

function uploadState() {
  if (U.projectId !== P.id && !U.busy) {
    for (const it of U.items) if (it.url) URL.revokeObjectURL(it.url);
    Object.assign(U, { projectId: P.id, items: [], imagesName: '', done: null });
  }
  return U;
}

function renderUpload() {
  uploadState();
  const main = $('#main');
  main.innerHTML = `<div class="page">
    ${pageHead('Upload', 'Add pictures and videos to this project. They go to the dataset unlabeled, ready to label.')}
    <div class="drop" id="drop">
      <div class="drop-icon">${icon('upload')}</div>
      <h3>Drag and drop images and videos</h3>
      <p>Videos are cut into frames at the rate you choose; each frame becomes an image to label.</p>
      <div class="btns">
        <button class="btn btn-primary" id="pickFiles">${icon('file')} Select files</button>
        <button class="btn" id="pickFolder">${icon('folder')} Select folder</button>
      </div>
      <div class="formats">Images: JPG, PNG, BMP, WEBP, TIFF · Videos: MP4, MOV, M4V, AVI, MKV, WEBM</div>
      <input type="file" id="fileInput" multiple accept="image/*,video/*,.mkv,.avi" hidden>
      <input type="file" id="folderInput" webkitdirectory multiple hidden>
    </div>
    <div id="staged"></div>
  </div>`;
  const drop = $('#drop');
  $('#pickFiles').addEventListener('click', () => $('#fileInput').click());
  $('#pickFolder').addEventListener('click', () => $('#folderInput').click());
  for (const input of [$('#fileInput'), $('#folderInput')]) {
    input.addEventListener('change', (e) => { addFiles([...e.target.files]); e.target.value = ''; });
  }
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', (e) => { if (!drop.contains(e.relatedTarget)) drop.classList.remove('over'); });
  drop.addEventListener('drop', async (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    addFiles(await droppedFiles(e.dataTransfer));
  });
  renderStaged();
}

// Files of a drop, including the contents of dropped folders.
async function droppedFiles(dt) {
  const entries = [...dt.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const out = [];
  const walk = async (entry) => {
    if (entry.isFile) {
      out.push(await new Promise((res, rej) => entry.file(res, rej)));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child);
      } while (batch.length);
    }
  };
  for (const entry of entries) await walk(entry);
  return out;
}

function addFiles(files) {
  if (U.busy) { toast('Wait for the current upload to finish'); return; }
  if (U.done) { U.items = []; U.done = null; }
  let skipped = 0;
  for (const file of files) {
    if (file.name.startsWith('.')) continue;
    const kind = IMAGE_EXT.test(file.name) ? 'image' : VIDEO_EXT.test(file.name) ? 'video' : null;
    if (!kind) { skipped++; continue; }
    const item = { id: ++U.seq, kind, file, name: file.name, state: 'ready', rate: pref('rate') ?? 1 };
    if (kind === 'video') {
      item.batchName = file.name;
      readVideoMeta(item);
    } else if (U.items.filter((i) => i.kind === 'image').length < 4) {
      item.url = URL.createObjectURL(file);
    }
    U.items.push(item);
  }
  if (!U.imagesName) {
    const d = new Date();
    U.imagesName = `Uploaded on ${d.toLocaleDateString()} at ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  }
  if (skipped) toast(`Skipped ${plural(skipped, 'file')} that are not images or videos`);
  renderStaged();
}

// Duration, size and a poster frame, read by the browser (formats it
// can't play just show less).
function readVideoMeta(item) {
  item.url = URL.createObjectURL(item.file);
  const v = document.createElement('video');
  v.muted = true;
  v.preload = 'metadata';
  v.src = item.url;
  v.addEventListener('loadedmetadata', () => {
    item.duration = v.duration;
    item.width = v.videoWidth;
    item.height = v.videoHeight;
    v.currentTime = Math.min(1, v.duration / 2 || 0);
    renderStaged();
  });
  v.addEventListener('seeked', () => {
    try {
      const c = document.createElement('canvas');
      c.width = 192;
      c.height = Math.round((192 * v.videoHeight) / v.videoWidth) || 120;
      c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
      item.poster = c.toDataURL('image/jpeg', 0.7);
      renderStaged();
    } catch { /* not decodable here */ }
  });
}

function estimate(item) {
  if (!item.duration) return 'length unknown';
  if (!item.rate) return `≈ ${fmtInt(Math.round(item.duration * 30))}+ images`;
  return `≈ ${fmtInt(Math.max(1, Math.round(item.duration * item.rate)))} images`;
}

function itemState(item) {
  switch (item.state) {
    case 'uploading': return `<div class="state"><span>Uploading… ${Math.round((item.progress || 0) * 100)}%</span><div class="meter"><i style="width:${(item.progress || 0) * 100}%"></i></div></div>`;
    case 'processing': return `<div class="state ok">${icon('check-circle', 'sm')} Uploaded · ${fmtInt(item.frames)} frames being extracted</div>`;
    case 'done': return `<div class="state ok">${icon('check-circle', 'sm')} Added</div>`;
    case 'error': return `<div class="state err" title="${esc(item.error)}">${icon('alert', 'sm')} ${esc(item.error)}</div>`;
    default: return '';
  }
}

function renderStaged() {
  const box = $('#staged');
  if (!box || U.projectId !== P.id) return;
  const images = U.items.filter((i) => i.kind === 'image');
  const videos = U.items.filter((i) => i.kind === 'video');
  if (!U.items.length) { box.innerHTML = ''; return; }
  const ready = !U.busy && !U.done;
  let html = '<div class="staged">';
  if (images.length) {
    const done = images.filter((i) => i.state === 'done').length;
    const failed = images.filter((i) => i.state === 'error').length;
    const thumbs = images.filter((i) => i.url).slice(0, 4).map((i) => `<img alt="" src="${i.url}">`).join('');
    let state = '';
    if (U.busy || U.done) {
      state = done + failed < images.length
        ? `<div class="state"><span>Uploading… ${done} of ${images.length}</span><div class="meter"><i style="width:${(100 * (done + failed)) / images.length}%"></i></div></div>`
        : `<div class="state ${failed ? 'err' : 'ok'}">${icon(failed ? 'alert' : 'check-circle', 'sm')} ${failed ? `${done} added, ${failed} failed` : `${plural(done, 'image')} added`}</div>`;
    }
    html += `<div class="srow"><div class="thumb"><div class="mosaic">${thumbs}</div><span class="badge">${images.length}</span></div>`
      + `<div class="info"><input class="input name-input" id="imagesName" value="${esc(U.imagesName)}" ${ready ? '' : 'disabled'} title="Batch name">`
      + `<div class="meta">${plural(images.length, 'image')} · one batch</div></div>`
      + (state || (ready ? `<button class="icon-btn" data-remove="images" title="Remove">${icon('x')}</button>` : ''))
      + `</div>`;
  }
  for (const v of videos) {
    const meta = [v.duration ? fmtDuration(v.duration) : null, v.width ? `${v.width}×${v.height}` : null, `${(v.file.size / 1e6).toFixed(1)} MB`].filter(Boolean).join(' · ');
    html += `<div class="srow"><div class="thumb">${v.poster ? `<img alt="" src="${v.poster}">` : icon('film')}${v.duration ? `<span class="badge">${fmtDuration(v.duration)}</span>` : ''}</div>`
      + `<div class="info"><input class="input name-input" data-name="${v.id}" value="${esc(v.batchName)}" ${ready ? '' : 'disabled'} title="Batch name">`
      + `<div class="meta">${icon('film', 'sm')} ${esc(v.name)} · ${meta}</div></div>`
      + (ready
        ? `<div class="opts"><select class="select" data-rate="${v.id}" title="How many frames to keep">${RATES.map(([r, label]) => `<option value="${r}" ${r === v.rate ? 'selected' : ''}>${label}</option>`).join('')}</select>`
          + `<span class="estimate">${estimate(v)}</span><button class="icon-btn" data-remove="${v.id}" title="Remove">${icon('x')}</button></div>`
        : itemState(v))
      + `</div>`;
  }
  html += '</div>';
  const count = images.length + videos.reduce((n, v) => n + (v.duration ? (v.rate ? Math.round(v.duration * v.rate) : Math.round(v.duration * 30)) : 0), 0);
  if (ready) {
    html += `<div class="upload-foot"><span class="summary">${[images.length && plural(images.length, 'image'), videos.length && plural(videos.length, 'video')].filter(Boolean).join(' and ')}${count ? ` · about <b>${fmtInt(count)}</b> images to label` : ''}</span><span class="grow"></span>`
      + `<button class="btn btn-ghost" id="clearStaged">Clear</button><button class="btn btn-primary" id="startUpload">${icon('upload')} Add to dataset</button></div>`;
    if (count > 3000) html += `<div class="callout warn">${icon('alert')}<div>That's a lot of images to label. A lower frame rate keeps nearly the same variety: neighbouring frames of a video look alike.</div></div>`;
  } else if (U.done) {
    const processing = videos.some((v) => v.state === 'processing');
    html += `<div class="callout ok">${icon('check-circle')}<div style="flex:1"><b>${U.items.some((i) => i.state === 'error') ? 'The rest was added to the dataset.' : 'Added to the dataset.'}</b> ${processing ? 'Video frames are still being extracted; they appear on the Annotate page as they come in.' : 'It is ready to label.'}</div>`
      + `<button class="btn btn-sm" id="uploadMore">Upload more</button><a class="btn btn-sm btn-primary" href="#/p/${P.id}/annotate">${icon('pen', 'sm')} Go to Annotate</a></div>`;
  }
  box.innerHTML = html;
  $('#imagesName')?.addEventListener('input', (e) => { U.imagesName = e.target.value; });
  for (const el of box.querySelectorAll('[data-name]')) {
    el.addEventListener('input', (e) => { U.items.find((i) => i.id === Number(el.dataset.name)).batchName = e.target.value; });
  }
  for (const el of box.querySelectorAll('[data-rate]')) {
    el.addEventListener('change', (e) => {
      const it = U.items.find((i) => i.id === Number(el.dataset.rate));
      it.rate = Number(e.target.value);
      pref('rate', it.rate);
      renderStaged();
    });
  }
  for (const el of box.querySelectorAll('[data-remove]')) {
    el.addEventListener('click', () => {
      const which = el.dataset.remove;
      U.items = U.items.filter((i) => {
        const gone = which === 'images' ? i.kind === 'image' : i.id === Number(which);
        if (gone && i.url) URL.revokeObjectURL(i.url);
        return !gone;
      });
      renderStaged();
    });
  }
  $('#clearStaged')?.addEventListener('click', () => { U.items = []; renderStaged(); });
  $('#startUpload')?.addEventListener('click', startUpload);
  $('#uploadMore')?.addEventListener('click', () => { U.items = []; U.done = null; renderStaged(); });
}

function sendFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.detail || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('the connection to the server was lost'));
    xhr.send(file);
  });
}

async function startUpload() {
  const projectId = P.id;
  const images = U.items.filter((i) => i.kind === 'image');
  const videos = U.items.filter((i) => i.kind === 'video');
  U.busy = true;
  renderStaged();
  const base = `/api/ws/projects/${projectId}/batches`;
  try {
    if (images.length) {
      const batch = await api(base, { method: 'POST', body: { name: U.imagesName.trim() || 'Upload', kind: 'images' } });
      let next = 0;
      const worker = async () => {
        while (next < images.length) {
          const item = images[next++];
          try {
            await sendFile(`${base}/${batch.id}/files?name=${encodeURIComponent(item.name)}`, item.file);
            item.state = 'done';
          } catch (err) {
            item.state = 'error';
            item.error = err.message;
          }
          renderStaged();
        }
      };
      await Promise.all(Array.from({ length: UPLOAD_CONCURRENCY }, worker));
    }
    for (const v of videos) {
      let batch = null;
      try {
        batch = await api(base, { method: 'POST', body: { name: v.batchName.trim() || v.name, kind: 'video' } });
        v.state = 'uploading';
        renderStaged();
        const q = new URLSearchParams({ name: v.name });
        if (v.rate) q.set('rate', v.rate);
        const r = await sendFile(`${base}/${batch.id}/files?${q}`, v.file, (f) => { v.progress = f; renderStaged(); });
        v.state = 'processing';
        v.frames = r.frames;
      } catch (err) {
        v.state = 'error';
        v.error = err.message;
        // Don't leave an empty batch behind.
        if (batch) api(`${base}/${batch.id}`, { method: 'DELETE' }).catch(() => {});
      }
      renderStaged();
    }
  } catch (err) {
    toast(err.message);
  } finally {
    U.busy = false;
    U.done = true;
  }
  const failed = U.items.filter((i) => i.state === 'error').length;
  toast(failed ? `${plural(failed, 'file')} could not be added` : 'Added to the dataset');
  if (P.id === projectId) {
    await refreshProject();
    schedulePoll();
  }
  renderStaged();
}

window.addEventListener('beforeunload', (e) => {
  if (U.busy) e.preventDefault();
});

// ------------------------------------------------------------------
// Annotate: batches and their progress
// ------------------------------------------------------------------

function batchAction(b) {
  if (b.state === 'processing' && !b.images) return '';
  if (!b.images) return '';
  if (b.autolabeling) return `<button class="btn btn-sm" data-act="stop-al">${icon('stop', 'sm')} Stop</button>`;
  const auto = b.unlabeled && b.state !== 'processing'
    ? `<button class="btn btn-sm" data-act="autolabel" title="Find the objects on every unlabeled image with a model">${icon('sparkle', 'sm')} Auto label</button>` : '';
  return auto + reviewAction(b);
}

function reviewAction(b) {
  if (b.unlabeled) return `<a class="btn btn-sm btn-primary" href="${annotateUrl(P.id, b.id)}">${icon('pen', 'sm')} Annotate</a>`;
  if (b.predicted) return `<a class="btn btn-sm btn-primary" href="${annotateUrl(P.id, b.id)}">${icon('check-circle', 'sm')} Review</a>`;
  return `<a class="btn btn-sm" href="${annotateUrl(P.id, b.id)}">Open</a>`;
}

function batchMeta(b) {
  const src = b.source || {};
  const parts = [];
  if (b.kind === 'video') {
    if (src.rate) parts.push(`${src.rate < 1 ? `1 frame every ${Math.round(1 / src.rate)} s` : `${src.rate} frames/s`}`);
    else if (src.imported) parts.push(`from ${src.detector || 'a detector'}`);
    else if (src.video) parts.push('every frame');
  }
  parts.push(plural(b.images, 'image'));
  if (b.boxes) parts.push(plural(b.boxes, 'box', 'boxes'));
  parts.push(`added ${timeAgo(b.created)}`);
  const al = b.autolabel;
  if (al && !b.autolabeling) {
    parts.push(al.error
      ? `<span style="color:var(--danger)" title="${esc(al.error)}">auto label failed</span>`
      : `auto-labeled with ${esc(al.model || al.source)} ${timeAgo(al.finished)}`);
  }
  return parts.join(' · ');
}

function batchCard(b) {
  const el = document.createElement('div');
  el.className = 'card bcard';
  const kind = b.kind === 'video'
    ? `<span class="kind-chip" title="${esc(b.source?.video || '')}">${icon('film')} Video</span>`
    : `<span class="kind-chip">${icon('images')} Images</span>`;
  const run = b.autolabeling;
  let state = '';
  if (b.state === 'processing') state = '<span class="state-chip processing">Extracting frames</span>';
  else if (run) state = '<span class="state-chip processing">Auto-labeling</span>';
  else if (b.state === 'interrupted') state = '<span class="state-chip interrupted" title="The server stopped while extracting frames; the ones extracted are kept">Interrupted</span>';
  else if (b.state === 'failed') state = `<span class="state-chip failed" title="${esc(b.error || '')}">Failed</span>`;
  else if (b.images && b.labeled === b.images) state = `<span class="state-chip done">${icon('check', 'sm')} Done</span>`;
  let progress;
  if (b.state === 'processing' && b.progress) {
    progress = `<div style="max-width:520px;margin-top:8px"><div class="meter"><i style="width:${(100 * b.progress.done) / Math.max(1, b.progress.total)}%"></i></div><div class="meta">${fmtInt(b.progress.done)} of ${fmtInt(b.progress.total)} frames</div></div>`;
  } else if (run) {
    const finished = run.done + run.failed + run.skipped;
    const left = run.done ? ((Date.now() / 1000 - run.started) / finished) * (run.total - finished) : null;
    progress = `<div style="max-width:520px;margin-top:8px"><div class="meter${finished ? '' : ' indeterminate'}"><i style="width:${(100 * finished) / Math.max(1, run.total)}%"></i></div>`
      + `<div class="meta">${icon('sparkle', 'sm')} Auto-labeling with ${esc(run.model)}: ${fmtInt(finished)} of ${fmtInt(run.total)} images · ${plural(run.boxes, 'box', 'boxes')}`
      + `${run.failed ? ` · <span style="color:var(--danger)" title="${esc(run.last_error || '')}">${fmtInt(run.failed)} failed</span>` : ''}`
      + `${left != null && finished < run.total ? ` · about ${fmtDuration(left)} left` : ''}</div></div>`;
  } else {
    progress = b.images ? stackBar(b) + legend(b) : '';
  }
  // The cover and the name open the batch in the annotator (once it has images).
  const href = b.images ? annotateUrl(P.id, b.id) : null;
  const cover = b.cover != null ? `<img loading="lazy" alt="" src="${thumbUrl(P.id, b.cover)}">` : icon(b.kind === 'video' ? 'film' : 'images');
  el.innerHTML = (href ? `<a class="cover-img" href="${href}" title="Open in the annotator">${cover}</a>` : `<div class="cover-img">${cover}</div>`)
    + `<div class="binfo"><div class="bname"><h3 title="${esc(b.name)}">${href ? `<a href="${href}">${esc(b.name)}</a>` : esc(b.name)}</h3>${kind}${state}</div>`
    + `<div class="meta">${batchMeta(b)}</div>${progress}</div>`
    + `<div class="bactions">${batchAction(b)}<button class="icon-btn" data-menu title="More">${icon('more')}</button></div>`;
  el.querySelector('[data-act=autolabel]')?.addEventListener('click', () => autoLabelDialog(b));
  el.querySelector('[data-act=stop-al]')?.addEventListener('click', async (e) => {
    e.currentTarget.disabled = true;
    try {
      await api(`/api/ws/projects/${P.id}/batches/${b.id}/autolabel/stop`, { method: 'POST' });
      toast('Stopping after the current image…');
    } catch (err) { toast(err.message); }
  });
  el.querySelector('[data-menu]').addEventListener('click', (e) => {
    e.stopPropagation();
    const items = [
      { label: 'Rename…', icon: 'edit', action: () => renameBatch(b) },
      { label: 'View in dataset', icon: 'grid', action: () => { DS.reset(); DS.batch = b.id; location.hash = `#/p/${P.id}/dataset`; } },
    ];
    // The row has an Auto label button while there are unlabeled images;
    // after that, this re-runs it on the pre-labeled ones.
    if (!b.autolabeling && !b.unlabeled && b.predicted && b.state !== 'processing') {
      items.push({ label: 'Auto label again…', icon: 'sparkle', action: () => autoLabelDialog(b) });
    }
    if (b.predicted) items.push({ label: `Accept ${plural(b.predicted, 'pre-label')}…`, icon: 'check-circle', action: () => acceptPredicted(b) });
    items.push('-', { label: 'Delete batch…', icon: 'trash', danger: true, action: () => deleteBatch(b) });
    openMenu(e.currentTarget, items);
  });
  return el;
}

// Auto label: run a detector (Label Assist's) on every image of a
// batch still to label; what it finds is saved as pre-labels.
const AL_SOURCES = [
  ['locate', 'Locate Anything', 'Local model, any class name. About 5 s an image.'],
  ['cloud', 'Claude', 'Cloud model, any class name. One API call per image.'],
  ['rfdetr', 'RF-DETR (COCO)', 'Local and fast, but only the 80 COCO classes.'],
  ['trained', 'Trained model', 'A model you trained; it finds its own classes.'],
];

async function autoLabelDialog(b) {
  let data;
  try { data = await api(`/api/ws/projects/${P.id}/assist`); } catch (err) { toast(err.message); return; }
  const src = data.sources;
  const saved = pref('autolabel') || {};
  const st = {
    source: [saved.source, 'locate', 'cloud', 'trained', 'rfdetr'].find((s) => s && src[s]?.available) || 'locate',
    model: { cloud: src.cloud.default, rfdetr: src.rfdetr.default, trained: src.trained.models?.[0]?.path, ...(saved.model || {}) },
    classes: P.data.classes.map((c) => c.name),
    images: b.unlabeled ? 'unlabeled' : 'todo',
    minConf: '',
  };
  if (!src.trained.models?.some((m) => m.path === st.model.trained)) st.model.trained = src.trained.models?.[0]?.path;
  const trainedModel = () => src.trained.models?.find((m) => m.path === st.model.trained);
  if (!st.classes.length && saved.classes?.length) st.classes = [...saved.classes];
  const count = () => (st.images === 'unlabeled' ? b.unlabeled : b.unlabeled + b.predicted);
  const threshold = () => ({ cloud: src.cloud.threshold, rfdetr: src.rfdetr.threshold, trained: trainedModel()?.threshold }[st.source]);

  const draw = (body) => {
    const n = count();
    const suggestions = st.source === 'rfdetr' ? data.coco
      : st.source === 'trained' ? (trainedModel()?.classes || []) : P.data.classes.map((c) => c.name);
    let model = '';
    if (st.source === 'cloud') model = src.cloud.models.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join('');
    if (st.source === 'rfdetr') model = src.rfdetr.sizes.map((m) => `<option value="${m}">RF-DETR ${m}</option>`).join('');
    if (st.source === 'trained') model = (src.trained.models || []).map((m) => `<option value="${esc(m.path)}">${esc(m.name)} · ${esc(m.classes.join(', '))}</option>`).join('');
    const eta = st.source === 'locate' ? ` · about ${fmtDuration(n * 5)}` : st.source === 'cloud' ? ` · ${plural(n, 'API call')}` : '';
    body.innerHTML = `<p>Finds the objects on the images of <b>${esc(b.name)}</b> and saves them as <b>pre-labels</b>, to check in the annotator (<kbd>M</kbd> accepts an image). Images you labeled are never changed.</p>`
      + `<div class="field"><span>Detector</span><div class="radio-cards">${AL_SOURCES.map(([key, label, hint]) => {
        const s = src[key];
        const why = s.available ? hint : s.reason;
        return `<label class="radio-card${st.source === key ? ' on' : ''}${s.available ? '' : ' off'}" title="${esc(why || '')}"><input type="radio" name="source" value="${key}" ${st.source === key ? 'checked' : ''} ${s.available ? '' : 'disabled'}>`
          + `<div><b>${label}${key === 'locate' && s.model ? ` <small class="mono">${esc(s.model)}</small>` : ''}</b><small>${esc(why || '')}</small></div></label>`;
      }).join('')}</div></div>`
      + (model ? `<label class="field"><span>Model</span><select class="select" name="model">${model}</select></label>` : '')
      + `<div class="field"><span>Classes to find</span><div class="classes-row" id="alChips">${st.classes.map((c, i) => `<span class="chip">${esc(c)}<button type="button" class="chip-x" data-rm="${i}" title="Remove">${icon('x', 'sm')}</button></span>`).join('') || '<span class="muted">None yet: add the classes to look for.</span>'}</div>`
      + `<div class="add-row"><input class="input" id="alAdd" list="alList" placeholder="${st.source === 'rfdetr' ? 'A COCO class, e.g. person' : 'Add a class, e.g. hand'}" autocomplete="off" spellcheck="false"><button type="button" class="btn" id="alAddBtn">${icon('plus', 'sm')} Add</button>`
      + `<datalist id="alList">${suggestions.filter((c) => !st.classes.some((x) => norm(x) === norm(c))).map((c) => `<option value="${esc(c)}">`).join('')}</datalist></div></div>`
      + `<div class="field"><span>Images</span><div class="radio-cards">`
      + `<label class="radio-card${st.images === 'unlabeled' ? ' on' : ''}${b.unlabeled ? '' : ' off'}"><input type="radio" name="images" value="unlabeled" ${st.images === 'unlabeled' ? 'checked' : ''} ${b.unlabeled ? '' : 'disabled'}><div><b>Unlabeled images</b><small>${plural(b.unlabeled, 'image')}</small></div></label>`
      + `<label class="radio-card${st.images === 'todo' ? ' on' : ''}${b.predicted ? '' : ' off'}"><input type="radio" name="images" value="todo" ${st.images === 'todo' ? 'checked' : ''} ${b.predicted ? '' : 'disabled'}><div><b>Unlabeled and pre-labeled images</b><small>${plural(b.unlabeled + b.predicted, 'image')}; the pre-labels there are replaced</small></div></label>`
      + `</div></div>`
      + (st.source === 'locate' ? '' : `<label class="field"><span>Min. confidence</span><input class="input" type="number" name="minConf" min="0" max="1" step="0.05" value="${esc(st.minConf)}" placeholder="${threshold() ?? 'auto'}" title="Boxes below it are left out. Empty: the detector's own threshold."></label>`)
      + `<p class="muted" style="font-size:13px">${plural(n, 'image')}${eta}. It runs in the background: you can keep working, or stop it from the batch.</p>`;
    const ok = body.closest('form').querySelector('[type=submit]');
    ok.innerHTML = `${icon('sparkle', 'sm')} Auto label ${plural(n, 'image')}`;
    ok.disabled = !n || !st.classes.length || !src[st.source]?.available;
    const sel = body.querySelector('[name=model]');
    if (sel) {
      sel.value = st.model[st.source] ?? sel.value;
      st.model[st.source] = sel.value;
      sel.addEventListener('change', () => {
        st.model[st.source] = sel.value;
        if (st.source === 'trained') {
          const own = trainedModel()?.classes || [];
          // A trained model only finds its own classes.
          const keep = st.classes.filter((c) => own.some((o) => norm(o) === norm(c)));
          st.classes = keep.length ? keep : [...own];
        }
        draw(body);
      });
    }
    for (const r of body.querySelectorAll('[name=source]')) {
      r.addEventListener('change', () => {
        st.source = r.value;
        if (st.source === 'trained' && !st.classes.some((c) => (trainedModel()?.classes || []).some((o) => norm(o) === norm(c)))) {
          st.classes = [...(trainedModel()?.classes || [])];
        }
        draw(body);
      });
    }
    for (const r of body.querySelectorAll('[name=images]')) r.addEventListener('change', () => { st.images = r.value; draw(body); });
    body.querySelector('[name=minConf]')?.addEventListener('input', (e) => { st.minConf = e.target.value; });
    for (const x of body.querySelectorAll('[data-rm]')) {
      x.addEventListener('click', () => { st.classes.splice(Number(x.dataset.rm), 1); draw(body); });
    }
    const add = () => {
      const name = body.querySelector('#alAdd').value.trim();
      if (name && !st.classes.some((c) => norm(c) === norm(name))) st.classes.push(name);
      draw(body);
      body.querySelector('#alAdd').focus();
    };
    body.querySelector('#alAddBtn').addEventListener('click', add);
    // A typed class is enough to start (it is added on submit).
    body.querySelector('#alAdd').addEventListener('input', (e) => {
      ok.disabled = !n || !src[st.source]?.available || (!st.classes.length && !e.target.value.trim());
    });
    body.querySelector('#alAdd').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } });
  };

  const started = await dialog({
    title: 'Auto label',
    okText: 'Auto label',
    wide: true,
    render: draw,
    onOk: async () => {
      // A class typed but not added yet counts too.
      const typed = $('#alAdd')?.value.trim();
      if (typed && !st.classes.some((c) => norm(c) === norm(typed))) st.classes.push(typed);
      if (!st.classes.length) throw new Error('Add at least one class to find.');
      const body = { source: st.source, classes: st.classes.map((name) => ({ name })), images: st.images };
      if (st.source !== 'locate') body.model = st.model[st.source];
      if (st.minConf !== '' && st.source !== 'locate') body.min_confidence = Number(st.minConf);
      const r = await api(`/api/ws/projects/${P.id}/batches/${b.id}/autolabel`, { method: 'POST', body });
      pref('autolabel', { source: st.source, model: st.model, classes: st.classes });
      return r;
    },
  });
  if (!started) return;
  toast(`Auto-labeling ${plural(started.total, 'image')}`);
  await refreshProject();
  schedulePoll();
}

async function renameBatch(b) {
  const name = await askName('Rename batch', b.name);
  if (!name || name === b.name) return;
  try {
    await api(`/api/ws/projects/${P.id}/batches/${b.id}`, { method: 'PATCH', body: { name } });
    refreshProject();
  } catch (err) { toast(err.message); }
}

async function acceptPredicted(b) {
  const ok = await confirmDialog('Accept the pre-labels?',
    `<p>Mark the ${plural(b.predicted, 'pre-labeled image')} of <b>${esc(b.name)}</b> as labeled, with their boxes as they are?</p>`
    + `<p class="muted">Only do this if you've checked them: labeled images are trained on with all their boxes, whatever their confidence.</p>`,
    'Mark as labeled', false);
  if (!ok) return;
  try {
    const r = await api(`/api/ws/projects/${P.id}/images/status`, { method: 'POST', body: { batch: b.id, from_status: 'predicted', status: 'labeled' } });
    toast(`${plural(r.changed, 'image')} marked as labeled`);
    refreshProject();
  } catch (err) { toast(err.message); }
}

async function deleteBatch(b) {
  const ok = await confirmDialog('Delete batch?',
    `<p>Delete <b>${esc(b.name)}</b> and its ${plural(b.images, 'image')}${b.boxes ? `, with ${plural(b.boxes, 'box', 'boxes')}` : ''}?</p><p class="muted">Models already trained on them are kept. This can't be undone.</p>`);
  if (!ok) return;
  try {
    await api(`/api/ws/projects/${P.id}/batches/${b.id}`, { method: 'DELETE' });
    toast(`Deleted “${b.name}”`);
    refreshProject();
  } catch (err) { toast(err.message); }
}

function renderAnnotate() {
  const d = P.data;
  const s = d.stats || {};
  const main = $('#main');
  const keep = main.scrollTop;
  const todo = d.batches.filter((b) => b.state === 'processing' || b.unlabeled || b.predicted);
  const done = d.batches.filter((b) => !todo.includes(b));
  const next = todo.find((b) => b.unlabeled) || todo.find((b) => b.predicted);
  let html = `<div class="page">${pageHead('Annotate', 'Each upload is a batch. Open one to draw boxes; <b>pre-labeled</b> images have boxes from a model that still need a look.',
    `<a class="btn" href="#/p/${d.id}/upload">${icon('upload')} Upload more</a>`)}`;
  if (!d.batches.length) {
    html += `<div class="empty">${icon('upload', 'big')}<h3>Nothing to label yet</h3><p>Upload pictures or videos first; they show up here as batches.</p><div class="btns"><a class="btn btn-primary" href="#/p/${d.id}/upload">${icon('upload')} Upload</a></div></div></div>`;
    main.innerHTML = html;
    return;
  }
  const pct = s.images ? Math.round((100 * s.labeled) / s.images) : 0;
  html += `<div class="overview"><div><div class="big-num">${pct}%<small>${fmtInt(s.labeled)} of ${plural(s.images, 'image')} labeled</small></div>${stackBar(s, true)}${legend(s)}</div>`
    + `<div>${next ? `<a class="btn btn-primary" href="${annotateUrl(d.id, next.id)}">${icon('pen')} ${next.unlabeled ? 'Continue labeling' : 'Review pre-labels'}</a>` : `<a class="btn" href="${annotateUrl(d.id)}">${icon('pen')} Open all images</a>`}</div></div>`;
  if (todo.length) html += `<h4 class="section-title">To do · ${todo.length}</h4><div class="batches" id="todoList"></div>`;
  if (done.length) html += `<h4 class="section-title">Done · ${done.length}</h4><div class="batches" id="doneList"></div>`;
  html += '</div>';
  main.innerHTML = html;
  for (const b of todo) $('#todoList').append(batchCard(b));
  for (const b of done) $('#doneList').append(batchCard(b));
  main.scrollTop = keep;
}

// ------------------------------------------------------------------
// Dataset: every image, filtered
// ------------------------------------------------------------------

const DS = {
  status: '', batch: '', cls: '', tag: '', q: '', sort: 'newest', page: 0, per: pref('per') || 100,
  show: pref('showBoxes') ?? true, selected: new Set(), data: null, seq: 0,
  reset() { Object.assign(this, { status: '', batch: '', cls: '', tag: '', q: '', page: 0, selected: new Set(), data: null }); },
};

function renderDataset() {
  const d = P.data;
  const s = d.stats || {};
  if (P.query?.get('status')) { DS.status = P.query.get('status'); DS.page = 0; P.query = null; }
  const main = $('#main');
  const chip = (value, label, n) => `<button class="fchip${DS.status === value ? ' on' : ''}" data-status="${value}">${value ? `<i class="sw ${value}"></i>` : ''}${label} <span class="n">${fmtInt(n)}</span></button>`;
  main.innerHTML = `<div class="page">
    ${pageHead('Dataset', `${plural(s.images || 0, 'image')} in ${plural(d.batches.length, 'batch', 'batches')}`, `<a class="btn" href="#/p/${d.id}/upload">${icon('upload')} Upload</a>`)}
    <div class="chips-row">${chip('', 'All', s.images)}${chip('unlabeled', 'Unlabeled', s.unlabeled)}${chip('predicted', 'Pre-labeled', s.predicted)}${chip('labeled', 'Labeled', s.labeled)}</div>
    <div class="ds-toolbar">
      <label class="search">${icon('search')}<input id="dsSearch" placeholder="Search by file name" value="${esc(DS.q)}" autocomplete="off" spellcheck="false"></label>
      <select class="select" id="dsBatch"><option value="">All batches</option>${d.batches.map((b) => `<option value="${b.id}">${esc(b.name)}</option>`).join('')}</select>
      <select class="select" id="dsClass"><option value="">All classes</option>${d.classes.map((c) => `<option value="${esc(c.name)}">${esc(c.name)} (${fmtInt(c.boxes)})</option>`).join('')}<option value="" disabled>──</option><option value="__empty">Labeled, no boxes</option></select>
      <select class="select" id="dsTag" title="Tags added to images in the annotator"><option value="">All tags</option>${d.tags.map((t) => `<option value="${esc(t.name)}">${esc(t.name)} (${fmtInt(t.images)})</option>`).join('')}<option value="" disabled>──</option><option value="__none">No tags</option></select>
      <select class="select" id="dsSort"><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="name">File name</option></select>
      <span class="grow"></span>
      <label class="toggle"><input type="checkbox" id="dsShow" ${DS.show ? 'checked' : ''}><span class="knob"></span> Show annotations</label>
    </div>
    <div id="selbar"></div>
    <div id="dsGrid"><div class="igrid">${'<div class="skeleton" style="aspect-ratio:16/10"></div>'.repeat(10)}</div></div>
    <div id="dsPager"></div>
  </div>`;
  $('#dsBatch').value = DS.batch;
  $('#dsClass').value = DS.cls === '' && DS.status === 'empty' ? '__empty' : DS.cls;
  $('#dsSort').value = DS.sort;
  if (DS.tag && DS.tag !== '__none' && !d.tags.some((t) => t.name === DS.tag)) DS.tag = '';
  $('#dsTag').value = DS.tag;
  $('#dsTag').addEventListener('change', (e) => { DS.tag = e.target.value; DS.page = 0; loadImages(); });
  for (const b of $$('[data-status]')) {
    b.addEventListener('click', () => { DS.status = b.dataset.status; DS.page = 0; DS.selected.clear(); renderDataset(); });
  }
  let timer = null;
  $('#dsSearch').addEventListener('input', (e) => { clearTimeout(timer); timer = setTimeout(() => { DS.q = e.target.value; DS.page = 0; loadImages(); }, 200); });
  $('#dsBatch').addEventListener('change', (e) => { DS.batch = e.target.value; DS.page = 0; loadImages(); });
  $('#dsClass').addEventListener('change', (e) => {
    if (e.target.value === '__empty') { DS.status = 'empty'; DS.cls = ''; } else { DS.cls = e.target.value; if (DS.status === 'empty') DS.status = ''; }
    DS.page = 0;
    renderDataset();
  });
  $('#dsSort').addEventListener('change', (e) => { DS.sort = e.target.value; DS.page = 0; loadImages(); });
  $('#dsShow').addEventListener('change', (e) => { DS.show = e.target.checked; pref('showBoxes', DS.show); renderGrid(); });
  loadImages();
}

async function loadImages() {
  const seq = ++DS.seq;
  const q = new URLSearchParams({ offset: DS.page * DS.per, limit: DS.per, sort: DS.sort === 'oldest' ? 'oldest' : DS.sort });
  if (DS.status) q.set('status', DS.status);
  if (DS.batch) q.set('batch', DS.batch);
  if (DS.cls) q.set('cls', DS.cls);
  if (DS.tag) q.set('tag', DS.tag);
  if (DS.q.trim()) q.set('q', DS.q.trim());
  try {
    const data = await api(`/api/ws/projects/${P.id}/images?${q}`);
    if (seq !== DS.seq) return;
    DS.data = data;
  } catch (err) {
    if (seq === DS.seq) $('#dsGrid').innerHTML = `<div class="empty">${icon('alert', 'big')}<h3>Could not load the images</h3><p>${esc(err.message)}</p></div>`;
    return;
  }
  renderGrid();
  renderPager();
}

function boxesSvg(img, classes) {
  if (!DS.show || !img.boxes.length) return '';
  const sw = Math.max(img.width, img.height) / 160;
  return `<svg class="boxes" viewBox="0 0 ${img.width} ${img.height}" preserveAspectRatio="xMidYMid meet">`
    + img.boxes.map(([label, x1, y1, x2, y2]) => {
      const c = colorOf(label, classes);
      return `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" fill="${c}" fill-opacity=".14" stroke="${c}" stroke-width="${sw}"/>`;
    }).join('')
    + '</svg>';
}

function renderGrid() {
  const box = $('#dsGrid');
  if (!box || !DS.data) return;
  const classes = P.data.classes.map((c) => c.name);
  if (!DS.data.images.length) {
    const filtered = DS.status || DS.batch || DS.cls || DS.q.trim();
    box.innerHTML = filtered
      ? `<div class="empty">${icon('search', 'big')}<h3>No images match</h3><p>Try other filters.</p></div>`
      : `<div class="empty">${icon('images', 'big')}<h3>No images yet</h3><p>Upload pictures or videos to start the dataset.</p><div class="btns"><a class="btn btn-primary" href="#/p/${P.id}/upload">${icon('upload')} Upload</a></div></div>`;
    renderSelbar();
    return;
  }
  const grid = document.createElement('div');
  grid.className = `igrid${DS.selected.size ? ' selecting' : ''}`;
  for (const img of DS.data.images) {
    const tile = document.createElement('div');
    tile.className = `tile${DS.selected.has(img.id) ? ' selected' : ''}`;
    tile.innerHTML = `<div class="frame" title="${esc(img.name)} · ${esc(img.batch_name || '')}">`
      + `<img loading="lazy" decoding="async" alt="" src="${thumbUrl(P.id, img.id)}">${boxesSvg(img, classes)}`
      + `<span class="pick">${icon('check')}</span>`
      + `<span class="st"><i class="sw ${img.status}"></i>${STATUS_NAMES[img.status]}</span>`
      + (img.boxes.length ? `<span class="count">${img.boxes.length}</span>` : '')
      + `</div><div class="name" title="${esc(img.name)}">${esc(midTrim(img.name))}</div>`
      + (img.tags.length ? `<div class="tile-tags">${img.tags.map((t) => `<button type="button" class="tag-pill" data-tag="${esc(t)}" title="Show the images tagged “${esc(t)}”">${esc(t)}</button>`).join('')}</div>` : '');
    for (const b of tile.querySelectorAll('[data-tag]')) {
      b.addEventListener('click', () => { DS.tag = b.dataset.tag; DS.page = 0; renderDataset(); });
    }
    const toggleSel = () => {
      if (DS.selected.has(img.id)) DS.selected.delete(img.id); else DS.selected.add(img.id);
      tile.classList.toggle('selected', DS.selected.has(img.id));
      grid.classList.toggle('selecting', DS.selected.size > 0);
      renderSelbar();
    };
    tile.querySelector('.frame').addEventListener('click', (e) => {
      if (e.target.closest('.pick') || e.metaKey || e.ctrlKey || e.shiftKey || DS.selected.size) { toggleSel(); return; }
      location.href = annotateUrl(P.id, img.batch, img.id);
    });
    grid.append(tile);
  }
  box.replaceChildren(grid);
  renderSelbar();
}

function renderSelbar() {
  const bar = $('#selbar');
  if (!bar) return;
  if (!DS.selected.size) { bar.innerHTML = ''; return; }
  const n = DS.selected.size;
  bar.innerHTML = `<div class="selbar"><b>${plural(n, 'image')} selected</b><span class="grow"></span>`
    + `<button class="btn btn-sm" data-act="all">Select page</button>`
    + `<button class="btn btn-sm" data-act="labeled">${icon('check-circle', 'sm')} Mark labeled</button>`
    + `<button class="btn btn-sm" data-act="unlabeled">Mark unlabeled</button>`
    + `<button class="btn btn-sm" data-act="tag">${icon('tag', 'sm')} Tag…</button>`
    + `<button class="btn btn-sm danger" data-act="delete">${icon('trash', 'sm')} Delete</button>`
    + `<button class="icon-btn" data-act="clear" title="Clear the selection" style="color:#fff">${icon('x')}</button></div>`;
  bar.querySelector('[data-act=all]').addEventListener('click', () => { for (const i of DS.data.images) DS.selected.add(i.id); renderGrid(); });
  bar.querySelector('[data-act=clear]').addEventListener('click', () => { DS.selected.clear(); renderGrid(); });
  for (const status of ['labeled', 'unlabeled']) {
    bar.querySelector(`[data-act=${status}]`).addEventListener('click', async () => {
      try {
        const r = await api(`/api/ws/projects/${P.id}/images/status`, { method: 'POST', body: { ids: [...DS.selected], status } });
        toast(`${plural(r.changed, 'image')} marked ${STATUS_NAMES[status].toLowerCase()}${status === 'unlabeled' ? ' (their boxes are kept)' : ''}`);
        DS.selected.clear();
        await refreshProject();
      } catch (err) { toast(err.message); }
    });
  }
  bar.querySelector('[data-act=tag]').addEventListener('click', () => tagSelected());
  bar.querySelector('[data-act=delete]').addEventListener('click', async () => {
    if (!await confirmDialog('Delete images?', `<p>Delete ${plural(n, 'image')} and their boxes from the project?</p><p class="muted">This can't be undone.</p>`)) return;
    try {
      const r = await api(`/api/ws/projects/${P.id}/images/delete`, { method: 'POST', body: { ids: [...DS.selected] } });
      toast(`Deleted ${plural(r.deleted, 'image')}`);
      DS.selected.clear();
      await refreshProject();
    } catch (err) { toast(err.message); }
  });
}

// Add a tag to the selected images, or take one off them.
async function tagSelected() {
  const n = DS.selected.size;
  let mode = 'add';
  const r = await dialog({
    title: `Tag ${plural(n, 'image')}`,
    okText: 'Apply',
    render: (body) => {
      body.innerHTML = `<div class="radio-cards">`
        + `<label class="radio-card on"><input type="radio" name="mode" value="add" checked><div><b>Add a tag</b><small>To every selected image</small></div></label>`
        + `<label class="radio-card"><input type="radio" name="mode" value="remove"><div><b>Remove a tag</b><small>From the selected images that have it</small></div></label></div>`
        + `<label class="field"><span>Tag</span><input class="input" name="tag" list="tagSuggest" placeholder="e.g. one hand" autofocus autocomplete="off" spellcheck="false" maxlength="60"></label>`
        + `<datalist id="tagSuggest">${P.data.tags.map((t) => `<option value="${esc(t.name)}">`).join('')}</datalist>`;
      for (const radio of body.querySelectorAll('[name=mode]')) {
        radio.addEventListener('change', () => {
          mode = radio.value;
          for (const c of body.querySelectorAll('.radio-card')) c.classList.toggle('on', c.querySelector('input').checked);
        });
      }
    },
    onOk: async (body) => {
      const tag = body.querySelector('[name=tag]').value.trim();
      if (!tag) throw new Error('Enter a tag.');
      return api(`/api/ws/projects/${P.id}/images/tags`, {
        method: 'POST',
        body: { ids: [...DS.selected], add: mode === 'add' ? [tag] : [], remove: mode === 'remove' ? [tag] : [] },
      }).then((res) => ({ ...res, tag }));
    },
  });
  if (!r) return;
  toast(`${mode === 'add' ? 'Tagged' : 'Untagged'} ${plural(r.changed, 'image')}${mode === 'add' ? ` “${r.tag}”` : ''}`);
  DS.selected.clear();
  await refreshProject();
}

function renderPager() {
  const box = $('#dsPager');
  if (!box || !DS.data) return;
  const { total } = DS.data;
  if (!total) { box.innerHTML = ''; return; }
  const pages = Math.ceil(total / DS.per);
  const from = DS.page * DS.per + 1, to = Math.min(total, (DS.page + 1) * DS.per);
  const nums = new Set([0, pages - 1, DS.page - 1, DS.page, DS.page + 1].filter((p) => p >= 0 && p < pages));
  let last = -1, btns = '';
  for (const p of [...nums].sort((a, b) => a - b)) {
    if (p - last > 1) btns += '<span class="muted">…</span>';
    btns += `<button data-page="${p}" class="${p === DS.page ? 'on' : ''}">${p + 1}</button>`;
    last = p;
  }
  box.innerHTML = `<div class="pager-row"><label class="select-label">Per page <select class="select" id="dsPer">${[50, 100, 200, 500].map((n) => `<option ${n === DS.per ? 'selected' : ''}>${n}</option>`).join('')}</select></label>`
    + `<span>${fmtInt(from)}–${fmtInt(to)} of ${fmtInt(total)}</span>`
    + `<div class="pages"><button data-page="${DS.page - 1}" ${DS.page ? '' : 'disabled'} title="Previous">‹</button>${btns}<button data-page="${DS.page + 1}" ${DS.page < pages - 1 ? '' : 'disabled'} title="Next">›</button></div></div>`;
  $('#dsPer').addEventListener('change', (e) => { DS.per = Number(e.target.value); pref('per', DS.per); DS.page = 0; loadImages(); });
  for (const b of box.querySelectorAll('[data-page]')) {
    b.addEventListener('click', () => { DS.page = Number(b.dataset.page); loadImages(); $('#main').scrollTop = 0; });
  }
}

// ------------------------------------------------------------------
// Classes
// ------------------------------------------------------------------

function renderClasses() {
  const d = P.data;
  const names = d.classes.map((c) => c.name);
  const max = Math.max(1, ...d.classes.map((c) => c.boxes));
  const main = $('#main');
  let html = `<div class="page">${pageHead('Classes', 'What the boxes can be. Rename a class into another to merge them.', `<button class="btn btn-primary" id="addClass">${icon('plus')} Add class</button>`)}`;
  if (!d.classes.length) {
    html += `<div class="empty">${icon('tag', 'big')}<h3>No classes yet</h3><p>Add them here, or as you draw boxes in the annotator.</p></div></div>`;
  } else {
    html += `<table class="table"><thead><tr><th>Class</th><th>Boxes</th><th></th><th></th></tr></thead><tbody>`
      + d.classes.map((c) => `<tr><td><span class="cdot" style="background:${colorOf(c.name, names)}"></span><b>${esc(c.name)}</b></td>`
        + `<td class="bar-cell"><div class="meter"><i style="width:${(100 * c.boxes) / max}%"></i></div></td>`
        + `<td class="num">${fmtInt(c.boxes)}</td>`
        + `<td class="acts"><button class="icon-btn" data-rename="${esc(c.name)}" title="Rename / merge">${icon('edit')}</button><button class="icon-btn" data-delete="${esc(c.name)}" data-n="${c.boxes}" title="Delete">${icon('trash')}</button></td></tr>`).join('')
      + `</tbody></table></div>`;
  }
  main.innerHTML = html;
  $('#addClass').addEventListener('click', async () => {
    const name = await askName('Add class', '', 'Add', 'Class name');
    if (!name) return;
    try {
      await api(`/api/ws/projects/${P.id}/classes`, { method: 'PUT', body: { classes: [...names, name] } });
      refreshProject();
    } catch (err) { toast(err.message); }
  });
  for (const b of main.querySelectorAll('[data-rename]')) {
    b.addEventListener('click', async () => {
      const old = b.dataset.rename;
      const name = await askName(`Rename “${old}”`, old, 'Rename', 'New name (an existing class merges them)');
      if (!name || name === old) return;
      try {
        await api(`/api/ws/projects/${P.id}/classes/rename`, { method: 'POST', body: { old, new: name } });
        toast(names.some((n) => norm(n) === norm(name) && n !== old) ? `Merged “${old}” into “${name}”` : `Renamed to “${name}”`);
        refreshProject();
      } catch (err) { toast(err.message); }
    });
  }
  for (const b of main.querySelectorAll('[data-delete]')) {
    b.addEventListener('click', async () => {
      const name = b.dataset.delete;
      const n = Number(b.dataset.n);
      if (!await confirmDialog('Delete class?', `<p>Delete <b>${esc(name)}</b>${n ? ` and its ${plural(n, 'box', 'boxes')} on every image` : ''}?</p><p class="muted">This can't be undone.</p>`)) return;
      try {
        await api(`/api/ws/projects/${P.id}/classes?name=${encodeURIComponent(name)}`, { method: 'DELETE' });
        refreshProject();
      } catch (err) { toast(err.message); }
    });
  }
}

// ------------------------------------------------------------------
// Train
// ------------------------------------------------------------------

const TF = { projectId: null, use: 'all', classes: null };

async function renderTrain() {
  const d = P.data;
  const s = d.stats || {};
  if (!P.defaults) {
    try { P.defaults = (await api('/api/models')).defaults; } catch (err) { toast(err.message); return; }
  }
  if (TF.projectId !== d.id) Object.assign(TF, { projectId: d.id, use: s.labeled >= 20 || !s.predicted ? 'labeled' : 'all', classes: null });
  const df = P.defaults;
  const usable = (s.labeled || 0) + (s.predicted || 0);
  const running = P.jobs.find((j) => j.status === 'running');
  const main = $('#main');
  let html = `<div class="page">${pageHead('Train a model', 'Fine-tune a small, fast RF-DETR on this project’s images. It trains in the background; you can keep labeling.')}`;
  if (!usable) {
    html += `<div class="empty">${icon('pen', 'big')}<h3>Label some images first</h3><p>Training learns from labeled images (and pre-labeled ones, if you choose). This project has none yet.</p><div class="btns"><a class="btn btn-primary" href="#/p/${d.id}/annotate">${icon('pen')} Annotate</a></div></div></div>`;
    main.innerHTML = html;
    return;
  }
  if (running) html += `<div class="callout warn" style="margin:0 0 16px">${icon('clock')}<div style="flex:1">A model is training now (${esc(running.title)}). Only one job runs at a time.</div><a class="btn btn-sm" href="#/p/${d.id}/models">See progress</a></div>`;
  html += `<div class="form-card">
    <div class="fsec"><div><h4>Images</h4><div class="hint">Unlabeled images are never used: nothing says what is on them.</div></div><div>
      <div class="datastats" style="margin-bottom:12px">
        <div class="metric"><b>${fmtInt(s.labeled)}</b><span><i class="sw labeled"></i> labeled</span></div>
        <div class="metric"><b>${fmtInt(s.predicted)}</b><span><i class="sw predicted"></i> pre-labeled</span></div>
        <div class="metric"><b>${fmtInt(s.unlabeled)}</b><span><i class="sw unlabeled"></i> unlabeled (left out)</span></div>
      </div>
      <div class="radio-cards">
        <label class="radio-card"><input type="radio" name="use" value="labeled"><div><b>Only labeled images</b><small>${plural(s.labeled || 0, 'image')} you drew or checked. Best when you have a few hundred.</small></div></label>
        <label class="radio-card"><input type="radio" name="use" value="all" ${s.predicted ? '' : 'disabled'}><div><b>Labeled and pre-labeled images</b><small>${plural(usable, 'image')}. Pre-labels are used as they are, so their mistakes are learned too; boxes below the minimum confidence are left out.</small></div></label>
      </div>
      <div class="fgrid" style="margin-top:12px" id="minScoreRow"><label class="field"><span>Min. confidence of pre-labels</span><input class="input" type="number" id="tMinScore" min="0" max="1" step="0.05" value="${df.min_score}"></label></div>
    </div></div>
    <div class="fsec"><div><h4>Classes</h4><div class="hint">The model learns to find these. Click to leave one out.</div></div><div class="classes-row" id="tClasses"></div></div>
    <div class="fsec"><div><h4>Model</h4><div class="hint">Bigger models are slower to train and run, and need more images.</div></div><div class="fgrid">
      <label class="field"><span>Size</span><select class="select" id="tModel">${df.models.map((m) => `<option value="${m}" ${m === df.model ? 'selected' : ''}>RF-DETR ${m}${m === df.model ? ' (recommended)' : ''}</option>`).join('')}</select></label>
      <label class="field"><span>Epochs</span><input class="input" type="number" id="tEpochs" min="1" value="${df.epochs}"></label>
      <label class="field"><span>Batch size</span><input class="input" type="number" id="tBatch" min="1" value="${df.batch_size}"></label>
      <label class="field"><span>Held out to score it (%)</span><input class="input" type="number" id="tVal" min="5" max="50" step="5" value="${Math.round(df.val_fraction * 100)}"></label>
      <label class="field"><span>Use every Nth image</span><input class="input" type="number" id="tEvery" min="1" value="1"></label>
      <label class="field"><span>Stop early after (epochs without gain)</span><input class="input" type="number" id="tPatience" min="0" value="0" title="0 trains every epoch"></label>
    </div></div>
    <div class="fsec"><div><h4>Command line</h4><div class="hint">The same run, from a terminal.</div></div><pre class="cli mono" id="tCli">…</pre></div>
    <div class="form-actions">
      <div class="grow"><div class="summary" id="tSummary"></div><div class="form-error" id="tError" hidden></div></div>
      <button class="btn btn-primary" id="tGo" ${running ? 'disabled' : ''}>${icon('cpu')} Start training</button>
    </div>
  </div>
  </div>`;
  main.innerHTML = html;
  if (TF.use === 'all' && !s.predicted) TF.use = 'labeled';
  $(`[name=use][value=${TF.use}]`).checked = true;
  const classCounts = d.classes.filter((c) => c.boxes > 0);
  if (!TF.classes) TF.classes = new Set(classCounts.map((c) => c.name));
  const renderClassToggles = () => {
    $('#tClasses').innerHTML = classCounts.length
      ? classCounts.map((c) => `<button type="button" class="cls-toggle${TF.classes.has(c.name) ? ' on' : ''}" data-cls="${esc(c.name)}"><span class="cdot" style="background:${colorOf(c.name, d.classes.map((x) => x.name))};margin:0"></span>${esc(c.name)} <span class="n">${fmtInt(c.boxes)}</span></button>`).join('')
      : '<span class="muted">No boxes yet.</span>';
    for (const b of $$('[data-cls]')) {
      b.addEventListener('click', () => {
        if (TF.classes.has(b.dataset.cls)) TF.classes.delete(b.dataset.cls); else TF.classes.add(b.dataset.cls);
        renderClassToggles();
        update();
      });
    }
  };
  const body = () => ({
    project: d.id,
    only_edited: TF.use === 'labeled',
    classes: TF.classes.size === classCounts.length ? null : [...TF.classes],
    min_score: Number($('#tMinScore').value),
    model: $('#tModel').value,
    epochs: Number($('#tEpochs').value),
    batch_size: Number($('#tBatch').value),
    val_fraction: Number($('#tVal').value) / 100,
    every: Number($('#tEvery').value) || 1,
    patience: Number($('#tPatience').value) || 0,
  });
  let seq = 0;
  const update = async () => {
    for (const r of $$('.radio-card')) r.classList.toggle('on', r.querySelector('input').checked);
    $('#minScoreRow').hidden = TF.use !== 'all';
    const n = Math.ceil((TF.use === 'labeled' ? s.labeled : usable) / (Number($('#tEvery').value) || 1));
    $('#tSummary').innerHTML = `About <b>${fmtInt(n)}</b> images · ${TF.classes.size ? plural(TF.classes.size, 'class', 'classes') : '<span style="color:var(--danger)">no classes</span>'}`;
    $('#tGo').disabled = !!running || !TF.classes.size || !n;
    const mine = ++seq;
    try {
      const r = await api('/api/jobs/train?dry=1', { method: 'POST', body: body() });
      if (mine === seq) { $('#tCli').textContent = r.cli; $('#tError').hidden = true; }
    } catch (err) {
      if (mine === seq) { $('#tCli').textContent = ''; $('#tError').textContent = err.message; $('#tError').hidden = false; }
    }
  };
  for (const r of $$('[name=use]')) r.addEventListener('change', () => { TF.use = r.value; update(); });
  // The command follows what is typed (shortly after, not on every key).
  let typing = null;
  for (const el of $$('.form-card input, .form-card select')) {
    el.addEventListener('change', update);
    el.addEventListener('input', () => { clearTimeout(typing); typing = setTimeout(update, 300); });
  }
  renderClassToggles();
  update();
  $('#tGo').addEventListener('click', async () => {
    $('#tGo').disabled = true;
    try {
      await api('/api/jobs/train', { method: 'POST', body: body() });
      toast('Training started');
      location.hash = `#/p/${d.id}/models`;
      pollNavJobs();
    } catch (err) {
      $('#tError').textContent = err.message;
      $('#tError').hidden = false;
      $('#tGo').disabled = false;
    }
  });
}

// ------------------------------------------------------------------
// Models (and the training jobs of this project)
// ------------------------------------------------------------------

function jobPhase(job) {
  const p = job.progress || {};
  if (job.kind === 'detect') {
    if (job.status === 'done') return { text: 'Done', detail: p.bar ? `${fmtInt(p.bar.total)} frames` : '', frac: 1 };
    if (p.bar) return { text: p.bar.label, detail: `${fmtInt(p.bar.done)}/${fmtInt(p.bar.total)}`, frac: p.bar.done / Math.max(1, p.bar.total) };
    return { text: job.last_line || 'Starting…', frac: null };
  }
  switch (p.phase) {
    case 'dataset': return { text: 'Preparing the images', detail: `${p.done}/${p.total}`, frac: 0.05 * p.done / p.total };
    case 'dataset_done': case 'loading': return { text: `Loading RF-DETR${p.device ? ` on ${p.device}` : ''}`, frac: 0.05 };
    case 'training': {
      const frac = ((p.epoch - 1) + p.batch / p.batches) / p.epochs;
      return { text: `Epoch ${p.epoch} of ${p.epochs}`, detail: `batch ${p.batch}/${p.batches}${p.loss != null ? ` · loss ${fmtNum(p.loss)}` : ''}`, frac: 0.05 + 0.95 * frac };
    }
    case 'validated': {
      const last = p.history?.[p.history.length - 1];
      return { text: `Epoch ${last?.epoch ?? '?'} of ${p.epochs} scored`, frac: 0.05 + 0.95 * (last?.epoch ?? 0) / p.epochs };
    }
    case 'stopping': return { text: 'Stopping: scoring the model one last time…', frac: null };
    case 'calibrating': return { text: 'Choosing the confidence threshold', frac: 0.99 };
    case 'stopped': return { text: 'Stopped before a model was saved', frac: null };
    case 'done': return { text: p.stopped ? 'Stopped; the best model so far was kept' : 'Done', frac: 1 };
    default: return { text: job.last_line || 'Starting…', frac: null };
  }
}

const openLogs = new Set();

function jobCard(job) {
  const el = document.createElement('div');
  el.className = 'card jobcard';
  const phase = jobPhase(job);
  const elapsed = (job.ended || Date.now() / 1000) - job.started;
  const running = job.status === 'running';
  const chip = { running: 'blue', done: 'ok', failed: 'bad', stopped: 'warn' }[job.status] || '';
  const hist = job.progress?.history || [];
  const best = hist.reduce((a, b) => ((b.map50 ?? -1) > (a?.map50 ?? -1) ? b : a), null);
  el.innerHTML = `<div class="jrow">${running ? '<span class="pulse"></span>' : icon('cpu')}<b>${esc(job.title)}</b><span class="chip ${chip}">${esc(job.stopping ? 'stopping' : job.status)}</span><span class="grow"></span><span class="muted">${fmtDuration(elapsed)}</span>`
    + (running ? `<button class="btn btn-sm" data-act="stop" ${job.stopping ? 'disabled' : ''}>${icon('stop', 'sm')} ${job.kind === 'train' ? 'Stop, keep best' : 'Stop'}</button>` : `<button class="icon-btn" data-act="remove" title="Remove from the list">${icon('x')}</button>`)
    + `</div><div class="jphase"><span>${esc(job.status === 'failed' ? (job.last_line || 'Failed') : phase.text)}</span><span class="mono">${esc([phase.detail, best && `best mAP50 ${fmtNum(best.map50)}`].filter(Boolean).join(' · '))}</span></div>`
    + (running ? `<div class="meter${phase.frac == null ? ' indeterminate' : ''}"><i style="width:${((phase.frac ?? 0) * 100).toFixed(1)}%"></i></div>` : '')
    + (job.kind === 'train' ? trainingCharts(hist, job.progress?.epochs) : '')
    + (job.kind === 'detect' && job.status === 'done'
      ? `<div style="margin-top:8px"><a class="btn btn-sm btn-primary" href="/annotate?${new URLSearchParams({ detections: job.result.detections, video: job.result.video, from: P.id })}">${icon('pen', 'sm')} Open the detections</a></div>`
      : '')
    + `<details ${openLogs.has(job.id) ? 'open' : ''}><summary>Output</summary><pre class="mono" data-log="${job.id}">Loading…</pre></details>`;
  const details = el.querySelector('details');
  details.addEventListener('toggle', () => { if (details.open) { openLogs.add(job.id); loadLog(job.id); } else openLogs.delete(job.id); });
  if (details.open) loadLog(job.id);
  el.querySelector('[data-act=stop]')?.addEventListener('click', async (e) => {
    e.currentTarget.disabled = true;
    try { await api(`/api/jobs/${job.id}/stop`, { method: 'POST' }); } catch (err) { toast(err.message); }
    refreshProject();
  });
  el.querySelector('[data-act=remove]')?.addEventListener('click', async () => {
    try { await api(`/api/jobs/${job.id}`, { method: 'DELETE' }); } catch (err) { toast(err.message); }
    refreshProject();
  });
  return el;
}

async function loadLog(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    const pre = document.querySelector(`[data-log="${id}"]`);
    if (pre) { pre.textContent = job.log.join('\n'); pre.scrollTop = pre.scrollHeight; }
  } catch { /* the next refresh retries */ }
}

function modelCard(m) {
  const el = document.createElement('div');
  el.className = 'card mcard';
  const hist = m.history || [];
  const top = m.best || hist.reduce((a, b) => ((b.map ?? -1) > (a?.map ?? -1) ? b : a), null);
  const cal = m.calibration || {};
  const imgs = m.images ? `${fmtInt(m.images.train)} train + ${fmtInt(m.images.valid)} held out` : '';
  const weak = m.threshold != null && m.threshold < 0.1;
  el.innerHTML = `<div class="mhead"><h3>${esc(m.name)}</h3><button class="icon-btn sm" data-act="rename" title="Rename">${icon('edit', 'sm')}</button><span class="chip blue">RF-DETR ${esc(m.base_model || '?')}</span>`
    + (m.stopped_early ? '<span class="chip warn">stopped early</span>' : '')
    + (weak ? '<span class="chip bad" title="It only finds objects with a very low confidence threshold: train for more epochs">undertrained</span>' : '')
    + `<span class="meta">${m.created ? `Trained ${timeAgo(m.created)}` : ''}${m.minutes != null ? ` · ${m.minutes} min` : ''}</span></div>`
    + `<div class="metrics">`
    + `<div class="metric"><b>${fmtNum(top?.map50)}</b><span>mAP50 (best)</span></div>`
    + `<div class="metric"><b>${fmtNum(top?.map)}</b><span>mAP50-95</span></div>`
    + `<div class="metric"><b>${fmtNum(cal.f1)}</b><span>F1 at threshold ${m.threshold ?? '–'}</span></div>`
    + `<div class="metric"><b>${m.epochs_run ?? '–'}</b><span>epochs · ${imgs}</span></div>`
    + `<div class="metric spark">${sparkline(hist.map((e) => e.map50)) || '<span class="muted">–</span>'}<span>mAP50 per epoch</span></div>`
    + `</div>`
    + `<div class="mfoot"><div class="classes-row">${(m.classes || []).map((c) => `<span class="chip">${esc(c)}</span>`).join('')}</div><span class="grow"></span>`
    + `<a class="btn btn-sm btn-primary" href="#/p/${P.id}/detections?model=${encodeURIComponent(m.path)}">${icon('film', 'sm')} Run on a video</a>`
    + `<button class="btn btn-sm" data-act="copy">${icon('terminal', 'sm')} Copy command</button>`
    + `<button class="btn btn-sm btn-ghost danger" data-act="delete">${icon('trash', 'sm')} Delete</button></div>`
    + (hist.length > 1 ? `<details class="curves"><summary>Training curves</summary>${trainingCharts(hist)}</details>` : '')
    + `<div class="mfoot" style="margin-top:6px"><span class="path mono" title="${esc(m.path)}">${esc(m.path)}</span></div>`;
  el.querySelector('[data-act=rename]').addEventListener('click', async () => {
    const name = await askName('Rename model', m.name);
    if (!name || name === m.name) return;
    try {
      await api(`/api/models?path=${encodeURIComponent(m.path)}`, { method: 'PATCH', body: { name } });
      toast(`Renamed to “${name}”`);
      refreshProject();
    } catch (err) { toast(err.message); }
  });
  el.querySelector('[data-act=copy]').addEventListener('click', async () => {
    const q = (s) => (/^[\w./-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);
    const cmd = `python process_video.py VIDEO.mp4 --weights ${q(m.path)}`;
    try { await navigator.clipboard.writeText(cmd); toast('Copied: run it on a video with process_video.py'); } catch { toast(cmd, 8000); }
  });
  el.querySelector('[data-act=delete]').addEventListener('click', async () => {
    if (!await confirmDialog('Delete model?', `<p>Delete <b>${esc(m.name)}</b> (its weights and the images it was trained on)?</p><p class="muted">The project's images and labels are kept. This can't be undone.</p>`)) return;
    try {
      await api(`/api/models?path=${encodeURIComponent(m.path)}`, { method: 'DELETE' });
      toast(`Deleted “${m.name}”`);
      refreshProject();
    } catch (err) { toast(err.message); }
  });
  return el;
}

function renderModels() {
  const d = P.data;
  const main = $('#main');
  const keep = main.scrollTop;
  const trainJobs = P.jobs.filter((j) => j.kind === 'train');
  let html = `<div class="page">${pageHead('Models', 'Models trained on this project. The annotator’s Label Assist (Trained tab) can use them to pre-label new images.', `<a class="btn btn-primary" href="#/p/${d.id}/train">${icon('cpu')} Train a model</a>`)}`;
  if (trainJobs.length) html += `<h4 class="section-title">Training</h4><div class="batches" id="jobList"></div>`;
  if (d.models.length) html += `<h4 class="section-title">Trained · ${d.models.length}</h4><div class="batches" id="modelList"></div>`;
  else if (!trainJobs.length) {
    html += `<div class="empty">${icon('model', 'big')}<h3>No models yet</h3><p>Train one on the labeled images: it learns to find your classes, fast enough to run on every frame of a video.</p><div class="btns"><a class="btn btn-primary" href="#/p/${d.id}/train">${icon('cpu')} Train a model</a></div></div>`;
  }
  html += '</div>';
  main.innerHTML = html;
  for (const j of [...trainJobs].reverse()) $('#jobList').append(jobCard(j));
  for (const m of d.models) $('#modelList').append(modelCard(m));
  main.scrollTop = keep;
}

// ------------------------------------------------------------------
// Detections: run a model on a video; the detections files of the project
// ------------------------------------------------------------------

const DF = { projectId: null, model: '', video: '', rate: 0, threshold: '', data: null, uploading: null };
const DETECT_RATES = [[0, 'Every frame'], [10, '10 per second'], [5, '5 per second'], [2, '2 per second'], [1, '1 per second']];

async function renderDetections(opts = {}) {
  const d = P.data;
  if (DF.projectId !== d.id) Object.assign(DF, { projectId: d.id, model: '', video: '', rate: 0, threshold: '', data: null });
  if (P.query?.get('model')) { DF.model = P.query.get('model'); DF.threshold = ''; P.query = null; }
  if (!d.models.some((m) => m.path === DF.model)) DF.model = d.models[0]?.path || '';
  if (opts.poll && $('#detFiles')) {
    renderDetJobs();
    await loadDetections();
    return;
  }
  $('#main').innerHTML = `<div class="page">
    ${pageHead('Detections', 'Run a model of this project on a video to get a detections file: check it in the annotator, then render the video with <code>process_video.py</code>.')}
    <div id="runCard"></div>
    <div id="detJobs"></div>
    <div id="detFiles"><div class="batches">${'<div class="skeleton" style="height:66px"></div>'.repeat(3)}</div></div>
  </div>`;
  renderRunCard();
  renderDetJobs();
  await loadDetections();
}

async function loadDetections() {
  const id = P.id;
  try {
    const data = await api(`/api/ws/projects/${id}/detections`);
    if (id !== P.id || P.section !== 'detections') return;
    const fresh = !DF.data;
    DF.data = data;
    if (fresh) renderRunCard();
  } catch (err) {
    $('#detFiles').innerHTML = `<div class="empty">${icon('alert', 'big')}<h3>Could not list the detections files</h3><p>${esc(err.message)}</p></div>`;
    return;
  }
  renderDetFiles();
}

function renderRunCard() {
  const box = $('#runCard');
  if (!box) return;
  const d = P.data;
  if (!d.models.length) {
    box.innerHTML = `<div class="callout" style="margin:0 0 24px">${icon('model')}<div style="flex:1">Train a model first: it runs on every frame of a video, fast, and its detections show up here.</div><a class="btn btn-sm btn-primary" href="#/p/${d.id}/train">${icon('cpu', 'sm')} Train a model</a></div>`;
    return;
  }
  const model = d.models.find((m) => m.path === DF.model);
  const videos = DF.data?.videos || [];
  if (!DF.video || !videos.includes(DF.video)) DF.video = DF.video || videos[0] || '';
  const busy = P.jobs.some((j) => j.status === 'running');
  box.innerHTML = `<form class="run-card" novalidate>
    <h3>${icon('sparkle', 'sm')} Run a model on a video</h3>
    <div class="hint">The detections file is saved next to the video, as <span class="mono">&lt;video&gt;_&lt;model&gt;_dets.json</span>.</div>
    <div class="run-grid">
      <label class="field"><span>Model</span><select class="select" name="model">${d.models.map((m) => `<option value="${esc(m.path)}">${esc(m.name)}</option>`).join('')}</select></label>
      <label class="field"><span>Video</span><span class="video-pick">
        <select class="select" name="video" ${videos.length ? '' : 'disabled'}>${videos.length ? [...new Set([DF.video, ...videos])].filter(Boolean).map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('') : `<option>${DF.data ? 'No videos under the folder yet' : 'Loading…'}</option>`}</select>
        <button type="button" class="btn" data-act="upload" title="Upload a video from this computer" ${DF.uploading != null ? 'disabled' : ''}>${icon('upload')} ${DF.uploading != null ? `${Math.round(DF.uploading * 100)}%` : 'Upload'}</button>
        <input type="file" name="file" accept="video/*,.mkv,.avi" hidden></span></label>
      <label class="field"><span>Frames</span><select class="select" name="rate">${DETECT_RATES.map(([r, label]) => `<option value="${r}">${label}</option>`).join('')}</select></label>
      <label class="field"><span>Min. confidence</span><input class="input" name="threshold" type="number" min="0" max="1" step="0.01" placeholder="${model?.threshold ?? 'auto'}" title="Empty: the threshold chosen when the model was trained"></label>
      <button class="btn btn-primary" type="submit" ${busy || !DF.video ? 'disabled' : ''} title="${busy ? 'A job is running; one runs at a time' : ''}">${icon('sparkle')} Detect</button>
    </div>
  </form>`;
  const form = box.querySelector('form');
  form.model.value = DF.model;
  if (DF.video) form.video.value = DF.video;
  form.rate.value = String(DF.rate);
  form.threshold.value = DF.threshold;
  form.model.addEventListener('change', () => { DF.model = form.model.value; DF.threshold = ''; renderRunCard(); });
  form.video.addEventListener('change', () => { DF.video = form.video.value; });
  form.rate.addEventListener('change', () => { DF.rate = Number(form.rate.value); });
  form.threshold.addEventListener('input', () => { DF.threshold = form.threshold.value; });
  const up = form.querySelector('[data-act=upload]');
  up.addEventListener('click', () => form.file.click());
  form.file.addEventListener('change', async () => {
    const file = form.file.files[0];
    if (!file) return;
    DF.uploading = 0;
    renderRunCard();
    try {
      const r = await sendFile(`/api/videos?name=${encodeURIComponent(file.name)}`, file, (f) => {
        DF.uploading = f;
        const b = $('#runCard [data-act=upload]');
        if (b) b.innerHTML = `${icon('upload')} ${Math.round(f * 100)}%`;
      });
      DF.data.videos = [r.video, ...DF.data.videos.filter((v) => v !== r.video)];
      DF.video = r.video;
      toast(`Uploaded to ${r.video}`);
    } catch (err) {
      toast(err.message, 5000);
    }
    DF.uploading = null;
    renderRunCard();
  });
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = form.querySelector('[type=submit]');
    btn.disabled = true;
    const body = { model: DF.model, video: DF.video, project: P.id };
    if (DF.threshold !== '') body.threshold = Number(DF.threshold);
    if (DF.rate) body.sample_rate = DF.rate;
    try {
      await api('/api/jobs/detect', { method: 'POST', body });
      toast('Detection started');
      await refreshProject({ poll: true });
      renderRunCard();
      schedulePoll();
    } catch (err) {
      toast(err.message, 5000);
      btn.disabled = false;
    }
  });
}

function renderDetJobs() {
  const box = $('#detJobs');
  if (!box) return;
  const jobs = P.jobs.filter((j) => j.kind === 'detect');
  box.innerHTML = jobs.length ? '<h4 class="section-title">Runs</h4><div class="batches"></div>' : '';
  for (const j of [...jobs].reverse()) box.querySelector('.batches').append(jobCard(j));
  // The Detect button follows whether a job is running.
  const btn = $('#runCard [type=submit]');
  if (btn) btn.disabled = P.jobs.some((j) => j.status === 'running') || !DF.video;
}

function renderDetFiles() {
  const box = $('#detFiles');
  if (!box || !DF.data) return;
  const files = DF.data.files;
  if (!files.length) {
    box.innerHTML = `<div class="empty">${icon('film', 'big')}<h3>No detections files yet</h3><p>Run a model on a video above; the files made with this project's models, or imported into it, are listed here.</p></div>`;
    return;
  }
  box.innerHTML = `<h4 class="section-title">Detections files · ${files.length}</h4><div class="batches"></div>`;
  const list = box.querySelector('.batches');
  for (const f of files) {
    const el = document.createElement('div');
    el.className = 'card dfile';
    const origin = f.model
      ? `<span class="chip blue" title="Made with this project's model">${icon('model', 'sm')} ${esc(f.model)}</span>`
      : f.batch ? `<span class="chip" title="Its frames were imported into the batch “${esc(f.batch)}”">${icon('images', 'sm')} imported</span>` : '';
    const boxes = Object.values(f.classes || {}).reduce((a, b) => a + b, 0);
    const meta = [
      f.video ? esc(f.video) : `<span style="color:var(--danger)">video ${esc(f.source || '?')} not found</span>`,
      f.detector && esc(f.detector),
      `${plural(f.samples || 0, 'frame')}`,
      boxes && plural(boxes, 'box', 'boxes'),
      f.edited && `${fmtInt(f.edited)} edited`,
      timeAgo(f.modified),
    ].filter(Boolean).join(' · ');
    const q = new URLSearchParams({ detections: f.path, from: P.id });
    if (f.video) q.set('video', f.video);
    el.innerHTML = `<span class="ficon">${icon('file')}</span>`
      + `<div class="dinfo"><div class="dname"><b title="${esc(f.path)}">${esc(f.name)}</b>${origin}</div><div class="meta" title="${esc(f.path)}">${meta}</div></div>`
      + `<button class="btn btn-sm" data-act="copy" title="Copy the command that renders the video from this file">${icon('terminal', 'sm')} Render command</button>`
      + (f.video ? `<a class="btn btn-sm btn-primary" href="/annotate?${q}">${icon('pen', 'sm')} Open</a>` : '');
    el.querySelector('[data-act=copy]').addEventListener('click', async () => {
      const qt = (s) => (/^[\w./-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\\''")}'`);
      const cmd = `python process_video.py ${qt(f.video || f.source || 'VIDEO.mp4')} --load-detections ${qt(f.path)}`;
      try { await navigator.clipboard.writeText(cmd); toast('Copied the render command'); } catch { toast(cmd, 8000); }
    });
    list.append(el);
  }
}

// ------------------------------------------------------------------
// The running job, as a floating pill (except on the page showing it)
// ------------------------------------------------------------------

let navJobsTimer = null;
function updateNavJobs(jobs) {
  const running = jobs.find((j) => j.status === 'running');
  const box = $('#navJobs');
  clearTimeout(navJobsTimer);
  const section = running?.kind === 'train' ? 'models' : 'detections';
  const shown = running && running.result?.project === P.id && P.section === section;
  box.hidden = !running || shown;
  if (!running) return;
  const phase = jobPhase(running);
  box.href = running.result?.project ? `#/p/${running.result.project}/${section}` : '/annotate';
  box.innerHTML = `<div class="nj-title"><span class="pulse"></span> ${running.kind === 'train' ? 'Training' : 'Detecting'}</div>`
    + `<div class="nj-sub" title="${esc(running.title)}">${esc(running.title)}</div>`
    + `<div class="nj-sub">${esc(phase.text)}</div><div class="meter${phase.frac == null ? ' indeterminate' : ''}"><i style="width:${((phase.frac ?? 0) * 100).toFixed(1)}%"></i></div>`;
  navJobsTimer = setTimeout(pollNavJobs, P.id ? 4000 : 2000);
}

async function pollNavJobs() {
  try { updateNavJobs((await api('/api/jobs')).jobs); } catch { /* server gone */ }
}

// ------------------------------------------------------------------
// Router
// ------------------------------------------------------------------

function route() {
  closeMenus();
  for (const m of $$('.modal-back')) m.remove();
  const [path, qs] = (location.hash.replace(/^#/, '') || '/').split('?');
  const parts = path.split('/').filter(Boolean);
  const query = new URLSearchParams(qs || '');
  if (parts[0] === 'p' && parts[1]) {
    showProject(parts[1], parts[2], query);
  } else {
    showHome(parts[0] === 'f' ? parts[1] : null);
  }
}

window.addEventListener('hashchange', route);
route();
pollNavJobs();
