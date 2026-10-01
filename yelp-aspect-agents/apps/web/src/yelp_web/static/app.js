"use strict";
// Rendering only: every score, star prediction, label and metric comes from the Python API.

const ASPECTS = ["food", "service", "ambience"];
const TABS = ["analyze", "explore", "compare", "metrics"];
const GLYPHS = {
  food: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M7 3v8M4 3v5a3 3 0 0 0 6 0V3M7 11v10M17 21V3c-2.5 1.5-4 4-4 8h4"/></svg>',
  service: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 17h16M6 17a6 6 0 0 1 12 0M12 8V6M10 6h4M3 20h18"/></svg>',
  ambience: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z"/></svg>',
  lead: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 6h10M4 12h16M4 18h7"/><circle cx="18" cy="6" r="2"/><circle cx="15" cy="18" r="2"/></svg>',
  gavel: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="flex:none;margin-top:1px"><path d="m14 13-7.5 7.5a2.1 2.1 0 0 1-3-3L11 10M16 16l6-6M8 8l6-6M9 7l8 8M21 11l-8-8"/></svg>',
};
const POLARITY_ORDER = ["negative", "neutral", "positive"];
const POLARITY_LABEL = { negative: "Negative", neutral: "Neutral", positive: "Positive" };
const SIGN = { positive: 1, negative: -1, neutral: 0 };
const SYMBOL = { negative: "square", neutral: "diamond", positive: "circle" };

const state = {
  status: null,
  results: [],
  colorBy: "actual",
  selected: null,
  search: "",
  actual: null,          // real stars / label of the review loaded into the analyzer, if any
  actualLabel: null,
  loadedText: "",
  loadedId: null,
  pollTimer: null,
  compare: null,         // /api/compare payload
  compareById: new Map(),  // review_id -> analysis, for the restaurants in the comparison
  compareSel: null,      // business_id shown in the Compare detail
  compareSearch: "",
  compareSort: "contested",
  openReview: null,      // review expanded in the Compare side-by-side table
};

/* ---------------- helpers ---------------- */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
const signed = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(2);
const starsText = (s) => "★".repeat(Math.round(s)) + "☆".repeat(5 - Math.round(s));
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const pval = (p) => (p == null ? "—" : p < 0.001 ? "<0.001" : p.toFixed(3));

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function toast(msg, ms = 3500) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg; t.setAttribute("role", "status");
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

function chip(sentiment) {
  const label = sentiment === "not_mentioned" ? "Not mentioned" : POLARITY_LABEL[sentiment] || sentiment;
  return `<span class="chip ${esc(sentiment)}">${esc(label)}</span>`;
}

function meter(score) {
  const w = Math.min(1, Math.abs(score)) * 50;
  const cls = score >= 0 ? "pos" : "neg";
  return `<div class="meter" role="img" aria-label="score ${signed(score)}"><i class="${cls}" style="width:${w}%"></i></div>`;
}

function cellScore(a) {
  if (!a || a.sentiment === "not_mentioned") return '<span class="cell-score na">—</span>';
  return `<span class="cell-score ${esc(a.sentiment)}"><i class="dot"></i>${signed(a.score)}</span>`;
}

/** The arithmetic behind a Python-computed aspect score, for display. */
function formula(a) {
  const terms = a.mentions.map((m) => SIGN[m.polarity] * m.intensity);
  return `(${terms.map((v) => (v < 0 ? "−" : "+") + Math.abs(v)).join(" ")}) / (3 × ${terms.length}) = ${signed(a.score)}`;
}

/** Mark each quote at the character offsets Python found when it grounded the quote.
 *  Offsets are Unicode code points (Python str indices), so index by code point, not UTF-16 unit. */
function highlight(text, aspects) {
  const cps = Array.from(text);
  const spans = [];
  for (const name of ASPECTS) {
    for (const m of aspects[name]?.mentions || []) {
      (m.spans || []).forEach(([start, end], k) => spans.push({ start, end, name, polarity: m.polarity, first: k === 0 }));
    }
  }
  spans.sort((x, y) => x.start - y.start);
  let out = "", pos = 0;
  for (const s of spans) {
    if (s.start < pos) continue; // overlapping quotes (a passage kept for two aspects): mark once
    out += esc(cps.slice(pos, s.start).join(""));
    out += `<mark class="ev ${esc(s.polarity)}">${esc(cps.slice(s.start, s.end).join(""))}${s.first ? `<sup>${esc(s.name)}</sup>` : ""}</mark>`;
    pos = s.end;
  }
  return out + esc(cps.slice(pos).join(""));
}

function arbiterNote(x) {
  return `<div class="notice">${GLYPHS.gavel}<span><b>Arbiter:</b> “${esc(trunc(x.passage, 120))}” was cited by ${esc(x.candidates.join(", "))} → kept for ${esc(x.kept.join(", "))}</span></div>`;
}

/* ---------------- theme & tabs ---------------- */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (_) {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (_) {}
    renderScatter();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderScatter);
}

function showTab(name) {
  $$("nav.tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$("section.view").forEach((s) => (s.hidden = s.id !== `view-${name}`));
  try { localStorage.setItem("tab", name); } catch (_) {}
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  if (name === "explore") loadResults();
  if (name === "compare") loadCompare();
  if (name === "metrics") loadMetrics();
}

/* ---------------- status / provider ---------------- */
async function loadStatus() {
  state.status = await api("/api/status");
  const sel = $("#provider");
  sel.innerHTML = state.status.available
    .map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join("");
  sel.value = state.status.default.id || state.status.default.provider;
  if (state.status.job?.running) pollJob();
}
const provider = () => $("#provider").value || null;

/* ---------------- analyze ---------------- */
function agentCard(name) {
  const def = state.status?.aspects?.[name] || "";
  return `<article class="card agent" id="agent-${name}" aria-live="polite">
    <header><span class="glyph">${GLYPHS[name]}</span><b>${name[0].toUpperCase() + name.slice(1)} agent</b>
      <span class="status">Waiting</span></header>
    <p class="def">${esc(def)}</p>
    <div class="body"><div class="skeleton" style="width:40%"></div><div class="skeleton" style="margin-top:12px"></div></div>
  </article>`;
}

function renderPipeline() {
  $("#pipeline").innerHTML = ASPECTS.map(agentCard).join("") +
    `<div class="arrow" aria-hidden="true"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg></div>
     <article class="card agent verdict" id="agent-lead" aria-live="polite">
       <header><span class="glyph">${GLYPHS.lead}</span><b>Lead agent</b><span class="status">Waiting</span></header>
       <p class="def">Combines the three verdicts with its own reading of the review.</p>
       <div class="body"><div class="skeleton" style="width:30%;height:36px"></div><div class="skeleton" style="margin-top:12px"></div></div>
     </article>`;
  $("#arbiter-notes").innerHTML = "";
}

function setStatus(id, html, cls) {
  const card = $(`#agent-${id}`);
  card.classList.remove("thinking", "error", "done");
  if (cls) card.classList.add(cls);
  $(".status", card).innerHTML = html;
}

function aspectBody(a) {
  const dropped = a.dropped?.length
    ? `<p class="small muted" style="margin:0">${a.dropped.length} quote${a.dropped.length > 1 ? "s" : ""} excluded: ` +
      a.dropped.map((m) => `“${esc(trunc(m.quote, 60))}” (${esc(m.note)})`).join("; ") + "</p>"
    : "";
  if (a.sentiment === "not_mentioned") {
    return `${chip(a.sentiment)}<p class="small muted" style="margin:0">The review says nothing about ${esc(a.aspect)}, so this axis stays at 0.</p>${dropped}`;
  }
  return `
    <div class="row">${chip(a.sentiment)}<span class="spacer"></span><span class="small muted">${a.mentions.length} quote${a.mentions.length > 1 ? "s" : ""}</span></div>
    <div class="meter-row">${meter(a.score)}<span class="num">${signed(a.score)}</span></div>
    <div class="formula" title="mean of sign(polarity) × intensity / 3, computed in Python">${formula(a)}</div>
    ${a.mentions.map((m) => {
      const v = SIGN[m.polarity] * m.intensity;
      return `<p class="evidence ${esc(m.polarity)}">“${esc(m.quote)}”<span class="val">${v < 0 ? "−" : "+"}${Math.abs(v)}/3</span></p>`;
    }).join("")}
    ${dropped}`;
}

function fillAspect(a, latency) {
  const card = $(`#agent-${a.aspect}`);
  if (a.error) {
    setStatus(a.aspect, "Failed", "error");
    $(".body", card).innerHTML = `<p class="small">${esc(a.error)}</p>`;
    return;
  }
  setStatus(a.aspect, latency != null ? `${fmt(latency, 1)}s` : "Done", "done");
  $(".body", card).innerHTML = aspectBody(a);
}

function fillVerdict(o, done) {
  const card = $("#agent-lead");
  setStatus("lead", o.error ? "Fallback" : `${fmt(o.latency_s, 1)}s`, o.error ? "error" : "done");
  const actual = state.actual;
  const rows = [];
  if (actual != null) {
    rows.push(`<span>Reviewer gave</span><b>${actual}★ · ${POLARITY_LABEL[state.actualLabel]}</b>`);
    rows.push(`<span>Difference</span><b>${signed(o.stars_rule - actual)}★</b>`);
  }
  if (done.calibrated_stars != null) {
    rows.push(`<span title="Ridge model fit on ${done.n_train} batch results (never on this review)">Calibrated model</span><b>${fmt(done.calibrated_stars, 2)}★</b>`);
  }
  const vector = done.analysis.vector;
  $(".body", card).innerHTML = `
    <div class="hero"><span class="big">${fmt(o.stars_rule, 1)}</span><span class="unit">predicted stars</span></div>
    <div class="row"><span class="stars" aria-hidden="true">${starsText(o.stars_rule)}</span>${chip(o.sentiment)}</div>
    ${rows.length ? `<div class="compare">${rows.join("")}</div>` : ""}
    <div class="formula" title="rule-based stars from the lead agent's overall score, computed in Python">3 + 2 × ${signed(o.score)} = ${fmt(o.stars_rule, 2)}★</div>
    <p class="small" style="margin:0;color:var(--ink-2)">${esc(o.rationale)}</p>
    ${o.other_factors?.length ? `<p class="small muted" style="margin:0">Also weighed: ${esc(o.other_factors.join(", "))}</p>` : ""}
    <div class="vector" title="food, service, ambience">[${vector.map(signed).join(", ")}]</div>`;
}

async function randomReview() {
  try {
    const r = await api("/api/sample/random");
    $("#review-text").value = r.text;
    state.actual = r.stars; state.actualLabel = r.label; state.loadedText = r.text; state.loadedId = r.review_id;
    $("#review-meta").innerHTML = `<b>${esc(r.business_name)}</b> · ${esc(r.city)} · reviewer gave <b>${r.stars}★</b> <span class="muted">(${esc(r.date?.slice(0, 10))})</span>`;
  } catch (e) { toast(e.message); }
}

async function runAgents() {
  const text = $("#review-text").value.trim();
  if (text.length < 3) { toast("Enter a review first."); return; }
  const fromSample = state.loadedId && text === state.loadedText.trim();
  if (!fromSample) { state.actual = null; state.loadedId = null; $("#review-meta").textContent = "Custom review"; }
  const btn = $("#run-btn"); btn.disabled = true;
  renderPipeline();
  $("#highlight-card").hidden = true;
  const latency = {};
  try {
    const res = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, review_id: fromSample ? state.loadedId : null, provider: provider(),
                             few_shot: $("#few-shot").checked, arbitrate: $("#arbiter").checked }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", overall = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const ev = /^event: (.*)$/m.exec(chunk)?.[1];
        const data = JSON.parse(/^data: (.*)$/m.exec(chunk)?.[1] || "{}");
        if (ev === "agent_start") setStatus(data.aspect, '<span class="spin"></span>Thinking', "thinking");
        else if (ev === "aspect") { latency[data.aspect] = data.latency_s; fillAspect(data, data.latency_s); }
        else if (ev === "arbiter") $("#arbiter-notes").insertAdjacentHTML("beforeend", arbiterNote(data));
        else if (ev === "aggregator_start") setStatus("lead", '<span class="spin"></span>Combining', "thinking");
        else if (ev === "overall") overall = data;
        else if (ev === "done") {
          // final state from Python (includes any arbiter re-scoring)
          for (const a of ASPECTS) fillAspect(data.analysis.aspects[a], latency[a]);
          fillVerdict(overall, data);
          $("#highlighted").innerHTML = highlight(text, data.analysis.aspects);
          $("#highlight-card").hidden = false;
        }
      }
    }
  } catch (e) {
    toast(`Analysis failed: ${e.message}`);
  } finally { btn.disabled = false; }
}

/* ---------------- explore ---------------- */
async function loadResults() {
  try { state.results = await api("/api/results"); } catch (e) { toast(e.message); return; }
  renderScatter(); renderTable();
  if (state.selected) renderDetail(state.results.find((r) => r.review.review_id === state.selected));
}

function filtered() {
  const q = state.search.toLowerCase();
  if (!q) return state.results;
  return state.results.filter((r) =>
    `${r.review.business_name} ${r.review.city} ${r.review.text}`.toLowerCase().includes(q));
}

function groupOf(r) { return state.colorBy === "actual" ? r.label : r.overall.sentiment; }

function renderScatter() {
  const el = $("#scatter");
  if ($("#view-explore").hidden) return;
  if (!window.Plotly) { el.innerHTML = '<div class="empty">Plotly failed to load (offline?). The table below still works.</div>'; return; }
  const rows = filtered();
  if (!rows.length) {
    Plotly.purge(el);
    el.innerHTML = `<div class="empty">No analyses yet. Start a batch run above, or run <code>uv run yelp-agents run -n 30</code>.</div>`;
    return;
  }
  if (el.querySelector(".empty")) el.innerHTML = "";
  const colors = { negative: cssVar("--neg"), neutral: cssVar("--neu"), positive: cssVar("--pos") };
  const ink = cssVar("--ink"), ink2 = cssVar("--ink-2"), grid = cssVar("--grid"), axis = cssVar("--axis"), surface = cssVar("--surface");
  // Deterministic jitter: many reviews share exact coordinates (e.g. aspects not mentioned -> 0).
  const jit = (id, k) => { let h = 0; for (const c of id + k) h = (h * 31 + c.charCodeAt(0)) | 0; return ((h % 1000) / 1000) * 0.06 - 0.03; };
  const traces = POLARITY_ORDER.map((g) => {
    const rs = rows.filter((r) => groupOf(r) === g);
    return {
      type: "scatter3d", mode: "markers", name: POLARITY_LABEL[g] + (state.colorBy === "actual" ? { negative: " (1–2★)", neutral: " (3★)", positive: " (4–5★)" }[g] : ""),
      x: rs.map((r) => r.vector[0] + jit(r.review.review_id, "x")),
      y: rs.map((r) => r.vector[1] + jit(r.review.review_id, "y")),
      z: rs.map((r) => r.vector[2] + jit(r.review.review_id, "z")),
      customdata: rs.map((r) => r.review.review_id),
      text: rs.map((r) => `<b>${esc(r.review.business_name)}</b><br>${r.review.stars}★ actual · ${fmt(r.overall.stars_rule, 2)}★ predicted<br>food ${signed(r.vector[0])} · service ${signed(r.vector[1])} · ambience ${signed(r.vector[2])}`),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: rs.map((r) => (r.review.review_id === state.selected ? 11 : 6)),
        color: colors[g], symbol: SYMBOL[g], opacity: 0.9,
        line: { color: surface, width: 1 },
      },
    };
  });
  const ax = (t) => ({
    title: { text: t, font: { color: ink2, size: 12 } }, range: [-1.15, 1.15], tickvals: [-1, -0.5, 0, 0.5, 1],
    gridcolor: grid, zerolinecolor: axis, zerolinewidth: 2, showbackground: false, color: ink2, tickfont: { size: 10 },
  });
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", font: { color: ink, family: "system-ui, -apple-system, sans-serif" },
    margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, uirevision: "keep",
    hoverlabel: { bgcolor: surface, bordercolor: axis, font: { color: ink } },
    scene: { xaxis: ax("Food"), yaxis: ax("Service"), zaxis: ax("Ambience"), aspectmode: "cube",
      camera: { eye: { x: 1.25, y: 1.25, z: 0.9 } } },
  };
  Plotly.react(el, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["toImage", "resetCameraLastSave3d"] });
  if (!el._bound) {
    el.on("plotly_click", (ev) => { const id = ev.points?.[0]?.customdata; if (id) select(id); });
    el._bound = true;
  }
  const lg = state.colorBy === "actual"
    ? ["Negative (1–2★)", "Neutral (3★)", "Positive (4–5★)"]
    : ["Predicted negative", "Predicted neutral", "Predicted positive"];
  $("#legend").innerHTML = POLARITY_ORDER.map((g, i) => `<span><i class="sym ${g}"></i>${lg[i]}</span>`).join("");
}

function renderTable() {
  const rows = filtered();
  const tb = $("#results-table tbody");
  if (!rows.length) { tb.innerHTML = `<tr><td colspan="8" class="muted">No analyses to show.</td></tr>`; return; }
  tb.innerHTML = rows.map((r) => `
    <tr data-id="${esc(r.review.review_id)}" class="${r.review.review_id === state.selected ? "sel" : ""}" tabindex="0">
      <td><b>${esc(r.review.business_name)}</b><div class="small muted">${esc(r.review.city)}</div></td>
      <td><div class="excerpt">${esc(r.review.text)}</div></td>
      <td class="num">${r.review.stars}★</td>
      ${ASPECTS.map((a) => `<td>${cellScore(r.aspects[a])}</td>`).join("")}
      <td>${chip(r.overall.sentiment)}</td>
      <td class="num">${fmt(r.overall.stars_rule, 2)}</td>
    </tr>`).join("");
}

function select(id) {
  state.selected = id;
  renderDetail(state.results.find((r) => r.review.review_id === id));
  $$("#results-table tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.id === id));
  renderScatter();
}

function renderDetail(r) {
  const el = $("#detail");
  if (!r) { el.innerHTML = '<div class="empty small">Select a point to see the review and each agent\'s verdict.</div>'; return; }
  const o = r.overall;
  el.innerHTML = `
    <h3>${esc(r.review.business_name)}</h3>
    <div class="small muted">${esc(r.review.city)} · ${esc(r.review.date?.slice(0, 10))} · ${esc(r.model)}</div>
    <div class="compare" style="margin-top:12px"><span>Reviewer</span><b>${r.review.stars}★ ${chip(r.label)}</b>
      <span>Lead agent</span><b>${fmt(o.stars_rule, 2)}★ ${chip(o.sentiment)}</b>
      ${r.stars_calibrated != null ? `<span title="Out-of-fold: the model never saw this review">Calibrated</span><b>${fmt(r.stars_calibrated, 2)}★</b>` : ""}</div>
    <div class="aspects">${ASPECTS.map((a) => {
      const x = r.aspects[a];
      return `<div class="aspect-line"><span>${a[0].toUpperCase() + a.slice(1)}</span>${x.sentiment === "not_mentioned" ? '<span class="small muted">not mentioned</span>' : meter(x.score)}<span class="num">${x.sentiment === "not_mentioned" ? "—" : signed(x.score)}</span></div>`;
    }).join("")}</div>
    <p class="small" style="color:var(--ink-2)"><b>Lead agent:</b> ${esc(o.rationale)}</p>
    ${(r.meta?.arbitrations || []).map(arbiterNote).join("")}
    <p class="highlighted small" style="margin-top:12px">${highlight(r.review.text, r.aspects)}</p>`;
}

/* batch runs */
async function startBatch() {
  try {
    await api("/api/runs", { method: "POST", body: JSON.stringify({
      n: +$("#run-n").value, workers: +$("#run-workers").value, provider: provider(),
      few_shot: $("#few-shot").checked, arbitrate: $("#arbiter").checked,
    }) });
    pollJob();
  } catch (e) { toast(e.message); }
}

function pollJob() {
  clearTimeout(state.pollTimer);
  let lastDone = -1;
  const tick = async () => {
    let j;
    try { j = await api("/api/runs/current"); } catch (_) { state.pollTimer = setTimeout(tick, 3000); return; }
    const p = j.total ? (100 * j.done) / j.total : 0;
    $("#batch-bar").style.width = `${p}%`;
    $("#batch-btn").disabled = !!j.running;
    if (j.running) {
      $("#batch-status").textContent = `${j.done}/${j.total} · ${j.backend}` + (j.last ? ` · last: ${j.last.business}` : "");
      if (j.done !== lastDone && !$("#view-explore").hidden) { lastDone = j.done; loadResults(); }
      state.pollTimer = setTimeout(tick, 1500);
    } else if (j.total) {
      $("#batch-status").textContent = j.error ? `Stopped: ${j.error}` : `Done · ${j.done}/${j.total}`;
      loadResults();
      if (!j.error) toast(`Batch run finished: ${j.done} reviews analyzed.`);
    }
  };
  tick();
}

/* ---------------- compare ---------------- */
const CONSENSUS_LABEL = { positive: "Positive", negative: "Negative", neutral: "Neutral", mixed: "Mixed", too_few: "Too few mentions" };

function consensusChip(c) {
  return `<span class="chip ${c === "too_few" ? "not_mentioned" : esc(c)}">${esc(CONSENSUS_LABEL[c] || c)}</span>`;
}

/** A restaurant's mean score on one aspect, with the share of its reviewers on the majority side. */
function meanCell(b) {
  if (b.mean == null) return '<span class="cell-score na">—</span>';
  const cls = b.mean > 0 ? "positive" : b.mean < 0 ? "negative" : "neutral";
  return `<span class="cell-score ${cls}"><i class="dot"></i>${signed(b.mean)}</span>` +
    `<span class="agree" title="${b.mentioned} reviews mention it">${b.agreement == null ? `n=${b.mentioned}` : `${pct(b.agreement)} agree`}</span>`;
}

async function loadCompare() {
  const body = $("#compare-body");
  let c;
  try { c = await api("/api/compare"); }
  catch (e) {
    body.innerHTML = `<div class="card empty">Couldn't load the comparison (${esc(e.message)}).<br>If the server was started before the Compare tab existed, restart it: <code>uv run yelp-web</code></div>`;
    return;
  }
  state.compare = c;
  state.compareById = new Map(c.reviews.map((r) => [r.review.review_id, r]));
  if (!c.restaurants.length) {
    body.innerHTML = `<div class="card empty">No restaurant has two or more analyzed reviews yet. Draw a restaurant sample, run the agents on it, and open the UI on those results:<br><br>
      <code>uv run yelp-agents sample-restaurants -b 50 -k 10</code><br><br>
      <code>uv run yelp-agents run --sample data/restaurant_reviews.jsonl --out outputs/restaurant_results.jsonl</code><br><br>
      <code>uv run yelp-web</code></div>`;
    return;
  }
  if (!c.restaurants.some((x) => x.business_id === state.compareSel)) state.compareSel = c.restaurants[0].business_id;
  body.innerHTML = `${compareSummary(c.summary)}
    <div class="card list-card">
      <div class="row">
        <h3 class="section" style="margin:0">Restaurants</h3><span class="spacer"></span>
        <input type="search" id="compare-search" placeholder="Filter by name, city or cuisine…" aria-label="Filter restaurants" value="${esc(state.compareSearch)}">
        <select id="compare-sort" aria-label="Sort restaurants">
          <option value="contested">Most contested first</option><option value="yelp">Yelp rating</option><option value="name">Name</option>
        </select>
      </div>
      <div class="list-scroll"><table id="restaurant-table">
        <thead><tr><th>Restaurant</th><th class="num">Yelp ★</th><th class="num">Sampled ★</th>
          ${ASPECTS.map((a) => `<th>${cap(a)}</th>`).join("")}<th>Reviewers disagree on</th></tr></thead>
        <tbody></tbody></table></div>
      <p class="small muted" style="margin:10px 0 0">Each aspect shows the mean score over the reviews that mention it, and the share of those reviewers on the majority side. Reviewers disagree on an aspect when at least one is positive and one is negative. Select a restaurant to compare its reviews.</p>
    </div>
    <div id="restaurant-detail"></div>`;
  $("#compare-sort").value = state.compareSort;
  $("#compare-search").addEventListener("input", (e) => { state.compareSearch = e.target.value; renderRestaurantList(); });
  $("#compare-sort").addEventListener("change", (e) => { state.compareSort = e.target.value; renderRestaurantList(); });
  const pick = (tr) => {
    state.compareSel = tr.dataset.bid; state.openReview = null; renderRestaurantList(); renderRestaurant();
    $("#restaurant-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  };
  $("#restaurant-table tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-bid]"); if (tr) pick(tr); });
  $("#restaurant-table tbody").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-bid]"); if (tr && e.key === "Enter") pick(tr); });
  $("#restaurant-detail").addEventListener("click", (e) => {
    const dot = e.target.closest(".sdot");
    const tr = e.target.closest("tr[data-id]");
    const id = dot?.dataset.id || tr?.dataset.id;
    if (!id) return;
    state.openReview = dot || state.openReview !== id ? id : null;
    renderMatrix();
    if (dot) $(`#review-matrix tr[data-id="${CSS.escape(id)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  });
  renderRestaurantList();
  renderRestaurant();
}

function compareSummary(s) {
  const tile = (label, value, sub) => `<div class="card tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const st = s.stars, sy = s.sample_vs_yelp, sa = s.stars_anova;
  const rows = ASPECTS.map((a) => {
    const v = s.aspects[a], c = v.consensus;
    return `<tr><td><b>${cap(a)}</b></td>
      <td class="num">${v.restaurants_compared} / ${s.restaurants}</td>
      <td class="num">${fmt(v.within_sd)}</td><td class="num">${fmt(v.between_sd)}</td>
      <td class="num">${fmt(v.anova.icc1)}</td><td class="num">${pval(v.anova.p)}</td>
      <td class="num">${v.contested} / ${s.restaurants}</td>
      <td class="small">${c.positive} positive · ${c.negative} negative · ${c.neutral} neutral · ${c.mixed} mixed</td>
      <td class="num" title="Spearman over ${v.vs_yelp_stars.n} restaurants, p = ${pval(v.vs_yelp_stars.spearman_p)}">${fmt(v.vs_yelp_stars.spearman)}</td></tr>`;
  }).join("");
  return `
    <div class="tiles">
      ${tile("Restaurants", s.restaurants, `${s.reviews} reviews · ${fmt(s.reviews_per_restaurant, 0)} per restaurant`)}
      ${tile("With a contested aspect", s.restaurants_contested, "a positive and a negative reviewer on the same aspect")}
      ${tile("Star error per restaurant", fmt(st.restaurant_mae), `MAE of the mean prediction · ${fmt(st.review_mae)} per single review`)}
      ${tile("Sample vs Yelp rating", fmt(sy.pearson), `Pearson r, sampled mean vs Yelp's rating · MAE ${fmt(sy.mae)}`)}
    </div>
    <div class="card table-wrap"><h3 class="section">Do reviewers of the same restaurant agree?</h3>
      <table class="static"><thead><tr><th>Dimension</th><th class="num">Compared</th><th class="num">Within-restaurant sd</th><th class="num">Between-restaurant sd</th>
        <th class="num">ICC(1)</th><th class="num">ANOVA p</th><th class="num">Contested</th><th>Verdict per restaurant</th><th class="num">ρ vs Yelp ★</th></tr></thead>
      <tbody>${rows}
        <tr><td><b>Reviewer stars</b><div class="small muted">baseline, 1–5 scale</div></td><td class="num">${sa.groups} / ${s.restaurants}</td><td class="num">—</td><td class="num">—</td>
          <td class="num">${fmt(sa.icc1)}</td><td class="num">${pval(sa.p)}</td><td></td><td></td><td></td></tr>
      </tbody></table>
      <p class="small muted" style="margin:12px 0 0">Scores run from −1 to +1 and only count reviews that mention the aspect. <b>Compared</b> is the number of restaurants with at least ${s.min_compare} such reviews. <b>ICC(1)</b> is the share of score variance that lies between restaurants: 0 means reviewers of the same place agree no more than strangers, and 1 means they agree completely. A verdict needs ${pct(s.consensus_threshold)} of those reviewers on one side; below that it is <i>mixed</i>. <b>ρ vs Yelp ★</b> is the Spearman correlation between a restaurant's mean aspect score and its overall Yelp rating.</p>
    </div>`;
}

function sortedRestaurants() {
  const q = state.compareSearch.toLowerCase();
  let xs = state.compare.restaurants.filter((x) => !q || `${x.business} ${x.city} ${x.categories}`.toLowerCase().includes(q));
  if (state.compareSort === "yelp") xs = [...xs].sort((a, b) => (b.yelp_stars ?? -1) - (a.yelp_stars ?? -1) || a.business.localeCompare(b.business));
  if (state.compareSort === "name") xs = [...xs].sort((a, b) => a.business.localeCompare(b.business));
  return xs; // default: server order, most contested first
}

function renderRestaurantList() {
  const xs = sortedRestaurants();
  const tb = $("#restaurant-table tbody");
  if (!xs.length) { tb.innerHTML = '<tr><td colspan="7" class="muted">No restaurant matches.</td></tr>'; return; }
  tb.innerHTML = xs.map((x) => `
    <tr data-bid="${esc(x.business_id)}" class="${x.business_id === state.compareSel ? "sel" : ""}" tabindex="0">
      <td><b>${esc(x.business)}</b><div class="small muted">${esc(x.city)} · ${x.reviews} reviews</div></td>
      <td class="num">${x.yelp_stars == null ? "—" : fmt(x.yelp_stars, 1)}</td>
      <td class="num">${fmt(x.stars.mean, 1)}</td>
      ${ASPECTS.map((a) => `<td class="nowrap">${meanCell(x.aspects[a])}</td>`).join("")}
      <td class="small">${x.contested.length ? esc(x.contested.join(", ")) : '<span class="muted">—</span>'}</td>
    </tr>`).join("");
}

/** Dot plot on the −1…+1 axis: one dot per review that mentions the aspect; equal scores stack. */
function strip(aspect, dots, mean) {
  const placed = [];
  for (const d of [...dots].sort((p, q) => p.score - q.score)) {
    const x = ((d.score + 1) / 2) * 100;
    let lvl = 0;
    while (placed.some((p) => p.lvl === lvl && Math.abs(p.x - x) < 6)) lvl++;
    placed.push({ ...d, x, lvl });
  }
  const h = (Math.max(...placed.map((p) => p.lvl)) + 1) * 14 + 10;
  return `<div class="strip" style="height:${h}px" role="img" aria-label="${esc(`${cap(aspect)} scores of ${dots.length} reviews: ${dots.map((d) => signed(d.score)).join(", ")}`)}">
      <i class="zero"></i>
      ${mean == null ? "" : `<i class="mean" style="left:${((mean + 1) / 2) * 100}%"></i>`}
      ${placed.map((p) => `<button class="sdot ${esc(p.sentiment)}${p.id === state.openReview ? " on" : ""}" style="left:${p.x}%;bottom:${6 + p.lvl * 14}px"
          data-id="${esc(p.id)}" data-tip="${esc(`${signed(p.score)}|${p.stars}★ review · ${p.date}`)}"
          aria-label="${esc(`${signed(p.score)}, ${p.stars}-star review from ${p.date}`)}"></button>`).join("")}
    </div>
    <div class="strip-axis small muted" aria-hidden="true"><span>−1</span><span>0</span><span>+1</span></div>`;
}

function aspectPanel(a, b, reviews) {
  const dots = reviews.filter((r) => r.aspects[a].sentiment !== "not_mentioned" && !r.aspects[a].error)
    .map((r) => ({ id: r.review.review_id, score: r.aspects[a].score, sentiment: r.aspects[a].sentiment,
                   stars: r.review.stars, date: (r.review.date || "").slice(0, 10) }));
  const quote = (q, cls) => (q ? `<p class="evidence ${cls}">“${esc(trunc(q.quote, 180))}”<span class="val">${signed(q.value)} · from a ${q.stars}★ review · ${esc(q.date)}</span></p>` : "");
  const lab = b.labels;
  const stats = [`${b.mentioned} of ${b.reviews} reviews mention it`];
  if (b.agreement != null) stats.push(`${pct(b.agreement)} agree`);
  if (b.mean != null) stats.push(`mean ${signed(b.mean)}${b.sd == null ? "" : ` (sd ${fmt(b.sd)})`}`);
  return `<article class="card aspect-panel">
    <header class="row"><span class="glyph">${GLYPHS[a]}</span><b>${cap(a)}</b><span class="spacer"></span>${consensusChip(b.consensus)}</header>
    <p class="small muted" style="margin:0">${stats.join(" · ")}</p>
    ${dots.length ? strip(a, dots, b.mean) : `<p class="small muted strip-empty">No reviewer mentions ${esc(a)}.</p>`}
    <p class="small tally">${lab.positive} positive · ${lab.neutral} neutral · ${lab.negative} negative · ${lab.not_mentioned} silent</p>
    ${b.most_positive || b.most_negative ? `<h4>${b.contested ? "Where reviewers disagree" : "Strongest quote"}</h4>` : ""}
    ${quote(b.most_positive, "positive")}${quote(b.most_negative, "negative")}
  </article>`;
}

function renderRestaurant() {
  const el = $("#restaurant-detail");
  const x = state.compare.restaurants.find((r) => r.business_id === state.compareSel);
  if (!x) { el.innerHTML = '<div class="card empty small">Select a restaurant.</div>'; return; }
  const reviews = x.review_ids.map((id) => state.compareById.get(id)).filter(Boolean);
  const counts = [5, 4, 3, 2, 1].filter((s) => x.stars.counts[s]).map((s) => `${s}★ ×${x.stars.counts[s]}`).join(" · ");
  const o = x.overall_labels;
  el.innerHTML = `
    <div class="card">
      <h3 class="rname">${esc(x.business)}</h3>
      <div class="small muted">${esc(x.city)}${x.categories ? ` · ${esc(trunc(x.categories, 100))}` : ""}</div>
      <div class="rstats">
        <div><span class="label">Yelp rating</span><b>${x.yelp_stars == null ? "—" : `${fmt(x.yelp_stars, 1)}★`}</b>
          <span class="sub">${x.yelp_review_count ? `over ${x.yelp_review_count} reviews on Yelp` : "not recorded in this sample"}</span></div>
        <div><span class="label">These ${x.reviews} reviews</span><b>${fmt(x.stars.mean, 1)}★</b><span class="sub">${counts}</span></div>
        <div><span class="label">Lead agent, averaged</span><b>${fmt(x.predicted_stars, 1)}★</b>
          <span class="sub">${o.positive} positive · ${o.neutral} neutral · ${o.negative} negative</span></div>
      </div>
    </div>
    <div class="small muted strip-legend"><span>Each dot is one review that mentions the aspect:</span>
      <span><i class="sym positive"></i>positive</span><span><i class="sym neutral"></i>neutral</span><span><i class="sym negative"></i>negative</span>
      <span><i class="mean-key"></i>mean</span><span>· select a dot to open that review below</span></div>
    <div class="aspect-grid">${ASPECTS.map((a) => aspectPanel(a, x.aspects[a], reviews)).join("")}</div>
    <div class="card table-wrap"><h3 class="section">The ${x.reviews} reviews side by side</h3>
      <table id="review-matrix"><thead><tr><th>Date</th><th class="num">Stars</th><th>Review</th>
        ${ASPECTS.map((a) => `<th>${cap(a)}</th>`).join("")}<th>Overall</th><th class="num">Pred ★</th></tr></thead>
      <tbody></tbody></table>
    </div>`;
  renderMatrix();
}

function renderMatrix() {
  const x = state.compare.restaurants.find((r) => r.business_id === state.compareSel);
  $("#review-matrix tbody").innerHTML = x.review_ids.map((id) => state.compareById.get(id)).filter(Boolean).map((r) => {
    const id = r.review.review_id, open = id === state.openReview;
    return `<tr data-id="${esc(id)}" class="${open ? "sel" : ""}" tabindex="0" aria-expanded="${open}">
        <td class="nowrap">${esc((r.review.date || "").slice(0, 10))}</td><td class="num">${r.review.stars}★</td>
        <td><div class="excerpt">${esc(r.review.text)}</div></td>
        ${ASPECTS.map((a) => `<td>${cellScore(r.aspects[a])}</td>`).join("")}
        <td>${chip(r.overall.sentiment)}</td><td class="num">${fmt(r.overall.stars_rule, 2)}</td>
      </tr>` + (open ? `<tr class="expand"><td colspan="8">
        <p class="highlighted small">${highlight(r.review.text, r.aspects)}</p>
        <p class="small" style="margin:10px 0 0;color:var(--ink-2)"><b>Lead agent:</b> ${esc(r.overall.rationale)}</p>
        ${(r.meta?.arbitrations || []).map(arbiterNote).join("")}</td></tr>` : "");
  }).join("");
  $$(".sdot").forEach((d) => d.classList.toggle("on", d.dataset.id === state.openReview));
}

/* one tooltip for every [data-tip] mark: "value|context"; the value leads */
function initTip() {
  const tip = $("#tip");
  const show = (el) => {
    const [value, rest] = el.dataset.tip.split("|");
    const b = document.createElement("b");
    b.textContent = value;
    tip.replaceChildren(b, document.createTextNode(` ${rest || ""}`));
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    tip.style.left = `${Math.min(innerWidth - tip.offsetWidth - 8, Math.max(8, r.left + r.width / 2 - tip.offsetWidth / 2))}px`;
    tip.style.top = `${Math.max(8, r.top - tip.offsetHeight - 10)}px`;
  };
  const hide = () => { tip.hidden = true; };
  const on = (e) => { const el = e.target.closest?.("[data-tip]"); if (el) show(el); else hide(); };
  document.addEventListener("pointerover", on);
  document.addEventListener("focusin", on);
  document.addEventListener("focusout", hide);
  addEventListener("scroll", hide, { passive: true });
}

/* ---------------- metrics ---------------- */
async function loadMetrics() {
  const body = $("#metrics-body");
  let m;
  try { m = await api("/api/metrics"); } catch (e) { toast(e.message); return; }
  if (!m.n) { body.innerHTML = '<div class="card empty">No analyses yet. Run the agents on the sample from the Explore tab.</div>'; return; }
  const o = m.lead_agent, cal = m.calibrated, g = m.grounding;
  const ci = (c) => `95% CI ${pct(c[0])}–${pct(c[1])}`;
  const tile = (label, value, sub) => `<div class="card tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const labels = POLARITY_ORDER;  // confusion rows = actual, cols = predicted, in this order
  const maxC = Math.max(1, ...o.confusion.flat());
  const confusion = `<div class="confusion" role="table" aria-label="Confusion matrix">
      <div class="h"></div>${labels.map((p) => `<div class="h">pred ${p}</div>`).join("")}
      ${labels.map((a, i) => `<div class="h">${a}</div>` + labels.map((p, j) => {
        const v = o.confusion[i][j], shade = Math.round((v / maxC) * 85);
        return `<div class="cell" style="background:color-mix(in srgb, var(--accent) ${shade}%, var(--surface-2));color:${shade > 45 ? "#fff" : "var(--ink)"};${i === j ? "font-weight:700" : ""}" title="actual ${a}, predicted ${p}: ${v}">${v}</div>`;
      }).join("")).join("")}
    </div>
    <table style="margin-top:16px"><thead><tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">n</th></tr></thead>
    <tbody>${labels.map((l) => { const v = o.report[l]; return `<tr><td>${chip(l)}</td><td class="num">${fmt(v.precision)}</td><td class="num">${fmt(v.recall)}</td><td class="num">${fmt(v["f1-score"])}</td><td class="num">${v.support}</td></tr>`; }).join("")}</tbody></table>`;
  const aspects = `<table><thead><tr><th>Dimension</th><th>Mentioned</th><th>Mean score</th><th class="num">ρ vs stars</th><th class="num">Grounded</th></tr></thead><tbody>
    ${ASPECTS.map((a) => { const v = m.aspects[a]; return `<tr>
      <td><b>${a[0].toUpperCase() + a.slice(1)}</b></td>
      <td><div class="bar-cell"><div class="hbar"><i style="width:${v.mention_rate * 100}%"></i></div><span class="num">${pct(v.mention_rate)}</span></div></td>
      <td><div class="bar-cell">${v.mean_score == null ? "<span></span>" : meter(v.mean_score)}<span class="num">${v.mean_score == null ? "—" : signed(v.mean_score)}</span></div></td>
      <td class="num" title="Spearman, n = ${v.vs_stars.n}, p = ${v.vs_stars.spearman_p == null ? "—" : v.vs_stars.spearman_p.toExponential(1)}">${fmt(v.vs_stars.spearman)}</td>
      <td class="num" title="${v.mentions_kept} kept, ${v.mentions_dropped} excluded">${pct(v.grounding_rate)}</td></tr>`; }).join("")}
    </tbody></table>`;
  const starRows = [["Rule: 3 + 2 × overall score", m.stars_rule]];
  if (cal) starRows.push([`Calibrated ridge (${cal.folds}-fold, out-of-fold)`, cal.stars]);
  const stars = `<table><thead><tr><th>Method</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">Pearson r</th><th class="num">Spearman ρ</th></tr></thead><tbody>
    ${starRows.map(([name, s]) => `<tr><td>${name}</td><td class="num">${fmt(s.mae)}</td><td class="num">${fmt(s.rmse)}</td><td class="num">${fmt(s.pearson)}</td><td class="num">${fmt(s.spearman)}</td></tr>`).join("")}
    </tbody></table>`;
  const biz = (m.businesses || []).slice(0, 30);
  const bizTable = `<table><thead><tr><th>Business</th><th>City</th><th class="num">Reviews</th><th class="num">Mean ★</th><th>Food</th><th>Service</th><th>Ambience</th></tr></thead><tbody>
    ${biz.map((b) => `<tr><td><b>${esc(b.business)}</b></td><td>${esc(b.city)}</td><td class="num">${b.reviews}</td><td class="num">${fmt(b.mean_stars, 1)}</td>
      ${ASPECTS.map((a) => `<td>${b[a] == null ? '<span class="cell-score na">—</span>' : `<span class="cell-score ${b[a] > 0 ? "positive" : b[a] < 0 ? "negative" : "neutral"}"><i class="dot"></i>${signed(b[a])}</span>`}</td>`).join("")}</tr>`).join("")}
    </tbody></table>`;
  body.innerHTML = `
    <div class="tiles">
      ${tile("Reviews analyzed", m.n, Object.keys(m.providers).join(", "))}
      ${tile("Sentiment accuracy", pct(o.accuracy), `${ci(o.accuracy_ci)} · macro-F1 ${fmt(o.macro_f1)}`)}
      ${tile("Star error (MAE)", fmt(m.stars_rule.mae), cal ? `calibrated model ${fmt(cal.stars.mae)}` : "average error in predicted stars")}
      ${tile("Star correlation", fmt(m.stars_rule.pearson), "Pearson r, predicted vs actual")}
      ${tile("Quotes grounded", pct(g.rate), `${g.kept} kept · ${g.dropped} excluded`)}
      ${tile("Latency / review", `${fmt(m.latency_s.mean, 1)}s`, `${m.reviews_with_errors} with agent errors`)}
    </div>
    <div class="metrics-grid">
      <div class="card"><h3 class="section">Overall sentiment · actual (rows) vs predicted</h3>${confusion}</div>
      <div class="card"><h3 class="section">The three dimensions</h3><div class="table-wrap" style="margin:0">${aspects}</div>
        <p class="small muted" style="margin:12px 0 0">Mean score and ρ (Spearman) use only reviews that mention the aspect. "Grounded" is the share of agent quotes Python found in the review text.</p>
        <h3 class="section" style="margin-top:20px">Star prediction</h3><div class="table-wrap" style="margin:0">${stars}</div></div>
    </div>
    <div class="card table-wrap"><h3 class="section">Businesses in 3-D aspect space</h3>${bizTable}</div>`;
}

/* ---------------- boot ---------------- */
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  initTip();
  $$("nav.tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#random-btn").addEventListener("click", randomReview);
  $("#run-btn").addEventListener("click", runAgents);
  $("#review-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runAgents(); });
  $("#batch-btn").addEventListener("click", startBatch);
  $$(".seg button").forEach((b) => b.addEventListener("click", () => {
    state.colorBy = b.dataset.color;
    $$(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderScatter();
  }));
  $("#search").addEventListener("input", (e) => { state.search = e.target.value; renderScatter(); renderTable(); });
  $("#results-table tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) select(tr.dataset.id); });
  $("#results-table tbody").addEventListener("keydown", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr && e.key === "Enter") select(tr.dataset.id); });
  try { await loadStatus(); } catch (e) { toast(`Backend unreachable: ${e.message}`); }
  renderPipeline();
  let tab = location.hash.slice(1);
  if (!TABS.includes(tab)) {
    tab = "analyze";
    try { tab = localStorage.getItem("tab") || tab; } catch (_) {}
  }
  showTab(tab);
  addEventListener("hashchange", () => { const t = location.hash.slice(1); if (TABS.includes(t)) showTab(t); });
});
