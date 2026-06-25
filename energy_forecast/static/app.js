/* global Chart */

let charts = {};

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function longRunButtons() {
  return ["btn-run", "btn-train", "btn-forecast"]
    .map((id) => document.getElementById(id))
    .filter(Boolean);
}

function setLongRunBusy(isBusy, statusText) {
  longRunButtons().forEach((b) => { b.disabled = isBusy; });
  const st = document.getElementById("status");
  if (st && statusText !== undefined) st.textContent = statusText;
}

function formatTrainedAtForBanner(isoUtc) {
  if (!isoUtc) return null;
  try {
    const d = new Date(String(isoUtc));
    if (Number.isNaN(d.getTime())) return String(isoUtc);
    return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return null;
  }
}

function trainedAtFromPayload(j) {
  if (!j || typeof j !== "object") return null;
  if (j.metadata && j.metadata.trained_at_utc) return j.metadata.trained_at_utc;
  if (j.model_metadata && j.model_metadata.trained_at_utc) return j.model_metadata.trained_at_utc;
  if (j.trained_at_utc) return j.trained_at_utc;
  return null;
}

function updateLastTrainedBanner(j) {
  const el = document.getElementById("last-trained-date");
  if (!el) return;
  const raw = trainedAtFromPayload(j);
  const label = raw ? formatTrainedAtForBanner(raw) : null;
  el.textContent = label ? `Last trained: ${label}` : "Last trained: —";
}

async function refreshModelStatus() {
  const pre = document.getElementById("model-status-pre");
  if (!pre) return;
  try {
    const r = await fetch("/api/model-status");
    const j = await r.json();
    updateLastTrainedBanner(j);
    renderModelStatus(j);
  } catch (e) {
    pre.innerHTML = `<span class="err-text">Error loading model status: ${e.message}</span>`;
  }
}

function renderPill(val) {
  const s = String(val);
  if (s === "true")  return `<span class="pill pill-good">✓ yes</span>`;
  if (s === "false") return `<span class="pill pill-bad">✗ no</span>`;
  return `<span class="pill pill-neutral">${s}</span>`;
}

/** Escape text for safe insertion into HTML (attribute / text nodes). */
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Model-status metadata values can be nested objects/arrays — avoid String(obj). */
function formatMetaValue(v) {
  if (v === null || v === undefined) {
    return `<span class="kv-val kv-muted">—</span>`;
  }
  if (typeof v === "boolean") {
    return `<span class="kv-val">${renderPill(v)}</span>`;
  }
  if (typeof v === "number" || typeof v === "bigint") {
    return `<span class="kv-val">${escapeHtml(String(v))}</span>`;
  }
  if (typeof v === "string") {
    return `<span class="kv-val">${escapeHtml(v)}</span>`;
  }
  // Array of objects → compact HTML table
  if (Array.isArray(v) && v.length > 0 && typeof v[0] === "object" && v[0] !== null) {
    const keys = Object.keys(v[0]);
    return `<div class="meta-table-wrap">
      <table class="meta-compact-table">
        <thead><tr>${keys.map(k => `<th>${escapeHtml(k.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
        <tbody>${v.map(row => `<tr>${keys.map(k => `<td>${escapeHtml(String(row[k] ?? "—"))}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>`;
  }
  // Plain array of scalars → comma list
  if (Array.isArray(v)) {
    return `<span class="kv-val">${escapeHtml(v.join(", "))}</span>`;
  }
  // Plain object → inline KV pairs
  if (typeof v === "object") {
    const entries = Object.entries(v).filter(([, val]) => !Array.isArray(val) && typeof val !== "object");
    if (entries.length > 0) {
      return `<div class="meta-inline-kv">${entries.map(([k, val]) =>
        `<span class="meta-kv-pair"><span class="meta-kv-key">${escapeHtml(k.replace(/_/g, " "))}</span><span class="meta-kv-val">${escapeHtml(String(val ?? "—"))}</span></span>`
      ).join("")}</div>`;
    }
  }
  // Fallback: compact JSON (no indent)
  return `<code class="meta-json-inline">${escapeHtml(JSON.stringify(v))}</code>`;
}

function renderModelStatus(j) {
  const container = document.getElementById("model-status-pre");
  if (!container) return;

  const scalars = [];
  let metadata  = null;
  let artifacts = null;

  for (const [k, v] of Object.entries(j)) {
    if (k === "metadata" && typeof v === "object" && v !== null) { metadata = v; continue; }
    if (k === "artifacts" && typeof v === "object" && v !== null) { artifacts = v; continue; }
    scalars.push([k, v]);
  }

  const scalarHtml = scalars.map(([k, v]) => `
    <div class="kv-row">
      <span class="kv-key">${k.replace(/_/g, " ")}</span>
      ${renderPill(v)}
    </div>`).join("");

  let metaHtml = "";
  if (metadata) {
    if (metadata.profiles_meta && Array.isArray(metadata.profiles_meta)) {
      metaHtml += `<div class="ms-section">
        <div class="ms-section-title">Profiles Meta</div>
        <div class="meta-table-wrap">
          <table class="meta-compact-table">
            <thead><tr>${Object.keys(metadata.profiles_meta[0] || {}).map(k => `<th>${escapeHtml(k.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
            <tbody>
              ${metadata.profiles_meta.map(row => `<tr>${Object.values(row).map(v => `<td>${escapeHtml(String(v))}</td>`).join("")}</tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>`;
    }
    metaHtml += `<div class="ms-section">
      <div class="ms-section-title">Metadata</div>
      <div class="kv-grid">
        ${Object.entries(metadata).filter(([k]) => k !== "profiles_meta").map(([k, v]) => `
          <div class="kv-row kv-row-meta">
            <span class="kv-key">${escapeHtml(k.replace(/_/g, " "))}</span>
            ${formatMetaValue(v)}
          </div>`).join("")}
      </div>
    </div>`;
  }

  const artifactsHtml = artifacts ? `
    <div class="ms-section">
      <div class="ms-section-title">Artifact files</div>
      <div class="artifact-grid">
        ${Object.entries(artifacts).map(([k, v]) => `
          <div class="artifact-item">
            ${renderPill(v)}
            <span class="artifact-name">${k}</span>
          </div>`).join("")}
      </div>
    </div>` : "";

  container.innerHTML = `
    <div class="kv-grid">${scalarHtml}</div>
    ${metaHtml}
    ${artifactsHtml}`;
}

function scoreClass(value, goodLimit, warnLimit) {
  if (value <= goodLimit) return "good";
  if (value <= warnLimit) return "warn";
  return "bad";
}

function renderSummaryCards(payload) {
  const v = payload.validation || {};
  /* Hourly / short-term KPI cards (disabled for daily-only UI)
  const blocksHourly = [
    { label: "Validation hourly MAPE", value: `${Number(v.hourly_mape_pct || 0).toFixed(2)}%`, cls: scoreClass(Number(v.hourly_mape_pct || 100), 20, 30), icon: "⏱" },
    { label: "Active hours (MAPE)",    value: `${v.active_hours_for_mape || 0}`, cls: "good", icon: "⚡" },
    { label: "7d short-term MAPE",     value: `${Number(payload.hourly_short_term_metrics?.mape_pct || 0).toFixed(2)}%`, cls: scoreClass(Number(payload.hourly_short_term_metrics?.mape_pct || 100), 15, 22), icon: "📈" },
  ];
  */
  const blocks = [
    { label: "Validation daily MAPE",  value: `${Number(v.daily_mape_pct  || 0).toFixed(2)}%`, cls: scoreClass(Number(v.daily_mape_pct  || 100), 10, 25), icon: "📅" },
    { label: "Monthly total error ⚠",  value: `${Number(v.val_total_error_pct || 0).toFixed(2)}%`, cls: scoreClass(Math.abs(Number(v.val_total_error_pct || 0)), 2, 15), icon: "🎯", title: "Not an accuracy claim — this is the threshold optimizer's own residual, measured on the same validation data used to tune it. Use Daily MAPE for genuine accuracy." },
  ];
  if (v.classifier_accuracy !== undefined && v.classifier_accuracy !== null && !Number.isNaN(Number(v.classifier_accuracy))) {
    const accPct = Number(v.classifier_accuracy) * 100;
    blocks.push({
      label: "Classifier accuracy",
      value: `${accPct.toFixed(1)}%`,
      cls: scoreClass(100 - accPct, 10, 25),
      icon: "🎚",
    });
  }
  document.getElementById("summary-cards").innerHTML = blocks.map(b => `
    <div class="kpi"${b.title ? ` title="${b.title}"` : ""}>
      <div class="label"><span class="kpi-icon">${b.icon}</span>${b.label}</div>
      <div class="value ${b.cls}">${b.value}</div>
    </div>`).join("");
}

function renderFlowBrief(payload) {
  const v = payload.validation || {};
  const flow = [
    { title: "Read & clean", icon: "🗄", text: "Raw readings from MongoDB → hourly kWh series for daily totals, full routines(weekdays, weekends, holidays), daily shapes (morning spikes, evening dips etc.) and calendar flags." },
    { title: "Daily forecast", icon: "📊", text: "Two-stage model: active/shutdown classifier → daily kWh predictor with recency correction; multi-day horizon from saved models." },
    /* Hourly / hybrid steps (disabled for daily-only UI)
    { title: "Hourly shape",  icon: "🔄", text: "Day-ahead model uses 24h / 48h / 168h lag patterns to build realistic hour-by-hour load shape." },
    { title: "48h blend",     icon: "🔀", text: "Near-term hours lean on recursive short-term forecasts, then smoothly blend into daily-guided profile." },
    */
    { title: "Validate", icon: "✅", text: `Daily MAPE ${Number(v.daily_mape_pct || 0).toFixed(2)}% · Total error ${Number(v.val_total_error_pct || 0).toFixed(2)}%` },
  ];
  const flowContent = document.querySelector('#flow-brief .dropdown-content');
  const target = flowContent || document.getElementById("flow-brief");
  if (!target) return;
  target.innerHTML = `
    <div class="flow-steps">
      ${flow.map((s, i) => `
        <div class="flow-step">
          <div class="flow-step-header">
            <span class="flow-icon">${s.icon}</span>
            <span class="flow-num">${i + 1}</span>
            <h3>${s.title}</h3>
          </div>
          <p>${s.text}</p>
        </div>`).join("")}
    </div>`;
}

// Validation metrics panel
function renderValMetrics(v) {
  const el = document.getElementById("val-metrics");
  if (!v || !Object.keys(v).length) { el.innerHTML = `<span class="muted-text">No data yet.</span>`; return; }

  const rows = [
    /* { label: "Hourly MAPE", key: "hourly_mape_pct",     unit: "%", fmt: 2, good: 20, warn: 30 }, */
    { label: "Daily MAPE",  key: "daily_mape_pct",      unit: "%", fmt: 2, good: 10, warn: 25 },
    { label: "Total error ⚠", key: "val_total_error_pct", unit: "%", fmt: 2, abs: true, good: 2, warn: 10, caveat: true },
    /* { label: "Active hours",key: "active_hours_for_mape",unit: "", fmt: 0 }, */
    { label: "Val samples", key: "n_val",               unit: "", fmt: 0 },
  ];

  el.innerHTML = `<div class="metric-table">
    ${rows.map(r => {
      const raw = v[r.key];
      if (raw === undefined || raw === null) return "";
      const num = Number(raw);
      const display = `${num.toFixed(r.fmt)}${r.unit}`;
      const cls = r.good !== undefined ? scoreClass(r.abs ? Math.abs(num) : num, r.good, r.warn) : "";
      const bar = r.good !== undefined
        ? `<div class="mbar-wrap"><div class="mbar mbar-${cls}" style="width:${Math.min(100, (r.abs ? Math.abs(num) : num) / (r.warn * 1.5) * 100).toFixed(1)}%"></div></div>`
        : "";
      return `<div class="metric-row"${r.caveat ? ' title="Not a true accuracy claim — see note below."' : ''}>
        <span class="metric-label">${r.label}</span>
        <span class="metric-val ${cls}">${display}</span>
        ${bar}
      </div>`;
    }).join("")}
  </div>
  <p class="metric-caveat">
    <strong>⚠ Total error is not an accuracy claim.</strong>
    The threshold was tuned to minimise this exact figure on the same validation data — so it reflects the optimizer's residual, not generalisation. Calibration is also fit on the same set, pushing it further toward zero.
    Use <strong>Daily MAPE</strong> for a genuine measure of per-day forecast accuracy.
  </p>`;
}

// Short-term hourly metrics panel (disabled in daily-only UI — #hourly-metrics commented out in index.html)
function renderHourlyMetrics(_h) {
  const el = document.getElementById("hourly-metrics");
  if (!el) return;
  el.innerHTML = `<span class="muted-text">Short-term hourly metrics are not shown (daily-only mode).</span>`;
  /*
  if (!h || !Object.keys(h).length) { el.innerHTML = `<span class="muted-text">No data yet.</span>`; return; }

  const rows = [
    { label: "MAPE",       key: "mape_pct", unit: "%",    fmt: 2, good: 15, warn: 22 },
    { label: "MAE (kWh)",  key: "mae_kwh",  unit: " kWh", fmt: 3 },
    { label: "RMSE (kWh)", key: "rmse_kwh", unit: " kWh", fmt: 3 },
    { label: "R²",         key: "r2",       unit: "",     fmt: 4 },
    { label: "Samples",    key: "n",        unit: "",     fmt: 0 },
  ];

  el.innerHTML = `<div class="metric-table">
    ${rows.map(r => {
      const raw = h[r.key];
      if (raw === undefined || raw === null) return "";
      const num = Number(raw);
      const display = `${num.toFixed(r.fmt)}${r.unit}`;
      const cls = r.good !== undefined ? scoreClass(num, r.good, r.warn) : "";
      const bar = r.good !== undefined
        ? `<div class="mbar-wrap"><div class="mbar mbar-${cls}" style="width:${Math.min(100, num / (r.warn * 1.5) * 100).toFixed(1)}%"></div></div>`
        : "";
      return `<div class="metric-row">
        <span class="metric-label">${r.label}</span>
        <span class="metric-val ${cls}">${display}</span>
        ${bar}
      </div>`;
    }).join("")}
  </div>`;
  */
}

// Recency panel
function renderRecency(data) {
  const el = document.getElementById("recency");
  if (!data || !Object.keys(data).length) { el.innerHTML = `<span class="muted-text">No data yet.</span>`; return; }

  const fmt = (v) => {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return renderPill(v);
    if (typeof v === "string" && /T\d{2}:\d{2}/.test(v)) return v.replace("T", " ").slice(0, 19) + " UTC";
    if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(4);
    return String(v);
  };

  const scalars = [];
  const nested  = [];
  for (const [k, v] of Object.entries(data)) {
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) continue;
    if (typeof v === "object") { nested.push([k, v]); continue; }
    scalars.push([k, v]);
  }

  const scalarHtml = scalars.length ? `
    <div class="kv-grid">
      ${scalars.map(([k, v]) => `
        <div class="kv-row">
          <span class="kv-key">${k.replace(/_/g, " ")}</span>
          <span class="kv-val">${fmt(v)}</span>
        </div>`).join("")}
    </div>` : "";

  const nestedHtml = nested.map(([k, obj]) => `
    <div class="ms-section">
      <div class="ms-section-title">${k.replace(/_/g, " ")}</div>
      <div class="kv-grid">
        ${Object.entries(obj)
          .filter(([, v]) => !Array.isArray(v) && typeof v !== "object")
          .map(([ik, iv]) => `
            <div class="kv-row">
              <span class="kv-key">${ik.replace(/_/g, " ")}</span>
              <span class="kv-val">${fmt(iv)}</span>
            </div>`).join("")}
      </div>
    </div>`).join("");

  el.innerHTML = (scalarHtml + nestedHtml) || `<span class="muted-text">No data yet.</span>`;
}

// Chart renderers
/* Axes / legend tuned for navy–white cards; dataset colors unchanged below */
const chartDefaults = {
  plugins: { legend: { labels: { color: "#ffffff", font: { family: "'Space Mono', monospace" } } } },
  scales: {
    x: { ticks: { color: "#ffffff" }, grid: { color: "#e2e8f0" } },
    y: { ticks: { color: "#ffffff" }, grid: { color: "#e2e8f0" } },
  },
};

function renderFutureDaily(rows) {
  destroyChart("fd");
  const labels = rows.map((r) => String(r.ds).slice(0, 10));
  const data   = rows.map((r) => r.pred);
  charts.fd = new Chart(document.getElementById("chart-future-daily"), {
    type: "bar",
    data: { labels, datasets: [{ label: "Predicted kWh/day", data, backgroundColor: "#3d9cf0aa", borderRadius: 4 }] },
    options: { responsive: true, ...chartDefaults },
  });
}

function renderFutureHourly(rows) {
  const canvas = document.getElementById("chart-future-hourly");
  if (!canvas) return; // daily-only: canvas section commented in index.html
  void rows;
  /*
  destroyChart("fh");
  const labels = rows.map((r) => String(r.rtc_timestamp).replace("T", " ").slice(0, 16));
  const data   = rows.map((r) => r.forecast_kwh);
  charts.fh = new Chart(canvas, {
    type: "line",
    data: { labels, datasets: [{ label: "kWh/h", data, borderColor: "#6ee7b7", backgroundColor: "#6ee7b712", fill: true, tension: 0.3, pointRadius: 0 }] },
    options: { responsive: true, ...chartDefaults, scales: { ...chartDefaults.scales, x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxTicksLimit: 12 } } } },
  });
  */
}

function render48h(rows) {
  const canvas = document.getElementById("chart-48h");
  if (!canvas) return; // daily-only: canvas section commented in index.html
  void rows;
  /*
  destroyChart("h48");
  const labels = rows.map((r) => String(r.rtc_timestamp).replace("T", " ").slice(0, 16));
  const data   = rows.map((r) => r.forecast_kwh);
  charts.h48 = new Chart(canvas, {
    type: "line",
    data: { labels, datasets: [{ label: "Hybrid 48h kWh/h", data, borderColor: "#f59e0b", backgroundColor: "#f59e0b12", fill: true, tension: 0.2, pointRadius: 0 }] },
    options: { responsive: true, ...chartDefaults, scales: { ...chartDefaults.scales, x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxTicksLimit: 14 } } } },
  });
  */
}

function renderImportance(items) {
  const canvas = document.getElementById("chart-importance");
  if (!canvas) return; // daily-only: canvas section commented in index.html
  void items;
  /*
  destroyChart("imp");
  const labels = items.map((x) => x.feature).reverse();
  const data   = items.map((x) => x.importance).reverse();
  charts.imp = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ label: "Importance", data, backgroundColor: "#a78bfa99", borderRadius: 3 }] },
    options: {
      indexAxis: "y", responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#475569" }, grid: { color: "#e2e8f0" } },
        y: { ticks: { color: "#475569", font: { size: 10 } }, grid: { display: false } },
      },
    },
  });
  */
}

function renderValDaily(rows) {
  destroyChart("vd");
  const labels = rows.map((r) => r.ds);
  charts.vd = new Chart(document.getElementById("chart-val-daily"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Actual",    data: rows.map((r) => r.y),    backgroundColor: "#3d9cf0aa", borderRadius: 3 },
        { label: "Predicted", data: rows.map((r) => r.pred), backgroundColor: "#fb7185aa", borderRadius: 3 },
      ],
    },
    options: { responsive: true, ...chartDefaults, scales: { ...chartDefaults.scales, x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxRotation: 45 } } } },
  });
}

function getTrainDays() {
  const val = document.getElementById("train-days").value.trim();
  if (val.toLowerCase() === "full") return "full";
  const num = Number(val);
  return num >= 251 ? num : null;
}

// Shared result render helpers
function showTrainResult(j) {
  updateLastTrainedBanner(j);
  const v = j.validation || {};
  const h = j.hourly_short_term_metrics || {};
  renderSummaryCards({ validation: v, hourly_short_term_metrics: h });
  renderFlowBrief({ validation: v });
  renderValMetrics(v);
  renderHourlyMetrics(h);
  renderRecency({
    mode: "train_only",
    trained_at_utc: j.trained_at_utc,
    model_dir: j.model_dir,
    // hint: "Click Forecast to update 7-day and 48h charts.",
    hint: "Click Forecast to refresh the multi-day daily forecast chart.",
  });
}

function showResult(payload) {
  updateLastTrainedBanner(payload);
  renderSummaryCards(payload);
  renderFlowBrief(payload);
  renderValMetrics(payload.validation);
  renderHourlyMetrics(payload.hourly_short_term_metrics);
  const recencyData = {};
  const scalarKeys = ["recency_ratio_val", "recency_ratio_full", "hourly_rows", "meter_id", "data_start", "data_end", "spw"];
  for (const k of scalarKeys) {
    if (payload[k] !== undefined && payload[k] !== null) recencyData[k] = payload[k];
  }
  if (payload.model_metadata && typeof payload.model_metadata === "object") {
    recencyData.model_metadata = payload.model_metadata;
  }
  renderRecency(recencyData);
  renderFutureDaily(payload.future_daily || []);
  /* Intra-day / hybrid charts disabled for daily-only UI (canvases commented in index.html).
     Previous calls kept for reference:
  renderFutureHourly(payload.future_hourly || []);
  render48h(payload.forecast_48h || []);
  renderImportance(payload.hourly_feature_importance_top15 || []);
  */
  renderValDaily(payload.val_daily_table || []);
}

// Button handlers
document.getElementById("btn-run").addEventListener("click", async () => {
  setLongRunBusy(true, "Full run…");
  try {
    const r = await fetch("/api/run", { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    showResult(j);
    document.getElementById("status").textContent = "Full run done.";
    await refreshModelStatus();
  } catch (e) {
    document.getElementById("status").textContent = "Error: " + e.message;
  } finally {
    setLongRunBusy(false);
  }
});

function updateConfigTrainDays(trainDays) {
  const rows = document.querySelectorAll("#config .kv-row");
  rows.forEach(row => {
    const key = row.querySelector(".kv-key")?.textContent;
    if (key === "Train lookback") {
      row.querySelector(".kv-val").textContent = `${trainDays}d`;
    }
  });
}

document.getElementById("btn-train").addEventListener("click", async () => {
  // const trainDays = getTrainDays();
  const trainDays = "full";
  if (!trainDays) {
    document.getElementById("status").textContent = "Error: Train days must be at least 251.";
    return;
  }
  setLongRunBusy(true, "Training…");
  try {
    const r = await fetch("/api/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ train_days: trainDays })
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    showTrainResult(j);
    if (j.train_days) {
      updateConfigTrainDays(j.train_days);
    }
    document.getElementById("status").textContent = "Train done.";
    await refreshModelStatus();
  } catch (e) {
    document.getElementById("status").textContent = "Error: " + e.message;
  } finally {
    setLongRunBusy(false);
  }
});

document.getElementById("btn-forecast").addEventListener("click", async () => {
  setLongRunBusy(true, "Forecasting…");
  try {
    const r = await fetch("/api/forecast", { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    showResult(j);
    document.getElementById("status").textContent = "Forecast done.";
    await refreshModelStatus();
  } catch (e) {
    document.getElementById("status").textContent = "Error: " + e.message;
  } finally {
    setLongRunBusy(false);
  }
});

// btn-status is commented out in index.html; listener removed to prevent null crash.

// Config loader
async function loadConfig() {
  // const r = await fetch("/api/config?train_days=full");
  // const c = await r.json();

  // const configFields = [
  //   { label: "Meter ID",         value: c.meter_id },
  //   { label: "Forecast days",    value: c.forecast_days },
  //   { label: "Validation days",  value: c.val_days },
  //   { label: "Train lookback",   value: `${c.train_raw_lookback_days}d` },
  //   { label: "Infer lookback",   value: `${c.infer_raw_lookback_days}d` },
  //   { label: "Model directory",  value: c.model_dir },
  // ];

  // const configContent = document.querySelector('#config .dropdown-content');
  // if (configContent) {
  //   configContent.innerHTML = `
  //     <div class="kv-grid">
  //       ${configFields.map(f => `
  //         <div class="kv-row">
  //           <span class="kv-key">${f.label}</span>
  //           <span class="kv-val">${f.value ?? "—"}</span>
  //         </div>`).join("")}
  //     </div>`;
  // }

  /* Default banner before first run (hourly charts disabled — daily forecast only)
  document.getElementById("summary-cards").innerHTML = `
    <div class="kpi">
      <div class="label">Pipeline status</div>
      <div class="value warn">Use Full run, or Train → Forecast</div>
    </div>`;
  */
  document.getElementById("summary-cards").innerHTML = `
    <div class="kpi">
      <div class="label">Pipeline status</div>
      <div class="value good">Daily forecast: use Full run, or Train → Forecast</div>
    </div>`;

  const flowBriefContent = document.querySelector('#flow-brief .dropdown-content');
  if (flowBriefContent) {
    /* flowBriefContent.innerHTML = `
      <p class="sub"><strong>Full run</strong> does everything end-to-end. For production cadence, use <strong>Train only</strong> (scheduled) then <strong>Forecast</strong> (frequent).</p>`; */
    flowBriefContent.innerHTML = `
    <p class="sub">
      <strong>Full run</strong> retrains daily models and refreshes the <strong>multi-day daily</strong> chart.
    </p>
    <p class="sub">
      Use <strong>Train only</strong> (scheduled) then <strong>Forecast</strong> for frequent daily updates.
    </p>`;
  }

  await refreshModelStatus();
}

// Dropdowns
function setupDropdowns() {
  document.querySelectorAll('.dropdown-section').forEach(section => {
    const btn = section.querySelector('.dropdown-toggle');

    if (btn) {
      btn.addEventListener('click', () => {
        const isOpen = section.classList.toggle('open');

        btn.setAttribute('aria-expanded', String(isOpen));
        btn.textContent = isOpen ? '▼' : '►';
      });
    }
  });
}

// Init
function init() {
  setupDropdowns();
  loadConfig().catch(console.error);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}