/* global Chart */

let charts = {};

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

/* Axes / legend for white cards; bar & line dataset colors unchanged */
const chartDefaults = {
  plugins: { legend: { labels: { color: "#1e293b", font: { family: "'IBM Plex Mono', monospace" } } } },
  scales: {
    x: { ticks: { color: "#475569" }, grid: { color: "#e2e8f0" } },
    y: { ticks: { color: "#475569" }, grid: { color: "#e2e8f0" } },
  },
};

async function loadConfig() {
  const r = await fetch("/api/config");
  const c = await r.json();
  const brief = {
    meter_id: c.meter_id,
    forecast_days: c.forecast_days,
    val_days: c.val_days,
    mongo_collection: c.mongo_collection,
    model_dir: c.model_dir,
  };
  document.getElementById("config").innerHTML =
    `<h2>Configuration snapshot</h2><pre>${JSON.stringify(brief, null, 2)}</pre>`;
  document.getElementById("summary-cards").innerHTML =
    `<div class="kpi"><div class="label">Pipeline status</div><div class="value warn">Waiting for run</div></div>`;
  document.getElementById("flow-brief").innerHTML =
    `<h2>How this forecast works (simple flow)</h2><p class="sub">Run pipeline to see live metrics, charts, and the explained decision flow.</p>`;
}

function scoreClass(value, goodLimit, warnLimit) {
  if (value <= goodLimit) return "good";
  if (value <= warnLimit) return "warn";
  return "bad";
}

function renderSummaryCards(payload) {
  const v = payload.validation || {};
  const blocks = [
    {
      label: "Validation hourly MAPE",
      value: `${Number(v.hourly_mape_pct || 0).toFixed(2)}%`,
      cls: scoreClass(Number(v.hourly_mape_pct || 100), 20, 30),
    },
    {
      label: "Validation daily MAPE",
      value: `${Number(v.daily_mape_pct || 0).toFixed(2)}%`,
      cls: scoreClass(Number(v.daily_mape_pct || 100), 10, 20),
    },
    {
      label: "Monthly total error",
      value: `${Number(v.val_total_error_pct || 0).toFixed(2)}%`,
      cls: scoreClass(Math.abs(Number(v.val_total_error_pct || 0)), 2, 5),
    },
    {
      label: "Active hours (MAPE)",
      value: `${v.active_hours_for_mape || 0}`,
      cls: "good",
    },
    {
      label: "7d short-term MAPE",
      value: `${Number(payload.hourly_short_term_metrics?.mape_pct || 0).toFixed(2)}%`,
      cls: scoreClass(Number(payload.hourly_short_term_metrics?.mape_pct || 100), 15, 22),
    },
  ];
  document.getElementById("summary-cards").innerHTML = blocks
    .map(
      (b) => `<div class="kpi">
        <div class="label">${b.label}</div>
        <div class="value ${b.cls}">${b.value}</div>
      </div>`
    )
    .join("");
}

function renderFlowBrief(payload) {
  const v = payload.validation || {};
  const flow = [
    {
      title: "1) Read and clean meter data",
      text: "Raw minute-level energy readings are loaded from MongoDB, converted to hourly consumption, and enriched with calendar + machine-state features.",
    },
    {
      title: "2) Predict daily energy budget",
      text: "A two-stage daily model first decides if the day is active/shutdown, then predicts daily kWh for active days with recency correction.",
    },
    {
      title: "3) Build hourly shape",
      text: "The day-ahead hourly model uses lagged hourly patterns (24h, 48h, 168h) and operating signals to create realistic hour-by-hour load shape.",
    },
    {
      title: "4) Blend for near-term control",
      text: "For 48h output, the first few hours lean on recursive short-term hourly forecasting, then smoothly blend into the daily-guided hourly profile.",
    },
    {
      title: "5) Validate and explain",
      text: `Current validation: hourly MAPE ${Number(v.hourly_mape_pct || 0).toFixed(2)}%, daily MAPE ${Number(v.daily_mape_pct || 0).toFixed(2)}%, and total error ${Number(v.val_total_error_pct || 0).toFixed(2)}%.`,
    },
  ];
  document.getElementById("flow-brief").innerHTML = `
    <h2>How this forecast works (simple flow)</h2>
    <div class="flow-steps">
      ${flow
        .map(
          (s) => `<div class="flow-step">
            <h3>${s.title}</h3>
            <p>${s.text}</p>
          </div>`
        )
        .join("")}
    </div>
  `;
}

function renderFutureDaily(rows) {
  destroyChart("fd");
  const labels = rows.map((r) => String(r.ds).slice(0, 10));
  const data = rows.map((r) => r.pred);
  const ctx = document.getElementById("chart-future-daily");
  charts.fd = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Predicted kWh/day", data, backgroundColor: "#3d9cf0aa" }],
    },
    options: {
      responsive: true,
      ...chartDefaults,
    },
  });
}

function renderFutureHourly(rows) {
  destroyChart("fh");
  const labels = rows.map((r) => String(r.rtc_timestamp).replace("T", " ").slice(0, 16));
  const data = rows.map((r) => r.forecast_kwh);
  charts.fh = new Chart(document.getElementById("chart-future-hourly"), {
    type: "line",
    data: {
      labels,
      datasets: [{ label: "kWh/h", data, borderColor: "#6ee7b7", tension: 0.1, pointRadius: 0 }],
    },
    options: {
      responsive: true,
      plugins: chartDefaults.plugins,
      scales: {
        x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxTicksLimit: 12 } },
        y: chartDefaults.scales.y,
      },
    },
  });
}

function render48h(rows) {
  destroyChart("h48");
  const labels = rows.map((r) => String(r.rtc_timestamp).replace("T", " ").slice(0, 16));
  const data = rows.map((r) => r.forecast_kwh);
  charts.h48 = new Chart(document.getElementById("chart-48h"), {
    type: "line",
    data: {
      labels,
      datasets: [{ label: "Hybrid 48h kWh/h", data, borderColor: "#f59e0b", tension: 0.15, pointRadius: 0 }],
    },
    options: {
      responsive: true,
      plugins: chartDefaults.plugins,
      scales: {
        x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxTicksLimit: 14 } },
        y: chartDefaults.scales.y,
      },
    },
  });
}

function renderImportance(items) {
  destroyChart("imp");
  const labels = items.map((x) => x.feature).reverse();
  const data = items.map((x) => x.importance).reverse();
  charts.imp = new Chart(document.getElementById("chart-importance"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Importance", data, backgroundColor: "#a78bfa99" }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: chartDefaults.scales.x,
        y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, font: { size: 10 } }, grid: { display: false } },
      },
    },
  });
}

function renderValDaily(rows) {
  destroyChart("vd");
  const labels = rows.map((r) => r.ds);
  charts.vd = new Chart(document.getElementById("chart-val-daily"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Actual", data: rows.map((r) => r.y), backgroundColor: "#3d9cf0aa" },
        { label: "Predicted", data: rows.map((r) => r.pred), backgroundColor: "#fb7185aa" },
      ],
    },
    options: {
      responsive: true,
      plugins: chartDefaults.plugins,
      scales: {
        x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxRotation: 45 } },
        y: chartDefaults.scales.y,
      },
    },
  });
}

function showResult(payload) {
  renderSummaryCards(payload);
  renderFlowBrief(payload);
  document.getElementById("val-metrics").textContent = JSON.stringify(payload.validation, null, 2);
  document.getElementById("hourly-metrics").textContent = JSON.stringify(
    payload.hourly_short_term_metrics,
    null,
    2
  );
  document.getElementById("recency").textContent = JSON.stringify(
    {
      recency_ratio_val: payload.recency_ratio_val,
      recency_ratio_full: payload.recency_ratio_full,
      hourly_rows: payload.hourly_rows,
      meter_id: payload.meter_id,
      data_start: payload.data_start,
      data_end: payload.data_end,
    },
    null,
    2
  );
  renderFutureDaily(payload.future_daily || []);
  renderFutureHourly(payload.future_hourly || []);
  render48h(payload.forecast_48h || []);
  renderImportance(payload.hourly_feature_importance_top15 || []);
  renderValDaily(payload.val_daily_table || []);
}

document.getElementById("btn-run").addEventListener("click", async () => {
  const btn = document.getElementById("btn-run");
  const st = document.getElementById("status");
  btn.disabled = true;
  st.textContent = "Running…";
  try {
    const r = await fetch("/api/run", { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    showResult(j);
    st.textContent = "Done.";
  } catch (e) {
    st.textContent = "Error: " + e.message;
    console.error(e);
  } finally {
    btn.disabled = false;
  }
});

loadConfig().catch(console.error);
