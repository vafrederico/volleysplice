#!/usr/bin/env python3
"""Build a self-contained HTML report for the beach-exclusion experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/"
    "beach-exclusion-retraining-comparison.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/reports/"
    "beach-exclusion-retraining-comparison.html"
)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Beach-exclusion retraining comparison</title>
  <style>
    :root {
      --ink: #172033;
      --muted: #61708a;
      --line: #dce3ee;
      --surface: #ffffff;
      --soft: #f4f7fb;
      --navy: #14253f;
      --blue: #2878d6;
      --blue-soft: #e9f2ff;
      --green: #0b7a58;
      --green-soft: #e8f7f1;
      --red: #b54646;
      --red-soft: #fff0f0;
      --amber: #93630b;
      --amber-soft: #fff7e2;
      --shadow: 0 12px 32px rgba(20, 37, 63, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #eef2f7;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    a { color: var(--blue); }
    code, .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
    .shell { max-width: 1440px; margin: 0 auto; padding: 34px 24px 70px; }
    .hero {
      color: #fff;
      background: linear-gradient(135deg, #14253f 0%, #1e4b7d 58%, #2878d6 100%);
      border-radius: 24px;
      padding: 34px 38px;
      box-shadow: var(--shadow);
    }
    .eyebrow {
      margin: 0 0 8px;
      color: #bcdcff;
      font-size: 0.78rem;
      font-weight: 750;
      letter-spacing: 0.13em;
      text-transform: uppercase;
    }
    h1, h2, h3 { line-height: 1.15; }
    h1 { max-width: 900px; margin: 0; font-size: clamp(2rem, 4vw, 3.4rem); letter-spacing: -0.04em; }
    h2 { margin: 0 0 18px; font-size: 1.45rem; letter-spacing: -0.02em; }
    h3 { margin: 0 0 10px; font-size: 1.05rem; }
    .hero p { max-width: 900px; margin: 16px 0 0; color: #e3efff; font-size: 1.03rem; }
    .hero-meta { display: flex; flex-wrap: wrap; gap: 9px 18px; margin-top: 22px; color: #cce2ff; font-size: 0.88rem; }
    .hero-meta span { white-space: nowrap; }

    .section { margin-top: 26px; }
    .surface {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 5px 18px rgba(20, 37, 63, 0.04);
    }
    .section-intro { margin: -5px 0 16px; color: var(--muted); }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    .card { padding: 20px; }
    .card-label { color: var(--muted); font-size: 0.82rem; font-weight: 700; }
    .card-value { margin-top: 8px; font-size: 1.85rem; font-weight: 780; letter-spacing: -0.035em; }
    .card-detail { margin-top: 4px; color: var(--muted); font-size: 0.83rem; }
    .card.good .card-value { color: var(--green); }
    .card.warn .card-value { color: var(--amber); }

    .callout { padding: 22px 24px; border-left: 5px solid var(--blue); background: var(--blue-soft); }
    .callout strong { color: var(--navy); }
    .callout p { margin: 0; }

    .two-col { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .panel { padding: 22px; }
    .panel-subtitle { margin: -4px 0 14px; color: var(--muted); font-size: 0.88rem; }
    .highlight-list { display: grid; gap: 10px; }
    .highlight-row { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 10px 0; border-bottom: 1px solid var(--line); }
    .highlight-row:last-child { border-bottom: 0; }
    .highlight-name { min-width: 0; overflow-wrap: anywhere; font-size: 0.9rem; }
    .highlight-value { flex: 0 0 auto; font-weight: 760; }
    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .neutral { color: var(--muted); }
    .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 8px; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
    .badge.good { color: var(--green); background: var(--green-soft); }
    .badge.bad { color: var(--red); background: var(--red-soft); }
    .badge.neutral { color: var(--muted); background: var(--soft); }
    .badge.ood { color: var(--amber); background: var(--amber-soft); }

    .controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; margin-bottom: 16px; }
    .control { display: grid; gap: 5px; }
    .control label { color: var(--muted); font-size: 0.76rem; font-weight: 750; text-transform: uppercase; letter-spacing: 0.06em; }
    select, input {
      min-height: 40px;
      border: 1px solid #cbd5e2;
      border-radius: 9px;
      padding: 8px 11px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }
    input { min-width: 260px; }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
    tr:last-child td { border-bottom: 0; }
    th { color: #43536d; background: #f7f9fc; font-size: 0.75rem; font-weight: 780; letter-spacing: 0.03em; text-transform: uppercase; white-space: nowrap; }
    th button { border: 0; padding: 0; color: inherit; background: transparent; font: inherit; text-transform: inherit; letter-spacing: inherit; cursor: pointer; }
    th button:hover { color: var(--blue); }
    td.numeric, th.numeric { text-align: right; font-variant-numeric: tabular-nums; }
    td.model-cell { min-width: 260px; }
    .model-name { font-weight: 700; overflow-wrap: anywhere; }
    .model-family { margin-top: 2px; color: var(--muted); font-size: 0.76rem; overflow-wrap: anywhere; }
    .delta-cell { font-weight: 760; }
    .table-note { margin: 11px 0 0; color: var(--muted); font-size: 0.8rem; }

    .scorecard td:first-child { min-width: 210px; font-weight: 700; }
    .metric-definition { color: var(--muted); font-size: 0.78rem; font-weight: 400; }
    .small-table { font-size: 0.8rem; }
    .small-table th, .small-table td { padding: 9px 10px; }
    .coverage-summary { margin-top: 18px; }
    .coverage-summary h3 { margin-bottom: 12px; }
    .path { display: block; max-width: 420px; overflow: hidden; color: var(--muted); font-size: 0.7rem; text-overflow: ellipsis; white-space: nowrap; }
    .protocol-grid { display: grid; grid-template-columns: minmax(160px, 0.35fr) 1fr; gap: 0; }
    .protocol-grid dt, .protocol-grid dd { margin: 0; padding: 11px 0; border-bottom: 1px solid var(--line); }
    .protocol-grid dt { color: var(--muted); font-weight: 750; }
    .protocol-grid dd { padding-left: 20px; }
    .limitations { margin: 18px 0 0; padding: 16px 18px; color: #5b4a1e; background: var(--amber-soft); border-radius: 12px; }
    .limitations ul { margin: 8px 0 0; padding-left: 20px; }
    footer { margin-top: 26px; color: var(--muted); font-size: 0.78rem; }
    .empty { padding: 28px; color: var(--muted); text-align: center; }

    @media (max-width: 980px) {
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two-col { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      .shell { padding: 18px 12px 50px; }
      .hero { padding: 26px 22px; border-radius: 18px; }
      .cards { grid-template-columns: 1fr; }
      input { min-width: 0; width: 100%; }
      .control { flex: 1 1 180px; }
      .protocol-grid { grid-template-columns: 1fr; }
      .protocol-grid dd { padding-left: 0; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">VolleyCut AI · retraining study</p>
      <h1>What changed when beach videos left the training set?</h1>
      <p>Paired before/after comparison for every model version in the established lineage, plus fresh inference on every full recording.</p>
      <div class="hero-meta">
        <span id="created-at"></span>
        <span id="model-count"></span>
        <span id="scope-line"></span>
      </div>
    </header>

    <section class="section">
      <div class="cards" id="headline-cards"></div>
    </section>

    <section class="section surface callout" id="takeaway"></section>

    <section class="section">
      <h2>Largest Event F1 changes</h2>
      <p class="section-intro">Held-out test is the primary paired generalization view; validation is shown separately because it was used for tuning and selection.</p>
      <div class="two-col">
        <article class="surface panel">
          <h3>Test gains</h3>
          <p class="panel-subtitle">Without beach − original, sorted largest first.</p>
          <div class="highlight-list" id="test-gains"></div>
        </article>
        <article class="surface panel">
          <h3>Test losses / unchanged</h3>
          <p class="panel-subtitle">The bottom of the same Event F1 comparison.</p>
          <div class="highlight-list" id="test-losses"></div>
        </article>
      </div>
    </section>

    <section class="section surface panel">
      <h2>All metrics at a glance</h2>
      <p class="section-intro">Mean and median deltas are calculated across all 33 versions. Score metrics are displayed in percentage points; rally fields are aggregate counts.</p>
      <div class="table-wrap">
        <table class="scorecard">
          <thead>
            <tr>
              <th>Metric</th>
              <th class="numeric">Test mean Δ</th>
              <th class="numeric">Test median Δ</th>
              <th class="numeric">Test ↑ / ↓ / =</th>
              <th class="numeric">Validation mean Δ</th>
              <th class="numeric">Validation median Δ</th>
              <th class="numeric">Validation ↑ / ↓ / =</th>
            </tr>
          </thead>
          <tbody id="metric-scorecard"></tbody>
        </table>
      </div>
    </section>

    <section class="section surface panel">
      <h2>Model-by-model metric comparison</h2>
      <div class="controls">
        <div class="control">
          <label for="split-select">Evaluation split</label>
          <select id="split-select">
            <option value="test">Held-out test</option>
            <option value="validation">Validation / tuning</option>
          </select>
        </div>
        <div class="control">
          <label for="metric-select">Metric</label>
          <select id="metric-select"></select>
        </div>
        <div class="control" style="flex: 1 1 260px">
          <label for="model-filter">Filter models</label>
          <input id="model-filter" type="search" placeholder="Search model or family…">
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th><button data-sort="name">Model</button></th>
              <th><button data-sort="family">Family</button></th>
              <th class="numeric"><button data-sort="old">Original</button></th>
              <th class="numeric"><button data-sort="new">Without beach</button></th>
              <th class="numeric"><button data-sort="delta">Δ</button></th>
            </tr>
          </thead>
          <tbody id="model-table"></tbody>
        </table>
      </div>
      <p class="table-note" id="model-table-note"></p>
    </section>

    <section class="section surface panel">
      <h2>Inference for every video</h2>
      <p class="section-intro">Each beach-free model was run on all nine full recordings. Beach rows are qualitative out-of-domain inspection only and were not scored.</p>
      <div class="controls">
        <div class="control" style="flex: 1 1 360px">
          <label for="inference-model-select">Model</label>
          <select id="inference-model-select"></select>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Recording</th>
              <th>Environment</th>
              <th>Split</th>
              <th class="numeric">Predicted rallies</th>
              <th>Use in report</th>
              <th>Artifact</th>
            </tr>
          </thead>
          <tbody id="inference-table"></tbody>
        </table>
      </div>
      <p class="table-note" id="inference-note"></p>
      <div class="coverage-summary">
        <h3>Coverage audit across all models</h3>
        <div class="table-wrap">
          <table class="small-table">
            <thead>
              <tr>
                <th>Model</th>
                <th class="numeric">Fresh / expected</th>
                <th class="numeric">Beach predicted rallies</th>
                <th class="numeric">Non-beach predicted rallies</th>
              </tr>
            </thead>
            <tbody id="coverage-table"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="section surface panel">
      <h2>Method and limitations</h2>
      <dl class="protocol-grid" id="protocol"></dl>
      <div class="limitations">
        <strong>Read this as a diagnostic, not a clean prospective benchmark.</strong>
        <ul>
          <li>The test corpus had already been inspected, so the paired test comparison is retrospective.</li>
          <li>Beach recordings were removed from training; they remain in inference for qualitative out-of-domain review.</li>
          <li>After exclusion, boundary-head epoch selection used two source groups instead of the historical three-fold selector.</li>
          <li>Feature-order OOF research artifacts and the separate ball-presence detector were outside the deployable model lineage.</li>
        </ul>
      </div>
    </section>

    <footer id="footer"></footer>
  </main>

  <script id="report-data" type="application/json">__REPORT_JSON__</script>
  <script>
    (() => {
      "use strict";

      const REPORT = JSON.parse(document.getElementById("report-data").textContent);
      const SCORE_METRICS = new Set(["eventF1", "timeIoU", "liveTimeRecall", "liveTimePrecision"]);
      const METRICS = [
        { key: "eventF1", label: "Event F1", definition: "event-level detection balance" },
        { key: "timeIoU", label: "Temporal IoU", definition: "predicted/live-time overlap" },
        { key: "liveTimePrecision", label: "Live-time precision", definition: "predicted live time that was live" },
        { key: "liveTimeRecall", label: "Live-time recall", definition: "true live time recovered" },
        { key: "predictedRallies", label: "Predicted rallies", definition: "aggregate predicted-event count" },
        { key: "trueRallies", label: "True rallies", definition: "aggregate ground-truth count" },
        { key: "matchedRallies", label: "Matched rallies", definition: "aggregate matched-event count" },
      ];
      const METRIC_MAP = Object.fromEntries(METRICS.map((item) => [item.key, item]));
      const state = { split: "test", metric: "eventF1", query: "", sort: "delta", direction: -1 };

      const $ = (id) => document.getElementById(id);
      const escapeHtml = (value) => String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
      const score = (value) => Number(value).toFixed(3);
      const signed = (value, decimals = 3) => {
        const number = Number(value);
        if (Math.abs(number) < 1e-12) return decimals === 0 ? "0" : (0).toFixed(decimals);
        return `${number > 0 ? "+" : "−"}${Math.abs(number).toFixed(decimals)}`;
      };
      const valueText = (key, value) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
        return SCORE_METRICS.has(key)
          ? score(value)
          : Number(value).toLocaleString("en-US", { maximumFractionDigits: 3 });
      };
      const deltaText = (key, value) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
        return SCORE_METRICS.has(key) ? `${signed(Number(value) * 100, 1)} pp` : signed(Number(value), 0);
      };
      const deltaClass = (value) => Math.abs(Number(value)) < 1e-12 ? "neutral" : (Number(value) > 0 ? "positive" : "negative");
      const splitMetrics = (model, split) => ({
        old: model.old[split],
        withoutBeach: model.withoutBeach[split],
        delta: model.deltaWithoutBeachMinusOld[split],
      });
      const median = (values) => {
        const sorted = [...values].sort((a, b) => a - b);
        if (!sorted.length) return 0;
        const middle = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
      };
      const stats = (split, key) => {
        const values = REPORT.models
          .map((model) => Number(model.deltaWithoutBeachMinusOld[split][key]))
          .filter((value) => Number.isFinite(value));
        return {
          mean: values.reduce((sum, value) => sum + value, 0) / values.length,
          median: median(values),
          improved: values.filter((value) => value > 1e-12).length,
          declined: values.filter((value) => value < -1e-12).length,
          unchanged: values.filter((value) => Math.abs(value) <= 1e-12).length,
        };
      };
      const deltaStatText = (key, value) => SCORE_METRICS.has(key) ? `${signed(value * 100, 1)} pp` : signed(value, 1);

      function renderHeader() {
        const summary = REPORT.summary;
        $("created-at").textContent = `Generated ${new Date(REPORT.createdAt).toLocaleString()}`;
        $("model-count").textContent = `${summary.modelCount} model versions`;
        $("scope-line").textContent = `${summary.inferenceArtifactCount}/${summary.expectedInferenceArtifactCount} fresh inference artifacts`;
        $("footer").textContent = `Source: ${REPORT.title} · schema ${REPORT.schemaVersion} · generated ${new Date(REPORT.createdAt).toLocaleString()}`;

        const testF1 = summary.testEventF1;
        const validationF1 = summary.validationEventF1;
        $("headline-cards").innerHTML = [
          ["Model versions", summary.modelCount, "Every established lineage version retrained"],
          ["Test Event F1 improved", `${testF1.improved}/${summary.modelCount}`, `${testF1.declined} declined · ${testF1.unchanged} unchanged`, "good"],
          ["Mean test Event F1 Δ", `${signed(testF1.meanDelta * 100, 1)} pp`, `Median ${signed(testF1.medianDelta * 100, 1)} pp`, "good"],
          ["Validation Event F1 Δ", `${signed(validationF1.meanDelta * 100, 1)} pp`, `${validationF1.declined} of ${summary.modelCount} declined`, "warn"],
        ].map(([label, value, detail, tone = ""]) => `<article class="surface card ${tone}"><div class="card-label">${label}</div><div class="card-value">${value}</div><div class="card-detail">${detail}</div></article>`).join("");

        $("takeaway").innerHTML = `<p><strong>Bottom line:</strong> removing beach videos looks promising on the paired held-out test: ${testF1.improved} of ${summary.modelCount} versions improved Event F1, with a mean change of ${signed(testF1.meanDelta * 100, 1)} percentage points. Validation moved the other way (${validationF1.declined} declines), so the signal should be treated as a retrospective generalization diagnostic rather than a final deployment decision.</p>`;
      }

      function renderHighlights() {
        const ranked = [...REPORT.models].sort((a, b) => b.deltaWithoutBeachMinusOld.test.eventF1 - a.deltaWithoutBeachMinusOld.test.eventF1);
        const render = (model) => {
          const value = model.deltaWithoutBeachMinusOld.test.eventF1;
          return `<div class="highlight-row"><span class="highlight-name">${escapeHtml(model.name)}</span><span class="highlight-value ${deltaClass(value)}">${deltaText("eventF1", value)}</span></div>`;
        };
        $("test-gains").innerHTML = ranked.slice(0, 5).map(render).join("");
        $("test-losses").innerHTML = ranked.slice(-5).reverse().map(render).join("");
      }

      function renderMetricSelect() {
        $("metric-select").innerHTML = METRICS.map((item) => `<option value="${item.key}">${item.label}</option>`).join("");
        $("metric-select").value = state.metric;
      }

      function renderScorecard() {
        $("metric-scorecard").innerHTML = METRICS.map((item) => {
          const test = stats("test", item.key);
          const validation = stats("validation", item.key);
          const testClass = deltaClass(test.mean);
          const validationClass = deltaClass(validation.mean);
          return `<tr>
            <td>${item.label}<div class="metric-definition">${item.definition}</div></td>
            <td class="numeric ${testClass}">${deltaStatText(item.key, test.mean)}</td>
            <td class="numeric ${deltaClass(test.median)}">${deltaStatText(item.key, test.median)}</td>
            <td class="numeric"><span class="positive">${test.improved}</span> / <span class="negative">${test.declined}</span> / <span class="neutral">${test.unchanged}</span></td>
            <td class="numeric ${validationClass}">${deltaStatText(item.key, validation.mean)}</td>
            <td class="numeric ${deltaClass(validation.median)}">${deltaStatText(item.key, validation.median)}</td>
            <td class="numeric"><span class="positive">${validation.improved}</span> / <span class="negative">${validation.declined}</span> / <span class="neutral">${validation.unchanged}</span></td>
          </tr>`;
        }).join("");
      }

      function renderModelTable() {
        const key = state.metric;
        const query = state.query.trim().toLowerCase();
        const rows = REPORT.models.filter((model) => !query || `${model.name} ${model.family}`.toLowerCase().includes(query));
        rows.sort((a, b) => {
          const aMetric = splitMetrics(a, state.split);
          const bMetric = splitMetrics(b, state.split);
          let left = state.sort === "name" ? a.name : state.sort === "family" ? a.family : aMetric[state.sort][key];
          let right = state.sort === "name" ? b.name : state.sort === "family" ? b.family : bMetric[state.sort][key];
          if (typeof left === "string") return state.direction * left.localeCompare(right);
          return state.direction * (Number(left) - Number(right));
        });
        $("model-table").innerHTML = rows.length ? rows.map((model) => {
          const values = splitMetrics(model, state.split);
          const delta = values.delta[key];
          return `<tr>
            <td class="model-cell"><div class="model-name">${escapeHtml(model.name)}</div><div class="model-family">${escapeHtml(model.predictionTask || model.family)}</div></td>
            <td>${escapeHtml(model.family)}</td>
            <td class="numeric">${valueText(key, values.old[key])}</td>
            <td class="numeric">${valueText(key, values.withoutBeach[key])}</td>
            <td class="numeric delta-cell ${deltaClass(delta)}">${deltaText(key, delta)}</td>
          </tr>`;
        }).join("") : `<tr><td class="empty" colspan="5">No models match this filter.</td></tr>`;
        const metricLabel = METRIC_MAP[key].label;
        $("model-table-note").textContent = `${rows.length} of ${REPORT.models.length} models shown · ${metricLabel} · ${state.split === "test" ? "held-out test" : "validation / tuning"}. Click a column heading to sort.`;
      }

      function renderInferenceModelSelect() {
        const select = $("inference-model-select");
        select.innerHTML = REPORT.inferenceCoverage.map((coverage) => `<option value="${escapeHtml(coverage.model)}">${escapeHtml(coverage.model)}</option>`).join("");
        select.value = REPORT.inferenceCoverage[0]?.model || "";
      }

      function renderInference() {
        const modelName = $("inference-model-select").value;
        const coverage = REPORT.inferenceCoverage.find((item) => item.model === modelName);
        if (!coverage) {
          $("inference-table").innerHTML = `<tr><td class="empty" colspan="6">No inference coverage found.</td></tr>`;
          return;
        }
        $("inference-table").innerHTML = coverage.rows.map((row) => {
          const outOfDomain = Boolean(row.outOfDomainBeachInference);
          return `<tr>
            <td class="mono">${escapeHtml(row.recordingId)}</td>
            <td>${escapeHtml(row.environment)}</td>
            <td>${escapeHtml(row.split)}</td>
            <td class="numeric">${Number(row.predictedRallies).toLocaleString("en-US")}</td>
            <td><span class="badge ${outOfDomain ? "ood" : "good"}">${outOfDomain ? "qualitative / OOD" : "in-domain"}</span></td>
            <td><span class="path" title="${escapeHtml(row.analysisPath)}">${escapeHtml(row.analysisPath)}</span></td>
          </tr>`;
        }).join("");
        const beachCount = coverage.rows.filter((row) => row.outOfDomainBeachInference).length;
        $("inference-note").textContent = `${coverage.recordings}/${coverage.expectedRecordings} recordings · ${beachCount} beach recording(s) shown for qualitative inspection · ${coverage.complete ? "complete" : "incomplete"} artifact set.`;
      }

      function renderCoverage() {
        $("coverage-table").innerHTML = REPORT.inferenceCoverage.map((coverage) => `<tr>
          <td class="model-name">${escapeHtml(coverage.model)}</td>
          <td class="numeric">${coverage.recordings}/${coverage.expectedRecordings} ${coverage.complete ? "✓" : "!"}</td>
          <td class="numeric">${Number(coverage.beachPredictedRallies).toLocaleString("en-US")}</td>
          <td class="numeric">${Number(coverage.nonBeachPredictedRallies).toLocaleString("en-US")}</td>
        </tr>`).join("");
      }

      function renderProtocol() {
        const labels = {
          validation: "Validation",
          test: "Test",
          comparison: "Comparison rule",
          inference: "Inference protocol",
          boundaryHeadFoldChange: "Boundary-head selector",
        };
        $("protocol").innerHTML = Object.entries(REPORT.protocol).map(([key, value]) => `<dt>${labels[key] || key}</dt><dd>${escapeHtml(value)}</dd>`).join("");
      }

      $("split-select").addEventListener("change", (event) => { state.split = event.target.value; renderModelTable(); });
      $("metric-select").addEventListener("change", (event) => { state.metric = event.target.value; renderModelTable(); });
      $("model-filter").addEventListener("input", (event) => { state.query = event.target.value; renderModelTable(); });
      $("inference-model-select").addEventListener("change", renderInference);
      document.querySelectorAll("th button[data-sort]").forEach((button) => button.addEventListener("click", () => {
        const next = button.dataset.sort;
        if (state.sort === next) state.direction *= -1;
        else { state.sort = next; state.direction = next === "name" || next === "family" ? 1 : -1; }
        renderModelTable();
      }));

      renderHeader();
      renderHighlights();
      renderMetricSelect();
      renderScorecard();
      renderModelTable();
      renderInferenceModelSelect();
      renderInference();
      renderCoverage();
      renderProtocol();
    })();
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text())
    # Prevent an input value from prematurely closing the data script element.
    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML_TEMPLATE.replace("__REPORT_JSON__", report_json))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
