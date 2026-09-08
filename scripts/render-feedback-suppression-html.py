#!/usr/bin/env python3
"""Render the feedback-suppression experiment as a self-contained HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16/report.json"
)
DEFAULT_OUTPUT = Path(
    "/home/developer/server/docker/caddy/html/reports/"
    "volleycut-feedback-suppression-v3-2026-08-16.html"
)


def render(report: dict[str, object]) -> str:
    # Per-recording duration numerators are preserved in report.json but are not
    # needed by this interactive aggregate report. Removing them keeps the public
    # page small and makes first render fast on mobile devices.
    public_report = json.loads(json.dumps(report, allow_nan=False))

    def strip_per_recording(value: object) -> None:
        if isinstance(value, dict):
            value.pop("perRecording", None)
            for child in value.values():
                strip_per_recording(child)
        elif isinstance(value, list):
            for child in value:
                strip_per_recording(child)

    strip_per_recording(public_report)
    embedded = json.dumps(public_report, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    title = "VolleySplice · Feedback Suppression V3"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #07111f; --panel: #0d1b2e; --panel-2: #11243b; --text: #e8f0f8;
      --muted: #91a5bb; --line: #223952; --cyan: #51d7e8; --green: #67e8a5;
      --amber: #f5c66b; --red: #ff7b8b; --blue: #7da7ff; --shadow: 0 18px 50px #0007;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: radial-gradient(circle at 15% -10%, #173d58 0, transparent 32rem), var(--bg); color: var(--text); font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color: var(--cyan); }}
    .shell {{ width: min(1440px, calc(100% - 32px)); margin: 0 auto; }}
    header {{ padding: 64px 0 32px; }}
    .eyebrow {{ color: var(--cyan); text-transform: uppercase; letter-spacing: .14em; font-weight: 750; font-size: 12px; }}
    h1 {{ margin: 10px 0 14px; font-size: clamp(36px, 6vw, 72px); line-height: .98; max-width: 980px; letter-spacing: -.045em; }}
    h2 {{ font-size: 27px; margin: 0 0 18px; letter-spacing: -.025em; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    .lede {{ color: #bdd0e1; font-size: clamp(17px, 2vw, 21px); max-width: 920px; }}
    .tag-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }}
    .tag {{ border: 1px solid var(--line); background: #0b192a99; color: #bdd0e1; border-radius: 999px; padding: 7px 11px; font-size: 12px; }}
    nav {{ position: sticky; top: 0; z-index: 5; background: #07111fe8; backdrop-filter: blur(14px); border-block: 1px solid var(--line); }}
    nav .shell {{ display: flex; gap: 22px; overflow: auto; padding-block: 12px; }}
    nav a {{ color: var(--muted); text-decoration: none; white-space: nowrap; font-weight: 650; font-size: 13px; }}
    nav a:hover {{ color: var(--text); }}
    main {{ padding: 34px 0 80px; }}
    section {{ margin: 0 0 28px; scroll-margin-top: 72px; }}
    .panel {{ background: linear-gradient(145deg, #10233a, #0b1829 72%); border: 1px solid var(--line); border-radius: 18px; padding: clamp(18px, 3vw, 30px); box-shadow: var(--shadow); }}
    .verdict {{ border-color: #2d6a62; background: linear-gradient(135deg, #123e3a, #0d1b2e 66%); }}
    .verdict strong {{ color: var(--green); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .card {{ background: #091727aa; border: 1px solid var(--line); border-radius: 14px; padding: 17px; }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .card .value {{ font-size: clamp(25px, 3vw, 38px); font-weight: 780; margin-top: 4px; letter-spacing: -.035em; }}
    .positive {{ color: var(--green); }} .warning {{ color: var(--amber); }} .negative {{ color: var(--red); }}
    .subtle {{ color: var(--muted); }}
    .controls {{ display: flex; gap: 14px; align-items: end; flex-wrap: wrap; margin-bottom: 18px; }}
    label {{ display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }}
    select {{ color: var(--text); background: #081524; border: 1px solid #34506e; border-radius: 9px; padding: 10px 34px 10px 11px; font: inherit; min-width: min(440px, 88vw); }}
    input[type="search"] {{ color: var(--text); background: #081524; border: 1px solid #34506e; border-radius: 9px; padding: 10px 11px; font: inherit; min-width: min(360px, 88vw); }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; font-variant-numeric: tabular-nums; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #102239; color: #a9bdd1; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
    th[data-tip] {{ cursor: help; text-decoration: underline dotted #6f8eaa; text-underline-offset: 4px; }}
    th[data-tip]::after {{
      content: attr(data-tip); position: fixed; z-index: 20; left: 50%; bottom: 22px;
      width: min(560px, calc(100vw - 28px)); transform: translate(-50%, 8px);
      padding: 12px 14px; border: 1px solid #496784; border-radius: 10px;
      background: #06111eef; color: #e8f0f8; box-shadow: 0 12px 38px #000b;
      font-size: 13px; font-weight: 500; line-height: 1.45; letter-spacing: 0;
      text-align: left; text-transform: none; white-space: normal; pointer-events: none;
      opacity: 0; transition: opacity .14s ease, transform .14s ease;
    }}
    th[data-tip]:hover::after, th[data-tip]:focus::after {{ opacity: 1; transform: translate(-50%, 0); }}
    th[data-tip]:focus {{ outline: 2px solid var(--cyan); outline-offset: -2px; }}
    th:first-child, td:first-child {{ text-align: left; white-space: normal; min-width: 250px; }}
    tbody tr:hover {{ background: #18304b66; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    tr.recommended {{ background: #173d3766; }}
    tr.recommended td:first-child {{ color: var(--green); font-weight: 750; }}
    .callout {{ border-left: 4px solid var(--amber); padding: 3px 0 3px 17px; color: #d8caa9; }}
    .review-cta {{ display: inline-flex; align-items: center; gap: 10px; margin-top: 8px; padding: 12px 16px; border: 1px solid var(--cyan); border-radius: 10px; color: #06111e; background: var(--cyan); font-weight: 800; text-decoration: none; }}
    .review-cta:hover {{ background: #8cebf5; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    ul {{ padding-left: 21px; }} li + li {{ margin-top: 7px; }}
    code {{ color: #b8cffd; background: #091421; padding: 2px 5px; border-radius: 5px; }}
    footer {{ border-top: 1px solid var(--line); color: var(--muted); padding: 26px 0 50px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} .two-col {{ grid-template-columns: 1fr; }} header {{ padding-top: 42px; }} }}
    @media (max-width: 520px) {{ .shell {{ width: min(100% - 20px, 1440px); }} .grid {{ grid-template-columns: 1fr; }} .panel {{ border-radius: 13px; }} }}
  </style>
</head>
<body>
  <header class="shell">
    <div class="eyebrow">VolleySplice model research · 16 August 2026</div>
    <h1>Suppressing false-positive setup and transition footage</h1>
    <p class="lede">A controlled comparison of a feedback-augmented three-head v3, a four-head v3 with a suppression veto, and suppression applied to the current production ensemble.</p>
    <div class="tag-row">
      <span class="tag">30 videos inferred</span><span class="tag">28 videos scored</span>
      <span class="tag">11 feedback exports</span><span class="tag">2-second target padding</span>
      <span class="tag">gaps &lt; 3 seconds joined</span>
    </div>
  </header>
  <nav><div class="shell"><a href="#outcome">Outcome</a><a href="#recall-audit">Recall audit</a><a href="#correct-suppression">Correct suppressions</a><a href="#component-veto">One-component veto</a><a href="#visual-review">Visual review</a><a href="#grass-audit">Grass losses</a><a href="#missed-rallies">Missed rallies</a><a href="#held">Held feedback</a><a href="#comparison">Comparison</a><a href="#sensitivity">Padding sensitivity</a><a href="#design">Design</a><a href="#artifacts">Artifacts</a></div></nav>
  <main class="shell">
    <section id="outcome" class="panel verdict">
      <div class="eyebrow">Recall-first recommendation</div>
      <h2>Keep current production unchanged. Do not deploy this suppression veto.</h2>
      <p>The aggregate precision gain hides expensive errors: the selected suppression decoder completely removes reviewed rallies. <strong>No tested threshold from 0.75 through 0.999 improved precision while preserving both unpadded Core Recall and canonical R_core on development.</strong></p>
      <p class="callout">With only the two components’ binary intervals, protecting every interval emitted by either component leaves nothing to veto. A useful guard must instead use stronger evidence—raw rally confidence, serve/rally anchors, and connected-rally continuity—to trim only weakly supported fringe while prohibiting deletion of an entire rally. Threshold adjustment alone is not sufficient.</p>
    </section>

    <section id="recall-audit">
      <h2>Recall-safety audit</h2>
      <div class="grid">
        <div class="card"><div class="label">Safe thresholds found</div><div class="value negative">0</div><div class="subtle">tested 0.75–0.999</div></div>
        <div class="card"><div class="label">Fully missed rallies</div><div class="value negative" id="full-miss-count">53</div><div class="subtle">after padding and gap joining</div></div>
        <div class="card"><div class="label">Inferred by both models</div><div class="value warning" id="both-model-count">44</div><div class="subtle">then removed by suppression</div></div>
        <div class="card"><div class="label">Full misses rescued by join</div><div class="value negative" id="join-rescue-count">0</div><div class="subtle">join helped only partial misses</div></div>
      </div>
      <div class="panel" style="margin-top:14px">
        <h3>Threshold sweep</h3>
        <p class="subtle">Every row freezes smoothing at 1 second, minimum suppression duration at 0.5 seconds, bridge gap at 0.5 seconds, and exit threshold at enter threshold minus 0.1. Selection labels were development-only.</p>
        <div class="table-wrap"><table>
          <thead><tr>
            <th tabindex="0" data-tip="Suppression enter threshold. Higher values veto only samples assigned greater suppression probability.">Threshold</th>
            <th tabindex="0" data-tip="P_pad on the development selection scope at 2-second symmetric padding.">Dev P_pad</th>
            <th tabindex="0" data-tip="Canonical R_core on development. The production baseline is 0.9987; a safe threshold must not reduce it.">Dev R_core</th>
            <th tabindex="0" data-tip="Unpadded Core Recall on development. The production baseline is 0.9381; a safe threshold must not reduce it.">Dev Core R</th>
            <th tabindex="0" data-tip="P_pad on the six held-feedback files. These labels were not used to choose a threshold.">Held P_pad</th>
            <th tabindex="0" data-tip="R_core on the six held-feedback files. Production is 1.0000 on this scope.">Held R_core</th>
            <th tabindex="0" data-tip="R_core on the protected-test recording, opened only for retrospective evaluation.">Test R_core</th>
          </tr></thead>
          <tbody id="threshold-table"></tbody>
        </table></div>
      </div>
    </section>

    <section id="correct-suppression">
      <h2>Which production model caused the correct suppressions?</h2>
      <p>These are false-positive production regions correctly removed by the specialist—not rallies. The primary view is conservative: removed time must lie outside the human export after 2-second padding and canonical short-gap joining.</p>
      <div class="grid">
        <div class="card"><div class="label">Strictly safe time removed</div><div class="value positive">1783.5s</div><div class="subtle">28 labeled recordings</div></div>
        <div class="card"><div class="label">Previous model only</div><div class="value warning">59.2%</div><div class="subtle">1056.0 seconds</div></div>
        <div class="card"><div class="label">Both models</div><div class="value">24.9%</div><div class="subtle">444.5 seconds</div></div>
        <div class="card"><div class="label">All-labels v2 only</div><div class="value">15.9%</div><div class="subtle">283.0 seconds</div></div>
      </div>
      <div class="panel" style="margin-top:14px">
        <p class="subtle">Duration attribution is pointwise, so every removed second belongs to exactly one component category. “Unknown” is export feedback whose schema does not encode environment.</p>
        <div class="table-wrap"><table>
          <thead><tr>
            <th tabindex="0" data-tip="Evaluation environment. All is pooled across all 28 recordings with reviewed labels.">Scope</th>
            <th tabindex="0" data-tip="Correctly removed seconds that were present in both production components before suppression, followed by their share of this scope's total.">Both models</th>
            <th tabindex="0" data-tip="Correctly removed seconds produced only by the previous-production model, followed by their share of this scope's total.">Previous only</th>
            <th tabindex="0" data-tip="Correctly removed seconds produced only by the all-labels-v2 model, followed by their share of this scope's total.">V2 only</th>
            <th tabindex="0" data-tip="Total production-positive seconds removed outside the padded human export target in this scope.">Total</th>
          </tr></thead>
          <tbody id="correct-suppression-table"></tbody>
        </table></div>
        <p class="subtle">Under the looser raw-core definition, 2370.1 non-rally seconds were removed: previous-only 1267.8s (53.5%), both 656.5s (27.7%), and v2-only 445.8s (18.8%).</p>
      </div>
    </section>

    <section id="component-veto" class="panel">
      <h2>What if suppression is applied to only one component?</h2>
      <h3>Suppress previous production only</h3>
      <p>This composition is <code>(previous production − suppression) ∪ unchanged all-labels v2</code>. It is substantially safer than vetoing the full union, but it is not recall-neutral.</p>
      <div class="grid">
        <div class="card"><div class="label">Completely lost rallies</div><div class="value negative">4</div><div class="subtle">after padding and gap joining</div></div>
        <div class="card"><div class="label">Partially affected rallies</div><div class="value warning">16</div><div class="subtle">20 affected in total</div></div>
        <div class="card"><div class="label">Additional core missed</div><div class="value warning">31.3s</div><div class="subtle">15.3 seconds on grass</div></div>
        <div class="card"><div class="label">All-video P_pad / R_core</div><div class="value" style="font-size:24px">.7507 / .9939</div><div class="subtle">production: .7008 / .9971</div></div>
      </div>
      <p class="subtle" style="margin-top:18px">The list includes every rally with additional missed core versus current production. Complete misses are shown first. Symmetric 2-second padding, ignored-range subtraction, and the strictly-less-than-3-second join are already applied.</p>
      <div class="table-wrap"><table>
        <thead><tr>
          <th tabindex="0" data-tip="Source video containing the reviewed rally whose coverage is reduced by old-only suppression.">File</th>
          <th tabindex="0" data-tip="Training, evaluation/validation/test-only, or export-feedback provenance of the source video.">Scope</th>
          <th tabindex="0" data-tip="One-based rally index in the reviewed label document or feedback correction list.">Rally</th>
          <th tabindex="0" data-tip="Reviewed human rally-core start and end in seconds from the source video start.">Core range</th>
          <th tabindex="0" data-tip="Evaluable rally-core duration after ignored intervals are removed.">Duration</th>
          <th tabindex="0" data-tip="Core seconds covered by current production after 2-second padding and canonical gap joining.">Production covered</th>
          <th tabindex="0" data-tip="Core seconds still covered when suppression is applied only to previous production and v2 remains unchanged.">Old-only veto covered</th>
          <th tabindex="0" data-tip="Additional core seconds missed by old-only suppression compared with current production.">Coverage lost</th>
          <th tabindex="0" data-tip="Complete means no core remains covered after padding and joining; partial means some coverage remains.">Outcome</th>
        </tr></thead>
        <tbody id="old-only-veto-table"></tbody>
      </table></div>
      <h3 style="margin-top:28px">Suppress all-labels v2 only</h3>
      <p>This mirror composition is <code>unchanged previous production ∪ (all-labels v2 − suppression)</code>. It affects fewer rallies and has no complete grass miss, but retains substantially more false-positive output.</p>
      <div class="grid">
        <div class="card"><div class="label">Completely lost rallies</div><div class="value negative">3</div><div class="subtle">one beach · two feedback</div></div>
        <div class="card"><div class="label">Partially affected rallies</div><div class="value warning">8</div><div class="subtle">11 affected in total</div></div>
        <div class="card"><div class="label">Additional core missed</div><div class="value warning">29.3s</div><div class="subtle">14.8 seconds on grass</div></div>
        <div class="card"><div class="label">All-video P_pad / R_core</div><div class="value" style="font-size:24px">.7156 / .9941</div><div class="subtle">production: .7008 / .9971</div></div>
      </div>
      <p class="subtle" style="margin-top:18px">The same evaluation contract is applied. All 11 affected rallies are listed below, with complete misses first.</p>
      <div class="table-wrap"><table>
        <thead><tr>
          <th tabindex="0" data-tip="Source video containing the reviewed rally whose coverage is reduced by v2-only suppression.">File</th>
          <th tabindex="0" data-tip="Training, evaluation/validation/test-only, or export-feedback provenance of the source video.">Scope</th>
          <th tabindex="0" data-tip="One-based rally index in the reviewed label document or feedback correction list.">Rally</th>
          <th tabindex="0" data-tip="Reviewed human rally-core start and end in seconds from the source video start.">Core range</th>
          <th tabindex="0" data-tip="Evaluable rally-core duration after ignored intervals are removed.">Duration</th>
          <th tabindex="0" data-tip="Core seconds covered by current production after 2-second padding and canonical gap joining.">Production covered</th>
          <th tabindex="0" data-tip="Core seconds still covered when suppression is applied only to all-labels v2 and previous production remains unchanged.">V2-only veto covered</th>
          <th tabindex="0" data-tip="Additional core seconds missed by v2-only suppression compared with current production.">Coverage lost</th>
          <th tabindex="0" data-tip="Complete means no core remains covered after padding and joining; partial means some coverage remains.">Outcome</th>
        </tr></thead>
        <tbody id="v2-only-veto-table"></tbody>
      </table></div>
    </section>

    <section id="visual-review" class="panel">
      <div class="eyebrow">Interactive missed-rally inspection</div>
      <h2>Compare pointwise overlap with full-span overlap protection</h2>
      <p>Both policies apply the specialist veto only to one-model-only production output. The pointwise policy protects only the exact seconds where previous production and all-labels v2 overlap. The full-span policy protects the entire overlap-connected union, including non-overlapping leading heads and trailing tails: for example, previous <code>15–20</code> plus v2 <code>10–18</code> protects <code>10–20</code>, including the <code>10–15</code> head and <code>18–20</code> tail.</p>
      <h3>Pointwise overlap</h3>
      <div class="grid">
        <div class="card"><div class="label">Export time saved</div><div class="value positive">2153.9s</div><div class="subtle">20099.4s → 17945.5s</div></div>
        <div class="card"><div class="label">Correct FP predictions removed</div><div class="value">135</div><div class="subtle">fully deleted · outside padded human</div></div>
        <div class="card"><div class="label">Core P / R / F1</div><div class="value" style="font-size:22px">.7650 / .9375 / .8426</div><div class="subtle">from .6783 / .9746 / .7999</div></div>
        <div class="card"><div class="label">Padded P / R / F1</div><div class="value" style="font-size:22px">.7691 / .9690 / .8576</div><div class="subtle">from .7008 / .9889 / .8203</div></div>
      </div>
      <p class="subtle">16 videos · 31 affected rallies · 7 complete misses · 24 partial misses · 60.6 seconds of core coverage lost.</p>
      <h3 style="margin-top:24px">Any overlap protects the full connected span</h3>
      <div class="grid">
        <div class="card"><div class="label">Export time saved</div><div class="value positive">1162.5s</div><div class="subtle">20099.4s → 18936.9s</div></div>
        <div class="card"><div class="label">Correct FP predictions removed</div><div class="value">135</div><div class="subtle">same complete false-positive deletions</div></div>
        <div class="card"><div class="label">Core P / R / F1</div><div class="value" style="font-size:22px">.7089 / .9714 / .8197</div><div class="subtle">from .6783 / .9746 / .7999</div></div>
        <div class="card"><div class="label">Padded P / R / F1</div><div class="value" style="font-size:22px">.7406 / .9846 / .8453</div><div class="subtle">from .7008 / .9889 / .8203</div></div>
      </div>
      <p class="subtle">7 videos · 10 affected rallies · 5 complete misses · 5 partial misses · 23.9 seconds of core coverage lost.</p>
      <p>The visual review page includes source video playback, rally seeking, raw/padded/joined interval layers, an explicit veto rail, and muted textured ranges outside each affected context. Raw previous-production, all-labels-v2, and suppression intervals show their uncalibrated decoder scores; derived union and padding ranges are explicitly marked as having no standalone confidence.</p>
      <a class="review-cta" href="https://internal.example/suppression-review">Open pointwise review →</a>
      <a class="review-cta" href="https://internal.example/suppression-review/any-overlap">Open full-span overlap review →</a>
    </section>

    <section id="grass-audit">
      <h2>Grass recall-loss attribution</h2>
      <p>At 2-second product padding, suppression lowers pooled grass <code>R_core</code> from 0.9959 to 0.9000, removing 191.5 seconds of rally-core coverage. The affected intervals were already present in the production ensemble; the columns below identify which component produced them before the veto.</p>
      <div class="grid">
        <div class="card"><div class="label">Affected grass rallies</div><div class="value negative">72</div><div class="subtle">32 complete · 40 partial misses</div></div>
        <div class="card"><div class="label">Produced by both models</div><div class="value warning">66</div><div class="subtle">91.7% of affected rallies</div></div>
        <div class="card"><div class="label">Both-model recall loss</div><div class="value warning">175.8s</div><div class="subtle">91.8% of lost core coverage</div></div>
        <div class="card"><div class="label">One-model-only loss</div><div class="value">15.7s</div><div class="subtle">old 10.7s · v2 5.0s</div></div>
      </div>
      <div class="panel" style="margin-top:14px">
        <p class="subtle">Only files with suppression-attributable grass recall loss are shown. Counts refer to reviewed rallies with newly uncovered core time after padding and canonical gap joining.</p>
        <div class="table-wrap"><table>
          <thead><tr>
            <th tabindex="0" data-tip="Grass recording whose reviewed rally-core coverage was reduced by suppression.">File</th>
            <th tabindex="0" data-tip="Whether the recording was part of the all-labels-v2 training data or reserved for evaluation, validation, or test.">Scope</th>
            <th tabindex="0" data-tip="R_core from the current production ensemble before applying suppression, shown as a percentage for this file.">Production R_core</th>
            <th tabindex="0" data-tip="R_core after the suppression specialist veto is applied, shown as a percentage for this file.">With suppression</th>
            <th tabindex="0" data-tip="Rally-core seconds covered by production padding but no longer covered after suppression.">Core coverage lost</th>
            <th tabindex="0" data-tip="Reviewed rallies in this file with additional uncovered core time caused by suppression.">Affected rallies</th>
            <th tabindex="0" data-tip="Affected rallies left with zero core coverage after 2-second padding and the strictly-less-than-3-second gap join.">Complete misses</th>
            <th tabindex="0" data-tip="Affected rallies that had core overlap from both production components before suppression.">Both models</th>
            <th tabindex="0" data-tip="Affected rallies that had core overlap only from the previous-production model before suppression.">Previous only</th>
            <th tabindex="0" data-tip="Affected rallies that had core overlap only from the all-labels-v2 model before suppression.">V2 only</th>
          </tr></thead>
          <tbody id="grass-audit-table"></tbody>
        </table></div>
      </div>
    </section>

    <section id="missed-rallies" class="panel">
      <h2>Rallies missed by current production + suppression</h2>
      <p>The default view lists complete misses after symmetric 2-second padding, clipping, merging, ignored-range subtraction, and the production rule that joins positive gaps strictly below 3 seconds. Switch the filter to inspect partial misses too.</p>
      <div class="controls">
        <label>Miss type<select id="miss-filter"><option value="full">Fully missed</option><option value="partial">Partially missed</option><option value="all">Any missed core</option></select></label>
        <label>Find file<input id="miss-search" type="search" placeholder="Filter by filename or recording ID"></label>
      </div>
      <p class="subtle" id="miss-count-note"></p>
      <div class="table-wrap"><table>
        <thead><tr>
          <th tabindex="0" data-tip="Source video filename containing the reviewed rally.">File</th>
          <th tabindex="0" data-tip="Whether this video belongs to all-labels-v2 training, evaluation/validation/test-only, or export feedback.">Scope</th>
          <th tabindex="0" data-tip="One-based rally index in the reviewed label document or feedback correction list.">Rally</th>
          <th tabindex="0" data-tip="Reviewed human core interval in seconds from the start of the source video.">Core range</th>
          <th tabindex="0" data-tip="Evaluable human rally-core duration after ignored intervals are removed.">Duration</th>
          <th tabindex="0" data-tip="Which production component had any unpadded core overlap before suppression: previous model, all-labels v2, both, or neither.">Production inference</th>
          <th tabindex="0" data-tip="Percent of this rally core overlapped by the previous-production three-head model before suppression.">Old coverage</th>
          <th tabindex="0" data-tip="Percent of this rally core overlapped by the all-labels-v2 three-head model before suppression.">V2 coverage</th>
          <th tabindex="0" data-tip="Core seconds covered after 2-second padding if positive gaps are not joined.">Before join</th>
          <th tabindex="0" data-tip="Core seconds covered after applying the production rule that joins positive padded gaps strictly below 3 seconds.">After &lt;3s join</th>
          <th tabindex="0" data-tip="Additional rally-core seconds recovered specifically by short-gap joining.">Join added</th>
        </tr></thead>
        <tbody id="miss-table"></tbody>
      </table></div>
    </section>

    <section id="held">
      <h2>Held-feedback result</h2>
      <div class="grid">
        <div class="card"><div class="label">F1_padP_coreR</div><div class="value positive">0.9267</div><div class="subtle">+0.0302 vs production</div></div>
        <div class="card"><div class="label">Padded precision</div><div class="value positive">0.8686</div><div class="subtle">+0.0562</div></div>
        <div class="card"><div class="label">R_core</div><div class="value warning">0.9932</div><div class="subtle">−0.0068 vs production</div></div>
        <div class="card"><div class="label">Export removed</div><div class="value positive">375.9s</div><div class="subtle">six unseen feedback files</div></div>
      </div>
      <div class="panel" style="margin-top:14px">
        <p class="subtle">Hover or focus any column heading for its definition.</p>
        <div class="table-wrap"><table>
          <thead><tr>
            <th tabindex="0" data-tip="The model or model composition evaluated on the six feedback files excluded from fitting and decoder selection.">Held feedback</th>
            <th tabindex="0" data-tip="Core precision: unpadded model–human overlap duration divided by unpadded predicted duration. Higher means fewer extra selected seconds.">Core P</th>
            <th tabindex="0" data-tip="Core recall: unpadded model–human overlap duration divided by unpadded human rally duration. Higher means less true rally time was missed.">Core R</th>
            <th tabindex="0" data-tip="The harmonic mean of unpadded core precision and unpadded core recall.">Core F1</th>
            <th tabindex="0" data-tip="Padded precision: overlap between equally padded model and human exports divided by padded model export duration.">Padded P</th>
            <th tabindex="0" data-tip="Padded recall: overlap between equally padded model and human exports divided by padded human export duration.">Padded R</th>
            <th tabindex="0" data-tip="The harmonic mean of padded precision and padded recall after symmetric 2-second padding.">Padded F1</th>
            <th tabindex="0" data-tip="Primary ranking metric: harmonic mean of padded-model precision (P_pad) and recall of unpadded human core time by the padded model (R_core).">F1_padP_coreR</th>
          </tr></thead>
          <tbody id="held-table"></tbody>
        </table></div>
      </div>
    </section>

    <section id="comparison" class="panel">
      <h2>Target-padding comparison</h2>
      <div class="controls"><label>Evaluation scope<select id="scope-select"></select></label></div>
      <p class="subtle">Hover or focus any column heading for its definition.</p>
      <div class="table-wrap"><table>
        <thead><tr>
          <th tabindex="0" data-tip="The model or ensemble composition whose predicted rally intervals are being evaluated.">Variant</th>
          <th tabindex="0" data-tip="Core precision: unpadded model–human overlap duration divided by unpadded predicted duration.">Core P</th>
          <th tabindex="0" data-tip="Core recall: unpadded model–human overlap duration divided by unpadded human rally duration.">Core R</th>
          <th tabindex="0" data-tip="The harmonic mean of unpadded core precision and unpadded core recall.">Core F1</th>
          <th tabindex="0" data-tip="Padded precision after applying the selected symmetric padding to both model and human ranges. At the target setting this equals P_pad.">Padded P</th>
          <th tabindex="0" data-tip="Padded recall: equally padded overlap duration divided by padded human export duration.">Padded R</th>
          <th tabindex="0" data-tip="The harmonic mean of padded precision and padded recall.">Padded F1</th>
          <th tabindex="0" data-tip="Canonical padded precision: duration(padded model ∩ padded human) divided by duration(padded model).">P_pad</th>
          <th tabindex="0" data-tip="Canonical core recall: duration(padded model ∩ unpadded human core) divided by duration(unpadded human core).">R_core</th>
          <th tabindex="0" data-tip="Primary ranking metric: the harmonic mean of P_pad and R_core.">F1_padP_coreR</th>
          <th tabindex="0" data-tip="Total padded model export duration in seconds after clipping, merging, short-gap joining, and ignored-interval subtraction.">Export s</th>
          <th tabindex="0" data-tip="Difference in padded export duration versus the current production ensemble on the same scope. Negative means fewer seconds exported.">Δ export s</th>
        </tr></thead>
        <tbody id="comparison-table"></tbody>
      </table></div>
      <p class="subtle" id="scope-note"></p>
    </section>

    <section id="sensitivity" class="panel">
      <h2>Required padding sensitivity</h2>
      <div class="controls"><label>Evaluation scope<select id="padding-scope-select"></select></label></div>
      <p class="subtle">Hover or focus any column heading for its definition.</p>
      <div class="table-wrap"><table>
        <thead><tr>
          <th tabindex="0" data-tip="The model or ensemble composition evaluated at each required padding setting.">Variant</th>
          <th tabindex="0" data-tip="Symmetric seconds added before and after every model and human interval. The required cases are 0, 1, 2, and 3 seconds.">Padding</th>
          <th tabindex="0" data-tip="Padded precision: duration(padded model ∩ padded human) divided by duration(padded model).">P_pad</th>
          <th tabindex="0" data-tip="Core recall: duration(padded model ∩ unpadded human core) divided by duration(unpadded human core).">R_core</th>
          <th tabindex="0" data-tip="Primary ranking metric: the harmonic mean of P_pad and R_core for this padding case.">F1_padP_coreR</th>
          <th tabindex="0" data-tip="Total model export duration in seconds after the listed padding, clipping, merging, short-gap joining, and ignored-interval subtraction.">Model export s</th>
          <th tabindex="0" data-tip="Total human-label export duration in seconds after identical padding and interval processing.">Human export s</th>
          <th tabindex="0" data-tip="Model export duration minus human export duration. Positive means the model exports extra time; negative means it exports less.">Difference s</th>
        </tr></thead>
        <tbody id="padding-table"></tbody>
      </table></div>
    </section>

    <section id="design" class="two-col">
      <div class="panel">
        <h2>Experiment design</h2>
        <ul>
          <li>Candidate 1 refits rally, serve, and dead-state heads on all-labels v2 plus five feedback files.</li>
          <li>Candidate 2 adds a head whose positive class is selected non-rally/setup/transition activity.</li>
          <li>The same suppression head is also evaluated as a veto on current production.</li>
          <li>Feedback features were decoded from JSON; no feedback video features were regenerated.</li>
          <li>The 11 feedback files were split 5 fit / 6 held-out with PCG64 seed <code>20260816</code>.</li>
        </ul>
      </div>
      <div class="panel">
        <h2>Evaluation contract</h2>
        <ul>
          <li>Primary rank: <code>F1_padP_coreR</code>, pooled by duration.</li>
          <li>Declared product padding: 2 seconds before and after.</li>
          <li>Symmetric 0, 1, 2, and 3 second cases are all reported.</li>
          <li>Identical padding is applied to model and human ranges.</li>
          <li>Ignored spans are outside the evaluation universe and never rejoined across.</li>
          <li>Protected test and held-feedback labels were closed during fitting and selection.</li>
        </ul>
      </div>
    </section>

    <section id="artifacts" class="panel">
      <h2>Models and coverage</h2>
      <div class="grid">
        <div class="card"><div class="label">v3 candidate 1</div><div class="value" style="font-size:21px">model-dfbb67c7c0c2</div><div class="subtle">three-head refit</div></div>
        <div class="card"><div class="label">v3 candidate 2</div><div class="value" style="font-size:21px">model-cacb15849ba8</div><div class="subtle">four heads with suppression</div></div>
        <div class="card"><div class="label">Inference artifacts</div><div class="value">180</div><div class="subtle">6 variants × 30 videos</div></div>
        <div class="card"><div class="label">Unscored inference</div><div class="value">2</div><div class="subtle">grass-source-02 and indoor-source-02</div></div>
      </div>
      <p class="subtle" style="margin-top:18px">Feedback environment is shown as “unknown” because the feedback schema does not encode environment; it was not inferred from filenames or appearance.</p>
    </section>
  </main>
  <footer><div class="shell">Generated from the immutable experiment report · VolleySplice feedback-suppression-v3-2026-08-16</div></footer>
  <script id="report-data" type="application/json">{embedded}</script>
  <script>
    const report = JSON.parse(document.getElementById('report-data').textContent);
    const thresholdAudit = report.thresholdRecallAudit;
    const missedAudit = report.missedRalliesAudit;
    const correctAudit = report.correctSuppressionAttribution;
    const oldOnlyAudit = report.oldOnlySuppressionAudit;
    const v2OnlyAudit = report.v2OnlySuppressionAudit;
    const order = report.variantOrder;
    const labels = report.variantLabels;
    const scopeLabels = {{
      'all-evaluable': 'All evaluable videos',
      'provenance:training-dataset': 'Training dataset',
      'provenance:evaluation-validation-test-only': 'Evaluation + validation + test only',
      'provenance:export-feedback': 'All export feedback',
      'feedback-partition:fit': 'Feedback fit files',
      'feedback-partition:held-out': 'Feedback held-out files',
      'environment:grass': 'Grass', 'environment:indoor': 'Indoor',
      'environment:beach': 'Beach', 'environment:unknown': 'Feedback environment unknown'
    }};
    const fmt = value => Number(value).toFixed(4);
    const secs = value => Number(value).toFixed(1);
    const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[character]));
    function metric(scope, variant, padding='2') {{ return report.metrics[scope][variant].padding[padding]; }}
    function rowClass(variant) {{ return variant === 'production-plus-suppression' ? 'recommended' : ''; }}
    function fillSelect(select) {{
      report.displayScopes.forEach(scope => {{ const option = document.createElement('option'); option.value = scope; option.textContent = scopeLabels[scope] || scope; select.append(option); }});
    }}
    function renderHeld() {{
      const target = document.getElementById('held-table'); target.innerHTML = '';
      order.forEach(variant => {{ const x = metric('feedback-partition:held-out', variant); const tr = document.createElement('tr'); tr.className = rowClass(variant); tr.innerHTML = `<td>${{labels[variant]}}</td><td>${{fmt(x.core.precision)}}</td><td>${{fmt(x.core.recall)}}</td><td>${{fmt(x.core.f1)}}</td><td>${{fmt(x.padded.precision)}}</td><td>${{fmt(x.padded.recall)}}</td><td>${{fmt(x.padded.f1)}}</td><td>${{fmt(x.F1_padP_coreR)}}</td>`; target.append(tr); }});
    }}
    function renderComparison(scope) {{
      const target = document.getElementById('comparison-table'); target.innerHTML = ''; const baseline = metric(scope, 'current-production-ensemble');
      order.forEach(variant => {{ const x = metric(scope, variant); const tr = document.createElement('tr'); tr.className = rowClass(variant); const delta = x.paddedModelExportSeconds - baseline.paddedModelExportSeconds; tr.innerHTML = `<td>${{labels[variant]}}</td><td>${{fmt(x.core.precision)}}</td><td>${{fmt(x.core.recall)}}</td><td>${{fmt(x.core.f1)}}</td><td>${{fmt(x.padded.precision)}}</td><td>${{fmt(x.padded.recall)}}</td><td>${{fmt(x.padded.f1)}}</td><td>${{fmt(x.P_pad)}}</td><td>${{fmt(x.R_core)}}</td><td>${{fmt(x.F1_padP_coreR)}}</td><td>${{secs(x.paddedModelExportSeconds)}}</td><td class="${{delta < 0 ? 'positive' : delta > 0 ? 'warning' : ''}}">${{delta >= 0 ? '+' : ''}}${{secs(delta)}}</td>`; target.append(tr); }});
      document.getElementById('scope-note').textContent = `${{metric(scope, order[0]).recordings}} labeled video${{metric(scope, order[0]).recordings === 1 ? '' : 's'}} in this scope. Values are pooled, not averaged per video.`;
    }}
    function renderPadding(scope) {{
      const target = document.getElementById('padding-table'); target.innerHTML = '';
      order.forEach(variant => ['0','1','2','3'].forEach(padding => {{ const x = metric(scope, variant, padding); const tr = document.createElement('tr'); tr.className = rowClass(variant); tr.innerHTML = `<td>${{labels[variant]}}</td><td>${{padding}}s</td><td>${{fmt(x.P_pad)}}</td><td>${{fmt(x.R_core)}}</td><td>${{fmt(x.F1_padP_coreR)}}</td><td>${{secs(x.paddedModelExportSeconds)}}</td><td>${{secs(x.paddedHumanExportSeconds)}}</td><td>${{x.paddedDurationDifferenceSeconds >= 0 ? '+' : ''}}${{secs(x.paddedDurationDifferenceSeconds)}}</td>`; target.append(tr); }}));
    }}
    function renderThresholds() {{
      const target = document.getElementById('threshold-table'); target.innerHTML = '';
      thresholdAudit.sweep.forEach(row => {{
        const development = row.scopes.development.candidate;
        const held = row.scopes['feedback-held'].candidate;
        const test = row.scopes['protected-test'].candidate;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{row.threshold}}</td><td>${{fmt(development.P_pad)}}</td><td>${{fmt(development.R_core)}}</td><td>${{fmt(development.coreRecall)}}</td><td>${{fmt(held.P_pad)}}</td><td>${{fmt(held.R_core)}}</td><td>${{fmt(test.R_core)}}</td>`;
        target.append(tr);
      }});
    }}
    const componentLabels = {{'both-models':'Both models','previous-production-only':'Previous production only','all-labels-v2-only':'All-labels v2 only','neither-model-core-overlap':'Neither model'}};
    function renderMisses() {{
      const filter = document.getElementById('miss-filter').value;
      const query = document.getElementById('miss-search').value.trim().toLowerCase();
      const rows = missedAudit.rallies.filter(row => {{
        const typeMatches = filter === 'all' || (filter === 'full' ? row.fullyMissedAfterPadding : !row.fullyMissedAfterPadding);
        const textMatches = !query || row.file.toLowerCase().includes(query) || row.recordingId.toLowerCase().includes(query);
        return typeMatches && textMatches;
      }});
      const target = document.getElementById('miss-table'); target.innerHTML = '';
      rows.forEach(row => {{
        const tr = document.createElement('tr');
        const scope = row.provenance === 'export-feedback' ? `Export feedback (${{row.feedbackPartition}})` : row.provenance === 'training-dataset' ? 'Training dataset' : 'Evaluation / validation / test';
        tr.innerHTML = `<td>${{escapeHtml(row.file)}}</td><td>${{scope}}</td><td>${{row.rallyNumber}}</td><td>${{Number(row.coreStart).toFixed(3)}}–${{Number(row.coreEnd).toFixed(3)}}</td><td>${{secs(row.evaluableCoreDurationSeconds)}}s</td><td>${{componentLabels[row.productionComponentAttribution]}}</td><td>${{Number(row.previousProductionCoreCoveragePercent).toFixed(1)}}%</td><td>${{Number(row.allLabelsV2CoreCoveragePercent).toFixed(1)}}%</td><td>${{secs(row.paddedCoverageWithoutShortGapJoinSeconds)}}s</td><td>${{secs(row.paddedCoverageWithShortGapJoinSeconds)}}s</td><td>${{secs(row.coreSecondsAddedByShortGapJoin)}}s</td>`;
        target.append(tr);
      }});
      document.getElementById('miss-count-note').textContent = `${{rows.length}} rallies shown. All ignored time and feedback footage outside the reviewed game window is excluded.`;
    }}
    function renderGrassAudit() {{
      const grass = report.metrics['environment:grass'];
      const baselineRows = grass['current-production-ensemble'].padding['2'].perRecording;
      const suppressedRows = grass['production-plus-suppression'].padding['2'].perRecording;
      const rows = baselineRows.map(baseline => {{
        const suppressed = suppressedRows.find(row => row.id === baseline.id);
        const affected = missedAudit.rallies.filter(row => row.recordingId === baseline.id && row.suppressionIntroducedAnyMiss);
        const lost = baseline.paddedVsCoreIntersectionSeconds - suppressed.paddedVsCoreIntersectionSeconds;
        return {{baseline, suppressed, affected, lost}};
      }}).filter(row => row.lost > 0.0005).sort((a, b) => b.lost - a.lost);
      const target = document.getElementById('grass-audit-table'); target.innerHTML = '';
      rows.forEach(row => {{
        const first = row.affected[0];
        const scope = first && first.provenance === 'training-dataset' ? 'Training dataset' : 'Evaluation / validation / test';
        const count = attribution => row.affected.filter(rally => rally.productionComponentAttribution === attribution).length;
        const full = row.affected.filter(rally => rally.fullyMissedAfterPadding).length;
        const baselineRecall = 100 * row.baseline.paddedVsCoreIntersectionSeconds / row.baseline.coreTruthSeconds;
        const suppressedRecall = 100 * row.suppressed.paddedVsCoreIntersectionSeconds / row.suppressed.coreTruthSeconds;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{escapeHtml(row.baseline.id)}}.mp4</td><td>${{scope}}</td><td>${{baselineRecall.toFixed(2)}}%</td><td>${{suppressedRecall.toFixed(2)}}%</td><td>${{secs(row.lost)}}s</td><td>${{row.affected.length}}</td><td>${{full}}</td><td>${{count('both-models')}}</td><td>${{count('previous-production-only')}}</td><td>${{count('all-labels-v2-only')}}</td>`;
        target.append(tr);
      }});
    }}
    function renderCorrectSuppression() {{
      const labels = {{all:'All labeled videos', grass:'Grass', indoor:'Indoor', beach:'Beach', 'unknown-feedback':'Export feedback (unknown)'}};
      const target = document.getElementById('correct-suppression-table'); target.innerHTML = '';
      Object.entries(correctAudit.strictProductSafeByEnvironmentSeconds).forEach(([scope, values]) => {{
        const total = values['both-models'] + values['previous-production-only'] + values['all-labels-v2-only'];
        const cell = key => `${{secs(values[key])}}s (${{(100 * values[key] / total).toFixed(1)}}%)`;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{labels[scope]}}</td><td>${{cell('both-models')}}</td><td>${{cell('previous-production-only')}}</td><td>${{cell('all-labels-v2-only')}}</td><td>${{secs(total)}}s</td>`;
        target.append(tr);
      }});
    }}
    function renderOldOnlyVeto() {{
      const target = document.getElementById('old-only-veto-table'); target.innerHTML = '';
      oldOnlyAudit.rallies.forEach(row => {{
        const scope = row.provenance === 'export-feedback' ? `Export feedback (${{row.feedbackPartition}})` : row.provenance === 'training-dataset' ? 'Training dataset' : 'Evaluation / validation / test';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{escapeHtml(row.file)}}</td><td>${{scope}}</td><td>${{row.rallyNumber}}</td><td>${{Number(row.coreStart).toFixed(3)}}–${{Number(row.coreEnd).toFixed(3)}}</td><td>${{secs(row.evaluableCoreDurationSeconds)}}s</td><td>${{secs(row.productionCoveredSeconds)}}s</td><td>${{secs(row.oldOnlySuppressionCoveredSeconds)}}s</td><td>${{secs(row.additionalMissedCoreSeconds)}}s</td><td class="${{row.fullyMissedAfterPadding ? 'negative' : 'warning'}}">${{row.fullyMissedAfterPadding ? 'Complete miss' : 'Partial miss'}}</td>`;
        target.append(tr);
      }});
    }}
    function renderV2OnlyVeto() {{
      const target = document.getElementById('v2-only-veto-table'); target.innerHTML = '';
      v2OnlyAudit.rallies.forEach(row => {{
        const scope = row.provenance === 'export-feedback' ? `Export feedback (${{row.feedbackPartition}})` : row.provenance === 'training-dataset' ? 'Training dataset' : 'Evaluation / validation / test';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{escapeHtml(row.file)}}</td><td>${{scope}}</td><td>${{row.rallyNumber}}</td><td>${{Number(row.coreStart).toFixed(3)}}–${{Number(row.coreEnd).toFixed(3)}}</td><td>${{secs(row.evaluableCoreDurationSeconds)}}s</td><td>${{secs(row.productionCoveredSeconds)}}s</td><td>${{secs(row.v2OnlySuppressionCoveredSeconds)}}s</td><td>${{secs(row.additionalMissedCoreSeconds)}}s</td><td class="${{row.fullyMissedAfterPadding ? 'negative' : 'warning'}}">${{row.fullyMissedAfterPadding ? 'Complete miss' : 'Partial miss'}}</td>`;
        target.append(tr);
      }});
    }}
    renderHeld();
    const scopeSelect = document.getElementById('scope-select'); const paddingSelect = document.getElementById('padding-scope-select'); fillSelect(scopeSelect); fillSelect(paddingSelect);
    scopeSelect.value = 'feedback-partition:held-out'; paddingSelect.value = 'feedback-partition:held-out';
    renderComparison(scopeSelect.value); renderPadding(paddingSelect.value);
    renderThresholds(); renderCorrectSuppression(); renderOldOnlyVeto(); renderV2OnlyVeto(); renderGrassAudit(); renderMisses();
    document.getElementById('full-miss-count').textContent = missedAudit.summary.fullyMissedAfterPadding;
    document.getElementById('both-model-count').textContent = missedAudit.summary.fullyMissedProductionComponentAttribution['both-models'];
    document.getElementById('join-rescue-count').textContent = missedAudit.summary.fullyMissedRalliesHelpedByShortGapJoin;
    scopeSelect.addEventListener('change', event => renderComparison(event.target.value));
    paddingSelect.addEventListener('change', event => renderPadding(event.target.value));
    document.getElementById('miss-filter').addEventListener('change', renderMisses);
    document.getElementById('miss-search').addEventListener('input', renderMisses);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.source.read_text(encoding="utf-8"))
    threshold_path = args.source.parent / "production-suppression-threshold-recall-constraint.json"
    missed_path = args.source.parent / "production-plus-suppression-missed-rallies.json"
    correct_path = args.source.parent / "production-plus-suppression-correct-attribution.json"
    old_only_path = args.source.parent / "old-only-suppression-rally-audit.json"
    v2_only_path = args.source.parent / "v2-only-suppression-rally-audit.json"
    report["thresholdRecallAudit"] = json.loads(threshold_path.read_text(encoding="utf-8"))
    report["missedRalliesAudit"] = json.loads(missed_path.read_text(encoding="utf-8"))
    report["correctSuppressionAttribution"] = json.loads(correct_path.read_text(encoding="utf-8"))
    report["oldOnlySuppressionAudit"] = json.loads(old_only_path.read_text(encoding="utf-8"))
    report["v2OnlySuppressionAudit"] = json.loads(v2_only_path.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
