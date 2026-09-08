#!/usr/bin/env python3
"""Render the corrected full-v3 comparison and product rules as standalone HTML."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-corrected-2026-08-18"
)
DEFAULT_OUTPUT = Path(
    "/home/developer/server/docker/caddy/html/reports/"
    "volleycut-corrected-v3-comparison-2026-08-18.html"
)
UI_ROOT = "http://192.0.2.1:3001/suppression-review/corrected-v3"


def esc(value: object) -> str:
    return html.escape(str(value))


def score(value: float) -> str:
    return f"{value:.4f}"


def th(label: str, tip: str) -> str:
    return f'<th tabindex="0" data-tip="{esc(tip)}">{esc(label)}</th>'


def metric_row(label: str, metric: dict[str, Any], baseline: dict[str, Any]) -> str:
    delta = metric["paddedModelExportSeconds"] - baseline["paddedModelExportSeconds"]
    return (
        f"<tr><td>{esc(label)}</td>"
        f"<td>{score(metric['core']['precision'])}</td>"
        f"<td>{score(metric['core']['recall'])}</td>"
        f"<td>{score(metric['core']['f1'])}</td>"
        f"<td>{score(metric['padded']['precision'])}</td>"
        f"<td>{score(metric['padded']['recall'])}</td>"
        f"<td>{score(metric['padded']['f1'])}</td>"
        f"<td>{score(metric['P_pad'])}</td>"
        f"<td>{score(metric['R_core'])}</td>"
        f"<td>{score(metric['F1_padP_coreR'])}</td>"
        f"<td>{metric['paddedModelExportSeconds']:.1f}s</td>"
        f"<td>{delta:+.1f}s</td></tr>"
    )


def non_exempt_misses(audit: dict[str, Any]) -> tuple[int, int]:
    videos = [
        video
        for video in audit["videos"]
        if video["file"] != "PXL_20260816_164327879.mp4"
    ]
    return (
        sum(video["completeMisses"] for video in videos),
        sum(video["partialMisses"] for video in videos),
    )


def render(report: dict[str, Any], audits: dict[str, dict[str, Any]]) -> str:
    variants = report["variantOrder"]
    labels = report["variantLabels"]
    metrics = report["metrics"]
    all_scope = metrics["all-evaluable"]
    dev_scope = metrics["selection:development"]
    test_scope = metrics["disclosure:protected-test"]
    baseline_all = all_scope["current-production-ensemble"]["padding"]["2"]
    baseline_dev = dev_scope["current-production-ensemble"]["padding"]["2"]
    held_suppression = audits["zero"]
    prod_decoder = {
        "smoothing_seconds": 1.0,
        "enter_threshold": 0.75,
        "exit_threshold": 0.65,
        "min_live_seconds": 0.5,
        "bridge_gap_seconds": 0.5,
        "short_event_min_seconds": 0.25,
        "short_event_threshold": 0.9,
    }
    v3_decoder = report["selection"]["v3Candidate2"]["selectedConfig"]
    product_audits = {
        "production-plus-suppression-raw-connected": audits["raw"],
        "production-plus-suppression-aggressive-intermediate": audits["aggressive"],
        "production-plus-suppression-zero-non-exempt-misses": audits["zero"],
    }
    audit_scope_ids = {
        "all-evaluable": "all-evaluable",
        "selection:development": "development",
        "disclosure:protected-test": "protected-test",
        "provenance:training-dataset": "training-dataset",
        "provenance:evaluation-validation-test-only": "evaluation-validation-test-only",
        "provenance:export-feedback": "export-feedback",
        "feedback-partition:fit": "feedback-fit",
        "feedback-partition:held-out": "feedback-held-out",
    }

    def selected_metric(scope: str, variant: str, padding: str) -> dict[str, Any]:
        audit = product_audits.get(variant)
        if audit is not None:
            return audit["scopeMetrics"][audit_scope_ids[scope]]["candidate"][padding]
        return metrics[scope][variant]["padding"][padding]

    metric_headers = "".join(
        [
            th("Variant", "Exact model or ensemble composition being evaluated."),
            th("Core P", "Unpadded model-human overlap divided by unpadded predicted duration."),
            th("Core R", "Unpadded model-human overlap divided by unpadded human rally duration."),
            th("Core F1", "Harmonic mean of unpadded Core precision and Core recall."),
            th("Padded P", "Equally padded model-human overlap divided by padded model export duration."),
            th("Padded R", "Equally padded model-human overlap divided by padded human export duration."),
            th("Padded F1", "Harmonic mean of symmetric padded precision and recall."),
            th("P_pad", "Padded precision used by the primary hybrid ranking metric."),
            th("R_core", "Human core duration covered by padded model output divided by human core duration."),
            th("F1_padP_coreR", "Primary pooled ranking metric: harmonic mean of P_pad and R_core."),
            th("Export", "Padded model export duration after joining positive gaps strictly below 3 seconds."),
            th("Δ export", "Change in padded export duration relative to current production on the same scope."),
        ]
    )
    all_rows = "".join(
        metric_row(
            labels[variant],
            selected_metric("all-evaluable", variant, "2"),
            baseline_all,
        )
        for variant in variants
    )
    dev_rows = "".join(
        metric_row(
            labels[variant],
            selected_metric("selection:development", variant, "2"),
            baseline_dev,
        )
        for variant in variants
    )

    policy_meta = [
        (
            "raw",
            "Raw connected",
            "No agreement padding or joining",
            f"{UI_ROOT}/raw",
        ),
        (
            "aggressive",
            "Aggressive intermediate",
            "1.5s agreement padding; join positive gaps <0.5s",
            f"{UI_ROOT}/aggressive",
        ),
        (
            "zero",
            "Zero non-exempt misses",
            "2s agreement padding; join positive gaps <0.5s",
            f"{UI_ROOT}/zero-miss",
        ),
    ]
    policy_rows = []
    for key, name, grouping, url in policy_meta:
        audit = audits[key]
        summary = audit["summary"]
        complete, partial = non_exempt_misses(audit)
        candidate = summary["candidate"]
        policy_rows.append(
            f'<tr><td><a href="{esc(url)}">{esc(name)}</a></td>'
            f"<td>{esc(grouping)}</td>"
            f"<td>{summary['correctlyRemovedFalsePositivePredictions']}</td>"
            f"<td>{summary['exportTimeSavedSeconds']:.1f}s</td>"
            f"<td>{complete}</td><td>{partial}</td>"
            f"<td>{score(candidate['core']['precision'])}</td>"
            f"<td>{score(candidate['core']['recall'])}</td>"
            f"<td>{score(candidate['core']['f1'])}</td>"
            f"<td>{score(candidate['padded']['precision'])}</td>"
            f"<td>{score(candidate['padded']['recall'])}</td>"
            f"<td>{score(candidate['padded']['f1'])}</td>"
            f"<td>{score(candidate['F1_padP_coreR'])}</td></tr>"
        )
    policy_headers = "".join(
        [
            th("Option", "Retained product suppression-aggressiveness option and its rally-level review UI."),
            th("Agreement grouping", "Rule used only to decide whether current-production output is supported by one model or both."),
            th("Correct FPs", "Production intervals fully removed with zero overlap against the padded human export target."),
            th("Export saved", "Reduction in final padded and joined export duration relative to current production."),
            th("Non-exempt complete", "New complete rally losses excluding the rotated-camera PXL exception."),
            th("Non-exempt partial", "New partial rally losses excluding the rotated-camera PXL exception."),
            th("Core P", "Unpadded precision pooled over all 28 evaluable recordings."),
            th("Core R", "Unpadded recall pooled over all 28 evaluable recordings."),
            th("Core F1", "Unpadded F1 pooled over all 28 evaluable recordings."),
            th("Padded P", "Symmetric padded precision pooled over all 28 evaluable recordings."),
            th("Padded R", "Symmetric padded recall pooled over all 28 evaluable recordings."),
            th("Padded F1", "Symmetric padded F1 pooled over all 28 evaluable recordings."),
            th("F1_padP_coreR", "Primary pooled metric at 2-second product padding."),
        ]
    )

    scope_rows = []
    scope_specs = [
        ("Training recordings", "provenance:training-dataset"),
        ("Evaluation / validation / test only", "provenance:evaluation-validation-test-only"),
        ("Export feedback", "provenance:export-feedback"),
        ("Feedback fit half", "feedback-partition:fit"),
        ("Feedback held-out half", "feedback-partition:held-out"),
    ]
    focus_variants = [
        "current-production-ensemble",
        "production-plus-suppression-raw-connected",
        "production-plus-suppression-aggressive-intermediate",
        "production-plus-suppression-zero-non-exempt-misses",
        "v3-candidate1-three-head",
        "v3-candidate2-four-head",
    ]
    for scope_label, scope_id in scope_specs:
        for variant in focus_variants:
            row = selected_metric(scope_id, variant, "2")
            scope_rows.append(
                f"<tr><td>{esc(scope_label)}</td><td>{esc(labels[variant])}</td>"
                f"<td>{score(row['core']['precision'])}</td>"
                f"<td>{score(row['core']['recall'])}</td>"
                f"<td>{score(row['core']['f1'])}</td>"
                f"<td>{score(row['padded']['precision'])}</td>"
                f"<td>{score(row['padded']['recall'])}</td>"
                f"<td>{score(row['padded']['f1'])}</td>"
                f"<td>{score(row['F1_padP_coreR'])}</td></tr>"
            )

    sensitivity_rows = []
    for variant in variants:
        for padding in ("0", "1", "2", "3"):
            row = selected_metric("all-evaluable", variant, padding)
            human_export = baseline_all["paddedHumanExportSeconds"]
            if padding != "2":
                human_export = all_scope["current-production-ensemble"]["padding"][padding][
                    "paddedHumanExportSeconds"
                ]
            sensitivity_rows.append(
                f"<tr><td>{esc(labels[variant])}</td><td>{padding}s</td>"
                f"<td>{score(row['P_pad'])}</td><td>{score(row['R_core'])}</td>"
                f"<td>{score(row['F1_padP_coreR'])}</td>"
                f"<td>{row['paddedModelExportSeconds']:.1f}s</td>"
                f"<td>{human_export:.1f}s</td>"
                f"<td>{row['paddedModelExportSeconds'] - human_export:+.1f}s</td></tr>"
            )

    candidate1 = report["models"]["candidate1"]
    candidate2 = report["models"]["candidate2"]
    model_rows = []
    for role, model in (("v3 candidate 1", candidate1), ("v3 candidate 2", candidate2)):
        for head_name, head in model["heads"].items():
            model_rows.append(
                f"<tr><td>{esc(role)}</td><td>{esc(model['modelId'])}</td>"
                f"<td>{esc(head_name)}</td><td><code>{esc(head['sha256'])}</code></td>"
                f"<td><code>{esc(head['path'])}</code></td></tr>"
            )

    test_best = max(
        variants,
        key=lambda variant: selected_metric(
            "disclosure:protected-test", variant, "2"
        )["F1_padP_coreR"],
    )
    corrected_intervals = [
        interval
        for row in report["suppressionTargets"]["perRecording"]
        for interval in row["excludedFalsePositiveIntervals"]
    ]
    excluded_samples = sum(
        row["excludedPositiveSamples"]
        for row in report["suppressionTargets"]["perRecording"]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VolleySplice corrected v3 comparison</title>
<style>
:root{{--ink:#171813;--paper:#f3f0e7;--acid:#dfff35;--line:#aaa99f;--muted:#5f6259;--red:#b52222;--blue:#173f73}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.55 system-ui,sans-serif}}
a{{color:var(--blue)}} code{{font:12px ui-monospace,monospace;overflow-wrap:anywhere}} .shell{{width:min(1500px,92vw);margin:auto}}
header{{padding:64px 0 40px;border-bottom:2px solid var(--ink)}} .eyebrow{{font:700 11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.12em}}
h1{{max-width:1100px;margin:.15em 0;font-size:clamp(44px,7vw,92px);line-height:.9;letter-spacing:-.065em}} h2{{font-size:30px;letter-spacing:-.035em}}
.lede{{max-width:900px;font-size:20px}} nav{{position:sticky;top:0;z-index:3;border-bottom:1px solid var(--ink);background:#f3f0e7ed;backdrop-filter:blur(10px)}}
nav .shell{{display:flex;gap:24px;overflow:auto;padding:12px 0}} nav a{{color:inherit;font:700 11px ui-monospace,monospace;text-transform:uppercase;white-space:nowrap}}
main{{padding:34px 0 80px}} section{{margin:0 0 36px;scroll-margin-top:60px}} .panel{{padding:26px;border:1px solid var(--ink);background:#fff}}
.verdict{{background:var(--ink);color:#fff}} .verdict strong{{color:var(--acid)}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.stat{{padding:18px;border:1px solid var(--line);background:#fff}} .stat span{{display:block;color:var(--muted);font:700 10px ui-monospace,monospace;text-transform:uppercase}}
.stat b{{display:block;margin-top:6px;font-size:28px}} .warning{{color:var(--red)}} .table-wrap{{overflow:auto;border:1px solid var(--ink);background:#fff}}
table{{width:100%;border-collapse:collapse;min-width:1100px;font-variant-numeric:tabular-nums}} th,td{{padding:10px 11px;border-bottom:1px solid #d2d0c6;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:var(--ink);color:#fff;font:700 10px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.04em}}
th[data-tip]{{cursor:help;text-decoration:underline dotted var(--acid);text-underline-offset:3px}} th[data-tip]::after{{content:attr(data-tip);position:fixed;left:50%;bottom:18px;z-index:9;width:min(620px,90vw);transform:translate(-50%,8px);padding:13px;background:#111;color:#fff;border:1px solid var(--acid);font:13px/1.45 system-ui,sans-serif;text-align:left;text-transform:none;white-space:normal;opacity:0;pointer-events:none;transition:.15s}}
th[data-tip]:hover::after,th[data-tip]:focus::after{{opacity:1;transform:translate(-50%,0)}} th:first-child,td:first-child{{text-align:left;white-space:normal;min-width:230px}} tbody tr:hover{{background:#f0ffd0}}
.rules{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}} .rules article{{padding:20px;border:1px solid var(--ink);background:#fff}} .rules h3{{margin-top:0}}
.callout{{border-left:5px solid var(--red);padding:4px 18px}} details{{border:1px solid var(--line);background:#fff}} summary{{cursor:pointer;padding:16px;font-weight:700}} details .table-wrap{{border:0;border-top:1px solid var(--line)}}
@media(max-width:900px){{.grid,.rules{{grid-template-columns:1fr 1fr}}}} @media(max-width:560px){{.grid,.rules{{grid-template-columns:1fr}}}}
</style></head><body>
<header class="shell"><div class="eyebrow">VolleySplice · suppression product decision · 18 August 2026</div><h1>Hold the prior decoder. Do not advance either v3 candidate.</h1>
<p class="lede">The current production ensemble remains the base model. Product suppression uses the corrected overlap-safe weights with the prior production decoder, while both retrained v3 candidates remain research artifacts only.</p></header>
<nav><div class="shell"><a href="#outcome">Outcome</a><a href="#policies">Product policies</a><a href="#all">All recordings</a><a href="#development">Development</a><a href="#scopes">Scopes</a><a href="#rules">Implementation</a><a href="#models">Models</a><a href="#sensitivity">Padding</a></div></nav>
<main class="shell">
<section id="outcome" class="panel verdict"><div class="eyebrow">Recorded decision</div><h2>Keep current production, hold the prior suppression decoder, and reject both v3 candidates.</h2>
<p><strong>v3 candidate 1 is not moving forward</strong> because all-evaluable Core Recall falls from {score(baseline_all['core']['recall'])} to {score(all_scope['v3-candidate1-three-head']['padding']['2']['core']['recall'])}. <strong>v3 candidate 2 is also rejected</strong>: despite leading the primary metric on development ({score(dev_scope['v3-candidate2-four-head']['padding']['2']['F1_padP_coreR'])}) and all evaluable recordings ({score(all_scope['v3-candidate2-four-head']['padding']['2']['F1_padP_coreR'])}), its all-evaluable Core Recall is only {score(all_scope['v3-candidate2-four-head']['padding']['2']['core']['recall'])}.</p>
<p class="callout">Holding the previous decoder restores the conservative 2s / &lt;0.5s option to zero non-exempt complete and partial misses. Suppression remains initially disabled and removals remain reviewable.</p></section>
<section class="grid"><div class="stat"><span>Corrected target samples removed</span><b>{excluded_samples}</b></div><div class="stat"><span>Zero-option non-exempt misses</span><b>0 / 0</b></div><div class="stat"><span>v3 candidate status</span><b>Rejected</b></div><div class="stat"><span>Protected-test best (disclosure)</span><b>{score(selected_metric('disclosure:protected-test', test_best, '2')['F1_padP_coreR'])}</b></div></section>

<section id="policies"><h2>Three retained production + suppression options</h2><p>Every row uses corrected overlap-safe weights and the held previous production decoder. The rotated-camera <code>PXL_20260816_164327879.mp4</code> is excluded only from the non-exempt miss columns, never from aggregate metrics.</p><div class="table-wrap"><table><thead><tr>{policy_headers}</tr></thead><tbody>{''.join(policy_rows)}</tbody></table></div><p><a href="{UI_ROOT}">Open the held-decoder visual comparison UI</a>.</p></section>

<section id="all"><h2>All 28 evaluable recordings</h2><div class="table-wrap"><table><thead><tr>{metric_headers}</tr></thead><tbody>{all_rows}</tbody></table></div></section>
<section id="development"><h2>Predeclared development ranking scope</h2><p>This is the only tuning-safe selection scope. The protected test is disclosed separately and was not used for selection.</p><div class="table-wrap"><table><thead><tr>{metric_headers}</tr></thead><tbody>{dev_rows}</tbody></table></div></section>

<section id="scopes"><h2>Training, untouched, and feedback breakdown</h2><div class="table-wrap"><table><thead><tr>{th('Scope','Recording provenance or feedback partition pooled for this row.')}{th('Variant','Model or production-suppression composition.')}{th('Core P','Unpadded pooled precision.')}{th('Core R','Unpadded pooled recall.')}{th('Core F1','Unpadded pooled F1.')}{th('Padded P','Symmetric padded pooled precision.')}{th('Padded R','Symmetric padded pooled recall.')}{th('Padded F1','Symmetric padded pooled F1.')}{th('F1_padP_coreR','Primary pooled metric at 2-second product padding.')}</tr></thead><tbody>{''.join(scope_rows)}</tbody></table></div></section>

<section id="rules"><h2>Exact product combination contract</h2><div class="rules">
<article><h3>1. Models and decoding</h3><p>Run the unchanged current production ensemble: previous-production three-head model and all-labels-v2 three-head model. Their raw prediction union is <code>P</code>. Separately run suppression artifact <code>{esc(held_suppression['suppressionModelSha256'])}</code> at <code>{esc(held_suppression['suppressionModelPath'])}</code>.</p><p>For production composition decode suppression with the held configuration <code>{esc(json.dumps(prod_decoder, separators=(',', ':')))}</code>. Do not use the re-tuned production decoder or the v3 standalone decoder (<code>{esc(json.dumps(v3_decoder, separators=(',', ':')))}</code>) for this product path.</p></article>
<article><h3>2. Agreement protection</h3><p>Tag raw intervals by source. Raw connected uses overlap-connected raw components. The 1.5s and 2s policies first pad each source interval symmetrically by the named amount and join positive gaps strictly below 0.5s. A grouped component is protected in full when any previous-production interval and any all-labels-v2 interval occur anywhere in it—including non-overlapping heads and tails.</p></article>
<article><h3>3. Apply only eligible cuts</h3><p>Let <code>S</code> be decoded suppression, clipped to <code>P</code>. Let <code>E</code> be raw production time belonging to one-model-only agreement components. The only applied cut is <code>C = P ∩ S ∩ E</code>; the unsmoothed candidate is <code>P − C</code>. Suppression never cuts a component protected as “both models.”</p></article>
<article><h3>4. Final export</h3><p>Pad candidate intervals by 2 seconds before and after, clip to video bounds, merge overlapping or touching ranges, then join positive gaps strictly below 3 seconds. Subtract <code>ignoredIntervals</code> from predictions and truth and never rejoin across an ignored interval.</p></article>
</div><p class="callout">Product UI: suppression starts disabled; affected rallies are highlighted for review. Show an aggressiveness selector only when the three policies actually produce different affected rallies in that game.</p></section>

<section id="models"><h2>Rejected v3 research artifacts</h2><p>These artifacts are retained for reproducibility only. Neither candidate is approved for product use.</p><div class="table-wrap"><table><thead><tr>{th('Candidate','Rejected retrained candidate containing this head.')}{th('Model ID','Bundle-level identifier for the rejected candidate.')}{th('Head','Rally, serve, dead-state, or suppression head.')}{th('SHA-256','Exact immutable artifact digest.')}{th('Path','NAS path retained for research reproducibility.')}</tr></thead><tbody>{''.join(model_rows)}</tbody></table></div></section>

<section id="sensitivity"><h2>Required 0/1/2/3-second padding sensitivity</h2><details><summary>Show all-evaluable padding cases for all eight variants</summary><div class="table-wrap"><table><thead><tr>{th('Variant','Exact model or ensemble composition.')}{th('Padding','Symmetric seconds added before and after model and padded-human intervals.')}{th('P_pad','Padded precision at this padding case.')}{th('R_core','Recall of human core by padded model output at this padding case.')}{th('F1_padP_coreR','Primary hybrid metric at this padding case.')}{th('Model export','Padded model export duration.')}{th('Human export','Padded human export duration.')}{th('Difference','Model export minus padded human export duration.')}</tr></thead><tbody>{''.join(sensitivity_rows)}</tbody></table></div></details></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = json.loads((args.experiment / "report.json").read_text(encoding="utf-8"))
    audits = {
        "raw": json.loads((Path("data") / "corrected-v3-suppression-raw-connected.json").read_text(encoding="utf-8")),
        "aggressive": json.loads((Path("data") / "corrected-v3-suppression-aggressive.json").read_text(encoding="utf-8")),
        "zero": json.loads((Path("data") / "corrected-v3-suppression-zero-miss.json").read_text(encoding="utf-8")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report, audits), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
