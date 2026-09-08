import type { Metadata } from "next";
import Link from "next/link";

import { Brand } from "@/components/brand";
import sweepData from "@/data/single-model-agreement-policy-sweep.json";

import styles from "./tuning.module.css";

export const metadata: Metadata = {
  title: "Suppression agreement tuning · VolleySplice",
  description:
    "Compare how agreement padding and join thresholds change false-positive removal and rally retention.",
  robots: { index: false, follow: false },
};

const VARIANTS = [
  {
    id: "pad-0-join-0-support-0",
    label: "Raw connected",
    detail: "No agreement padding or joining",
    href: "/suppression-review/tuning/raw",
  },
  {
    id: "pad-1.5-join-0.5-support-0",
    label: "Aggressive intermediate",
    detail: "1.5s agreement padding · joins <0.5s",
    href: "/suppression-review/tuning/pad-1_5-join-0_5",
  },
  {
    id: "pad-2-join-0-support-0",
    label: "Padding only",
    detail: "2s agreement padding · no joining",
    href: "/suppression-review/tuning/pad-2-join-0",
  },
  {
    id: "pad-2-join-0.5-support-0",
    label: "Zero non-exempt misses",
    detail: "2s agreement padding · joins <0.5s",
    href: "/suppression-review/tuning/pad-2-join-0_5",
  },
  {
    id: "pad-2-join-3-support-0",
    label: "Full export-component rule",
    detail: "2s agreement padding · joins <3s",
    href: "/suppression-review/any-overlap/retrained",
  },
] as const;

function seconds(value: number): string {
  return `${value.toFixed(1)}s`;
}

function score(value: number): string {
  return value.toFixed(4);
}

export default function SuppressionAgreementTuningPage() {
  const results = new Map(
    sweepData.results.map((result) => [result.id, result]),
  );
  const baseline = sweepData.baseline.allEvaluable;

  return (
    <main className={styles.shell}>
      <nav className={styles.topbar}>
        <Brand className={styles.brand} label="Agreement tuning" priority />
        <Link href="/suppression-review/any-overlap/retrained">
          Current review
        </Link>
      </nav>

      <header className={styles.hero}>
        <p>Retrained suppression specialist · fixed product export contract</p>
        <h1>How close must the two models be to protect the region?</h1>
        <div className={styles.callout}>
          <strong>Best zero-non-exempt-miss diagnostic</strong>
          <span>2s agreement padding + joins below 0.5s</span>
          <b>71 correct FP predictions removed</b>
          <small>Current full-component rule removes 44</small>
        </div>
      </header>

      <section className={styles.contract}>
        <div>
          <strong>Fixed export rule</strong>
          <span>
            Every candidate is exported with +2s padding and joins &lt;3s.
          </span>
        </div>
        <div>
          <strong>What changes</strong>
          <span>
            Only the grouping used to call a region “one model” or “both
            models.”
          </span>
        </div>
        <div>
          <strong>Camera exception</strong>
          <span>
            PXL_20260816_164327879.mp4 is excluded from the no-miss guardrail.
          </span>
        </div>
      </section>

      <section className={styles.baseline} aria-labelledby="baseline-title">
        <div className={styles.baselineIntro}>
          <p>Reference metrics</p>
          <h2 id="baseline-title">Before any suppression</h2>
          <span>
            Current production ensemble · all evaluable labeled and feedback
            videos
          </span>
        </div>
        <div className={styles.baselineGroups}>
          <article>
            <header>
              <strong>Core</strong>
              <small>Unpadded prediction vs. human core</small>
            </header>
            <dl>
              <div title="Unpadded model-human overlap duration divided by unpadded predicted duration before suppression.">
                <dt>Precision</dt>
                <dd>{score(baseline.core.precision)}</dd>
              </div>
              <div title="Unpadded model-human overlap duration divided by unpadded human rally duration before suppression.">
                <dt>Recall</dt>
                <dd>{score(baseline.core.recall)}</dd>
              </div>
              <div title="Harmonic mean of unpadded core precision and recall before suppression.">
                <dt>F1</dt>
                <dd>{score(baseline.core.f1)}</dd>
              </div>
            </dl>
          </article>
          <article>
            <header>
              <strong>Padded</strong>
              <small>Both sides +2s, then joins &lt;3s</small>
            </header>
            <dl>
              <div title="Equally padded model-human overlap divided by padded model export duration before suppression.">
                <dt>Precision</dt>
                <dd>{score(baseline.padded.precision)}</dd>
              </div>
              <div title="Equally padded model-human overlap divided by padded human export duration before suppression.">
                <dt>Recall</dt>
                <dd>{score(baseline.padded.recall)}</dd>
              </div>
              <div title="Harmonic mean of padded precision and padded recall before suppression.">
                <dt>F1</dt>
                <dd>{score(baseline.padded.f1)}</dd>
              </div>
            </dl>
          </article>
        </div>
      </section>

      <section className={styles.tableWrap}>
        <table>
          <thead>
            <tr>
              <th title="Named agreement-grouping policy and a link to its missed-rally visual inspection.">
                Version
              </th>
              <th title="Maximum positive raw gap that can connect model intervals under the grouping padding and strict join threshold.">
                Effective raw gap
              </th>
              <th title="Production prediction ranges fully deleted with zero overlap against the padded human export target.">
                Correct FPs removed
              </th>
              <th title="Reduction in final exported duration after fixed +2s padding and joins below 3 seconds.">
                Export saved
              </th>
              <th title="Complete human rallies newly reduced to zero coverage, excluding the rotated-camera PXL exception.">
                Complete misses
              </th>
              <th title="Human rallies that retain some coverage but lose core seconds, excluding the rotated-camera PXL exception.">
                Partial misses
              </th>
              <th title="Padded precision measured across all evaluable labeled and feedback recordings.">
                All P_pad
              </th>
              <th title="Core recall measured using padded model exports across all evaluable labeled and feedback recordings.">
                All R_core
              </th>
              <th title="Primary hybrid metric pooled across all evaluable labeled and feedback recordings: padded precision combined with recall of human core by the padded model export.">
                All F1_padP_coreR
              </th>
              <th title="Primary ranking metric measured only on the predeclared development recordings; protected test is not used for selection.">
                Dev F1_padP_coreR
              </th>
            </tr>
          </thead>
          <tbody>
            <tr data-baseline="true">
              <td>
                <strong>No suppression</strong>
                <small>Current production reference</small>
              </td>
              <td>—</td>
              <td>0</td>
              <td>0.0s</td>
              <td>0</td>
              <td>0</td>
              <td>{score(baseline.P_pad)}</td>
              <td>{score(baseline.R_core)}</td>
              <td>{score(baseline.F1_padP_coreR)}</td>
              <td>{score(sweepData.baseline.development.F1_padP_coreR)}</td>
            </tr>
            {VARIANTS.map((variant) => {
              const result = results.get(variant.id);
              if (!result) return null;
              const misses = result.allEvaluableExceptRotatedCamera.misses;
              return (
                <tr
                  data-safe={misses.affectedRallies === 0 ? "true" : "false"}
                  key={variant.id}
                >
                  <td>
                    <Link href={variant.href}>{variant.label}</Link>
                    <small>{variant.detail}</small>
                  </td>
                  <td>{seconds(result.effectiveRawGapSeconds)}</td>
                  <td>{result.correctlyRemovedFalsePositivePredictions}</td>
                  <td>{seconds(result.exportTimeSavedSeconds)}</td>
                  <td>{misses.completeMisses}</td>
                  <td>{misses.partialMisses}</td>
                  <td>{score(result.allEvaluable.metric.P_pad)}</td>
                  <td>{score(result.allEvaluable.metric.R_core)}</td>
                  <td>{score(result.allEvaluable.metric.F1_padP_coreR)}</td>
                  <td>{score(result.development.metric.F1_padP_coreR)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className={styles.notes}>
        <p>
          The{" "}
          <strong>development column is the only tuning-safe ranking</strong>.
          The zero-miss label is a full-set post-hoc diagnostic—including held
          feedback—and needs confirmation on fresh untouched footage before a
          production choice.
        </p>
        <p>
          Requiring 0.5s or 1s of raw support from each model did not change any
          result in this sweep, so the review versions keep the simpler “any raw
          support” criterion.
        </p>
      </section>
    </main>
  );
}
