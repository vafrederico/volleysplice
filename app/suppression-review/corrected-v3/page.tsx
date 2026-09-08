import type { Metadata } from "next";
import Link from "next/link";

import { Brand } from "@/components/brand";
import aggressiveData from "@/data/corrected-v3-suppression-aggressive.json";
import rawData from "@/data/corrected-v3-suppression-raw-connected.json";
import zeroData from "@/data/corrected-v3-suppression-zero-miss.json";
import styles from "../tuning/tuning.module.css";
import type { SuppressionReviewDataset } from "../types";

export const metadata: Metadata = {
  title: "Held-decoder suppression options · VolleySplice",
  description:
    "Compare the three retained production suppression rules with corrected targets and the held production decoder.",
  robots: { index: false, follow: false },
};

const OPTIONS = [
  {
    label: "Raw connected",
    detail: "No agreement padding or joining",
    href: "/suppression-review/corrected-v3/raw",
    dataset: rawData as SuppressionReviewDataset,
  },
  {
    label: "Aggressive intermediate",
    detail: "1.5s agreement padding · joins <0.5s",
    href: "/suppression-review/corrected-v3/aggressive",
    dataset: aggressiveData as SuppressionReviewDataset,
  },
  {
    label: "Zero non-exempt misses",
    detail: "2s agreement padding · joins <0.5s",
    href: "/suppression-review/corrected-v3/zero-miss",
    dataset: zeroData as SuppressionReviewDataset,
  },
] as const;

function score(value: number): string {
  return value.toFixed(4);
}

function nonExemptMisses(dataset: SuppressionReviewDataset) {
  return dataset.videos
    .filter((video) => video.file !== "PXL_20260816_164327879.mp4")
    .reduce(
      (total, video) => ({
        complete: total.complete + video.completeMisses,
        partial: total.partial + video.partialMisses,
      }),
      { complete: 0, partial: 0 },
    );
}

export default function CorrectedV3SuppressionOptionsPage() {
  const baseline = OPTIONS[0].dataset.summary.production;
  return (
    <main className={styles.shell}>
      <nav className={styles.topbar}>
        <Brand className={styles.brand} label="Held decoder" priority />
        <Link href="/suppression-review/tuning">Earlier tuning</Link>
      </nav>

      <header className={styles.hero}>
        <p>Current production · corrected targets · previous decoder held</p>
        <h1>Three suppression rules with the recall-safe decoder frozen.</h1>
        <div className={styles.callout}>
          <strong>Product decision</strong>
          <span>Hold the prior production suppression decoder.</span>
          <b>Zero non-exempt misses at the conservative option</b>
          <small>Neither standalone v3 candidate is moving forward.</small>
        </div>
      </header>

      <section className={styles.contract}>
        <div>
          <strong>Production base</strong>
          <span>Previous-production prediction union all-labels-v2.</span>
        </div>
        <div>
          <strong>Fixed export rule</strong>
          <span>+2s symmetric padding, then positive gaps &lt;3s joined.</span>
        </div>
        <div>
          <strong>Frozen suppression</strong>
          <span>Corrected weights; 1s smoothing and 0.75 enter threshold.</span>
        </div>
      </section>

      <section className={styles.baseline} aria-labelledby="baseline-title">
        <div className={styles.baselineIntro}>
          <p>Reference metrics</p>
          <h2 id="baseline-title">Before suppression</h2>
          <span>
            All 28 evaluable recordings at the 2-second product padding.
          </span>
        </div>
        <div className={styles.baselineGroups}>
          <article>
            <header>
              <strong>Core</strong>
              <small>Unpadded prediction vs. human core</small>
            </header>
            <dl>
              <div title="Core precision before suppression.">
                <dt>Precision</dt>
                <dd>{score(baseline.core.precision)}</dd>
              </div>
              <div title="Core recall before suppression.">
                <dt>Recall</dt>
                <dd>{score(baseline.core.recall)}</dd>
              </div>
              <div title="Core F1 before suppression.">
                <dt>F1</dt>
                <dd>{score(baseline.core.f1)}</dd>
              </div>
            </dl>
          </article>
          <article>
            <header>
              <strong>Padded</strong>
              <small>Both sides +2s, joins &lt;3s</small>
            </header>
            <dl>
              <div title="Padded precision before suppression.">
                <dt>Precision</dt>
                <dd>{score(baseline.padded.precision)}</dd>
              </div>
              <div title="Padded recall before suppression.">
                <dt>Recall</dt>
                <dd>{score(baseline.padded.recall)}</dd>
              </div>
              <div title="Padded F1 before suppression.">
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
              <th title="Agreement-grouping policy and link to its rally-level visual inspection.">
                Option
              </th>
              <th title="Production false-positive intervals fully removed outside the padded human target.">
                Correct FPs removed
              </th>
              <th title="Reduction in final export duration after the fixed product padding and join rule.">
                Export saved
              </th>
              <th title="New complete human-rally losses, excluding the rotated-camera PXL exception.">
                Non-exempt complete
              </th>
              <th title="New partial human-rally losses, excluding the rotated-camera PXL exception.">
                Non-exempt partial
              </th>
              <th title="Unpadded precision after this suppression policy on all evaluable recordings.">
                Core P
              </th>
              <th title="Unpadded recall after this suppression policy on all evaluable recordings.">
                Core R
              </th>
              <th title="Unpadded F1 after this suppression policy on all evaluable recordings.">
                Core F1
              </th>
              <th title="Equally padded precision after this suppression policy on all evaluable recordings.">
                Padded P
              </th>
              <th title="Equally padded recall after this suppression policy on all evaluable recordings.">
                Padded R
              </th>
              <th title="Equally padded F1 after this suppression policy on all evaluable recordings.">
                Padded F1
              </th>
              <th title="Primary pooled metric combining padded precision and recall of human core by padded prediction.">
                F1_padP_coreR
              </th>
            </tr>
          </thead>
          <tbody>
            {OPTIONS.map((option) => {
              const summary = option.dataset.summary;
              const misses = nonExemptMisses(option.dataset);
              return (
                <tr key={option.href}>
                  <td>
                    <Link href={option.href}>{option.label}</Link>
                    <small>{option.detail}</small>
                  </td>
                  <td>{summary.correctlyRemovedFalsePositivePredictions}</td>
                  <td>{summary.exportTimeSavedSeconds.toFixed(1)}s</td>
                  <td>{misses.complete}</td>
                  <td>{misses.partial}</td>
                  <td>{score(summary.candidate.core.precision)}</td>
                  <td>{score(summary.candidate.core.recall)}</td>
                  <td>{score(summary.candidate.core.f1)}</td>
                  <td>{score(summary.candidate.padded.precision)}</td>
                  <td>{score(summary.candidate.padded.recall)}</td>
                  <td>{score(summary.candidate.padded.f1)}</td>
                  <td>{score(summary.candidate.F1_padP_coreR)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className={styles.notes}>
        <p>
          All three reviews use corrected suppression artifact{" "}
          <code>39eddf581639…</code>, weights <code>a943749b69c9…</code>, and
          the held production decoder: 1s smoothing, 0.75 enter, 0.65 exit, 0.5s
          minimum live duration, and 0.5s bridging. The protected test was not
          used for the original decoder selection.
        </p>
      </section>
    </main>
  );
}
