import { useEffect, useMemo, useState } from "react";

import {
  completedInferenceStepCount,
  inferenceStepElapsedSeconds,
  inferenceStepEtaSeconds,
  overallInferenceProgress,
  type InferenceProgressStep,
} from "@/lib/inference-progress";

import styles from "./InferenceProgressPanel.module.css";

type InferenceProgressPanelProps = {
  steps: readonly InferenceProgressStep[];
  compact?: boolean;
  wakeLockActive?: boolean;
};

function formatDuration(seconds: number): string {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatRate(step: InferenceProgressStep, nowMs: number): string {
  if (step.status === "pending") return "Waiting";
  if (step.status === "error") return "Stopped";
  if (step.status === "skipped") return "Not needed";
  if (step.rate !== null && Number.isFinite(step.rate) && step.rate > 0) {
    if (step.rateUnit === "realtime")
      return `${step.rate.toFixed(2)}× realtime`;
    const digits = step.rate >= 10 ? 1 : 2;
    return `${step.rate.toFixed(digits)} ${step.rateUnit}`;
  }
  const elapsed = inferenceStepElapsedSeconds(step, nowMs);
  if (step.status === "running" && elapsed >= 0.5 && step.fraction > 0) {
    return `${((step.fraction * 100) / elapsed).toFixed(1)}%/s`;
  }
  return step.status === "complete" ? "Completed" : "Measuring…";
}

function formatEta(step: InferenceProgressStep, nowMs: number): string {
  if (step.status === "pending") return "Waiting";
  if (step.status === "error") return "Stopped";
  const eta = inferenceStepEtaSeconds(step, nowMs);
  if (eta === 0) return "Done";
  if (eta === null) return "Estimating…";
  if (eta <= 1) return "Finishing…";
  return `About ${formatDuration(eta)}`;
}

function statusLabel(step: InferenceProgressStep): string {
  if (step.status === "complete") return "Done";
  if (step.status === "skipped") return "Skipped";
  if (step.status === "error") return "Error";
  if (step.status === "running") return "Running";
  return "Queued";
}

export function InferenceProgressPanel({
  steps,
  compact = false,
  wakeLockActive = false,
}: InferenceProgressPanelProps) {
  const [nowMs, setNowMs] = useState(() => performance.now());
  const running = steps.some((step) => step.status === "running");
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setNowMs(performance.now()), 500);
    return () => window.clearInterval(timer);
  }, [running]);

  const completeCount = completedInferenceStepCount(steps);
  const overallPercent = Math.round(overallInferenceProgress(steps) * 100);
  const activeStepIndex = steps.findIndex((step) => step.status === "running");
  const activeStep = activeStepIndex >= 0 ? steps[activeStepIndex] : null;
  const totalElapsed = useMemo(() => {
    const started = steps
      .map((step) => step.startedAtMs)
      .filter((value): value is number => value !== null);
    if (started.length === 0) return 0;
    const allFinished = steps.every(
      (step) =>
        step.status === "complete" ||
        step.status === "skipped" ||
        step.status === "error",
    );
    const finished = steps
      .map((step) => step.finishedAtMs)
      .filter((value): value is number => value !== null);
    const end =
      allFinished && finished.length > 0 ? Math.max(...finished) : nowMs;
    return Math.max(0, (end - Math.min(...started)) / 1000);
  }, [nowMs, steps]);

  if (steps.length === 0) return null;
  return (
    <section
      className={styles.panel}
      data-compact={compact || undefined}
      aria-live="polite"
    >
      <header className={styles.header}>
        <div>
          <p>
            LOCAL INFERENCE · {completeCount}/{steps.length} DONE
          </p>
          <strong>
            {activeStep
              ? `Step ${activeStepIndex + 1} of ${steps.length} · ${activeStep.detail}`
              : completeCount === steps.length
                ? "All inference steps complete"
                : "Preparing the next inference step"}
          </strong>
        </div>
        <output>{overallPercent}%</output>
      </header>

      <div className={styles.steps}>
        {steps.map((step, index) => {
          const stepPercent = Math.round(step.fraction * 100);
          return (
            <article
              key={step.id}
              className={styles.step}
              data-status={step.status}
            >
              <header>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{step.label}</strong>
                <em>{statusLabel(step)}</em>
              </header>
              <div
                className={styles.track}
                role="progressbar"
                aria-label={`${step.label} progress`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={stepPercent}
              >
                <i style={{ width: `${stepPercent}%` }} />
              </div>
              <div className={styles.metrics}>
                <span>
                  <b>Progress</b>
                  {stepPercent}%
                </span>
                <span>
                  <b>Speed</b>
                  {formatRate(step, nowMs)}
                </span>
                <span>
                  <b>ETA</b>
                  {formatEta(step, nowMs)}
                </span>
                <span>
                  <b>Elapsed</b>
                  {formatDuration(inferenceStepElapsedSeconds(step, nowMs))}
                </span>
              </div>
              <small>{step.detail}</small>
            </article>
          );
        })}
      </div>

      <footer>
        <span>Total elapsed · {formatDuration(totalElapsed)}</span>
        {wakeLockActive && <span>Screen wake lock active</span>}
      </footer>
    </section>
  );
}
