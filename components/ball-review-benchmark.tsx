"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useMemo, useState, type CSSProperties } from "react";

import {
  benchmarkConfigurationIds,
  type BallReviewBenchmarkBundle,
  type BenchmarkConfigurationId,
  type BenchmarkFrameAnnotation,
  type BenchmarkSimilarityMetrics,
} from "@/lib/ball-review-benchmark";
import styles from "./ball-review-benchmark.module.css";

type ViewMode = "grid" | "focus" | "overlay" | "metrics" | "heatmap";
type LayerChoice = BenchmarkConfigurationId | "reference";
type SimilarityKey =
  | "stateAccuracy"
  | "stateMacroF1"
  | "primaryPresenceF1"
  | "boxF1Iou25"
  | "boxF1Iou50"
  | "matchedPrimaryMeanIou"
  | "matchedPrimaryVisibilityAccuracyIou25"
  | "objectCountAccuracy";
type MetricSort =
  | "elapsedSeconds"
  | "speedupVsFreshSolXhigh"
  | "stateAccuracy"
  | "primaryPresenceF1"
  | "boxF1Iou25"
  | "matchedPrimaryMeanIou";

const configurationPresentation: Record<
  BenchmarkConfigurationId,
  { label: string; short: string; color: string; family: "sol" | "terra" | "luna" }
> = {
  "sol-low": { label: "Sol · low", short: "S-L", color: "#d64d2f", family: "sol" },
  "sol-medium": { label: "Sol · medium", short: "S-M", color: "#e37a25", family: "sol" },
  "sol-high": { label: "Sol · high", short: "S-H", color: "#d5a600", family: "sol" },
  "sol-xhigh": { label: "Sol · xhigh", short: "S-X", color: "#6b8f24", family: "sol" },
  "sol-max": { label: "Sol · max", short: "S-MX", color: "#26725e", family: "sol" },
  "terra-xhigh": { label: "Terra · xhigh", short: "T-X", color: "#277bb8", family: "terra" },
  "terra-max": { label: "Terra · max", short: "T-MX", color: "#565cc8", family: "terra" },
  "luna-xhigh": { label: "Luna · xhigh", short: "L-X", color: "#8c4fbd", family: "luna" },
  "luna-max": { label: "Luna · max", short: "L-MX", color: "#c13f88", family: "luna" },
};

const stateLabels: Record<BenchmarkFrameAnnotation["primaryBallState"], string> = {
  localizable: "Localizable",
  fully_occluded: "Fully occluded",
  out_of_frame: "Out of frame",
  indeterminate: "Indeterminate",
};

const similarityLabels: Record<SimilarityKey, string> = {
  stateAccuracy: "Pseudo-ref state agreement",
  stateMacroF1: "Active-state macro-F1",
  primaryPresenceF1: "Pseudo-ref presence F1",
  boxF1Iou25: "Pseudo-ref box F1 · IoU .25",
  boxF1Iou50: "Pseudo-ref box F1 · IoU .50",
  matchedPrimaryMeanIou: "Matched-primary mean IoU",
  matchedPrimaryVisibilityAccuracyIou25: "Matched visibility agreement",
  objectCountAccuracy: "Object-count agreement",
};

const viewLabels: Array<{ id: ViewMode; label: string; key: string }> = [
  { id: "overlay", label: "All overlays", key: "O" },
  { id: "grid", label: "9-up grid", key: "G" },
  { id: "focus", label: "A / B", key: "F" },
  { id: "metrics", label: "Metrics", key: "M" },
  { id: "heatmap", label: "Pairwise", key: "H" },
];

function formatPercent(value: number | null, digits = 0): string {
  return value === null ? "N/A" : `${(value * 100).toFixed(digits)}%`;
}

function formatSeconds(value: number): string {
  return value < 100 ? `${value.toFixed(1)}s` : `${Math.round(value)}s`;
}

function primaryObject(annotation: BenchmarkFrameAnnotation) {
  return annotation.objects.find((object) => object.role === "primary-court") ?? null;
}

function boxIou(
  first: BenchmarkFrameAnnotation["objects"][number]["bbox"],
  second: BenchmarkFrameAnnotation["objects"][number]["bbox"],
): number {
  const left = Math.max(first.x, second.x);
  const top = Math.max(first.y, second.y);
  const right = Math.min(first.x + first.width, second.x + second.width);
  const bottom = Math.min(first.y + first.height, second.y + second.height);
  const overlap = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = first.width * first.height + second.width * second.height - overlap;
  return union > 0 ? Math.max(0, Math.min(1, overlap / union)) : 0;
}

function centerError(
  first: BenchmarkFrameAnnotation["objects"][number]["bbox"],
  second: BenchmarkFrameAnnotation["objects"][number]["bbox"],
  width: number,
  height: number,
): number {
  const firstX = (first.x + first.width / 2) * width;
  const firstY = (first.y + first.height / 2) * height;
  const secondX = (second.x + second.width / 2) * width;
  const secondY = (second.y + second.height / 2) * height;
  return Math.hypot(firstX - secondX, firstY - secondY);
}

function compareFrameAnnotations(
  first: BenchmarkFrameAnnotation,
  second: BenchmarkFrameAnnotation,
  width: number,
  height: number,
) {
  const firstPrimary = primaryObject(first);
  const secondPrimary = primaryObject(second);
  return {
    stateAgreement: first.primaryBallState === second.primaryBallState,
    presenceAgreement:
      (first.primaryBallState === "localizable") ===
      (second.primaryBallState === "localizable"),
    primaryIou:
      firstPrimary && secondPrimary ? boxIou(firstPrimary.bbox, secondPrimary.bbox) : null,
    centerError:
      firstPrimary && secondPrimary
        ? centerError(firstPrimary.bbox, secondPrimary.bbox, width, height)
        : null,
    objectDelta: second.objects.length - first.objects.length,
  };
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      target.closest("a, button, input, select, textarea, summary, [role='button']") !== null)
  );
}

function layerLabel(choice: LayerChoice): string {
  return choice === "reference"
    ? "Prior Sol-xhigh pseudo-reference"
    : configurationPresentation[choice].label;
}

type OverlayLayer = {
  id: string;
  label: string;
  short: string;
  color: string;
  dashed?: boolean;
  annotation: BenchmarkFrameAnnotation;
};

function AnnotationStage({
  imageUrl,
  frameId,
  width,
  height,
  layers,
  zoom,
  showLabels = false,
  eager = false,
  highlightedLayerId = null,
}: {
  imageUrl: string;
  frameId: string;
  width: number;
  height: number;
  layers: OverlayLayer[];
  zoom: 1 | 2 | 4;
  showLabels?: boolean;
  eager?: boolean;
  highlightedLayerId?: string | null;
}) {
  const orderedLayers = highlightedLayerId
    ? [
        ...layers.filter((layer) => layer.id !== highlightedLayerId),
        ...layers.filter((layer) => layer.id === highlightedLayerId),
      ]
    : layers;
  return (
    <div className={styles.stageViewport} data-zoom={zoom}>
      <div
        className={styles.stage}
        style={{ width: `${zoom * 100}%`, minWidth: `${zoom * 100}%` }}
      >
        <Image
          src={imageUrl}
          width={width}
          height={height}
          sizes="(max-width: 760px) 96vw, (max-width: 1200px) 46vw, 31vw"
          unoptimized
          loading={eager ? "eager" : "lazy"}
          alt={`Blind benchmark frame ${frameId}`}
        />
        <svg
          viewBox={`0 0 ${width} ${height}`}
          aria-hidden="true"
          className={styles.boxOverlay}
        >
          {orderedLayers.flatMap((layer) =>
            layer.annotation.objects.map((object, objectIndex) => {
              const highlighted = highlightedLayerId === layer.id;
              const muted = highlightedLayerId !== null && !highlighted;
              const x = object.bbox.x * width;
              const y = object.bbox.y * height;
              const boxWidth = object.bbox.width * width;
              const boxHeight = object.bbox.height * height;
              const roleColor =
                object.role === "primary-court"
                  ? layer.color
                  : object.role === "other-court"
                    ? "#ff744f"
                    : "#54d7dd";
              const labelY = Math.max(12, y - 4);
              return (
                <g
                  key={`${layer.id}-${object.id}-${objectIndex}`}
                  opacity={muted ? 0.13 : 1}
                >
                  <rect
                    x={x}
                    y={y}
                    width={boxWidth}
                    height={boxHeight}
                    fill={`${roleColor}22`}
                    stroke={roleColor}
                    strokeWidth={
                      highlighted ? 4.5 : layer.dashed ? 2.8 : 2.2
                    }
                    strokeDasharray={layer.dashed ? "7 5" : undefined}
                    vectorEffect="non-scaling-stroke"
                  />
                  {showLabels && (!muted || highlightedLayerId === null) && (
                    <>
                      <rect
                        x={x}
                        y={labelY - 11}
                        width={Math.max(29, layer.short.length * 7 + 8)}
                        height={13}
                        fill="#10120f"
                        opacity={0.88}
                      />
                      <text
                        x={x + 4}
                        y={labelY - 1}
                        fill={roleColor}
                        fontSize={9}
                        fontFamily="DM Mono, monospace"
                      >
                        {layer.short}
                      </text>
                    </>
                  )}
                </g>
              );
            }),
          )}
        </svg>
      </div>
    </div>
  );
}

function AnnotationFacts({ annotation }: { annotation: BenchmarkFrameAnnotation }) {
  return (
    <dl className={styles.annotationFacts}>
      <div>
        <dt>State</dt>
        <dd>{stateLabels[annotation.primaryBallState]}</dd>
      </div>
      <div>
        <dt>Objects</dt>
        <dd>{annotation.objects.length}</dd>
      </div>
      <div>
        <dt>Roles</dt>
        <dd>
          {annotation.objects.length
            ? annotation.objects.map((object) => object.role).join(" · ")
            : "None"}
        </dd>
      </div>
      <div>
        <dt>Visibility</dt>
        <dd>
          {annotation.objects.length
            ? annotation.objects
                .map((object) =>
                  `${object.visibility}${object.truncated ? " (truncated)" : ""}`,
                )
                .join(" · ")
            : "None"}
        </dd>
      </div>
      <div>
        <dt>Boxes</dt>
        <dd>
          {annotation.objects.length
            ? annotation.objects
                .map(
                  (object) =>
                    `${Math.round(object.bbox.x * 1000) / 10}%,${Math.round(object.bbox.y * 1000) / 10}% · ${Math.round(object.bbox.width * 1000) / 10}%×${Math.round(object.bbox.height * 1000) / 10}%`,
                )
                .join(" · ")
            : "None"}
        </dd>
      </div>
    </dl>
  );
}

function RunTile({
  id,
  benchmark,
  frameIndex,
  zoom,
  showReference,
  eager,
  onFocus,
}: {
  id: BenchmarkConfigurationId;
  benchmark: BallReviewBenchmarkBundle;
  frameIndex: number;
  zoom: 1 | 2 | 4;
  showReference: boolean;
  eager: boolean;
  onFocus: () => void;
}) {
  const frame = benchmark.frames[frameIndex];
  const annotation = frame.outputs[id];
  const config = benchmark.configurations[id];
  const presentation = configurationPresentation[id];
  const layers: OverlayLayer[] = [
    {
      id,
      label: presentation.label,
      short: presentation.short,
      color: presentation.color,
      annotation,
    },
  ];
  if (showReference) {
    layers.push({
      id: "reference",
      label: "Prior pseudo-reference",
      short: "REF",
      color: "#ffffff",
      dashed: true,
      annotation: frame.reference,
    });
  }
  const frameComparison = compareFrameAnnotations(
    frame.reference,
    annotation,
    benchmark.width,
    benchmark.height,
  );

  return (
    <article
      className={styles.runTile}
      style={{ "--run-color": presentation.color } as CSSProperties}
    >
      <header className={styles.tileHeader}>
        <div>
          <span>{presentation.short}</span>
          <strong>{presentation.label}</strong>
        </div>
        <button onClick={onFocus} aria-label={`Focus ${presentation.label}`}>
          Compare
        </button>
      </header>
      <AnnotationStage
        imageUrl={frame.imageUrl}
        frameId={frame.frameId}
        width={benchmark.width}
        height={benchmark.height}
        layers={layers}
        zoom={zoom}
        eager={eager}
      />
      <div className={styles.tileState} data-state={annotation.primaryBallState}>
        <strong>{stateLabels[annotation.primaryBallState]}</strong>
        <span>{annotation.objects.length} object{annotation.objects.length === 1 ? "" : "s"}</span>
      </div>
      <div className={styles.tileSignals}>
        <span data-pass={frameComparison.stateAgreement}>
          Pseudo-ref state {frameComparison.stateAgreement ? "matches" : "differs"}
        </span>
        <span>{formatSeconds(config.elapsedSeconds)}</span>
        <span>{config.speedupVsFreshSolXhigh.toFixed(2)}×</span>
      </div>
      <details className={styles.notes}>
        <summary>Annotation details</summary>
        <AnnotationFacts annotation={annotation} />
        <p>{annotation.notes || "No note was supplied."}</p>
      </details>
    </article>
  );
}

function ScatterPlot({
  benchmark,
  metric,
}: {
  benchmark: BallReviewBenchmarkBundle;
  metric: SimilarityKey;
}) {
  const width = 900;
  const height = 330;
  const padding = { left: 58, right: 30, top: 28, bottom: 48 };
  const values = benchmarkConfigurationIds.map((id) => ({
    id,
    x: benchmark.configurations[id].speedupVsFreshSolXhigh,
    y: benchmark.configurations[id].similarityToPriorSolXhigh[metric] ?? 0,
  }));
  const maximumX = Math.max(...values.map((value) => value.x), 1);
  const scaleX = (value: number) =>
    padding.left + (value / maximumX) * (width - padding.left - padding.right);
  const scaleY = (value: number) =>
    height - padding.bottom - value * (height - padding.top - padding.bottom);

  return (
    <div className={styles.scatterWrap}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="scatter-title">
        <title id="scatter-title">
          Speedup versus {similarityLabels[metric]} relative to the prior pseudo-reference
        </title>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={scaleY(tick)}
              y2={scaleY(tick)}
            />
            <text x={padding.left - 10} y={scaleY(tick) + 4} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {[1, 2, 3, 4, 5].filter((tick) => tick <= maximumX + 0.15).map((tick) => (
          <text key={tick} x={scaleX(tick)} y={height - 18} textAnchor="middle">
            {tick}×
          </text>
        ))}
        {values.map(({ id, x, y }) => {
          const presentation = configurationPresentation[id];
          return (
            <g key={id} transform={`translate(${scaleX(x)} ${scaleY(y)})`}>
              <circle r={9} fill={presentation.color} stroke="#161813" strokeWidth={2} />
              <text x={12} y={4} className={styles.pointLabel}>
                {presentation.short}
              </text>
            </g>
          );
        })}
        <text x={width / 2} y={height - 2} textAnchor="middle" className={styles.axisLabel}>
          Speedup versus fresh Sol xhigh →
        </text>
        <text
          x={14}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
          className={styles.axisLabel}
        >
          {similarityLabels[metric]} →
        </text>
      </svg>
    </div>
  );
}

export function BallReviewBenchmark({
  benchmark,
}: {
  benchmark: BallReviewBenchmarkBundle;
}) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [view, setView] = useState<ViewMode>("overlay");
  const [zoom, setZoom] = useState<1 | 2 | 4>(1);
  const [showReference, setShowReference] = useState(false);
  const [focusA, setFocusA] = useState<LayerChoice>("sol-high");
  const [focusB, setFocusB] = useState<LayerChoice>("sol-xhigh");
  const [overlayIds, setOverlayIds] = useState<BenchmarkConfigurationId[]>([
    ...benchmarkConfigurationIds,
  ]);
  const [highlightedOverlayId, setHighlightedOverlayId] = useState<
    BenchmarkConfigurationId | "reference" | null
  >(null);
  const [metricSort, setMetricSort] = useState<MetricSort>("elapsedSeconds");
  const [scatterMetric, setScatterMetric] = useState<SimilarityKey>("boxF1Iou25");
  const [heatMetric, setHeatMetric] = useState<SimilarityKey>("stateAccuracy");

  const frame = benchmark.frames[frameIndex];
  const fastestId = benchmarkConfigurationIds.reduce((fastest, candidate) =>
    benchmark.configurations[candidate].elapsedSeconds <
    benchmark.configurations[fastest].elapsedSeconds
      ? candidate
      : fastest,
  );
  const fastest = benchmark.configurations[fastestId];
  const recommendationId: BenchmarkConfigurationId = "sol-high";
  const recommendation = benchmark.configurations[recommendationId];

  const stateVotes = useMemo(() => {
    const votes = new Map<BenchmarkFrameAnnotation["primaryBallState"], number>();
    benchmarkConfigurationIds.forEach((id) => {
      const state = frame.outputs[id].primaryBallState;
      votes.set(state, (votes.get(state) ?? 0) + 1);
    });
    return [...votes.entries()].sort((left, right) => right[1] - left[1]);
  }, [frame]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (isInteractiveTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key.toLowerCase();
      if (key === "arrowleft") {
        event.preventDefault();
        setFrameIndex((current) => Math.max(0, current - 1));
      } else if (key === "arrowright") {
        event.preventDefault();
        setFrameIndex((current) => Math.min(benchmark.frames.length - 1, current + 1));
      } else if (key === "g") setView("grid");
      else if (key === "f") setView("focus");
      else if (key === "o") setView("overlay");
      else if (key === "m") setView("metrics");
      else if (key === "h") setView("heatmap");
      else if (key === "r") setShowReference((current) => !current);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [benchmark.frames.length]);

  const sortedConfigurations = useMemo(() => {
    return [...benchmarkConfigurationIds].sort((left, right) => {
      const leftConfig = benchmark.configurations[left];
      const rightConfig = benchmark.configurations[right];
      if (metricSort === "elapsedSeconds") {
        return leftConfig.elapsedSeconds - rightConfig.elapsedSeconds;
      }
      if (metricSort === "speedupVsFreshSolXhigh") {
        return rightConfig.speedupVsFreshSolXhigh - leftConfig.speedupVsFreshSolXhigh;
      }
      return (
        (rightConfig.similarityToPriorSolXhigh[metricSort] ?? -1) -
        (leftConfig.similarityToPriorSolXhigh[metricSort] ?? -1)
      );
    });
  }, [benchmark.configurations, metricSort]);

  function annotationFor(choice: LayerChoice): BenchmarkFrameAnnotation {
    return choice === "reference" ? frame.reference : frame.outputs[choice];
  }

  function layerFor(choice: LayerChoice, dashed = false): OverlayLayer {
    if (choice === "reference") {
      return {
        id: "reference",
        label: "Prior pseudo-reference",
        short: "REF",
        color: "#ffffff",
        dashed: true,
        annotation: frame.reference,
      };
    }
    const presentation = configurationPresentation[choice];
    return {
      id: choice,
      label: presentation.label,
      short: presentation.short,
      color: presentation.color,
      dashed,
      annotation: frame.outputs[choice],
    };
  }

  function focusPair(first: LayerChoice, second: LayerChoice) {
    setFocusA(first);
    setFocusB(second);
    setView("focus");
  }

  function toggleOverlay(id: BenchmarkConfigurationId) {
    setOverlayIds((current) => {
      if (current.includes(id)) return current.filter((candidate) => candidate !== id);
      return [...current, id];
    });
    if (highlightedOverlayId === id) setHighlightedOverlayId(null);
  }

  const focusComparison = compareFrameAnnotations(
    annotationFor(focusA),
    annotationFor(focusB),
    benchmark.width,
    benchmark.height,
  );

  const focusOverall: BenchmarkSimilarityMetrics | null =
    focusA === "reference" && focusB !== "reference"
      ? benchmark.configurations[focusB].similarityToPriorSolXhigh
      : focusB === "reference" && focusA !== "reference"
        ? benchmark.configurations[focusA].similarityToPriorSolXhigh
        : focusA !== "reference" && focusB !== "reference"
          ? benchmark.pairwise[focusA][focusB]
          : null;
  const visibleOverlayIds =
    highlightedOverlayId &&
    highlightedOverlayId !== "reference" &&
    !overlayIds.includes(highlightedOverlayId)
      ? [...overlayIds, highlightedOverlayId]
      : overlayIds;

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>BLIND MODEL / EFFORT SCREEN</p>
          <h1>Nine ways to see the ball</h1>
          <p>
            Compare every box, state, note, latency, and pairwise signal on the same
            twelve frames.
          </p>
        </div>
        <nav>
          <Link href="/label/ball">Human review</Link>
          <Link href="/">Review dashboard</Link>
        </nav>
      </header>

      <section className={styles.caveat}>
        <strong>Pseudo-reference, not truth.</strong>
        <span>
          One blinded run per configuration · independent stills · prior contextual
          Sol-xhigh labels are shown only as a comparison layer.
        </span>
        <code>{benchmark.reportSha256.slice(0, 12)}</code>
      </section>

      <section className={styles.summaryCards} aria-label="Benchmark summary">
        <article>
          <span>Fastest</span>
          <strong>{configurationPresentation[fastestId].label}</strong>
          <b>{formatSeconds(fastest.elapsedSeconds)} · {fastest.speedupVsFreshSolXhigh.toFixed(2)}×</b>
        </article>
        <article data-accent="recommended">
          <span>Confirmation candidate</span>
          <strong>{configurationPresentation[recommendationId].label}</strong>
          <b>{formatSeconds(recommendation.elapsedSeconds)} · {recommendation.speedupVsFreshSolXhigh.toFixed(2)}×</b>
        </article>
        <article>
          <span>Sample</span>
          <strong>{benchmark.frames.length} frames</strong>
          <b>{benchmark.sample.recordingCount} recordings · 3 environments</b>
        </article>
        <article>
          <span>Current frame consensus</span>
          <strong>{stateLabels[stateVotes[0][0]]}</strong>
          <b>{stateVotes[0][1]} of 9 runs</b>
        </article>
      </section>

      <section className={styles.controlBar}>
        <div className={styles.viewTabs} role="toolbar" aria-label="Comparison view">
          {viewLabels.map((candidate) => (
            <button
              key={candidate.id}
              type="button"
              aria-pressed={view === candidate.id}
              className={view === candidate.id ? styles.activeTab : ""}
              onClick={() => setView(candidate.id)}
            >
              {candidate.label} <kbd>{candidate.key}</kbd>
            </button>
          ))}
        </div>
        <label>
          Zoom
          <select value={zoom} onChange={(event) => setZoom(Number(event.target.value) as 1 | 2 | 4)}>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
        <label className={styles.referenceToggle}>
          <input
            type="checkbox"
            checked={showReference}
            onChange={(event) => setShowReference(event.target.checked)}
          />
          Dashed pseudo-reference <kbd>R</kbd>
        </label>
      </section>

      <section className={styles.frameRail} aria-label="Benchmark frames">
        <button
          className={styles.railStep}
          disabled={frameIndex === 0}
          onClick={() => setFrameIndex((current) => Math.max(0, current - 1))}
          aria-label="Previous frame"
        >
          ←
        </button>
        <div>
          {benchmark.frames.map((candidate, index) => {
            const matchingStates = benchmarkConfigurationIds.filter(
              (id) =>
                candidate.outputs[id].primaryBallState ===
                candidate.reference.primaryBallState,
            ).length;
            return (
              <button
                key={candidate.frameId}
                className={index === frameIndex ? styles.activeFrame : ""}
                onClick={() => setFrameIndex(index)}
                aria-current={index === frameIndex ? "true" : undefined}
                title={`${candidate.frameId}: ${matchingStates}/9 states match pseudo-reference`}
              >
                <span>{String(candidate.attachmentIndex).padStart(2, "0")}</span>
                <small>{matchingStates}/9</small>
              </button>
            );
          })}
        </div>
        <button
          className={styles.railStep}
          disabled={frameIndex === benchmark.frames.length - 1}
          onClick={() =>
            setFrameIndex((current) => Math.min(benchmark.frames.length - 1, current + 1))
          }
          aria-label="Next frame"
        >
          →
        </button>
      </section>
      <p className={styles.railLegend}>
        Frame rail counts show runs whose state agrees with the prior Sol-xhigh
        pseudo-reference; they are not human-label accuracy.
      </p>

      <div className={styles.frameHeading} aria-live="polite">
        <div>
          <span>FRAME {frameIndex + 1} / {benchmark.frames.length}</span>
          <strong>{frame.frameId}</strong>
        </div>
        <div className={styles.voteLine}>
          {stateVotes.map(([state, count]) => (
            <span key={state} data-state={state}>
              {stateLabels[state]} <b>{count}</b>
            </span>
          ))}
        </div>
      </div>

      {view === "grid" && (
        <section className={styles.runGrid} aria-label="All nine benchmark outputs">
          {benchmarkConfigurationIds.map((id, index) => (
            <RunTile
              key={id}
              id={id}
              benchmark={benchmark}
              frameIndex={frameIndex}
              zoom={zoom}
              showReference={showReference}
              eager={index < 3}
              onFocus={() => focusPair(id, "reference")}
            />
          ))}
        </section>
      )}

      {view === "focus" && (
        <section className={styles.focusView}>
          <div className={styles.focusSelectors}>
            {([focusA, focusB] as const).map((choice, index) => (
              <label key={index}>
                {index === 0 ? "A" : "B"}
                <select
                  value={choice}
                  onChange={(event) =>
                    (index === 0 ? setFocusA : setFocusB)(event.target.value as LayerChoice)
                  }
                >
                  <option value="reference">Prior pseudo-reference</option>
                  {benchmarkConfigurationIds.map((id) => (
                    <option key={id} value={id}>{configurationPresentation[id].label}</option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className={styles.focusGrid}>
            {[focusA, focusB].map((choice, index) => (
              <article key={`${index}-${choice}`} className={styles.focusCard}>
                <h2>{layerLabel(choice)}</h2>
                <AnnotationStage
                  imageUrl={frame.imageUrl}
                  frameId={frame.frameId}
                  width={benchmark.width}
                  height={benchmark.height}
                  layers={[layerFor(choice)]}
                  zoom={zoom}
                  showLabels
                  eager
                />
                <AnnotationFacts annotation={annotationFor(choice)} />
                <p>{annotationFor(choice).notes || "No note was supplied."}</p>
              </article>
            ))}
          </div>
          <article className={styles.differenceStrip}>
            <div>
              <span>THIS FRAME</span>
              <strong>{focusComparison.stateAgreement ? "Same state" : "Different state"}</strong>
            </div>
            <dl>
              <div><dt>Presence</dt><dd>{focusComparison.presenceAgreement ? "Agrees" : "Differs"}</dd></div>
              <div><dt>Primary IoU</dt><dd>{focusComparison.primaryIou === null ? "N/A" : focusComparison.primaryIou.toFixed(3)}</dd></div>
              <div><dt>Center error</dt><dd>{focusComparison.centerError === null ? "N/A" : `${focusComparison.centerError.toFixed(1)}px`}</dd></div>
              <div><dt>Object Δ B−A</dt><dd>{focusComparison.objectDelta > 0 ? "+" : ""}{focusComparison.objectDelta}</dd></div>
            </dl>
            {focusOverall && (
              <dl>
                <div><dt>12-frame state agreement</dt><dd>{formatPercent(focusOverall.stateAccuracy)}</dd></div>
                <div><dt>12-frame presence F1</dt><dd>{formatPercent(focusOverall.primaryPresenceF1)}</dd></div>
                <div><dt>12-frame box F1 @ .25</dt><dd>{formatPercent(focusOverall.boxF1Iou25)}</dd></div>
                <div><dt>12-frame box F1 @ .50</dt><dd>{formatPercent(focusOverall.boxF1Iou50)}</dd></div>
              </dl>
            )}
          </article>
        </section>
      )}

      {view === "overlay" && (
        <section className={styles.overlayView}>
          <div className={styles.overlayPicker}>
            <div>
              <strong>All nine label sets</strong>
              <span>
                Toggle any run; hover or focus one to emphasize its boxes and dim the rest.
              </span>
              <span className={styles.overlayBulkActions}>
                <button
                  type="button"
                  onClick={() => setOverlayIds([...benchmarkConfigurationIds])}
                >
                  Show all
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setOverlayIds([]);
                    setHighlightedOverlayId(null);
                  }}
                >
                  Hide all
                </button>
              </span>
            </div>
            {benchmarkConfigurationIds.map((id) => (
              <label
                key={id}
                data-highlighted={highlightedOverlayId === id}
                style={{ "--run-color": configurationPresentation[id].color } as CSSProperties}
                onMouseEnter={() => setHighlightedOverlayId(id)}
                onMouseLeave={() => setHighlightedOverlayId(null)}
                onFocus={() => setHighlightedOverlayId(id)}
                onBlur={() => setHighlightedOverlayId(null)}
              >
                <input
                  type="checkbox"
                  checked={overlayIds.includes(id)}
                  onChange={() => toggleOverlay(id)}
                />
                {configurationPresentation[id].label}
              </label>
            ))}
          </div>
          <div className={styles.overlayWorkspace}>
            <AnnotationStage
              imageUrl={frame.imageUrl}
              frameId={frame.frameId}
              width={benchmark.width}
              height={benchmark.height}
              layers={[
                ...visibleOverlayIds.map((id) => layerFor(id)),
                ...(showReference ? [layerFor("reference")] : []),
              ]}
              zoom={zoom}
              showLabels
              eager
              highlightedLayerId={highlightedOverlayId}
            />
            <div className={styles.overlayNotes}>
              {overlayIds.map((id) => (
                <article
                  key={id}
                  tabIndex={0}
                  data-highlighted={highlightedOverlayId === id}
                  style={{ "--run-color": configurationPresentation[id].color } as CSSProperties}
                  onMouseEnter={() => setHighlightedOverlayId(id)}
                  onMouseLeave={() => setHighlightedOverlayId(null)}
                  onFocus={() => setHighlightedOverlayId(id)}
                  onBlur={() => setHighlightedOverlayId(null)}
                >
                  <strong>{configurationPresentation[id].label}</strong>
                  <span>{stateLabels[frame.outputs[id].primaryBallState]} · {frame.outputs[id].objects.length} objects</span>
                  <p>{frame.outputs[id].notes || "No note."}</p>
                </article>
              ))}
              {showReference && (
                <article
                  data-reference="true"
                  data-highlighted={highlightedOverlayId === "reference"}
                  tabIndex={0}
                  onMouseEnter={() => setHighlightedOverlayId("reference")}
                  onMouseLeave={() => setHighlightedOverlayId(null)}
                  onFocus={() => setHighlightedOverlayId("reference")}
                  onBlur={() => setHighlightedOverlayId(null)}
                >
                  <strong>Prior pseudo-reference</strong>
                  <span>{stateLabels[frame.reference.primaryBallState]} · {frame.reference.objects.length} objects</span>
                  <p>{frame.reference.notes || "No note."}</p>
                </article>
              )}
            </div>
          </div>
        </section>
      )}

      {view === "metrics" && (
        <section className={styles.metricsView}>
          <h2>All 12 frames · agreement with the Sol pseudo-reference</h2>
          <div className={styles.metricToolbar}>
            <label>
              Sort table
              <select value={metricSort} onChange={(event) => setMetricSort(event.target.value as MetricSort)}>
                <option value="elapsedSeconds">Fastest</option>
                <option value="speedupVsFreshSolXhigh">Largest speedup</option>
                <option value="stateAccuracy">Pseudo-ref state agreement</option>
                <option value="primaryPresenceF1">Pseudo-ref presence F1</option>
                <option value="boxF1Iou25">Pseudo-ref box F1 @ .25</option>
                <option value="matchedPrimaryMeanIou">Matched-primary mean IoU</option>
              </select>
            </label>
            <label>
              Scatter quality axis
              <select value={scatterMetric} onChange={(event) => setScatterMetric(event.target.value as SimilarityKey)}>
                {(Object.keys(similarityLabels) as SimilarityKey[]).map((metric) => (
                  <option key={metric} value={metric}>{similarityLabels[metric]}</option>
                ))}
              </select>
            </label>
          </div>
          <div className={styles.tableWrap}>
            <table>
              <caption>
                Timings and agreement metrics over all 12 frames. Quality columns
                compare with the prior Sol-xhigh pseudo-reference, not human truth.
              </caption>
              <thead>
                <tr>
                  <th>Run</th><th>Time</th><th>Sec/frame</th><th>Speedup</th><th>Reasoning</th>
                  <th>Ref state</th><th>Ref presence</th><th>Ref box .25</th><th>Ref box .50</th>
                  <th>Matches</th><th>Mean IoU</th><th>Center px</th><th>Visibility</th><th>Objects</th>
                </tr>
              </thead>
              <tbody>
                {sortedConfigurations.map((id) => {
                  const config = benchmark.configurations[id];
                  const score = config.similarityToPriorSolXhigh;
                  return (
                    <tr key={id}>
                      <th>
                        <button onClick={() => focusPair(id, "reference")}>
                          <i style={{ background: configurationPresentation[id].color }} />
                          {configurationPresentation[id].label}
                        </button>
                      </th>
                      <td>{formatSeconds(config.elapsedSeconds)}</td>
                      <td>{(config.elapsedSeconds / benchmark.frames.length).toFixed(2)}</td>
                      <td>{config.speedupVsFreshSolXhigh.toFixed(2)}×</td>
                      <td>{config.usage.reasoning_output_tokens.toLocaleString()}</td>
                      <td>{formatPercent(score.stateAccuracy)}</td>
                      <td>{formatPercent(score.primaryPresenceF1)}</td>
                      <td>{formatPercent(score.boxF1Iou25)}</td>
                      <td>{formatPercent(score.boxF1Iou50)}</td>
                      <td>{score.matchedPrimaryCount}</td>
                      <td>{formatPercent(score.matchedPrimaryMeanIou)}</td>
                      <td>{score.matchedPrimaryMedianCenterErrorPixels === null ? "N/A" : score.matchedPrimaryMedianCenterErrorPixels.toFixed(1)}</td>
                      <td>{formatPercent(score.matchedPrimaryVisibilityAccuracyIou25)}</td>
                      <td>{formatPercent(score.objectCountAccuracy)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <ScatterPlot benchmark={benchmark} metric={scatterMetric} />
          <p className={styles.metricFootnote}>{benchmark.metricSemantics.similarityToPriorSolXhigh}</p>
        </section>
      )}

      {view === "heatmap" && (
        <section className={styles.heatmapView}>
          <h2>All 12 frames · run-to-run agreement</h2>
          <div className={styles.metricToolbar}>
            <label>
              Pairwise signal
              <select value={heatMetric} onChange={(event) => setHeatMetric(event.target.value as SimilarityKey)}>
                {(Object.keys(similarityLabels) as SimilarityKey[]).map((metric) => (
                  <option key={metric} value={metric}>{similarityLabels[metric]}</option>
                ))}
              </select>
            </label>
            <p>Click a cell to open that pair in A/B view.</p>
          </div>
          <div className={styles.heatmapScroll}>
            <table className={styles.heatmap}>
              <caption>
                Symmetric pairwise agreement between benchmark runs; no human truth
                is used.
              </caption>
              <thead>
                <tr>
                  <th>A \ B</th>
                  {benchmarkConfigurationIds.map((id) => <th key={id}>{configurationPresentation[id].short}</th>)}
                </tr>
              </thead>
              <tbody>
                {benchmarkConfigurationIds.map((left) => (
                  <tr key={left}>
                    <th>{configurationPresentation[left].short}</th>
                    {benchmarkConfigurationIds.map((right) => {
                      const value = benchmark.pairwise[left][right][heatMetric] ?? 0;
                      return (
                        <td key={right}>
                          <button
                            style={{ "--heat": value } as CSSProperties}
                            onClick={() => focusPair(left, right)}
                            aria-label={`${configurationPresentation[left].label} versus ${configurationPresentation[right].label}: ${similarityLabels[heatMetric]} ${formatPercent(value)}`}
                          >
                            {value.toFixed(2)}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.metricFootnote}>{benchmark.metricSemantics.pairwise}</p>
        </section>
      )}

      <footer className={styles.footer}>
        <div>
          <span>LIMITATIONS</span>
          <ul>{benchmark.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
        </div>
        <p><kbd>←</kbd><kbd>→</kbd> frame · <kbd>G</kbd><kbd>F</kbd><kbd>O</kbd><kbd>M</kbd><kbd>H</kbd> view · <kbd>R</kbd> reference</p>
      </footer>
    </main>
  );
}
