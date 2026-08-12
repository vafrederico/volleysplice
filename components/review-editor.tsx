"use client";

import Link from "next/link";
import { useMemo, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { RallyTimeline, type TimelineTrack } from "@/components/rally-timeline";
import type {
  AnalysisKind,
  AnalysisOption,
  ReviewAnalysis,
  ReviewVideoOption,
} from "@/lib/analysis-types";
import { buildEditList, formatTime, type Rally } from "@/lib/edit-list";

const demoRallies: Rally[] = [
  { id: "R01", start: 24, end: 37, confidence: 0.78, included: true },
  { id: "R02", start: 52, end: 71, confidence: 0.74, included: true },
  { id: "R03", start: 91, end: 104, confidence: 0.57, included: true },
];

const demoAnalysis: ReviewAnalysis = {
  id: "demo",
  recordingId: "demo",
  title: "Example analysis",
  variantLabel: "Demo",
  variantDescription: null,
  kind: "unknown",
  method: "demo",
  modelVersion: null,
  datasetRole: "not-applicable",
  datasetRoleLabel: "Not applicable",
  duration: 150,
  width: 16,
  height: 9,
  sourceFilename: "Add a video to begin",
  videoUrl: null,
  courtPreviewUrl: null,
  courtConfidence: 0,
  courtSource: "demo",
  courtLines: [],
  cameraStability: 1,
  warnings: ["This is sample data. Run an analyzer to review a real recording."],
  rallies: demoRallies,
};

type ReviewEditorProps = {
  initialAnalysis: ReviewAnalysis | null;
  analysisOptions: AnalysisOption[];
  videoOptions: ReviewVideoOption[];
  comparisonAnalyses: ReviewAnalysis[];
  initialTime: number;
};

function tone(kind: AnalysisKind): "model" | "heuristic" | "sol" | "gold" {
  if (kind === "model" || kind === "sol" || kind === "gold") return kind;
  return "heuristic";
}

function preferredAnalysis(video: ReviewVideoOption): AnalysisOption | undefined {
  return (
    video.analyses.find(
      (analysis) =>
        analysis.id === `model-full-percentile-v1--${video.id}`,
    ) ??
    video.analyses.find(
      (analysis) =>
        analysis.kind === "model" && analysis.modelVersion === "full-percentile-v1",
    ) ??
    video.analyses.find((analysis) => analysis.kind === "model") ??
    video.analyses.find(
      (analysis) => analysis.kind === "heuristic" && analysis.id.endsWith("-v2"),
    ) ??
    video.analyses.find((analysis) => analysis.kind === "heuristic") ??
    video.analyses[0]
  );
}

function analysisDescription(kind: AnalysisKind): string {
  if (kind === "model") {
    return "The trained temporal classifier produced these live-play intervals. Compare them against the verified reference and other analyzers below.";
  }
  if (kind === "heuristic") {
    return "Court motion and supporting audio produced these no-model activity suggestions. Compare them against the trained model and verified reference.";
  }
  if (kind === "sol") {
    return "Blind audiovisual frame and audio review produced these Sol candidates before continuous human verification.";
  }
  if (kind === "gold") {
    return "These are the continuously reviewed serve-contact-to-dead-ball reference intervals used for training and evaluation.";
  }
  return "Review the selected rally suggestions against the shared timeline.";
}

export function ReviewEditor({
  initialAnalysis,
  analysisOptions,
  videoOptions,
  comparisonAnalyses,
  initialTime,
}: ReviewEditorProps) {
  const analysis = initialAnalysis ?? demoAnalysis;
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [rallies, setRallies] = useState(analysis.rallies);
  const [selectedId, setSelectedId] = useState(analysis.rallies[0]?.id ?? "");
  const [preRoll, setPreRoll] = useState(3);
  const [postRoll, setPostRoll] = useState(2);
  const [playbackTime, setPlaybackTime] = useState(initialTime);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const intervals = useMemo(
    () => buildEditList(rallies, preRoll, postRoll, analysis.duration),
    [rallies, preRoll, postRoll, analysis.duration],
  );
  const selected = rallies.find((rally) => rally.id === selectedId) ?? rallies[0] ?? null;
  const keptSeconds = intervals.reduce(
    (total, interval) => total + interval.keptEnd - interval.keptStart,
    0,
  );
  const keptCount = rallies.filter((rally) => rally.included).length;
  const currentVideo =
    videoOptions.find((video) => video.id === analysis.recordingId) ?? null;
  const analysisIndex = analysisOptions.findIndex((option) => option.id === analysis.id);
  const tracks = useMemo<TimelineTrack[]>(
    () =>
      comparisonAnalyses.map((candidate) => {
        const trackRallies = candidate.id === analysis.id ? rallies : candidate.rallies;
        return {
          id: candidate.id,
          label: candidate.variantLabel,
          title: candidate.variantDescription ?? undefined,
          detail: `${candidate.datasetRoleLabel} · ${trackRallies.length} rallies`,
          active: candidate.id === analysis.id,
          intervals: trackRallies.map((rally) => ({
            id: rally.id,
            start: rally.start,
            end: rally.end,
            confidence: rally.confidence,
            tone: tone(candidate.kind),
            title: `${candidate.variantLabel} · ${rally.id} · ${formatTime(rally.start)}–${formatTime(rally.end)}`,
          })),
        };
      }),
    [analysis.id, comparisonAnalyses, rallies],
  );

  function navigate(videoId: string, analysisId: string, time = 0) {
    const query = new URLSearchParams({ video: videoId, analysis: analysisId });
    if (time > 0) query.set("time", time.toFixed(3));
    startTransition(() => router.push(`/?${query.toString()}`));
  }

  function selectVideo(id: string) {
    const video = videoOptions.find((candidate) => candidate.id === id);
    const next = video ? preferredAnalysis(video) : undefined;
    if (video && next) navigate(video.id, next.id);
  }

  function selectAnalysis(id: string, time = playbackTime) {
    if (!currentVideo || !id || id === analysis.id) return;
    navigate(currentVideo.id, id, time);
  }

  function seekTo(time: number, intervalId?: string) {
    const clamped = Math.max(0, Math.min(analysis.duration, time));
    setPlaybackTime(clamped);
    if (intervalId) setSelectedId(intervalId);
    if (videoRef.current) videoRef.current.currentTime = clamped;
  }

  function toggleRally(id: string) {
    setRallies((current) =>
      current.map((rally) =>
        rally.id === id ? { ...rally, included: !rally.included } : rally,
      ),
    );
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) await video.play();
    else video.pause();
  }

  const confidenceLabel =
    analysis.kind === "gold"
      ? "VERIFICATION"
      : analysis.kind === "model"
        ? "MODEL SCORE"
        : analysis.kind === "sol"
          ? "SOL CONFIDENCE"
          : "HEURISTIC CONFIDENCE";

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#">VOLLEYCUT <span>LAB</span></a>
        <div className="top-actions">
          <Link href="/label">Open labeling station →</Link>
          <div className="project-state">
            <i /> {initialAnalysis ? `${videoOptions.length} videos ready` : "Demo mode"}
          </div>
        </div>
      </header>

      {videoOptions.length > 0 && (
        <nav className="analysis-picker" aria-label="Video and analysis selection">
          <div>
            <span className="picker-kicker">COMPARISON DATASET</span>
            <strong>Choose video and analyzer</strong>
          </div>
          <div className="picker-fields">
            <label>
              <span>Video</span>
              <select
                aria-label="Video"
                value={analysis.recordingId}
                disabled={isPending}
                onChange={(event) => selectVideo(event.target.value)}
              >
                {videoOptions.map((video) => (
                  <option key={video.id} value={video.id}>
                    {video.environment} · {video.title}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Analysis source</span>
              <select
                aria-label="Analysis source"
                title={analysis.variantDescription ?? undefined}
                value={analysis.id}
                disabled={isPending}
                onChange={(event) => selectAnalysis(event.target.value)}
              >
                {analysisOptions.map((option) => (
                  <option
                    key={option.id}
                    value={option.id}
                    title={option.variantDescription ?? undefined}
                  >
                    {option.variantLabel} · {option.datasetRoleLabel} · {option.rallyCount}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="analysis-position" aria-live="polite">
            <strong>{Math.max(0, analysisIndex) + 1}</strong>
            <span>/ {analysisOptions.length}</span>
            <small>{isPending ? "LOADING…" : analysis.recordingId}</small>
          </div>
        </nav>
      )}

      <section className="hero">
        <div>
          <div className="analysis-badges">
            <span
              data-kind={analysis.kind}
              title={analysis.variantDescription ?? undefined}
            >
              {analysis.variantLabel}
            </span>
            <span data-role={analysis.datasetRole}>{analysis.datasetRoleLabel}</span>
          </div>
          <h1>Find the rallies.<br /><em>Compare the evidence.</em></h1>
          <p className="intro">{analysisDescription(analysis.kind)}</p>
          <p className="source-name">SOURCE / {analysis.sourceFilename}</p>
        </div>
        <div className="scorecard">
          <span>RALLIES KEPT</span>
          <strong>{keptCount}<small> / {rallies.length}</small></strong>
          <div className="progress">
            <b style={{ width: `${rallies.length ? (keptCount / rallies.length) * 100 : 0}%` }} />
          </div>
          <p>{formatTime(keptSeconds)} kept from {formatTime(analysis.duration)}</p>
        </div>
      </section>

      {analysis.warnings.length > 0 && (
        <section className="warnings" aria-label="Analysis warnings">
          {analysis.warnings.map((warning) => <p key={warning}>⚑ {warning}</p>)}
        </section>
      )}

      <section className="workspace">
        <div className="viewer">
          <div
            className={`court video-stage ${analysis.videoUrl ? "has-video" : ""}`}
            style={{ aspectRatio: `${analysis.width} / ${analysis.height}` }}
          >
            {analysis.videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={analysis.videoUrl}
                  preload="metadata"
                  controls
                  onLoadedMetadata={(event) => {
                    event.currentTarget.currentTime = initialTime;
                    setPlaybackTime(initialTime);
                  }}
                  onTimeUpdate={(event) => setPlaybackTime(event.currentTarget.currentTime)}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
                <svg className="court-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                  {analysis.courtLines.map((line, index) => (
                    <line key={index} x1={line.x1 * 100} y1={line.y1 * 100} x2={line.x2 * 100} y2={line.y2 * 100} />
                  ))}
                </svg>
              </>
            ) : (
              <div className="court-lines"><span /><span /><span /></div>
            )}
            <button
              className={`play ${isPlaying ? "playing" : ""}`}
              aria-label={isPlaying ? "Pause preview" : "Play preview"}
              onClick={togglePlayback}
              disabled={!analysis.videoUrl}
            >
              {isPlaying ? "Ⅱ" : "▶"}
            </button>
            <div className="timecode">{formatTime(playbackTime)} <span>/ {formatTime(analysis.duration)}</span></div>
            <div className="court-status">
              COURT / {analysis.courtSource} / {analysis.courtSource === "manual-roi"
                ? "PROVIDED"
                : `${Math.round(analysis.courtConfidence * 100)}%`}
            </div>
          </div>

          <div className="comparison-heading">
            <div>
              <strong>All analysis tracks</strong>
              <span>Same video clock · vertical line is the current playhead</span>
            </div>
            <span>{tracks.length} sources</span>
          </div>
          <RallyTimeline
            duration={analysis.duration}
            currentTime={playbackTime}
            tracks={tracks}
            selectedTrackId={analysis.id}
            selectedIntervalId={selectedId}
            onTrackSelect={(trackId) => selectAnalysis(trackId)}
            onSeek={(time, trackId, intervalId) => {
              if (trackId !== analysis.id) {
                if (currentVideo) navigate(currentVideo.id, trackId, time);
                return;
              }
              seekTo(time, intervalId);
            }}
            ariaLabel="Comparison of verified, heuristic, Sol, and trained-model rallies"
          />
        </div>

        <aside>
          {selected ? (
            <>
              <div className="aside-heading"><span>SELECTED RALLY</span><strong>{selected.id}</strong></div>
              <div className={`confidence ${selected.confidence < 0.7 ? "warn" : ""}`}>
                <span>{confidenceLabel}</span>
                <strong>{analysis.kind === "gold" ? "VERIFIED" : `${Math.round(selected.confidence * 100)}%`}</strong>
              </div>
              <dl>
                <div><dt>Suggested start</dt><dd>{formatTime(selected.start)}</dd></div>
                <div><dt>Suggested end</dt><dd>{formatTime(selected.end)}</dd></div>
                <div><dt>Core duration</dt><dd>{(selected.end - selected.start).toFixed(1)}s</dd></div>
                <div><dt>Dataset role</dt><dd>{analysis.datasetRoleLabel}</dd></div>
                <div><dt>Method</dt><dd>{analysis.method}</dd></div>
              </dl>
              <button className="include" onClick={() => toggleRally(selected.id)}>
                {selected.included ? "✓ Included in export" : "+ Restore to export"}
              </button>
              {analysis.courtPreviewUrl && (
                <a className="diagnostic-link" href={analysis.courtPreviewUrl} target="_blank">
                  Open court diagnostic ↗
                </a>
              )}
            </>
          ) : (
            <div className="empty-state"><span>NO RALLIES FOUND</span><p>This source produced no rally candidates for the selected video.</p></div>
          )}
        </aside>
      </section>

      <section className="controls">
        <div><p className="eyebrow">EDIT DECISION LIST</p><h2>Give every point<br />room to breathe.</h2></div>
        <label>Before activity <output>{preRoll}s</output><input type="range" min="0" max="8" value={preRoll} onChange={(event) => setPreRoll(Number(event.target.value))} /></label>
        <label>After activity <output>{postRoll}s</output><input type="range" min="0" max="8" value={postRoll} onChange={(event) => setPostRoll(Number(event.target.value))} /></label>
        <div className="export"><span>ESTIMATED EXPORT</span><strong>{formatTime(keptSeconds)}</strong><button disabled>Export coming next</button></div>
      </section>
    </main>
  );
}
