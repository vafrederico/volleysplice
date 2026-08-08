"use client";

import { useMemo, useRef, useState } from "react";
import type { ReviewAnalysis } from "@/lib/analysis-types";
import { buildEditList, formatTime, timelinePercent, timelineTicks, type Rally } from "@/lib/edit-list";

const demoRallies: Rally[] = [
  { id: "R01", start: 24, end: 37, confidence: 0.78, included: true },
  { id: "R02", start: 52, end: 71, confidence: 0.74, included: true },
  { id: "R03", start: 91, end: 104, confidence: 0.57, included: true },
  { id: "R04", start: 126, end: 151, confidence: 0.71, included: true },
];

const demoAnalysis: ReviewAnalysis = {
  id: "demo",
  title: "Example analysis",
  duration: 182,
  width: 16,
  height: 7,
  sourceFilename: "Add a video to begin",
  videoUrl: null,
  courtPreviewUrl: null,
  courtConfidence: 0,
  courtSource: "demo",
  courtLines: [],
  cameraStability: 1,
  warnings: ["This is sample data. Run the local analyzer to review a real recording."],
  rallies: demoRallies,
};

export function ReviewEditor({ initialAnalysis }: { initialAnalysis: ReviewAnalysis | null }) {
  const analysis = initialAnalysis ?? demoAnalysis;
  const [rallies, setRallies] = useState(analysis.rallies);
  const [selectedId, setSelectedId] = useState(analysis.rallies[0]?.id ?? "");
  const [preRoll, setPreRoll] = useState(3);
  const [postRoll, setPostRoll] = useState(2);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const intervals = useMemo(
    () => buildEditList(rallies, preRoll, postRoll, analysis.duration),
    [rallies, preRoll, postRoll, analysis.duration],
  );
  const selected = rallies.find((rally) => rally.id === selectedId) ?? rallies[0] ?? null;
  const keptSeconds = intervals.reduce((total, interval) => total + interval.keptEnd - interval.keptStart, 0);
  const keptCount = rallies.filter((rally) => rally.included).length;
  const ticks = timelineTicks(analysis.duration);

  function selectRally(rally: Rally) {
    setSelectedId(rally.id);
    const seekTime = Math.max(0, rally.start - preRoll);
    setPlaybackTime(seekTime);
    if (videoRef.current) videoRef.current.currentTime = seekTime;
  }

  function toggleRally(id: string) {
    setRallies((current) =>
      current.map((rally) => (rally.id === id ? { ...rally, included: !rally.included } : rally)),
    );
  }

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) await video.play();
    else video.pause();
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#">VOLLEYCUT <span>LAB</span></a>
        <div className="project-state"><i /> {initialAnalysis ? "Latest local analysis" : "Demo mode"}</div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">{analysis.title.toUpperCase()} / REVIEW</p>
          <h1>Find the rallies.<br /><em>Keep the game.</em></h1>
          <p className="intro">Court motion and audio produced conservative activity suggestions. Review every cut before using it.</p>
          <p className="source-name">SOURCE / {analysis.sourceFilename}</p>
        </div>
        <div className="scorecard">
          <span>RALLIES KEPT</span>
          <strong>{keptCount}<small> / {rallies.length}</small></strong>
          <div className="progress"><b style={{ width: `${rallies.length ? (keptCount / rallies.length) * 100 : 0}%` }} /></div>
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
          <div className={`court video-stage ${analysis.videoUrl ? "has-video" : ""}`} style={{ aspectRatio: `${analysis.width} / ${analysis.height}` }}>
            {analysis.videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={analysis.videoUrl}
                  preload="metadata"
                  controls
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
            <button className={`play ${isPlaying ? "playing" : ""}`} aria-label={isPlaying ? "Pause preview" : "Play preview"} onClick={togglePlayback} disabled={!analysis.videoUrl}>
              {isPlaying ? "Ⅱ" : "▶"}
            </button>
            <div className="timecode">{formatTime(playbackTime)} <span>/ {formatTime(analysis.duration)}</span></div>
            <div className="court-status">COURT / {analysis.courtSource} / {Math.round(analysis.courtConfidence * 100)}%</div>
          </div>
          <div className="timeline-labels">
            {ticks.map((tick) => <span key={tick}>{formatTime(tick)}</span>)}
          </div>
          <div className="timeline" aria-label="Suggested rally timeline">
            {rallies.map((rally) => (
              <button
                key={rally.id}
                className={`${rally.id === selectedId ? "selected" : ""} ${rally.confidence < 0.7 ? "uncertain" : ""} ${!rally.included ? "excluded" : ""}`}
                style={{ left: `${timelinePercent(rally.start, analysis.duration)}%`, width: `${timelinePercent(rally.end - rally.start, analysis.duration)}%` }}
                onClick={() => selectRally(rally)}
                title={`${rally.id}: ${Math.round(rally.confidence * 100)}% heuristic confidence`}
              />
            ))}
          </div>
        </div>

        <aside>
          {selected ? (
            <>
              <div className="aside-heading"><span>SELECTED RALLY</span><strong>{selected.id}</strong></div>
              <div className={`confidence ${selected.confidence < 0.7 ? "warn" : ""}`}>
                <span>HEURISTIC CONFIDENCE</span><strong>{Math.round(selected.confidence * 100)}%</strong>
              </div>
              <dl>
                <div><dt>Suggested start</dt><dd>{formatTime(selected.start)}</dd></div>
                <div><dt>Suggested end</dt><dd>{formatTime(selected.end)}</dd></div>
                <div><dt>Core duration</dt><dd>{(selected.end - selected.start).toFixed(1)}s</dd></div>
                <div><dt>Camera stability</dt><dd>{Math.round(analysis.cameraStability * 100)}%</dd></div>
              </dl>
              <button className="include" onClick={() => toggleRally(selected.id)}>
                {selected.included ? "✓ Included in export" : "+ Restore to export"}
              </button>
              {analysis.courtPreviewUrl && <a className="diagnostic-link" href={analysis.courtPreviewUrl} target="_blank">Open court diagnostic ↗</a>}
            </>
          ) : (
            <div className="empty-state"><span>NO RALLIES FOUND</span><p>The analyzer found no conservative candidates. Inspect the diagnostic and try representative end-line footage.</p></div>
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
