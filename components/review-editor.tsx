"use client";

import { useMemo, useState } from "react";
import { buildEditList, formatTime, type Rally } from "@/lib/edit-list";

const duration = 342;
const initialRallies: Rally[] = [
  { id: "R01", start: 24, end: 37, confidence: 0.98, included: true },
  { id: "R02", start: 52, end: 71, confidence: 0.94, included: true },
  { id: "R03", start: 91, end: 104, confidence: 0.67, included: true },
  { id: "R04", start: 126, end: 151, confidence: 0.91, included: true },
  { id: "R05", start: 174, end: 182, confidence: 0.55, included: true },
  { id: "R06", start: 207, end: 236, confidence: 0.96, included: true },
  { id: "R07", start: 259, end: 281, confidence: 0.87, included: true },
  { id: "R08", start: 307, end: 326, confidence: 0.93, included: true },
];

export function ReviewEditor() {
  const [rallies, setRallies] = useState(initialRallies);
  const [selectedId, setSelectedId] = useState("R03");
  const [preRoll, setPreRoll] = useState(3);
  const [postRoll, setPostRoll] = useState(2);
  const intervals = useMemo(
    () => buildEditList(rallies, preRoll, postRoll, duration),
    [rallies, preRoll, postRoll],
  );
  const selected = rallies.find((rally) => rally.id === selectedId) ?? rallies[0];
  const keptSeconds = intervals.reduce((total, interval) => total + interval.keptEnd - interval.keptStart, 0);

  function toggleRally(id: string) {
    setRallies((current) =>
      current.map((rally) => (rally.id === id ? { ...rally, included: !rally.included } : rally)),
    );
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#">VOLLEYCUT <span>LAB</span></a>
        <div className="project-state"><i /> Local prototype</div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">MATCH 001 / REVIEW</p>
          <h1>Find the rallies.<br /><em>Keep the game.</em></h1>
          <p className="intro">A first pass at conservative rally detection. Review the uncertain cuts, tune the breathing room, then export.</p>
        </div>
        <div className="scorecard">
          <span>REVIEW PROGRESS</span>
          <strong>{rallies.filter((rally) => rally.included).length}<small> / {rallies.length}</small></strong>
          <div className="progress"><b style={{ width: `${(rallies.filter((r) => r.included).length / rallies.length) * 100}%` }} /></div>
          <p>{formatTime(keptSeconds)} kept from {formatTime(duration)}</p>
        </div>
      </section>

      <section className="workspace">
        <div className="viewer">
          <div className="court">
            <div className="court-lines"><span /><span /><span /></div>
            <button className="play" aria-label="Play preview">▶</button>
            <div className="timecode">{formatTime(selected.start)} <span>/ {formatTime(duration)}</span></div>
          </div>
          <div className="timeline-labels"><span>0:00</span><span>1:00</span><span>2:00</span><span>3:00</span><span>4:00</span><span>5:00</span></div>
          <div className="timeline" aria-label="Suggested rally timeline">
            {rallies.map((rally) => (
              <button
                key={rally.id}
                className={`${rally.id === selectedId ? "selected" : ""} ${rally.confidence < 0.7 ? "uncertain" : ""} ${!rally.included ? "excluded" : ""}`}
                style={{ left: `${(rally.start / duration) * 100}%`, width: `${((rally.end - rally.start) / duration) * 100}%` }}
                onClick={() => setSelectedId(rally.id)}
                title={`${rally.id}: ${Math.round(rally.confidence * 100)}% confidence`}
              />
            ))}
          </div>
        </div>

        <aside>
          <div className="aside-heading"><span>SELECTED RALLY</span><strong>{selected.id}</strong></div>
          <div className={`confidence ${selected.confidence < 0.7 ? "warn" : ""}`}>
            <span>MODEL CONFIDENCE</span><strong>{Math.round(selected.confidence * 100)}%</strong>
          </div>
          <dl>
            <div><dt>Serve contact</dt><dd>{formatTime(selected.start)}</dd></div>
            <div><dt>End of play</dt><dd>{formatTime(selected.end)}</dd></div>
            <div><dt>Core duration</dt><dd>{selected.end - selected.start}s</dd></div>
          </dl>
          <button className="include" onClick={() => toggleRally(selected.id)}>
            {selected.included ? "✓ Included in export" : "+ Restore to export"}
          </button>
        </aside>
      </section>

      <section className="controls">
        <div><p className="eyebrow">EDIT DECISION LIST</p><h2>Give every point<br />room to breathe.</h2></div>
        <label>Before serve <output>{preRoll}s</output><input type="range" min="0" max="8" value={preRoll} onChange={(event) => setPreRoll(Number(event.target.value))} /></label>
        <label>After point <output>{postRoll}s</output><input type="range" min="0" max="8" value={postRoll} onChange={(event) => setPostRoll(Number(event.target.value))} /></label>
        <div className="export"><span>ESTIMATED EXPORT</span><strong>{formatTime(keptSeconds)}</strong><button disabled>Export coming next</button></div>
      </section>
    </main>
  );
}
