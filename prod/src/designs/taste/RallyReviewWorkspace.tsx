import { useEffect, useRef, useState, type CSSProperties, type ReactNode, type PointerEvent, type RefObject } from "react";
import { deriveScoreAt } from "@/lib/score-tracking";
import { scrollElementIntoContainer } from "@/lib/scroll-container";
import type { Clip, Prototype } from "./index";
import { ResizeHandle } from "./ResizableWorkspace";

function time(value: number) {
  const tenths = Math.round(Math.max(0, value) * 10);
  const seconds = Math.floor(tenths / 10);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}.${tenths % 10}`;
}

// One decision per rally, even when both the clip and its serve need attention.
function reviewRallies(state: Prototype) {
  const ids = new Set(state.clipReviewTaskIds);
  if (state.scoreEnabled) for (const marker of state.activeScoreMarkers) {
    if (marker.side !== "review") continue;
    const clip = state.clips.find(clip => clip.id === marker.rallyId)
      ?? state.clips.find(clip => marker.timestamp >= clip.start && marker.timestamp <= clip.end);
    if (clip) ids.add(clip.id);
  }
  return state.clips.filter(clip => ids.has(clip.id)).sort((a, b) => a.start - b.start);
}

function BoundsTimeline({ state, clip: sourceClip }: { state: Prototype; clip: Clip }) {
  const [preview, setPreview] = useState<{ id: string; start: number; end: number } | null>(null);
  const previewRef = useRef(preview);
  const clip = preview?.id === sourceClip.id ? { ...sourceClip, start: preview.start, end: preview.end } : sourceClip;
  const ordered = state.clips.map(item => item.id === clip.id ? clip : item).sort((a, b) => a.start - b.start);
  const index = ordered.findIndex(item => item.id === clip.id);
  const previous = ordered[index - 1];
  const next = ordered[index + 1];
  const start = Math.max(state.gameStart, Math.min(sourceClip.start - state.beforePadding - 4, previous?.start ?? sourceClip.start));
  const end = Math.min(state.gameEnd, Math.max(sourceClip.end + state.afterPadding + 4, next?.end ?? sourceClip.end));
  const duration = Math.max(.1, end - start);
  const rail = useRef<HTMLDivElement>(null);
  const drag = useRef<{ kind: "start" | "end" | "seek"; pointerId: number } | null>(null);
  const percent = (value: number) => Math.max(0, Math.min(100, (value - start) / duration * 100));
  function seek(event: PointerEvent<HTMLElement>, kind: "start" | "end" | "seek") {
    const rect = rail.current?.getBoundingClientRect();
    if (!rect || !rect.width) return;
    const value = Math.round(Math.max(start, Math.min(end, start + (event.clientX - rect.left) / rect.width * duration)) * 10) / 10;
    if (kind === "seek") state.setPlayhead(value);
    else {
      const next = { id: clip.id, start: clip.start, end: clip.end, ...(kind === "start"
        ? { start: Math.max(state.gameStart, Math.min(clip.end - .1, value)) }
        : { end: Math.min(state.gameEnd, Math.max(clip.start + .1, value)) }) };
      previewRef.current = next;
      setPreview(next);
    }
  }
  function begin(event: PointerEvent<HTMLElement>, kind: "start" | "end" | "seek") {
    if (event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    event.stopPropagation();
    state.setPlaying(false);
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { kind, pointerId: event.pointerId };
    previewRef.current = null;
    if (kind === "seek") seek(event, kind);
  }
  const finish = (commit: boolean) => {
    const current = drag.current;
    if (commit && current && current.kind !== "seek" && previewRef.current?.id === sourceClip.id) {
      state.updateSelected({ [current.kind]: previewRef.current[current.kind] });
    }
    drag.current = null;
    previewRef.current = null;
    setPreview(null);
  };
  return <section className="rw-bounds" data-tour="rw-trim" aria-label="Rally trim controls">
    <div className="rw-bound-head">
      <div><span>Start</span><div><button type="button" aria-label="Move rally start 1 second earlier" onClick={() => state.updateSelected({ start: Math.max(state.gameStart, clip.start - 1) })}>−1s</button><button type="button" aria-label="Move rally start 1 second later" onClick={() => state.updateSelected({ start: Math.min(clip.end - .1, clip.start + 1) })}>+1s</button></div></div>
      <h2>Rally {index + 1}</h2>
      <div><span>End</span><div><button type="button" aria-label="Move rally end 1 second earlier" onClick={() => state.updateSelected({ end: Math.max(clip.start + .1, clip.end - 1) })}>−1s</button><button type="button" aria-label="Move rally end 1 second later" onClick={() => state.updateSelected({ end: Math.min(state.gameEnd, clip.end + 1) })}>+1s</button></div></div>
    </div>
    <div className="rw-bounds-rail" ref={rail} role="slider" aria-label="Rally context playhead" tabIndex={0}
      aria-valuemin={start} aria-valuemax={end} aria-valuenow={Math.max(start, Math.min(end, state.playhead))} aria-valuetext={time(state.playhead)}
      onKeyDown={event => { if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); state.setPlayhead(Math.max(start, Math.min(end, state.playhead + (event.key === "ArrowLeft" ? -.1 : .1)))); } }}
      onPointerDown={event => begin(event, "seek")}
      onPointerMove={event => { if (drag.current?.pointerId === event.pointerId) seek(event, drag.current.kind); }}
      onPointerUp={() => finish(true)} onPointerCancel={() => finish(false)} onLostPointerCapture={() => finish(false)}>
      {ordered.filter(item => item.end + state.afterPadding >= start && item.start - state.beforePadding <= end).map(item => {
        const kept = state.effectiveKeptIds.has(item.id);
        return <div key={item.id} className="rw-context-rally" data-current={item.id === clip.id || undefined} data-kept={kept} data-pending={state.clipReviewTaskIds.has(item.id) || undefined}
          style={{ left: `${percent(item.start - state.beforePadding)}%`, width: `${percent(item.end + state.afterPadding) - percent(item.start - state.beforePadding)}%` }}>
          <span className="rw-context-core" style={{ left: `${(percent(item.start) - percent(item.start - state.beforePadding)) / Math.max(.01, percent(item.end + state.afterPadding) - percent(item.start - state.beforePadding)) * 100}%`, width: `${(percent(item.end) - percent(item.start)) / Math.max(.01, percent(item.end + state.afterPadding) - percent(item.start - state.beforePadding)) * 100}%` }} />
          {item.id !== clip.id && <small>Rally {ordered.indexOf(item) + 1}</small>}
        </div>;
      })}
      <span className="rw-list-marker rw-bounds-marker" role="img" aria-label={`Rally ${index + 1} location`} data-rally-location={clip.id} style={{ left: `${percent((clip.start + clip.end) / 2)}%` }} />
      {(["start", "end"] as const).map(boundary => <button key={boundary} type="button" className="rw-trim-handle" role="slider" aria-label={`Rally ${boundary}`} tabIndex={0}
        aria-valuemin={boundary === "start" ? state.gameStart : clip.start + .1} aria-valuemax={boundary === "start" ? clip.end - .1 : state.gameEnd} aria-valuenow={clip[boundary]} aria-valuetext={time(clip[boundary])}
        style={{ left: `${percent(clip[boundary])}%` }} onPointerDown={event => begin(event, boundary)}
        onKeyDown={event => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault(); event.stopPropagation();
          const min = boundary === "start" ? state.gameStart : clip.start + .1;
          const max = boundary === "start" ? clip.end - .1 : state.gameEnd;
          const value = event.key === "Home" ? min : event.key === "End" ? max : clip[boundary] + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 1 : .1);
          state.updateSelected({ [boundary]: Math.max(min, Math.min(max, value)) });
        }} />)}
      {state.playhead >= start && state.playhead <= end && <i className="rw-playhead" style={{ left: `${percent(state.playhead)}%` }} />}
    </div>
    <div className="rw-bound-times"><span>Start {time(clip.start)}</span><span>End {time(clip.end)}</span></div>
  </section>;
}

export type RalliesTourView = "review" | "edit" | "browse" | "add";

export function RallyReviewWorkspace({ state, nextReviewButtonRef, renderVideo, renderTimeline, tourView, sidebarWidth, onSidebarWidthChange }: {
  state: Prototype;
  nextReviewButtonRef: RefObject<HTMLButtonElement | null>;
  renderVideo: (range?: { start: number; end: number }) => ReactNode;
  renderTimeline: () => ReactNode;
  tourView?: RalliesTourView | null;
  sidebarWidth?: number;
  onSidebarWidthChange: (width: number | undefined) => void;
}) {
  const [mode, setMode] = useState<"review" | "browse" | "edit" | "add">("review");
  const [showTimeline, setShowTimeline] = useState(true);
  const body = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ width: window.innerWidth, height: window.innerHeight, available: window.innerWidth - 32 });
  useEffect(() => {
    const measure = () => setViewport({ width: window.innerWidth, height: window.innerHeight, available: body.current?.clientWidth || window.innerWidth - 32 });
    const observer = new ResizeObserver(measure);
    if (body.current) observer.observe(body.current);
    window.addEventListener("resize", measure);
    measure();
    return () => { observer.disconnect(); window.removeEventListener("resize", measure); };
  }, []);
  const sidebarDefault = viewport.width > 1100 ? 416 : viewport.height < 480 ? 340 : 360;
  // Keep room for the player, and retain the saved width when a smaller screen clamps it.
  const sidebarMax = Math.max(320, Math.min(640, viewport.available - 332));
  const fittedSidebarWidth = Math.max(320, Math.min(sidebarMax, sidebarWidth ?? sidebarDefault));
  const pending = reviewRallies(state);
  const orphanServes = state.scoreEnabled ? state.activeScoreMarkers.filter(marker => marker.side === "review" &&
    !state.clips.some(clip => clip.id === marker.rallyId || (!marker.rallyId && marker.timestamp >= clip.start && marker.timestamp <= clip.end))) : [];
  const [orphanId, setOrphanId] = useState<string | null>(null);
  const orphan = state.activeScoreMarkers.find(marker => marker.id === orphanId);
  const pendingCount = pending.length + orphanServes.length;
  const ordered = [...state.clips].sort((a, b) => a.start - b.start);
  const clip = state.selected;
  const index = ordered.findIndex(item => item.id === clip?.id);
  const browserClip = ordered.find(item => state.playhead >= Math.max(state.gameStart, item.start - state.beforePadding) && state.playhead < Math.min(state.gameEnd, item.end + state.afterPadding));
  const browserSelected = browserClip ?? clip;
  const browserIndex = ordered.findIndex(item => item.id === browserSelected?.id);
  const [followingSource, setFollowingSource] = useState(true);
  const [centeredId, setCenteredId] = useState<string | null>(null);
  const centered = ordered.find(item => item.id === centeredId) ?? browserSelected ?? ordered[0];
  const centeredIndex = ordered.findIndex(item => item.id === centered?.id);
  const locationRally = mode === "browse" ? centered : clip;
  const locationTime = locationRally ? (locationRally.start + locationRally.end) / 2 : null;
  const list = useRef<HTMLDivElement>(null);
  const initialized = useRef(false);
  const previousCount = useRef(state.clips.length);
  // A resolved item stays open until Next review, so the second decision never disappears.
  const [reviewingId, setReviewingId] = useState<string | null>(pending[0]?.id ?? null);
  // The walkthrough only navigates. In particular, opening Add never marks a boundary.
  useEffect(() => {
    if (!tourView) return;
    setMode(tourView);
    setOrphanId(null);
    setShowTimeline(true);
  }, [tourView]);
  function updateListCenter() {
    const container = list.current;
    if (!container || !container.clientHeight) return;
    const rect = container.getBoundingClientRect();
    const middle = rect.top + container.clientHeight / 2;
    const rows = [...container.querySelectorAll<HTMLElement>("[data-rally-id]")];
    const closest = rows.reduce<HTMLElement | null>((best, row) => {
      const bounds = row.getBoundingClientRect();
      const distance = Math.abs((bounds.top + bounds.bottom) / 2 - middle);
      const bestBounds = best?.getBoundingClientRect();
      return !bestBounds || distance < Math.abs((bestBounds.top + bestBounds.bottom) / 2 - middle) ? row : best;
    }, null);
    if (closest) setCenteredId(closest.dataset.rallyId ?? null);
  }
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    state.setFinalPreview(false);
    if (pending[0]) state.selectClip(pending[0].id);
    else if (orphanServes[0]) { setOrphanId(orphanServes[0].id); state.selectScoreMarker(orphanServes[0].id); }
  }, [state, pending]);
  useEffect(() => {
    if (orphanServes.some(marker => marker.id === state.selectedScoreMarkerId)) setOrphanId(state.selectedScoreMarkerId);
    else if (state.selectedScoreMarkerId && state.selectedScoreMarkerId !== orphanId) setOrphanId(null);
  }, [state.selectedScoreMarkerId]);
  useEffect(() => {
    const selected = list.current?.querySelector<HTMLElement>('[aria-current="true"]');
    if (followingSource && selected && list.current) {
      scrollElementIntoContainer(list.current, selected, "auto");
      setCenteredId(browserSelected?.id ?? null);
    }
    updateListCenter();
  }, [browserSelected?.id, mode, followingSource]);
  useEffect(() => {
    const container = list.current;
    if (!container) return;
    const observer = new ResizeObserver(updateListCenter);
    observer.observe(container);
    return () => observer.disconnect();
  }, [mode, ordered.length]);
  useEffect(() => { if (state.manualStart !== null && !tourView) setMode("add"); }, [state.manualStart, tourView]);
  useEffect(() => {
    if (state.clips.length > previousCount.current && mode === "add" && state.manualStart === null) setMode("edit");
    previousCount.current = state.clips.length;
  }, [state.clips.length, state.manualStart, mode]);
  const sourcePlayback = mode === "browse" || mode === "add";
  const cut = state.workingDraft.cuts.find(item => item.id === clip?.id);
  const range = sourcePlayback ? undefined : orphan ? { start: Math.max(state.gameStart, orphan.timestamp - 2), end: Math.min(state.gameEnd, orphan.timestamp + 5) } : cut ? { start: cut.keepStart, end: cut.keepEnd } : undefined;
  const markers = clip && state.scoreEnabled ? state.activeScoreMarkers.filter(marker => marker.rallyId === clip.id || (!marker.rallyId && marker.timestamp >= clip.start && marker.timestamp <= clip.end)) : [];
  const unresolved = clip ? pending.some(item => item.id === clip.id) : false;
  const reviewComplete = mode === "review" && pending.length === 0 && reviewingId !== clip?.id;
  function open(id: string, nextMode: "edit" | "review" = "edit") {
    setOrphanId(null);
    state.setPlaying(false); state.setFinalPreview(false); state.selectClip(id); setMode(nextMode);
    if (nextMode === "review") setReviewingId(id);
  }
  function nextReview() {
    const next = pending.find(item => item.id !== clip?.id && item.start > (clip?.start ?? -1)) ?? pending.find(item => item.id !== clip?.id) ?? pending[0];
    if (next) open(next.id, "review");
    else if (orphanServes[0]) { setMode("review"); setOrphanId(orphanServes[0].id); state.selectScoreMarker(orphanServes[0].id); }
    else { setReviewingId(null); setMode("review"); state.setPlaying(false); }
  }
  function serve(side: "near" | "far" | null, markerId?: string) {
    state.setRallyServe(clip?.id ?? "", side, markerId);
  }
  return <section className="rw-workspace" aria-label="Rally review workspace">
    <header className="rw-heading" data-tour="rw-review">
      <h1>{mode === "add" ? "Add missing rally" : mode === "browse" ? "All rallies" : mode === "edit" ? `Editing rally ${index + 1}` : `Reviewing ${pendingCount} ${pendingCount === 1 ? "rally" : "rallies"}`}</h1>
      <button type="button" data-tour="rw-all-rallies" onClick={() => {
        state.setPlaying(false);
        if (state.manualStart !== null) state.cancelManualBoundary();
        if (mode === "review") setMode("browse");
        else if (pending[0]) open(pending.find(item => item.id === reviewingId)?.id ?? pending[0].id, "review");
        else if (orphanServes[0]) { setMode("review"); setOrphanId(orphanServes[0].id); state.selectScoreMarker(orphanServes[0].id); }
        else { setOrphanId(null); setReviewingId(null); setMode("review"); }
      }}>{mode === "review" ? `All rallies (${ordered.length}) →` : "← Review rallies"}</button>
    </header>
    <div className="rw-body" ref={body} style={{ "--rw-sidebar-width": `${fittedSidebarWidth}px` } as CSSProperties}>
      <div className="rw-player">{renderVideo(range)}<label className="rw-source-seek"><span>Source playhead</span><span className="rw-source-slider"><input aria-label="Source playhead" type="range" min={state.gameStart} max={state.gameEnd} step="any" value={state.playhead} onChange={event => { setFollowingSource(true); state.setPlayhead(Number(event.currentTarget.value)); }} />{locationTime !== null && <span className="rw-list-marker" role="img" aria-label={`Rally ${ordered.indexOf(locationRally!) + 1} location`} data-rally-location={locationRally!.id} style={{ left: `${(locationTime - state.gameStart) / Math.max(.1, state.gameEnd - state.gameStart) * 100}%` }} />}</span></label></div>
      <ResizeHandle axis="horizontal" label="Resize rallies sidebar" className="rw-sidebar-resize" direction={-1}
        value={fittedSidebarWidth} min={320} max={sidebarMax} onChange={onSidebarWidthChange} onReset={() => onSidebarWidthChange(undefined)} />
      <aside className="rw-panel" aria-label={mode === "browse" ? "All rallies browser" : "Rally decisions"}>
        {mode === "browse" ? <>
          <header className="rw-browser-heading"><h2>{ordered.length} rallies</h2><button type="button" onClick={() => setShowTimeline(!showTimeline)}>{showTimeline ? "Hide timeline" : "Show timeline"}</button></header>
          <p className="rw-browser-position">{browserClip ? `Rally ${ordered.indexOf(browserClip) + 1}` : "No rally"} at {time(state.playhead)}</p>
          <div className="rw-browser-content">
            {showTimeline && <div className="rw-source-map" data-tour="rw-source" aria-label="Recording overview">
              {[0, 1].map(column => {
                const start = state.gameStart + (state.gameEnd - state.gameStart) * column / 2;
                const end = state.gameStart + (state.gameEnd - state.gameStart) * (column + 1) / 2;
                const contains = (value: number) => value >= start && (value < end || column === 1 && value <= end);
                const percent = (value: number) => (value - start) / Math.max(.1, end - start) * 100;
                return <div className="rw-source-column" key={column}>
                  <span>{time(start).replace(/\.\d$/, "")}</span>
                  <div className="rw-source-track"><input type="range" aria-label={`Recording overview playhead ${column + 1}`} min={start} max={end} step="any" value={Math.max(start, Math.min(end, state.playhead))} onChange={event => { setFollowingSource(true); state.setPlayhead(Number(event.currentTarget.value)); }} />
                    <div className="rw-source-marks" aria-hidden="true">
                      {ordered.filter(item => item.end + state.afterPadding > start && item.start - state.beforePadding < end).map(item => <i key={item.id} data-kept={state.effectiveKeptIds.has(item.id)} data-pending={pending.some(clip => clip.id === item.id) || undefined} style={{ top: `${Math.max(0, percent(item.start - state.beforePadding))}%`, height: `${Math.min(100, percent(item.end + state.afterPadding)) - Math.max(0, percent(item.start - state.beforePadding))}%` }} />)}
                      {contains(state.playhead) && <span className="rw-source-playhead" data-playhead-marker={column + 1} style={{ top: `${percent(state.playhead)}%` }} />}
                      {locationTime !== null && contains(locationTime) && <span className="rw-list-marker rw-source-location" data-rally-location={centered?.id} style={{ top: `${percent(locationTime)}%` }} />}
                    </div>
                  </div>
                  <span>{time(end).replace(/\.\d$/, "")}</span>
                </div>;
              })}
            </div>}
            <div className="rw-list-column" data-tour="rw-list"><div className="rw-browser-key"><span>● Play</span><span>◇ List · {centeredIndex >= 0 ? centeredIndex + 1 : "—"}</span></div>
            <div className="rw-rally-list" ref={list} onScroll={updateListCenter} onWheel={() => setFollowingSource(false)} onTouchMove={() => setFollowingSource(false)} onPointerDown={() => setFollowingSource(false)} onKeyDown={() => setFollowingSource(false)}>
              {ordered.length === 0 && <p>No rallies yet. Add a missing rally below.</p>}
              {ordered.map((item, i) => {
                const score = state.scoreEnabled ? deriveScoreAt(state.activeScoreTracking, item.start) : null;
                return <button type="button" key={item.id} data-rally-id={item.id} aria-current={browserSelected?.id === item.id ? "true" : undefined} aria-label={`Open rally ${i + 1}`} onClick={() => { setFollowingSource(true); state.setPlaying(false); state.selectClip(item.id); }} onDoubleClick={() => open(item.id)}>
                  <span><strong>Rally {i + 1}{centered?.id === item.id && <i className="rw-list-marker" role="img" aria-label="Timeline list position" data-rally-location={item.id} />}</strong><small>{time(item.start)}–{time(item.end)}</small></span>
                  <span>{score && <strong>{score.team1Score} : {score.team2Score}</strong>}<small>{pending.some(clip => clip.id === item.id) ? "Needs review" : state.effectiveKeptIds.has(item.id) ? "Keep" : "Removed"}</small></span>
                </button>;
              })}
            </div></div>
          </div>
          <footer className="rw-browser-actions"><button type="button" disabled={!browserSelected} onClick={() => browserSelected && open(browserSelected.id)}>Edit rally {browserIndex >= 0 ? browserIndex + 1 : ""}</button><button type="button" className="rw-primary" data-tour="rw-missing" onClick={() => { state.setFinalPreview(false); setMode("add"); }}>+ Add missing rally</button></footer>
        </> : mode === "add" ? <>
          <div className="rw-add-heading" data-tour="rw-add"><p>{state.manualStart === null ? "Seek to the rally start, then mark it below." : "Seek to the rally end, then mark and add it below."}</p><button type="button" onClick={() => { state.cancelManualBoundary(); setMode("browse"); }}>Cancel</button></div>
          <div className="rw-bound-times"><span>Start {state.manualStart === null ? "—" : time(state.manualStart)}</span><span>End {state.manualStart === null ? "—" : time(state.playhead)}</span></div>
          <div className="rw-missing-timeline">{renderTimeline()}</div>
          <button type="button" className="rw-primary" disabled={state.manualStart !== null && Math.abs(state.playhead - state.manualStart) < .3} onClick={() => {
            const finishing = state.manualStart !== null;
            state.markManualBoundary();
            if (finishing && Math.abs(state.playhead - (state.manualStart ?? state.playhead)) >= .3) setMode("edit");
          }}>{state.manualStart === null ? "Mark rally start" : "Mark end & add rally"}<kbd>Shift + R</kbd></button>
          {state.manualStart !== null && <button type="button" onClick={state.cancelManualBoundary}>Reset start</button>}
        </> : orphan ? <>
          <h2>Review footage</h2><p className="rw-pending-note">Serve at {time(orphan.timestamp)}</p>
          <div className="rw-choices" role="group" aria-label="Serve outside a rally">
            <button type="button" aria-pressed={orphan.side === "near"} onClick={() => state.updateScoreMarkerSide(orphan.id, "near")}>Near<kbd>N</kbd></button>
            <button type="button" aria-pressed={orphan.side === "far"} onClick={() => state.updateScoreMarkerSide(orphan.id, "far")}>Far<kbd>F</kbd></button>
            <button type="button" onClick={() => { state.removeServeById(orphan.id); setOrphanId(null); }}>Not a serve</button>
          </div>
          <button ref={nextReviewButtonRef} type="button" className="rw-primary" disabled={orphan.side === "review"} title="R" aria-keyshortcuts="R" onClick={() => { setOrphanId(null); nextReview(); }}>Next review</button>
        </> : reviewComplete || !clip ? <div className="rw-complete"><h2>{pendingCount === 0 ? "All checked." : "Continue reviewing"}</h2><p>{pendingCount === 0 ? "Browse the recording for anything missed." : `${pendingCount} rallies still need a decision.`}</p><button type="button" className="rw-primary" onClick={() => pendingCount ? nextReview() : setMode("browse")}>{pendingCount ? "Continue review" : "Browse rallies"}</button><button type="button" onClick={() => state.setStage("deliver")}>Export</button></div> : <>
          <BoundsTimeline state={state} clip={clip} />
          {state.suppressionReviewIds.has(clip.id) && <p className="rw-pending-note">Suggested for cleanup. Watch before deciding.</p>}
          <div className="rw-choices" data-tour="rw-decisions" role="group" aria-label="Keep or remove rally">
            <button type="button" aria-pressed={!state.clipReviewTaskIds.has(clip.id) && state.effectiveKeptIds.has(clip.id)} onClick={() => state.setCurrentRallyIncluded(true)}>Keep<kbd>K</kbd></button>
            <button type="button" aria-pressed={!state.clipReviewTaskIds.has(clip.id) && !state.effectiveKeptIds.has(clip.id)} onClick={() => state.setCurrentRallyIncluded(false)}>Remove<kbd>⌫</kbd></button>
          </div>
          {state.scoreEnabled && (markers.length ? markers : [null]).map(marker => <div key={marker?.id ?? "no-serve"} className="rw-serve-group" data-tour="rw-serve">
            {markers.length > 1 && <span>Serve {time(marker!.timestamp)}</span>}
            <div className="rw-choices" role="group" aria-label={marker ? `Serve at ${time(marker.timestamp)}` : "Rally serve"}>
              <button type="button" aria-pressed={marker?.side === "near"} onClick={() => serve("near", marker?.id)}>Near<kbd>N</kbd></button>
              <button type="button" aria-pressed={marker?.side === "far"} onClick={() => serve("far", marker?.id)}>Far<kbd>F</kbd></button>
              <button type="button" aria-pressed={!marker} onClick={() => serve(null, marker?.id)}>Not a serve</button>
            </div>
          </div>)}
          <div className="rw-navigation" data-tour="rw-navigation"><button type="button" disabled={index <= 0} title="Shift + Left arrow" onClick={() => open(ordered[index - 1].id)}>Previous rally</button><button ref={nextReviewButtonRef} type="button" className="rw-primary" disabled={unresolved} title="R" aria-keyshortcuts="R" onClick={nextReview}>Next review</button><button type="button" disabled={index < 0 || index >= ordered.length - 1} title="Shift + Right arrow" onClick={() => open(ordered[index + 1].id)}>Next rally</button></div>
          <details className="rw-more"><summary>More editing tools</summary><div><button type="button" onClick={() => state.setSelectedBoundaryAtPlayhead("start")}>Set start to playhead<kbd>Ctrl + Shift + S</kbd></button><button type="button" onClick={() => state.setSelectedBoundaryAtPlayhead("end")}>Set end to playhead<kbd>Ctrl + Shift + E</kbd></button><button type="button" onClick={state.splitSelectedAtPlayhead}>Split at playhead<kbd>Shift + S</kbd></button></div></details>
        </>}
        <p className="rw-status" role="status">{state.reviewMessage}</p>
      </aside>
    </div>
  </section>;
}
