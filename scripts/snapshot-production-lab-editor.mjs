import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// A deliberate, lab-owned snapshot of the editor currently rendered by prod/App.
// Run manually to refresh it. This does not modify the production application.
const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceRoot = path.join(repository, "prod/src");
const destinationRoot = path.join(repository, "components/production-lab/editor");
const pending = [path.join(sourceRoot, "designs/taste/index.tsx")];
const files = new Map();
const importPattern = /(?:from\s+|import\s*\(\s*|import\s+)["']([^"']+)["']/g;
const sha256 = (text) => createHash("sha256").update(text).digest("hex");

const labDisplayLabelsSource = `type NamedRally = { id: string; start: number; end: number };

function suffix(index: number): string {
  let value = index + 1;
  let text = "";
  while (value > 0) { value -= 1; text = String.fromCharCode(97 + value % 26) + text; value = Math.floor(value / 26); }
  return text;
}

/** Short presentation labels; immutable model IDs remain the identity for every edit. */
export function labRallyDisplayLabels(rallies: readonly NamedRally[]): Map<string, string> {
  const labels = new Map<string, string>();
  const groups = new Map<string, NamedRally[]>();
  const ordered = [...rallies].sort((a, b) => a.start - b.start || a.end - b.end || a.id.localeCompare(b.id));
  for (const [index, rally] of ordered.entries()) {
    const candidate = /^candidate:(R\\d+)/.exec(rally.id);
    const compact = /^compact:(\\d+)/.exec(rally.id);
    const base = candidate?.[1] ?? (compact ? "C" + compact[1].padStart(3, "0") : rally.id.startsWith("human:") ? rally.id.slice(6) : null);
    if (base) {
      const group = groups.get(base) ?? [];
      group.push(rally); groups.set(base, group);
    } else {
      const restored = /^restored:removed:(R\\d+)/.exec(rally.id);
      labels.set(rally.id, restored ? restored[1] + " restored" : rally.id.includes(":") ? "Rally " + (index + 1) : rally.id);
    }
  }
  for (const [base, group] of groups) group.forEach((rally, index) => labels.set(rally.id, base + (group.length > 1 ? suffix(index) : "")));
  return labels;
}
`;

const labSplitSource = `import type { ScoreTracking } from "./score-tracking.ts";

/** Preserve reviewed serves while giving a genuinely new rally an unresolved start. */
export function splitLabServeMarkers(
  scoreTracking: ScoreTracking,
  parentId: string,
  left: { id: string; start: number },
  right: { id: string; start: number },
): ScoreTracking {
  const serveMarkers = scoreTracking.serveMarkers.map(marker => marker.rallyId === parentId
    ? { ...marker, rallyId: left.id, timestamp: left.start } : marker);
  const rightMarkerId = "lab-split-serve:" + right.id;
  if (!serveMarkers.some(marker => marker.rallyId === right.id || marker.id === rightMarkerId)
    && !scoreTracking.sideSwitchMarkers.some(marker => marker.id === rightMarkerId)
    && !scoreTracking.removedModelMarkerIds.includes(rightMarkerId)) {
    serveMarkers.push({ id: rightMarkerId, rallyId: right.id, timestamp: right.start,
      side: "review", modelSide: "review", origin: "model", ignorePreviousPoint: false });
  }
  return { ...scoreTracking,
    serveMarkers: serveMarkers.sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id)) };
}
`;

const sourceReviewPlayback = `
  function playSourceRange(start: number, end: number): () => void {
    const from = Math.max(gameStart, Math.min(gameEnd, start));
    const to = Math.max(from, Math.min(gameEnd, end));
    videoElementRef.current?.pause();
    setPlaying(false);
    setFinalPreview(false);
    setPlayhead(from);
    const id = ++sourceRangeSequence.current;
    setSourceRange(to > from ? { id, start: from, end: to } : null);
    return () => {
      if (sourceRangeSequence.current !== id) return;
      sourceRangeSequence.current += 1;
      setSourceRange(null);
      videoElementRef.current?.pause();
      setPlaying(false);
    };
  }

  // Start only after final-preview=false has committed, so removed footage is
  // never skipped by the production player's final-cut timeupdate handler.
  useEffect(() => {
    const video = videoElementRef.current;
    if (!sourceRange) return;
    if (stage !== "review" || finalPreview || !video || !videoUrl) {
      setSourceRange(null);
      return;
    }
    const request = sourceRange;
    let active = true;
    let started = false;
    let frame = 0;
    function detach() {
      active = false;
      if (frame) window.cancelAnimationFrame(frame);
      video!.removeEventListener("loadedmetadata", start);
      video!.removeEventListener("play", onPlay);
      video!.removeEventListener("pause", onPause);
      video!.removeEventListener("ended", onEnded);
      video!.removeEventListener("timeupdate", checkEnd);
    }
    function stop(atEnd: boolean) {
      if (!active) return;
      detach();
      video!.pause();
      setPlaying(false);
      if (atEnd) {
        video!.currentTime = request.end;
        setPlayheadValue(request.end);
      }
      setSourceRange((current) => current?.id === request.id ? null : current);
    }
    function checkEnd() {
      if (active && started && video!.currentTime >= request.end) stop(true);
    }
    function checkFrame() {
      checkEnd();
      if (active) frame = window.requestAnimationFrame(checkFrame);
    }
    function onPlay() {
      if (!active || started) return;
      started = true;
      frame = window.requestAnimationFrame(checkFrame);
    }
    function onPause() { if (started) stop(false); }
    function onEnded() { stop(false); }
    function start() {
      if (!active || sourceRangeSequence.current !== request.id) return;
      video!.currentTime = request.start;
      setPlayheadValue(request.start);
      const bounds = video!.getBoundingClientRect?.();
      if (bounds && (bounds.top < 0 || bounds.bottom > window.innerHeight)) {
        video!.scrollIntoView?.({ block: "nearest", inline: "nearest", behavior: "smooth" });
      }
      // Let the previous pause event settle before VideoStage starts again.
      frame = window.requestAnimationFrame(() => {
        if (active && sourceRangeSequence.current === request.id) setPlaying(true);
      });
    }
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onEnded);
    video.addEventListener("timeupdate", checkEnd);
    if (video.readyState > 0) start();
    else video.addEventListener("loadedmetadata", start, { once: true });
    return () => { detach(); video.pause(); };
  }, [sourceRange, stage, videoUrl, finalPreview]);
`;

function resolveImport(importer, specifier) {
  const base = specifier.startsWith("@/")
    ? path.join(sourceRoot, specifier.slice(2))
    : specifier.startsWith(".") ? path.resolve(path.dirname(importer), specifier) : null;
  if (!base) return null;
  const candidates = [base, `${base}.ts`, `${base}.tsx`, path.join(base, "index.ts"), path.join(base, "index.tsx")];
  const found = candidates.find((candidate) => existsSync(candidate) && path.extname(candidate));
  if (!found) throw new Error(`Unresolved snapshot import: ${importer}: ${specifier}`);
  if (!path.relative(sourceRoot, found) || path.relative(sourceRoot, found).startsWith("..")) {
    throw new Error(`Snapshot import escaped source root: ${found}`);
  }
  return found;
}

while (pending.length) {
  const filename = pending.pop();
  if (files.has(filename)) continue;
  const source = readFileSync(filename, "utf8").replace(/\r\n/g, "\n");
  files.set(filename, source);
  if (!/\.tsx?$/.test(filename)) continue;
  for (const [, specifier] of source.matchAll(importPattern)) {
    const resolved = resolveImport(filename, specifier);
    if (resolved) pending.push(resolved);
  }
}

const manifest = [];
for (const [filename, source] of [...files].sort(([a], [b]) => a.localeCompare(b))) {
  const relative = path.relative(sourceRoot, filename).replaceAll("\\", "/");
  let output = source.replace(/(["'])@\/([^"']+)\1/g, (_match, quote, specifier) => {
    let target = path.relative(path.dirname(filename), path.join(sourceRoot, specifier)).replaceAll("\\", "/");
    if (!target.startsWith(".")) target = `./${target}`;
    return `${quote}${target}${quote}`;
  });
  // Never share production review state, history, tours, or layout preferences.
  output = output.replaceAll("volleycut:", "volleycut:production-lab:")
    .replaceAll("volleysplice:", "volleycut:production-lab:");

  if (relative === "designs/useDesignReview.ts") {
    output = `// Lab adapter contract copied from production. No project-store hook or writes.\n`
      + `import type { CutDraft, CutDraftSeed } from "../lib/cut-draft";\n`
      + `import type { VolleySpliceProject } from "../lib/project-store";\n`
      + `import type { OnDeviceSuppression } from "../lib/on-device/types";\n`
      + `import type { ExportInterval, ScoreOverlayOptions } from "../lib/on-device/export";\n`
      + `import type { ProductAnalysis } from "../lib/product-analysis";\n\n`
      + source.slice(source.indexOf("export type DesignProjectOption"), source.indexOf("function reloadSelectedProject"));
    output = output.replace('export type ReadyDesignReview = {\n  state: "ready";', 'export type ReadyDesignReview = {\n  /** Recorded-model UX trial; no production inference or project persistence. */\n  labMode?: boolean;\n  state: "ready";');
  }
  if (relative === "lib/project-store.ts") {
    output = `// Type-only production shapes. The lab never opens the production project database.\n`
      + `import type { AnalysisWindow } from "./on-device/analysis-window";\n`
      + `import type { NormalizedRoi, OnDeviceAnalysis, OnDeviceMediaInfo } from "./on-device/types";\n`
      + `import type { CutDraft } from "./cut-draft";\n\n`
      + source.slice(source.indexOf("export type ProjectSource"), source.indexOf("function requestResult"));
  }
  if (relative === "lib/runtime-assets.ts") {
    output = output.slice(0, output.indexOf("export function runtimeAssetUrl"))
      + `export function runtimeAssetUrl(asset: RuntimeAsset): string {\n`
      + `  return \`/production-lab/\${asset}\`;\n}\n`;
  }
  if (relative === "designs/taste/index.tsx") {
    output = output.replace("    restoreHistory, historyStorageFailed, resetProjectChanges,", "    restoreHistory, historyStorageFailed, resetProjectChanges, applyReviewDraft,")
      .replace('import { GOOGLE_PLAY_URL } from "../../lib/android-app";', 'import { GOOGLE_PLAY_URL } from "../../lib/android-app";\nimport { splitLabServeMarkers } from "../../lib/lab-split-markers";')
      .replace('import { splitLabServeMarkers } from "../../lib/lab-split-markers";', 'import { splitLabServeMarkers } from "../../lib/lab-split-markers";\nimport { labRallyDisplayLabels } from "../../lib/lab-display-labels";')
      .replace('import "./taste-designs.css";', 'import "../../lab-base.css";\nimport "./taste-designs.css";')
      .replace("Saved analysis and review edits loaded from the live app.", "Saved lab model results and review edits loaded.")
      .replace("Live review data loaded", "Lab review data loaded")
      .replace("Saved to the live review on this device", "Saved to this lab configuration on this device")
      .replace("export function RallyDesk({\n  review,\n  onDeleteProject,", `export type RallyDeskLabTools = {\n  draft: CutDraft;\n  applyDraft: (draft: CutDraft) => void;\n  seek: (time: number) => void;\n  selectedCutId: string;\n};\n\nexport function RallyDesk({\n  review,\n  onDeleteProject,\n  labTools,`)
      .replace("  onDeleteProject?: () => void;\n}) {\n  const state = usePrototype(review, \"review\", false, true);", "  onDeleteProject?: () => void;\n  labTools?: (tools: RallyDeskLabTools) => ReactNode;\n}) {\n  const state = usePrototype(review, \"review\", false, true);")
      .replace("            <ReconnectNotice state={state} />\n            <div className=\"rd-mobile-events\">", "            {labTools?.({ draft: state.workingDraft, applyDraft: state.applyReviewDraft, seek: state.setPlayhead, selectedCutId: state.selectedId })}\n            <ReconnectNotice state={state} />\n            <div className=\"rd-mobile-events\">");
    output = output
      .replace('  const [playing, setPlaying] = useState(false);', '  const [playing, setPlaying] = useState(false);\n  const sourceRangeSequence = useRef(0);\n  const [sourceRange, setSourceRange] = useState<{ id: number; start: number; end: number } | null>(null);')
      .replace('  function setPlayhead(value: number) {\n    setOpenedClipId(null);', '  function setPlayhead(value: number) {\n    sourceRangeSequence.current += 1;\n    setSourceRange(null);\n    setOpenedClipId(null);')
      .replace('  function applyReviewDraft(draft: CutDraft) {\n    setBaseDraft(draft);', '  function applyReviewDraft(draft: CutDraft) {\n    setBaseDraft(draft);\n    setFinalPreview(draft.cutPreviewEnabled);\n    setPlaybackRate(draft.playbackRate);')
      .replace('    setClips((current) => current.flatMap((clip) => clip.id === selected.id ? [left, right] : [clip]));\n    const suppression =', '    setClips((current) => current.flatMap((clip) => clip.id === selected.id ? [left, right] : [clip]));\n    const splitScores = splitLabServeMarkers(workingDraft.scoreTracking, selected.id, left, right);\n    setScoreMarkers(splitScores.serveMarkers);\n    setRemovedModelMarkerIds(splitScores.removedModelMarkerIds);\n    const suppression =')
      .replace('  const workingDraft = useMemo<CutDraft>(() => {', sourceReviewPlayback + '\n  const workingDraft = useMemo<CutDraft>(() => {')
      .replace('    restoreHistory, historyStorageFailed, resetProjectChanges, applyReviewDraft,', '    restoreHistory, historyStorageFailed, resetProjectChanges, applyReviewDraft, playSourceRange,')
      .replace('  seek: (time: number) => void;\n  selectedCutId: string;', '  seek: (time: number) => void;\n  playSourceRange: (start: number, end: number) => () => void;\n  selectedCutId: string;')
      .replace('seek: state.setPlayhead, selectedCutId: state.selectedId', 'seek: state.setPlayhead, playSourceRange: state.playSourceRange, selectedCutId: state.selectedId');
    output = output
      .replace('    label:\n      cut.origin === "manual"', '    label:\n      review.labMode && cut.id.startsWith("candidate:") ? "Compact boundary proposal" :\n      cut.origin === "manual"')
      .replace('function rallyAgreementLabel(clip: Clip): string {', 'function rallyAgreementLabel(clip: Clip, labMode = false): string {\n  if (labMode && clip.id.startsWith("candidate:")) return "Compact boundary proposal";')
      .replace('{rallyAgreementLabel(clip)}', '{rallyAgreementLabel(clip, state.labMode)}')
      .replace('<span>review tasks</span>', '<span>{state.labMode ? "serve / clip checks" : "review tasks"}</span>')
      .replace('  const [clips, setClips] = useState<Clip[]>(initialClips);', '  const [clips, setClips] = useState<Clip[]>(initialClips);\n  const displayLabels = useMemo(() => labRallyDisplayLabels(clips), [clips]);\n  function rallyLabel(id: string) { return review.labMode ? displayLabels.get(id) ?? "Rally" : id; }\n  function serveLabel(marker: DesignServeMarker) { return review.labMode ? "Serve · " + (marker.rallyId ? rallyLabel(marker.rallyId) : formatPreciseTime(marker.timestamp)) : marker.id; }')
      .replace('  const [reviewMessage, setReviewMessage] = useState(', '  const [reviewMessage, setReviewMessageValue] = useState(')
      .replace('  const [scoreEnabled, setScoreEnabled] = useState(review.draft.scoreTracking.enabled);', '  function setReviewMessage(message: string) {\n    if (review.labMode) for (const [id, label] of [...displayLabels].sort((a, b) => b[0].length - a[0].length)) message = message.replaceAll(id, label);\n    setReviewMessageValue(message);\n  }\n  const [scoreEnabled, setScoreEnabled] = useState(review.draft.scoreTracking.enabled);')
      .replace('    stage, setStage, dark, setDark, helpOpen, setHelpOpen,', '    stage, setStage, dark, setDark, helpOpen, setHelpOpen, rallyLabel, serveLabel,')
      .replaceAll('>{clip.id}<', '>{state.rallyLabel(clip.id)}<')
      .replace('{clip.id} <span>{clip.label}</span>', '{state.rallyLabel(clip.id)} <span>{clip.label}</span>')
      .replace('aria-label={\x60\x24{clip.id},', 'aria-label={\x60\x24{state.rallyLabel(clip.id)},')
      .replace('title={\x60\x24{cut.id} ·', 'title={\x60\x24{state.rallyLabel(cut.id)} ·')
      .replace('<strong>{state.selectedScoreMarker.id}</strong>', '<strong>{state.serveLabel(state.selectedScoreMarker)}</strong>')
      .replace('setScoreMessage(\x60Serve \x24{marker.id} selected at', 'setScoreMessage(\x60\x24{serveLabel(marker)} selected at')
      .replace('<div className="rd-mobile-events"><RallyDeskEventRail state={state} /></div>', '{!state.labMode && <div className="rd-mobile-events"><RallyDeskEventRail state={state} /></div>}')
      .replace('<div className="rd-media-stack"><VideoStage state={state} desk resizeControl={videoResize} /><RallyDeskTimeline state={state} /></div>', '<div className="rd-media-stack"><VideoStage state={state} desk resizeControl={videoResize} /><RallyDeskTimeline state={state} /></div>\n            {state.labMode && <div className="rd-mobile-events"><RallyDeskEventRail state={state} /></div>}')
      .replace('  const [needsSource, setNeedsSource] = useState(review.sourceNeedsReconnect);', '  const [needsSource, setNeedsSource] = useState(review.labMode && review.videoUrl ? false : review.sourceNeedsReconnect);')
      .replaceAll('setNeedsSource(review.sourceNeedsReconnect);', 'setNeedsSource(review.labMode && review.videoUrl ? false : review.sourceNeedsReconnect);')
      .replace('  }, [review.sourceNeedsReconnect, review.videoUrl]);', '  }, [review.labMode, review.sourceNeedsReconnect, review.videoUrl]);')
      .replace('    stage, setStage, dark, setDark, helpOpen, setHelpOpen,', '    stage, setStage, dark, setDark, helpOpen, setHelpOpen,\n    labMode: Boolean(review.labMode), hasSourceFile: Boolean(review.sourceFile), hasProductAnalysis: Boolean(review.productAnalysis),')
      .replace('  if (!state.needsSource) return null;', '  if (!state.needsSource || (state.labMode && state.videoUrl)) return null;')
      .replace('<option value="__new__">＋ Start a new project…</option>', '{!state.labMode && <option value="__new__">＋ Start a new project…</option>}')
      .replace('case "new": state.selectProject(null); break;', 'case "new": if (!state.labMode) state.selectProject(null); break;')
      .replace('{state.stage === "review" && <RallyDeskGuidedTour />}', '{state.stage === "review" && !state.labMode && <RallyDeskGuidedTour />}')
      .replace('<button type="button" onClick={() => { state.setStage("source"); state.setHelpOpen(false); }}><strong>Set up a new game</strong><span>Choose the game window and camera options.</span></button>', '{!state.labMode && <button type="button" onClick={() => { state.setStage("source"); state.setHelpOpen(false); }}><strong>Set up a new game</strong><span>Choose the game window and camera options.</span></button>}')
      .replace('<label><span>Automatic cleanup</span><select value={state.cleanup} onChange={(event) => state.setCleanup(event.currentTarget.value as CleanupStrength)}><option value="off">Off</option><option value="light">Light</option><option value="standard">Standard</option><option value="strong">Strong</option></select></label>', '{state.labMode ? <p className="td-info-box">Cleanup is selected in the lab configuration above.</p> : <label><span>Automatic cleanup</span><select value={state.cleanup} onChange={(event) => state.setCleanup(event.currentTarget.value as CleanupStrength)}><option value="off">Off</option><option value="light">Light</option><option value="standard">Standard</option><option value="strong">Strong</option></select></label>}')
      .replace('  const actionLabel = state.exportKind === "video"', '  const unavailableReason = state.labMode && state.exportKind === "video" && !state.hasSourceFile\n    ? "MP4 export unavailable in lab"\n    : state.labMode && state.exportKind === "feedback" && !state.hasProductAnalysis\n      ? "Production project export unavailable" : null;\n  const actionLabel = state.exportKind === "video"')
      .replace('<p>Choose one output. Every option is prepared in this browser from the reviewed project.</p>', '<p>{state.labMode ? "YouTube chapters use the current review. Download lab edits from the configuration bar to save this experiment." : "Choose one output. Every option is prepared in this browser from the reviewed project."}</p>')
      .replace('<header><h2 id="td-readiness-title">Ready check</h2><span>Export remains available</span></header>', '<header><h2 id="td-readiness-title">Ready check</h2><span>{state.labMode ? "Available outputs below" : "Export remains available"}</span></header>')
      .replace('{state.needsSource ? "Reconnect for playback and MP4" : "Connected locally"}', '{state.labMode && !state.hasSourceFile ? "NAS video connected for playback" : state.needsSource ? "Reconnect for playback and MP4" : "Connected locally"}')
      .replace('<p className="td-privacy-line">The browser uses the original local video. Nothing is uploaded.</p>', '<p className="td-privacy-line">{state.labMode && !state.hasSourceFile ? "This lab plays the NAS source. MP4 rendering is not connected to the lab editor." : "The browser uses the original local video. Nothing is uploaded."}</p>')
      .replace('disabled={state.exportBusy || (state.exportKind === "video" && state.needsSource)}>{state.exportKind === "video" && state.needsSource ? "Reconnect source for MP4" : actionLabel}', 'disabled={Boolean(unavailableReason) || state.exportBusy || (state.exportKind === "video" && state.needsSource)}>{unavailableReason ?? (state.exportKind === "video" && state.needsSource ? "Reconnect source for MP4" : actionLabel)}');
  }
  if (relative === "designs/taste/index.tsx") {
    output = output
      .replace('  selectedCutId: string;', '  selectedCutId: string;\n  currentTime: number;')
      .replace('  labTools,\n', '  labTools,\n  labComparison,\n')
      .replace('  labTools?: (tools: RallyDeskLabTools) => ReactNode;', '  labTools?: (tools: RallyDeskLabTools) => ReactNode;\n  labComparison?: (tools: RallyDeskLabTools) => ReactNode;')
      .replace('selectedCutId: state.selectedId })}', 'selectedCutId: state.selectedId, currentTime: state.playhead })}')
      .replace('{state.labMode && <div className="rd-mobile-events">', '{labComparison?.({ draft: state.workingDraft, applyDraft: state.applyReviewDraft, seek: state.setPlayhead, playSourceRange: state.playSourceRange, selectedCutId: state.selectedId, currentTime: state.playhead })}\n            {state.labMode && <div className="rd-mobile-events">')
      .replace('    label:\n      review.labMode', '    label:\n      review.labMode && cut.id.startsWith("human:") ? "Human export region" :\n      review.labMode')
      .replace('function rallyAgreementLabel(clip: Clip, labMode = false): string {', 'function rallyAgreementLabel(clip: Clip, labMode = false): string {\n  if (labMode && clip.id.startsWith("human:")) return "Human export";')
      .replace('clip.origin === "manual" ? "Added by you"', 'state.labMode && clip.id.startsWith("human:") ? "Human export" : clip.origin === "manual" ? "Added by you"')
      .replace('clip.origin === "model" ? ` · ${Math.round(clip.confidence * 100)}%`', 'clip.origin === "model" && !clip.id.startsWith("human:") ? ` · ${Math.round(clip.confidence * 100)}%`');
  }
  output = output.replace(/\n+$/, "\n");
  const destination = path.join(destinationRoot, relative);
  mkdirSync(path.dirname(destination), { recursive: true });
  writeFileSync(destination, output, "utf8");
  manifest.push({ source: `prod/src/${relative}`, destination: `components/production-lab/editor/${relative}`, sourceSha256: sha256(source), snapshotSha256: sha256(output) });
}
writeFileSync(path.join(destinationRoot, "lib/lab-split-markers.ts"), labSplitSource);
writeFileSync(path.join(destinationRoot, "lib/lab-display-labels.ts"), labDisplayLabelsSource);
writeFileSync(path.join(destinationRoot, "lab-base.css"), `/* Production base styles scoped to the copied editor, plus resets for lab-wide headings. */
.production-lab-root { min-width: 320px; color: #14231f; background: #f6f7f2; font-family: Manrope, Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-synthesis: none; text-rendering: optimizeLegibility; }
.production-lab-root * { box-sizing: border-box; }
.production-lab-root button, .production-lab-root input, .production-lab-root select { font: inherit; }
.production-lab-root h1 { margin: .67em 0; font-size: 2em; line-height: normal; letter-spacing: normal; }
.production-lab-root h1 em { color: inherit; font-style: italic; }
.production-lab-root .rd-register-list > button { min-width: 0; grid-template-columns: 54px minmax(0, 1fr); }
.production-lab-root .rd-register-id { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.production-lab-root .td-clip-inspector > header > div { min-width: 0; }
.production-lab-root .td-clip-inspector h2 { overflow-wrap: anywhere; }
.production-lab-root .rd-event-editor > header { min-width: 0; flex-wrap: wrap; }
.production-lab-root .rd-event-editor > header strong { min-width: 0; overflow-wrap: anywhere; }
`);
const assetRoot = path.join(repository, "public/production-lab");
mkdirSync(assetRoot, { recursive: true });
for (const asset of ["volleysplice-logo.png", "volleysplice-icon-transparent.png"]) {
  copyFileSync(path.join(repository, "prod/public/runtime", asset), path.join(assetRoot, asset));
}
writeFileSync(path.join(destinationRoot, "snapshot.json"), JSON.stringify({
  schemaVersion: 1,
  hashEncoding: "UTF-8 with LF-normalized line endings",
  sourceEntry: "prod/src/designs/taste/index.tsx",
  productionEntry: "prod/src/App.tsx:ReadyProjectEditor",
  adaptations: ["relative imports", "lab-only storage keys", "type-only project and review adapter contracts", "Next-compatible lab asset URLs", "labTools review extension", "bounded source review playback", "preserve serve identities across native splits", "readable rally and serve presentation labels", "adjacent review and video on mobile", "lab persistence copy", "explicit unavailable export actions", "lab-owned cleanup configuration", "hide production project setup and automatic tour in lab mode"],
  generatedHelpers: [{ file: "components/production-lab/editor/lib/lab-split-markers.ts", sha256: sha256(labSplitSource) }, { file: "components/production-lab/editor/lib/lab-display-labels.ts", sha256: sha256(labDisplayLabelsSource) }],
  files: manifest,
}, null, 2) + "\n");
console.log(`Copied ${manifest.length} production editor dependencies into the lab.`);
