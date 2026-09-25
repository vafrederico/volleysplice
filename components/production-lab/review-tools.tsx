"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { applyLabProposal, restoreLabRemoval, undoLabSplit, type LabConfiguration, type LabInterval, type ProductionEditorLabTask } from "@/lib/production-editor-lab";
import { currentLabDecision, draftLabEvents, draftWithLabEvents, recordLabDecision } from "@/lib/production-editor-lab-draft";
import { servingDecisionExplanation } from "@/lib/labeling-serving";
import type { ResearchSignals } from "@/lib/labeling-research";
import type { RallyDeskLabTools } from "./editor/designs/taste";
import { materializeFinalCutIntervals, rallySuppressionDecisionKey } from "./editor/lib/cut-draft";
import styles from "./lab.module.css";

function stamp(value: number) {
  return `${Math.floor(value / 60)}:${(value % 60).toFixed(2).padStart(5, "0")}`;
}

type Tab = "removals" | "proposals" | "splits";
type Item = { id: string; start: number; end: number; parentId: string; label: string };

function Signals({ signals, range, seek, currentTime }: { signals: ResearchSignals; range: LabInterval; seek: (time: number) => void; currentTime: number }) {
  const heads = [{ key: "live", name: "Live play", color: "#278261" }, { key: "serve", name: "Serve / start", color: "#527cc2" },
    { key: "end", name: "Rally end", color: "#c15b47" }, { key: "keep", name: "Keep", color: "#9671b3" }] as const;
  const indices = signals.times.flatMap((time, i) => time >= range.start && time <= range.end ? [i] : []);
  const span = Math.max(.1, range.end - range.start);
  const inRange = currentTime >= range.start && currentTime <= range.end;
  const playheadX = 600 * (currentTime - range.start) / span;
  const sample = inRange && indices.length ? indices.reduce((best, i) => Math.abs(signals.times[i] - currentTime) < Math.abs(signals.times[best] - currentTime) ? i : best, indices[0]) : null;
  return <details className={styles.signals}>
    <summary>Model signals for this region</summary>
    <svg className={styles.signalSvg} viewBox="0 0 600 80" preserveAspectRatio="none" role="img" aria-label="Four model signals; click to seek"
      onClick={event => { const box = event.currentTarget.getBoundingClientRect(); const time = range.start + ((event.clientX - box.left) / box.width) * span;
        seek(time); }}>
      {heads.map(head => <polyline key={head.key} fill="none" stroke={head.color} strokeWidth="1.6" vectorEffect="non-scaling-stroke"
        points={indices.map(i => `${600 * (signals.times[i] - range.start) / span},${77 - 74 * signals[head.key][i]}`).join(" ")} />)}
      {inRange && <g pointerEvents="none" aria-label={`Video playhead ${stamp(currentTime)}`}>
        <line x1={playheadX} x2={playheadX} y1="0" y2="80" stroke="#fff" strokeWidth="4" vectorEffect="non-scaling-stroke" />
        <line data-testid="signal-playhead" x1={playheadX} x2={playheadX} y1="0" y2="80" stroke="#d81717" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <path d={`M ${playheadX - 4} 0 L ${playheadX + 4} 0 L ${playheadX} 6 Z`} fill="#d81717" />
      </g>}
    </svg>
    <div className={styles.regionTimes}><span>{stamp(range.start)}</span><output>Playhead {stamp(currentTime)}{inRange ? "" : " · outside this region"}</output><span>{stamp(range.end)}</span></div>
    <div className={styles.signalLegend}>{heads.map(head => <span key={head.key}><i style={{ background: head.color }} />{head.name}{sample === null ? "" : ` ${(signals[head.key][sample] * 100).toFixed(0)}%`}</span>)}</div>
    <p className={styles.notice}>Model scores help inspect timing; serving side is reviewed in the editor’s serve panel.</p>
  </details>;
}

export function ReviewTools({ task, configuration, tools, download }: {
  task: ProductionEditorLabTask; configuration: LabConfiguration; tools: RallyDeskLabTools; download: () => void;
}) {
  const initialTab: Tab = configuration.proposals.length && configuration.id === "compact-guidance" ? "proposals" : "removals";
  const [tab, setTab] = useState<Tab>(initialTab);
  const [selected, setSelected] = useState<Record<Tab, number>>({ removals: 0, proposals: 0, splits: 0 });
  const [message, setMessage] = useState("");
  const [expanded, setExpanded] = useState(true);
  const [showAllProposals, setShowAllProposals] = useState(false);
  const queues = useMemo<Record<Tab, Item[]>>(() => ({
    removals: configuration.removals.map(item => ({ ...item, label: `${item.kind === "whole" ? "Removed rally" : item.kind === "gap" ? "Removed between rallies" : item.kind === "prefix" ? "Trimmed start" : "Trimmed end"} · ${item.parentId}` })),
    proposals: configuration.proposals.filter(item => showAllProposals || item.recommended !== false).map(item => ({ id: item.id, parentId: item.parentId,
      start: Math.min(...item.before.map(event => event.start)), end: Math.max(...item.before.map(event => event.end)), label: `Boundary proposal · ${item.parentId}` })),
    splits: configuration.splits.map(item => ({ ...item, label: `Separate rallies · ${item.parentId}` })),
  }), [configuration, showAllProposals]);
  const tabs = (Object.keys(queues) as Tab[]).filter(key => queues[key].length && (key !== "proposals" || configuration.id === "compact-guidance"));
  const activeTab = tabs.includes(tab) ? tab : tabs[0];
  const items = activeTab ? queues[activeTab] : [];
  const index = Math.min(selected[activeTab] ?? 0, Math.max(0, items.length - 1));
  const item = items[index];
  const status = item ? currentLabDecision(tools.draft, item.id, item) : undefined;
  const activeProposal = configuration.proposals.find(proposal => proposal.id === item?.id);
  const activeRemoval = configuration.removals.find(removal => removal.id === item?.id);
  const suppressionEvidence = activeRemoval && configuration.suppressionPolicy ? configuration.suppression?.suggestions.filter(s =>
    s.eligiblePolicyIds.includes(configuration.suppressionPolicy!) && s.start < activeRemoval.end && s.end > activeRemoval.start) ?? [] : [];
  const activeSplit = configuration.splits.find(split => split.id === item?.id);
  const splitStillExists = activeSplit && [activeSplit.leftEventId, activeSplit.rightEventId].every(id => tools.draft.cuts.some(cut => cut.id === id && cut.included));
  const checked = items.filter(entry => currentLabDecision(tools.draft, entry.id, entry)).length;
  const selectedCut = tools.draft.cuts.find(cut => cut.id === tools.selectedCutId);
  const servePrediction = configuration.servingPredictions?.find(prediction => prediction.id === selectedCut?.id && Math.abs(prediction.anchor - selectedCut.coreStart) < 1e-6);
  const effectiveCutIds = new Set(materializeFinalCutIntervals(tools.draft, configuration.suppression).intervals.flatMap(interval => interval.cutIds));
  const context = item ? { start: Math.max(0, item.start - 2), end: Math.min(task.durationSeconds, item.end + 2) }
    : { start: Math.max(0, (selectedCut?.coreStart ?? configuration.events[0]?.start ?? 0) - 2), end: Math.min(task.durationSeconds, (selectedCut?.coreEnd ?? configuration.events[0]?.end ?? 20) + 2) };
  const playheadInContext = tools.currentTime >= context.start && tools.currentTime <= context.end;
  const focusedItem = useRef<string | null>(null);
  useEffect(() => {
    if (!item || focusedItem.current === item.id) return;
    focusedItem.current = item.id;
    tools.seek(context.start);
  }, [item, context.start, tools.seek]);

  function choose(next: number) {
    const safe = Math.max(0, Math.min(items.length - 1, next));
    setSelected(current => ({ ...current, [activeTab]: safe })); setMessage("");
    if (items[safe]) tools.seek(Math.max(0, items[safe].start - 2));
  }
  function nextPending() {
    for (let offset = 1; offset <= items.length; offset++) {
      const next = (index + offset) % items.length;
      if (!currentLabDecision(tools.draft, items[next].id, items[next])) { choose(next); return; }
    }
    setMessage("Every item in this queue has a decision. You can revisit any of them or adjust the timeline.");
  }
  function decide(action: "accept" | "restore" | "apply" | "dismiss" | "merge" | "adjusted") {
    if (!item) return;
    let draft = tools.draft;
    const events = draftLabEvents(draft, configuration);
    const nativeSuppression = configuration.suppressionPolicy && activeRemoval && (action === "restore" || action === "accept");
    const result = action === "restore" && activeRemoval && !nativeSuppression ? restoreLabRemoval(events, activeRemoval)
      : action === "apply" && activeProposal ? applyLabProposal(events, activeProposal)
      : action === "merge" && activeSplit ? undoLabSplit(events, activeSplit) : null;
    if (result && !result.ok) { setMessage(result.error); return; }
    if (result) draft = draftWithLabEvents(draft, result.events);
    if (nativeSuppression && activeRemoval) draft = { ...draft,
      cuts: action === "restore" ? draft.cuts.map(cut => cut.id === activeRemoval.parent.id ? { ...cut, included: true } : cut) : draft.cuts,
      suppressionDecisionOverrides: { ...draft.suppressionDecisionOverrides,
      [rallySuppressionDecisionKey(activeRemoval.parent.id)]: action === "restore" ? "keep" : "suppress" } };
    const label = action === "restore" ? "Footage restored" : action === "apply" ? "Proposal applied" : action === "dismiss" ? "Original boundaries kept"
      : action === "merge" ? "Split undone" : action === "adjusted" ? "Manually checked" : activeTab === "splits" ? "Split confirmed" : "Removal kept";
    tools.applyDraft(recordLabDecision(draft, item.id, item, label));
    setMessage(`${label}. The editor’s Undo also undoes this review decision.`);
  }
  function playContext() {
    tools.playSourceRange(context.start, context.end);
  }

  return <section className={styles.tools} aria-label="Model review workflow">
    <header className={styles.queueHeader}>
      <div><strong>{configuration.id === "compact-guidance" ? "Compact boundary guidance" : configuration.id === "boundary-undo" ? "Review what changed" : configuration.removals.length ? "Review suppressed rallies" : "Review and adjust"}</strong>
        <span> {items.length ? `${checked} of ${items.length} checked` : "Use the production controls to edit rallies and serves"}</span></div>
      <div className={styles.toolbar}><button type="button" onClick={download}>Download lab edits</button>
        {tabs.length > 0 && <button type="button" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? "Collapse review" : "Open review"}</button>}</div>
    </header>
    {expanded && tabs.length > 0 && <>
      <div className={styles.queueTabs}>{tabs.map(key => <button key={key} type="button" aria-pressed={activeTab === key}
        onClick={() => { setTab(key); setMessage(""); }}>
        {key === "removals" ? "Removed footage" : key === "proposals" ? "Boundary proposals" : "Rally splits"} ({queues[key].length})</button>)}
        {configuration.id === "compact-guidance" && <button type="button" onClick={() => { setShowAllProposals(value => !value); setSelected(current => ({ ...current, proposals: 0 })); }}>
          {showAllProposals ? "Recommended only" : `All ${configuration.proposals.length} flagged rallies`}</button>}</div>
      {item && <div className={styles.decision}>
        <div className={styles.decisionTop}>
          <select aria-label="Review region" value={index} onChange={event => choose(Number(event.target.value))}>
            {items.map((entry, i) => <option key={entry.id} value={i}>{currentLabDecision(tools.draft, entry.id, entry) ? "✓ " : ""}{i + 1}. {entry.label} · {stamp(entry.start)}–{stamp(entry.end)}</option>)}
          </select>
          <nav aria-label="Review navigation"><button type="button" disabled={index === 0} onClick={() => choose(index - 1)}>Previous</button><span>{index + 1} / {items.length}</span><button type="button" onClick={nextPending}>Next unchecked</button></nav>
        </div>
        {suppressionEvidence.length > 0 && <p>Suppression evidence: {suppressionEvidence.map(s => `${stamp(Math.max(s.start, item.start))}–${stamp(Math.min(s.end, item.end))} (${(s.score * 100).toFixed(0)}% mean head score)`).join(", ")}. Any eligible overlap removes the whole rally by default; the score is not a calibrated probability that the removal is correct.</p>}
        <p>{activeTab === "removals" ? `${stamp(item.start)}–${stamp(item.end)} · ${(item.end - item.start).toFixed(2)}s removed. Watch the source, then keep the removal or restore this portion.`
          : activeTab === "proposals" ? status === "Proposal applied" ? "The proposed boundaries are active. Adjust them in the editor or use Undo to return to the previous boundaries." : "Inspect the original and proposed starts and ends, apply the proposal, or keep the original."
          : "Confirm that these are separate rallies. This decision stays visible even when export padding joins the gap."}</p>
        {activeTab === "proposals" && activeProposal && <p>Proposed: {activeProposal.after.map(event => `${stamp(event.start)}–${stamp(event.end)}`).join(" · ")}</p>}
        <div className={styles.miniTimeline} role="img" aria-label="Current rally coverage with the review region highlighted">
          {tools.draft.cuts.filter(cut => cut.included && effectiveCutIds.has(cut.id) && cut.coreStart < context.end && cut.coreEnd > context.start).map(cut => <span key={cut.id}
            style={{ left: `${100 * (Math.max(context.start, cut.coreStart) - context.start) / (context.end - context.start)}%`, width: `${100 * (Math.min(context.end, cut.coreEnd) - Math.max(context.start, cut.coreStart)) / (context.end - context.start)}%` }} />)}
          <i style={{ left: `${100 * (item.start - context.start) / (context.end - context.start)}%`, width: `${100 * Math.max(.02, item.end - item.start) / (context.end - context.start)}%` }} />
          {playheadInContext && <b className={styles.regionPlayhead} data-testid="region-playhead" aria-label={`Video playhead ${stamp(tools.currentTime)}`}
            style={{ left: `${100 * (tools.currentTime - context.start) / (context.end - context.start)}%` }} />}
        </div>
        <div className={styles.regionTimes}><span>{stamp(context.start)}</span><output>Playhead {stamp(tools.currentTime)}{playheadInContext ? "" : " · outside this region"}</output><span>{stamp(context.end)}</span></div>
        <div className={styles.actions}>
          <button type="button" onClick={playContext}>Play with context</button>
          {activeTab === "proposals" ? <><button className={styles.primary} type="button" disabled={status === "Proposal applied"} onClick={() => decide("apply")}>Apply proposal</button><button type="button" disabled={status === "Proposal applied"} onClick={() => decide("dismiss")}>Keep original</button></>
            : activeTab === "splits" ? <><button className={styles.primary} type="button" disabled={!splitStillExists} onClick={() => decide("accept")}>Confirm split</button><button type="button" disabled={!splitStillExists} onClick={() => decide("merge")}>Undo split</button></>
              : <><button className={styles.primary} type="button" disabled={status === "Footage restored" && !configuration.suppressionPolicy} onClick={() => decide("accept")}>Keep removed</button><button type="button" disabled={status === "Footage restored"} onClick={() => decide("restore")}>Restore footage</button></>}
          <button type="button" onClick={() => {
            const adjacentId = activeRemoval?.rightEventId ?? activeRemoval?.leftEventId ?? activeSplit?.leftEventId;
            const adjacent = tools.draft.cuts.find(cut => cut.id === adjacentId) ?? tools.draft.cuts.find(cut => cut.id === item.parentId);
            tools.seek(adjacent ? adjacent.coreStart + Math.min(.01, (adjacent.coreEnd - adjacent.coreStart) / 2) : item.start);
            setMessage("Use the Start/End controls or drag the timeline boundaries, then choose Mark checked."); }}>Adjust in editor</button>
          <button type="button" onClick={() => decide("adjusted")}>Mark checked</button>
          <span className={styles.state}>{status ?? "Needs your decision"}</span>
        </div>
      </div>}
      {message && <p className={styles.message} role="status">{message}</p>}
    </>}
    {!tabs.length && <p className={styles.notice}>{configuration.humanReference ? "Saved human regions and serve markers are loaded. Play the final cut or adjust this lab copy; the saved labels stay unchanged." : configuration.signals ? "Model rally boundaries are loaded. Inspect its four timing signals, adjust boundaries, or add a missed rally. Serving-side predictions have not been prepared for these anchors." : configuration.id === "compact-standalone" ? "Compact’s own rallies are loaded. Review uncertain starts and serving sides, adjust boundaries, or add a missed rally." : "The ensemble rallies are loaded with production serve predictions. Use the review list, boundary controls and serve decisions as you would in production."}</p>}
    {(configuration.signals || configuration.id.startsWith("compact") || configuration.id === "boundary-undo") && <Signals key={`${context.start}:${context.end}`} signals={configuration.signals ?? task.signals} range={context} seek={tools.seek} currentTime={tools.currentTime} />}
    {servePrediction && <p className={styles.notice}>Serve model at {stamp(servePrediction.anchor)}: {servePrediction.verdict === "not-serve" ? "no serve marker" : servePrediction.verdict === "review" ? "needs review" : `${servePrediction.side} side`}. {servingDecisionExplanation(servePrediction)}</p>}
  </section>;
}
