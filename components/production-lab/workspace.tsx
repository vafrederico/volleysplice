"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LabConfiguration, ProductionEditorLabTask } from "@/lib/production-editor-lab";
import { hasVisibleTime, initialLabDraft, labSeed, labStorageKey, type LabDraft } from "@/lib/production-editor-lab-draft";
import { RallyDesk } from "./editor/designs/taste";
import { parseCutDraft, type CutDraft } from "./editor/lib/cut-draft";
import type { ReadyDesignReview } from "./editor/designs/useDesignReview";
import type { OnDeviceSuppression } from "./editor/lib/on-device/types";
import { ReviewTools } from "./review-tools";
import { ComparisonRail } from "./comparison-rail";
import styles from "./lab.module.css";
import { canCombineSuppression, LAB_SUPPRESSION_OPTIONS, withLabSuppression, type LabSuppressionMode } from "@/lib/production-editor-lab-suppression";

function saveJson(value: unknown, filename: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename;
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Trial({ task, configuration, reportSave }: { task: ProductionEditorLabTask; configuration: LabConfiguration; reportSave: (message: string) => void }) {
  const storageKey = labStorageKey(task, configuration);
  const seed = useMemo(() => labSeed(task, configuration), [task, configuration]);
  const modelDraft = useMemo(() => initialLabDraft(task, configuration), [task, configuration]);
  const [initial] = useState<LabDraft>(() => {
    try { const raw = localStorage.getItem(storageKey); return (raw && parseCutDraft(raw, seed)) || modelDraft; }
    catch { return modelDraft; }
  });
  const latest = useRef<CutDraft>(initial);
  const saveDraft = useCallback((draft: CutDraft) => {
    latest.current = draft;
    try { localStorage.setItem(storageKey, JSON.stringify(draft)); reportSave("This mode’s edits are saved in this browser."); }
    catch { reportSave("Browser storage is unavailable. Download lab edits to keep this session."); }
  }, [storageKey, reportSave]);
  const download = useCallback(() => saveJson({ kind: "volleycut-editor-lab-review", schemaVersion: 1,
    recordingId: task.id, configurationId: configuration.id, sourceRevision: configuration.humanReference?.revision ?? configuration.sourceRevision ?? task.sourceRevision,
    savedAt: new Date().toISOString(), draft: latest.current }, `${task.name.replace(/\.[^.]+$/, "")}-${configuration.id}-lab.json`), [task, configuration]);
  const suppression = useMemo<OnDeviceSuppression | undefined>(() => configuration.suppression ? {
    modelId: "frozen-production-suppression", artifactSha256: task.provenance.sourceHashes["production-replay.json"] ?? task.sourceRevision,
    weightsSha256: "", decoderVersion: "frozen-production", policyContractVersion: modelDraft.suppressionContractVersion,
    probabilities: new Float32Array(), decodedIntervals: [], ...configuration.suppression,
  } : undefined, [configuration, task, modelDraft]);
  const review = useMemo<ReadyDesignReview>(() => ({
    state: "ready", labMode: true, projectId: storageKey, projectName: task.name, sourceName: task.name,
    sourceSize: 0, sourceNeedsReconnect: false, sourceFile: null, videoUrl: task.mediaUrl, productAnalysis: null,
    duration: task.durationSeconds, width: 1920, height: 1080, gameStart: 0, gameEnd: task.durationSeconds,
    cropCourt: false, sideSwitchEnabled: false, projectStatus: "ready", draft: initial, modelDraft, draftSeed: seed,
    suppression,
    cleanupSuggestions: (suppression?.suggestions ?? []).map(suggestion => ({
      id: suggestion.id, logicalId: suggestion.logicalId, eligiblePolicyIds: suggestion.eligiblePolicyIds,
      start: suggestion.start, end: suggestion.end, score: suggestion.score, cutId: null, decision: "pending",
    })),
    projects: [{ id: storageKey, name: task.name, status: "ready", exportJob: null }], workActivity: [], exportJob: null,
    selectProject: () => {}, attachSource: async () => ({ ok: false, message: "This lab recording is served directly from the NAS.", videoUrl: task.mediaUrl }),
    queueVideoExport: () => {}, saveDraft,
  }), [storageKey, task, initial, modelDraft, seed, suppression, saveDraft]);
  return <RallyDesk review={review} labTools={tools => <ReviewTools task={task} configuration={configuration} tools={tools} download={download} />}
    labComparison={tools => <ComparisonRail task={task} configuration={configuration} tools={tools} />} />;
}

export default function Workspace() {
  const [task, setTask] = useState<ProductionEditorLabTask | null>(null);
  const [error, setError] = useState("");
  const [configurationId, setConfigurationId] = useState(() => new URLSearchParams(location.search).get("mode") ?? "production");
  const [suppressionMode, setSuppressionMode] = useState<LabSuppressionMode>(() => {
    const value = new URLSearchParams(location.search).get("suppression") ?? "none";
    return Object.hasOwn(LAB_SUPPRESSION_OPTIONS, value) ? value as LabSuppressionMode : "none";
  });
  const [saveMessage, setSaveMessage] = useState("Each mode has its own editable draft.");
  const [taskId, setTaskId] = useState(() => new URLSearchParams(location.search).get("task") ?? "");
  const [recordings, setRecordings] = useState<Array<{ id: string; name: string; tier: string }>>([]);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/editor-lab/tasks", { signal: controller.signal }).then(async response => {
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error);
      setRecordings(payload.tasks);
      setTaskId(current => current || payload.tasks[0]?.id || "");
    }).catch(reason => { if (reason.name !== "AbortError") setError(reason.message); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!taskId) return;
    const controller = new AbortController();
    setTask(null); setError("");
    fetch(`/api/editor-lab/tasks/${encodeURIComponent(taskId)}`, { signal: controller.signal }).then(async response => {
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "The lab recording could not be loaded.");
      setTask(payload);
    }).catch(reason => { if (reason.name !== "AbortError") setError(reason.message); });
    return () => controller.abort();
  }, [taskId]);
  const baseConfiguration = task?.configurations.find(item => item.id === configurationId) ?? task?.configurations[0];
  const configuration = useMemo(() => task && baseConfiguration ? withLabSuppression(task, baseConfiguration, suppressionMode) : undefined,
    [task, baseConfiguration, suppressionMode]);
  const switchConfiguration = (id: string) => {
    setConfigurationId(id); const url = new URL(location.href); url.searchParams.set("task", taskId); url.searchParams.set("mode", id);
    history.replaceState(null, "", url); setSaveMessage("Each mode has its own editable draft.");
  };
  return <div className={`production-lab-root ${styles.root}`}>
    <div className={styles.bar}>
      <div className={styles.heading}><strong>Editor lab</strong><span>{task?.name ?? "Production review experience"}</span>
        <a href={`/labelv2?task=${encodeURIComponent(taskId)}`}>Back to labels</a></div>
      {!!recordings.length && <label>Video <select aria-label="Editor lab video" value={taskId} onChange={event => {
        const id = event.target.value; setTaskId(id);
        const url = new URL(location.href); url.searchParams.set("task", id); history.replaceState(null, "", url);
        setSaveMessage("Each mode has its own editable draft.");
      }}>{recordings.map(recording => <option key={recording.id} value={recording.id}>{recording.name} · {recording.tier}</option>)}</select></label>}
      {task && <div className={styles.modes} role="group" aria-label="Model configuration">{task.configurations.map(item => <button key={item.id} type="button"
        aria-pressed={baseConfiguration?.id === item.id} onClick={() => switchConfiguration(item.id)}>{item.label}</button>)}</div>}
      {task && baseConfiguration && <label>Suppression combination <select aria-label="Suppression combination"
        disabled={!task.suppressionSource || !canCombineSuppression(baseConfiguration)}
        value={configuration === baseConfiguration ? "none" : suppressionMode} onChange={event => {
          const mode = event.target.value as LabSuppressionMode; setSuppressionMode(mode);
          const url = new URL(location.href); url.searchParams.set("suppression", mode); history.replaceState(null, "", url);
          setSaveMessage("Each model and suppression combination has its own saved draft.");
        }}>{Object.entries(LAB_SUPPRESSION_OPTIONS).filter(([key]) => key !== "direct" || !baseConfiguration.id.startsWith("production"))
          .map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
        <small>{!canCombineSuppression(baseConfiguration) ? " This reference or review mode keeps its existing policy."
          : !task.suppressionSource ? " Frozen suppression predictions are unavailable for this video." : " Compare with Off; review removals to restore rallies."}</small></label>}
      <div className={styles.description}><p>{configuration?.description ?? "Loading the prepared model results…"}</p>
        <small role="status">{saveMessage}</small></div>
      {task && configuration && <div className={styles.description}><p>{configuration.events.filter(event => hasVisibleTime(event, configuration.ignoredIntervals ?? task.ignoredIntervals)).length} {configuration.humanReference ? "saved human regions" : configuration.suppressionBaseId ? "initially kept rallies" : "starting rally candidates"}
        {configuration.suppressionBaseId && ` · ${new Set(configuration.removals.map(r => r.parent.id)).size} rallies flagged for removal review`}
        {" · Original ignored footage excluded · Changes stay in the lab"}</p></div>}
    </div>
    {error ? <div className={styles.error} role="alert">{error} <a href={`/labelv2?task=${encodeURIComponent(taskId)}`}>Return to labels</a></div>
      : task && configuration ? <Trial key={labStorageKey(task, configuration)} task={task} configuration={configuration} reportSave={setSaveMessage} />
        : <p className={styles.loading}>Loading video and model configurations…</p>}
  </div>;
}
