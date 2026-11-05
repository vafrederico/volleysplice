import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./GuidedTour.module.css";

export type RalliesTourDestination = "review" | "edit" | "browse" | "add" | "editor" | "export" | "settings";
const STORAGE_KEY = "volleycut:rallies-tour:v1";
let memoryState: string | null = null;

function readProgress() {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (value) return value;
    // Respect people who already closed or completed the previous web walkthrough.
    const previous = window.localStorage.getItem("volleycut:rally-desk-tour:v1");
    return previous === "done" || previous === "dismissed" ? previous : null;
  } catch { return memoryState; }
}

function writeProgress(value: string) {
  memoryState = value;
  try { window.localStorage.setItem(STORAGE_KEY, value); } catch { /* Session progress still works. */ }
}

export function RalliesGuidedTour({ hasRallies, scoreEnabled, sourceAvailable, restartToken, onNavigate, onActiveChange }: {
  hasRallies: boolean;
  scoreEnabled: boolean;
  sourceAvailable: boolean;
  restartToken: number;
  onNavigate: (destination: RalliesTourDestination) => void;
  onActiveChange: (active: boolean) => void;
}) {
  const steps = useMemo(() => [
    ...(!sourceAvailable ? [{ id: "reconnect", target: "rw-reconnect", view: "review", title: "Reconnect the original video", body: "Your analysis and review edits are saved. Choose Reconnect video and select the original recording to watch it again. Playback stays paused until the video is available." }] : []),
    { id: "review", target: "rw-review", view: "review", title: "Review one rally at a time", body: "Rallies opens first. The reviewing count shows what still needs a decision. Watch each rally here; playback stops at its padded end. Space plays or pauses, and playback speed is shared with Editor." },
    ...(hasRallies ? [
      { id: "decisions", target: "rw-decisions", view: "edit", title: "Keep or remove a rally", body: "Choose Keep or Remove after watching. The web shortcuts stay the same: K keeps the rally, Backspace removes it, and Ctrl+Z undoes your last edit." },
      { id: "trim", target: "rw-trim", view: "edit", title: "Adjust the rally edges", body: "Drag either trim handle, use its arrow keys for fine adjustments, or nudge an edge by one second. The outlined diamond marks this rally’s position; the moving line follows playback." },
      ...(scoreEnabled ? [{ id: "serve", target: "rw-serve", view: "edit", title: "Check the serving side", body: "Choose Near, Far, or Not a serve. N and F keep their web bindings. The serving side determines the point history and score; team names and score options are available in Editor." }] : []),
      { id: "navigation", target: "rw-navigation", view: "edit", title: "Move without making a decision", body: "Previous rally and Next rally browse the recording. R clicks Next review and respects its disabled state: finish the current keep/remove and serve decisions before continuing." },
    ] : []),
    { id: "all-rallies", target: "rw-all-rallies", view: "review", title: "Open all rallies", body: "All rallies opens the recording browser. You can return to the review queue whenever you want; browsing never keeps, removes, or trims a rally for you." },
    { id: "list", target: "rw-list", view: "browse", title: "Browse the rally list", body: "Scroll the time and score rows. Matching outlined diamonds identify the rally at the center of the list and its position on the timelines. Scrolling the list leaves the video where it is. Select a row to seek, then choose Edit rally to adjust it." },
    { id: "source", target: "rw-source", view: "browse", title: "Follow the recording", body: "The two columns show the first and second halves of the recording. A single filled circle follows the video and moves to the other column in the second half. The outlined diamond follows the list. Seek on either column to follow the video again." },
    { id: "missing", target: "rw-missing", view: "browse", title: "Find something the analysis missed", body: "Choose Add missing rally to keep the video visible beside the source timeline. You can look through gaps between detected rallies without changing the saved clips." },
    { id: "add", target: "rw-add", view: "add", title: "Mark a missing rally explicitly", body: "Seek to the first frame and choose Mark rally start. Seek to the last frame and choose Mark end & add rally. Shift+R uses the same two steps. Opening this screen or continuing this tutorial never marks a boundary or adds a clip." },
    { id: "editor", target: "rd-video", view: "editor", title: "Use the full Editor", body: "The Editor tab opens the existing workspace for score markers, side switches, excluded footage, cleanup, and detailed controls. Rallies, Editor, and Export share your edits and playback position." },
    { id: "export", target: "rw-export-tab", view: "export", title: "Save the finished game", body: "Export contains the final video, YouTube chapters, and Save project. Your keep/remove decisions, trims, padding, and score corrections carry through to the saved result." },
    { id: "settings", target: "rw-settings", view: "settings", title: "Fine-tune when you need to", body: "Settings controls rally padding, short joined gaps, and optional cleanup. Open the app menu to restart this tutorial at any time. This walkthrough has only navigated the interface; it has made no review edits." },
  ] as { id: string; target: string; view: RalliesTourDestination; title: string; body: string }[], [hasRallies, scoreEnabled, sourceAvailable]);
  const [progress, setProgress] = useState<string | null>(null);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  const callbacks = useRef({ onNavigate, onActiveChange });
  callbacks.current = { onNavigate, onActiveChange };
  const dialog = useRef<HTMLElement>(null);
  const lastRestart = useRef(restartToken);
  const index = steps.findIndex(step => step.id === progress);
  const step = steps[index];
  const active = Boolean(step);

  useEffect(() => {
    const stored = readProgress();
    setProgress(stored === "done" || stored === "dismissed" || steps.some(item => item.id === stored) ? stored : steps[0].id);
  }, []);
  useEffect(() => {
    if (lastRestart.current === restartToken) return;
    lastRestart.current = restartToken;
    writeProgress(steps[0].id);
    setProgress(steps[0].id);
  }, [restartToken, steps]);
  useEffect(() => {
    callbacks.current.onActiveChange(active);
    return () => callbacks.current.onActiveChange(false);
  }, [active]);
  useEffect(() => {
    if (!step) return;
    callbacks.current.onNavigate(step.view);
    dialog.current?.focus({ preventScroll: true });
  }, [step?.id, restartToken]);
  // A source can be reconnected, or score tracking turned off, during the tour.
  useEffect(() => {
    if (progress && progress !== "done" && progress !== "dismissed" && !step) setProgress("review");
  }, [progress, step]);
  useEffect(() => {
    if (!step) { setTargetRect(null); return; }
    let frame = 0;
    let revealed: Element | null = null;
    let observed: Element | null = null;
    const resize = new ResizeObserver(schedule);
    function update() {
      const target = [...document.querySelectorAll(`[data-tour="${step.target}"]`)].find(node => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      });
      if (target && target !== observed) { resize.disconnect(); resize.observe(target); observed = target; }
      const rect = target?.getBoundingClientRect() ?? null;
      setTargetRect(rect);
      if (target && target !== revealed && rect && (rect.top < 12 || rect.bottom > window.innerHeight - 12)) {
        revealed = target;
        target.scrollIntoView({ behavior: "instant", block: "center" });
      }
    }
    function schedule() { window.cancelAnimationFrame(frame); frame = window.requestAnimationFrame(update); }
    const mutations = new window.MutationObserver(schedule);
    mutations.observe(document.body, { childList: true, subtree: true });
    schedule();
    window.addEventListener("resize", schedule);
    window.addEventListener("scroll", schedule, true);
    return () => {
      window.cancelAnimationFrame(frame); resize.disconnect(); mutations.disconnect();
      window.removeEventListener("resize", schedule); window.removeEventListener("scroll", schedule, true);
    };
  }, [step?.id]);

  function go(value: string) { writeProgress(value); setProgress(value); }
  useEffect(() => {
    if (!active) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault(); event.stopImmediatePropagation(); go("dismissed");
    };
    window.addEventListener("keydown", escape, true);
    return () => window.removeEventListener("keydown", escape, true);
  }, [active]);

  if (!step) return null;
  const wide = window.innerWidth >= 900 || window.innerWidth >= 560 && window.innerHeight < 480;
  const targetOnRight = targetRect && targetRect.left + targetRect.width / 2 > window.innerWidth / 2;
  const targetBelow = targetRect && targetRect.top + targetRect.height / 2 > window.innerHeight / 2;
  const spotlightStyle = targetRect ? {
    top: Math.max(4, targetRect.top - 5), left: Math.max(4, targetRect.left - 5),
    width: Math.min(window.innerWidth - Math.max(4, targetRect.left - 5) - 4, targetRect.width + 10),
    height: Math.min(window.innerHeight - Math.max(4, targetRect.top - 5) - 4, targetRect.height + 10),
  } : undefined;
  return <>
    {spotlightStyle && <div className={styles.spotlight} style={spotlightStyle} aria-hidden="true" />}
    <section ref={dialog} tabIndex={-1} className={`${styles.dialog} ${styles.ralliesDialog}`} data-rallies-tour-step={step.id}
      data-side={wide && targetOnRight ? "left" : "right"} data-position={!wide && targetBelow ? "top" : "bottom"}
      role="dialog" aria-modal="false" aria-labelledby="rallies-tour-title" aria-describedby="rallies-tour-body">
      <button className={styles.close} type="button" onClick={() => go("dismissed")} aria-label="Close tutorial">×</button>
      <span className={styles.kicker}>Tutorial · {index + 1} of {steps.length}</span>
      <h2 id="rallies-tour-title">{step.title}</h2>
      <p id="rallies-tour-body">{step.body}</p>
      <div className={styles.progress} aria-hidden="true">{steps.map(item => <i key={item.id} data-active={step.id === item.id || undefined} />)}</div>
      <div className={styles.actions}>
        <button className={styles.skip} type="button" disabled={index === 0} onClick={() => go(steps[index - 1].id)}>Back</button>
        <button className={styles.next} type="button" onClick={() => go(steps[index + 1]?.id ?? "done")}>{index === steps.length - 1 ? "Finish tutorial" : "Next"}</button>
      </div>
      <small className={styles.escapeHint}>Press Esc to close · Restart from the app menu</small>
    </section>
  </>;
}
