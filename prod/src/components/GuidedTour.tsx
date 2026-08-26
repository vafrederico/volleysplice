import { useEffect, useMemo, useState } from "react";

import styles from "./GuidedTour.module.css";

const TOUR_STORAGE_KEY = "volleycut:guided-tour:v11";

const SOURCE_STEPS = [
  "source-select",
  "source-window",
  "source-create",
] as const;

const EDITOR_STEPS = [
  "editor-settings",
  "editor-video",
  "editor-overview",
  "editor-focus",
  "editor-marking",
  "editor-score",
  "editor-score-panel",
  "editor-export-video",
] as const;

const LEGACY_EDITOR_STEPS = new Set([
  "editor-header",
  "editor-source",
  "editor-suppression",
  "editor-padding",
  "editor-join-gaps",
  "editor-play-final-cut",
  "editor-score-overlay",
  "editor-transport",
  "editor-score-markers",
  "editor-change-duration",
  "editor-split-rally",
  "editor-cuts",
]);

type SourceStep = (typeof SOURCE_STEPS)[number];
type EditorStep = (typeof EDITOR_STEPS)[number];
type TourState = SourceStep | EditorStep | "done" | "dismissed";
type TourStage = "source" | "editor";

const SCORE_EDITOR_STEPS = new Set<EditorStep>(["editor-score-panel"]);

function isSourceStep(state: TourState | null): state is SourceStep {
  return SOURCE_STEPS.some((step) => step === state);
}

function isEditorStep(state: TourState | null): state is EditorStep {
  return EDITOR_STEPS.some((step) => step === state);
}

let memoryTourState: TourState | null = null;

function readTourState(): TourState | null {
  try {
    const value = window.localStorage.getItem(TOUR_STORAGE_KEY);
    if (value === "source-camera") return "source-window";
    if (value && LEGACY_EDITOR_STEPS.has(value)) return "editor-settings";
    return value === "done" ||
      value === "dismissed" ||
      SOURCE_STEPS.some((step) => step === value) ||
      EDITOR_STEPS.some((step) => step === value)
      ? (value as TourState)
      : null;
  } catch {
    return memoryTourState;
  }
}

function writeTourState(state: TourState): void {
  memoryTourState = state;
  try {
    window.localStorage.setItem(TOUR_STORAGE_KEY, state);
  } catch {
    // The in-memory value keeps the tutorial usable when storage is restricted.
  }
}

type GuidedTourProps = {
  stage: TourStage;
  sourceReady?: boolean;
  scoreTrackingEnabled?: boolean;
};

export function GuidedTour({
  stage,
  sourceReady = false,
  scoreTrackingEnabled = false,
}: GuidedTourProps) {
  const [tourState, setTourState] = useState<TourState | null>(null);
  const [holdSourceSelect, setHoldSourceSelect] = useState(false);
  const activeEditorSteps = useMemo<EditorStep[]>(
    () =>
      scoreTrackingEnabled
        ? [...EDITOR_STEPS]
        : EDITOR_STEPS.filter((step) => !SCORE_EDITOR_STEPS.has(step)),
    [scoreTrackingEnabled],
  );

  const sourceStep = stage === "source" && isSourceStep(tourState)
    ? tourState
    : null;
  const editorStep =
    stage === "editor" &&
    isEditorStep(tourState) &&
    activeEditorSteps.includes(tourState)
      ? tourState
      : null;
  const activeSteps = stage === "source" ? SOURCE_STEPS : activeEditorSteps;
  const activeStep = stage === "source"
    ? sourceStep
      ? SOURCE_STEPS.indexOf(sourceStep)
      : 0
    : editorStep
      ? activeEditorSteps.indexOf(editorStep)
      : 0;
  const targetName = stage === "source"
    ? sourceStep === "source-window"
      ? sourceReady
        ? "source-game-window"
        : "source-picker"
      : sourceStep === "source-create"
        ? sourceReady
          ? "source-create"
          : "source-picker"
        : "source-picker"
    : editorStep ?? "editor-settings";

  const stepLabel = `TUTORIAL · ${activeStep + 1} OF ${activeSteps.length}`;
  const copy = useMemo(() => {
    if (stage === "source") {
      if (sourceStep === "source-window") {
        return sourceReady
          ? {
              label: stepLabel,
              title: "Choose the part with the game",
              body: "If the whole video is game footage, leave it set to the full video. Otherwise, move to the game’s start and end and use the nearby buttons to mark them.",
              action: "Next",
            }
          : {
              label: stepLabel,
              title: "Choose a video first",
              body: "After you choose a video, VolleyCut will show simple controls for marking where the game starts and ends.",
              action: "Next",
            };
      }
      if (sourceStep === "source-create") {
        return sourceReady
          ? {
              label: stepLabel,
              title: "Let VolleyCut find the rallies",
              body: "Select Find the rallies. Keep this page open while VolleyCut prepares the clips; you can come back when the project is ready.",
              action: "Continue to editor",
            }
          : {
              label: stepLabel,
              title: "Choose a video first",
              body: "Once a video is ready, you can ask VolleyCut to find the rallies and prepare the review.",
              action: "Continue",
            };
      }
      return {
        label: stepLabel,
        title: "Choose your video",
        body: "Pick a game video from this device, or open a saved VolleyCut project. Your video stays on this device and is not uploaded.",
        action: sourceReady ? "Next" : "Choose a video first",
      };
    }

    switch (editorStep ?? "editor-settings") {
      case "editor-settings":
        return {
          label: stepLabel,
          title: "Check the final video options",
          body: "See the planned length here. Play only the final video skips removed sections. Add scores includes the checked scoreboard in the saved video. Open Fine-tune only if you want to change cleanup or extra time around clips.",
          action: "Next",
        };
      case "editor-video":
        return {
          label: stepLabel,
          title: "Watch the video",
          body: "Use the player to play, pause, or move to any moment. When Play only the final video is on, the player skips anything that will not be saved.",
          action: "Next",
        };
      case "editor-overview":
        return {
          label: stepLabel,
          title: "Review the suggested clips",
          body: "Each block is a clip planned for the final video. Select a block to check it. Start with anything marked Check; the rest can be reviewed only if something looks wrong.",
          action: "Next",
        };
      case "editor-focus":
        return {
          label: stepLabel,
          title: "Fix one clip",
          body: "Use this area to include or leave out the selected clip, watch it, and adjust where it starts or ends. Mark it checked when it looks right.",
          action: "Next",
        };
      case "editor-marking":
        return {
          label: stepLabel,
          title: "Add anything VolleyCut missed",
          body: "For a missed rally, mark its start and end. Use Leave out a section for camera gaps, breaks, or other footage that should not appear in the final video.",
          action: "Next",
        };
      case "editor-score":
        return {
          label: stepLabel,
          title: "Add a scoreboard if you want one",
          body: "The score is built from serve markers. Add any missing serves, and add a team side switch whenever the teams change court sides so Near and Far still point to the right team.",
          action: scoreTrackingEnabled ? "Next: check the score" : "Next: save video",
        };
      case "editor-score-panel":
        return {
          label: stepLabel,
          title: "Check the score markers",
          body: "The first serve sets who starts serving. Each later serve gives the previous rally’s point to the team now serving. Correct Near or Far, add missed serves, and add side switches when teams change court sides.",
          action: "Next: save video",
        };
      case "editor-export-video":
        return {
          label: stepLabel,
          title: "Save the finished video",
          body: "Select Save final video when the review looks right. If the original video is not attached, choose it first. Use Save project under Project options when you want to keep your edits for later.",
          action: "Finish tutorial",
        };
    }
  }, [editorStep, scoreTrackingEnabled, sourceReady, sourceStep, stage, stepLabel]);

  useEffect(() => {
    const current = readTourState();
    if (stage === "editor" && current !== "done" && current !== "dismissed") {
      const next = isEditorStep(current) && activeEditorSteps.includes(current)
        ? current
        : activeEditorSteps[0];
      writeTourState(next);
      setTourState(next);
      return;
    }
    setTourState(current ?? "source-select");
  }, [activeEditorSteps, stage]);

  useEffect(() => {
    if (
      stage === "source" &&
      !sourceReady &&
      (tourState === "source-window" || tourState === "source-create")
    ) {
      setHoldSourceSelect(false);
      writeTourState("source-select");
      setTourState("source-select");
    }
  }, [sourceReady, stage, tourState]);

  useEffect(() => {
    if (
      stage !== "source" ||
      !sourceReady ||
      tourState !== "source-select" ||
      holdSourceSelect
    ) return;
    writeTourState("source-window");
    setTourState("source-window");
  }, [holdSourceSelect, sourceReady, stage, tourState]);

  const visible = stage === "source" ? sourceStep !== null : editorStep !== null;
  const restartAriaLabel = stage === "source"
    ? "Restart setup tutorial"
    : "Restart editor tutorial";

  useEffect(() => {
    if (!visible) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      writeTourState("dismissed");
      setTourState("dismissed");
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [visible]);

  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  useEffect(() => {
    if (!visible) return;
    let frame = 0;
    let target: Element | null = null;
    let revealed = false;
    const update = () => {
      target = document.querySelector(`[data-tour="${targetName}"]`);
      const rect = target?.getBoundingClientRect() ?? null;
      setTargetRect(rect && rect.width > 0 && rect.height > 0 ? rect : null);
      if (
        !revealed &&
        target &&
        rect &&
        rect.width > 0 &&
        rect.height > 0 &&
        (rect.top < 90 || rect.bottom > window.innerHeight - 90)
      ) {
        revealed = true;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    };
    const scheduleUpdate = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(update);
    };
    update();
    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("scroll", scheduleUpdate, true);
    const observer = new ResizeObserver(scheduleUpdate);
    if (target) observer.observe(target);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", scheduleUpdate);
      window.removeEventListener("scroll", scheduleUpdate, true);
      observer.disconnect();
    };
  }, [targetName, visible]);

  function dismiss() {
    writeTourState("dismissed");
    setTourState("dismissed");
  }

  function restart() {
    const initialState: TourState = stage === "source"
      ? SOURCE_STEPS[0]
      : activeEditorSteps[0];
    setHoldSourceSelect(stage === "source");
    writeTourState(initialState);
    setTourState(initialState);
  }

  function advance() {
    if (sourceStep === "source-select" && !sourceReady) return;
    if (sourceStep === "source-select") setHoldSourceSelect(false);

    if (stage === "source") {
      const index = sourceStep ? SOURCE_STEPS.indexOf(sourceStep) : 0;
      const next = index >= SOURCE_STEPS.length - 1
        ? activeEditorSteps[0]
        : SOURCE_STEPS[index + 1];
      writeTourState(next);
      setTourState(next);
      return;
    }

    const index = editorStep ? activeEditorSteps.indexOf(editorStep) : 0;
    const next: TourState = index >= activeEditorSteps.length - 1
      ? "done"
      : activeEditorSteps[index + 1];
    writeTourState(next);
    setTourState(next);
  }

  const sourceSelectBlocked =
    stage === "source" && sourceStep === "source-select" && !sourceReady;
  const spotlightStyle = targetRect
    ? {
        top: `${Math.max(8, targetRect.top - 7)}px`,
        left: `${Math.max(8, targetRect.left - 7)}px`,
        width: `${targetRect.width + 14}px`,
        height: `${targetRect.height + 14}px`,
      }
    : undefined;

  if (tourState === null) return null;
  if (!visible) {
    return (
      <button
        className={styles.launcher}
        type="button"
        onClick={restart}
        aria-label={restartAriaLabel}
      >
        <span aria-hidden="true">?</span>
        Restart tutorial
      </button>
    );
  }

  return (
    <>
      {spotlightStyle && (
        <div
          className={styles.spotlight}
          style={spotlightStyle}
          aria-hidden="true"
        />
      )}
      <section
        className={styles.dialog}
        role="dialog"
        aria-modal="false"
        aria-labelledby="volleycut-tour-title"
        aria-describedby="volleycut-tour-body"
      >
        <button
          className={styles.close}
          type="button"
          onClick={dismiss}
          aria-label="Close tutorial"
        >
          ×
        </button>
        <span className={styles.kicker}>{copy.label}</span>
        <h2 id="volleycut-tour-title">{copy.title}</h2>
        <p id="volleycut-tour-body">{copy.body}</p>
        <div className={styles.progress} aria-hidden="true">
          {Array.from({ length: activeSteps.length }, (_, step) => (
            <i
              key={`tour-step-${step}`}
              data-active={activeStep === step || undefined}
            />
          ))}
        </div>
        <div className={styles.actions}>
          <button className={styles.skip} type="button" onClick={dismiss}>
            Close tutorial
          </button>
          <button
            className={styles.next}
            type="button"
            onClick={advance}
            disabled={sourceSelectBlocked}
          >
            {copy.action}
          </button>
        </div>
        <small className={styles.escapeHint}>Press Esc anytime to close</small>
      </section>
    </>
  );
}
