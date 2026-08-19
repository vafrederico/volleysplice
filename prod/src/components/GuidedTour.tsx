import { useEffect, useMemo, useState } from "react";

import styles from "./GuidedTour.module.css";

const TOUR_STORAGE_KEY = "volleycut:guided-tour:v8";

const EDITOR_STEPS = [
  "editor-header",
  "editor-source",
  "editor-settings",
  "editor-suppression",
  "editor-padding",
  "editor-join-gaps",
  "editor-play-final-cut",
  "editor-export-video",
  "editor-video",
  "editor-transport",
  "editor-overview",
  "editor-focus",
  "editor-change-duration",
  "editor-split-rally",
  "editor-marking",
  "editor-cuts",
] as const;
const SOURCE_STEP_COUNT = 4;
const TOTAL_TOUR_STEPS = SOURCE_STEP_COUNT + EDITOR_STEPS.length;

type TourState =
  | "source-select"
  | "source-camera"
  | "source-window"
  | "source-create"
  | (typeof EDITOR_STEPS)[number]
  | "done"
  | "dismissed";
type TourStage = "source" | "editor";
type SourceStep =
  | "source-select"
  | "source-camera"
  | "source-window"
  | "source-create";
type EditorStep = (typeof EDITOR_STEPS)[number];

function isEditorStep(state: TourState | null): state is EditorStep {
  return EDITOR_STEPS.some((step) => step === state);
}

let memoryTourState: TourState | null = null;

function readTourState(): TourState | null {
  try {
    const value = window.localStorage.getItem(TOUR_STORAGE_KEY);
    return value === "source-select" ||
      value === "source-camera" ||
      value === "source-window" ||
      value === "source-create" ||
      EDITOR_STEPS.some((step) => step === value) ||
      value === "done" ||
      value === "dismissed"
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
    // The in-memory value keeps the tour usable when storage is restricted.
  }
}

type GuidedTourProps = {
  stage: TourStage;
  sourceReady?: boolean;
};

export function GuidedTour({ stage, sourceReady = false }: GuidedTourProps) {
  const [tourState, setTourState] = useState<TourState | null>(null);
  const [holdSourceSelect, setHoldSourceSelect] = useState(false);
  const sourceStep: SourceStep | null =
    stage === "source" &&
    (tourState === "source-select" ||
      tourState === "source-camera" ||
      tourState === "source-window" ||
      tourState === "source-create")
      ? tourState
      : null;
  const editorStep: EditorStep | null =
    stage === "editor" && isEditorStep(tourState) ? tourState : null;
  const targetName =
    stage === "editor"
      ? (editorStep ?? "editor-header")
      : sourceStep === "source-camera"
        ? sourceReady
          ? "source-camera"
          : "source-picker"
        : sourceStep === "source-window"
          ? sourceReady
            ? "source-game-window"
            : "source-picker"
          : sourceStep === "source-create"
            ? sourceReady
              ? "source-create"
              : "source-picker"
            : "source-picker";

  const copy = useMemo(() => {
    if (stage === "source" && sourceStep === "source-camera" && sourceReady) {
      return {
        label: `WELCOME TOUR · 2 OF ${TOTAL_TOUR_STEPS}`,
        title: "Set the camera viewport",
        body: "The yellow FEATURE CROP box is the part of the video the models analyze. New projects start with the full frame; keep the court and players inside it, then use the x, y, width, and height sliders to provide a tighter crop when needed. Full frame resets the crop.",
        action: "Next: start & end",
      };
    }
    if (stage === "source" && sourceStep === "source-camera") {
      return {
        label: `WELCOME TOUR · 2 OF ${TOTAL_TOUR_STEPS}`,
        title: "Reveal the camera controls",
        body: "Choose video to reveal the preview and camera viewport controls. The yellow FEATURE CROP box will show what the models analyze; new projects start at full frame, and the sliders let you provide a tighter crop.",
        action: "Next: start & end",
      };
    }
    if (stage === "source" && sourceStep === "source-window" && sourceReady) {
      return {
        label: `WELCOME TOUR · 3 OF ${TOTAL_TOUR_STEPS}`,
        title: "Choose the game start and end",
        body: "Seek the preview, then use Set to playhead under Game start and Game end to copy the current time. Use full video to reset both boundaries. Only this analysis window generates features, so set it before continuing.",
        action: "Next: create project",
      };
    }
    if (stage === "source" && sourceStep === "source-window") {
      return {
        label: `WELCOME TOUR · 3 OF ${TOTAL_TOUR_STEPS}`,
        title: "Set the game window",
        body: "Choose video first to reveal the Game start and Game end controls. You will be able to seek the preview, copy the playhead into either boundary, reset to the full video, and then queue local analysis.",
        action: "Next: create project",
      };
    }
    if (stage === "source" && sourceStep === "source-create" && sourceReady) {
      return {
        label: `WELCOME TOUR · 4 OF ${TOTAL_TOUR_STEPS}`,
        title: "Create the project and queue inference",
        body: "When the camera crop and game start/end look right, click Create project & queue inference. Your source settings are saved on this device and local model analysis begins; the editor opens when the project is ready.",
        action: "Next: editor",
      };
    }
    if (stage === "source") {
      return {
        label: `WELCOME TOUR · 1 OF ${TOTAL_TOUR_STEPS}`,
        title: "Select a volleyball video",
        body: "Choose video to load one recording. The file stays in this browser and is not uploaded. Once it opens, the camera viewport and game start/end sections appear for you to configure before running analysis.",
        action: sourceReady ? "Next: camera setup" : "Choose video first",
      };
    }
    switch (editorStep ?? "editor-header") {
      case "editor-source":
        return {
          label: `WELCOME TOUR · 6 OF ${TOTAL_TOUR_STEPS}`,
          title: "Know what this video represents",
          body: "The top source bar names the local video and the inference that produced its ranges. The Project menu switches between saved videos; reconnect the source there when playback or export needs the original file.",
          action: "Next: final settings",
        };
      case "editor-header":
        return {
          label: `WELCOME TOUR · 5 OF ${TOTAL_TOUR_STEPS}`,
          title: "Navigate projects from the header",
          body: "The VolleyCut header keeps project navigation in one place. Use the Project menu to switch saved videos, watch the queue status while inference runs, and delete the selected project when you no longer need its local data.",
          action: "Next: local source",
        };
      case "editor-settings":
        return {
          label: `WELCOME TOUR · 7 OF ${TOTAL_TOUR_STEPS}`,
          title: "Build the final edit",
          body: "This panel is the control center for the final edit. It summarizes kept and removed time, previews the selected result, and contains the controls for suppression, padding, gap joining, playback, and exports.",
          action: "Next: suppression",
        };
      case "editor-suppression":
        return {
          label: `WELCOME TOUR · 8 OF ${TOTAL_TOUR_STEPS}`,
          title: "Choose a suppression policy",
          body: "Suppression levels automatically remove model ranges that look like false positives. None preserves the existing output; the available policy levels apply increasingly strong suggestions. Untouched suggestions are suppressed until you choose Keep while reviewing. When a suggestion is selected, choose Whole rally to veto the inferred rally and padding, or Veto region to remove only the highlighted red span; Whole rally is the default.",
          action: "Next: padding",
        };
      case "editor-padding":
        return {
          label: `WELCOME TOUR · 9 OF ${TOTAL_TOUR_STEPS}`,
          title: "Add padding around each cut",
          body: "The Before and After sliders add extra seconds around every inferred cut. Use them to keep context around a rally; the current values are shown beside each slider and are included in the final edit timing.",
          action: "Next: gap joining",
        };
      case "editor-join-gaps":
        return {
          label: `WELCOME TOUR · 10 OF ${TOTAL_TOUR_STEPS}`,
          title: "Join short gaps between cuts",
          body: "Join gaps under controls when nearby kept ranges should become one continuous export. Light-gray gaps shorter than this threshold are retained; set it to Off when every gap should remain a cut.",
          action: "Next: final-cut preview",
        };
      case "editor-play-final-cut":
        return {
          label: `WELCOME TOUR · 11 OF ${TOTAL_TOUR_STEPS}`,
          title: "Preview only the final cut",
          body: "Play final cut only skips removed rallies, ignored sections, and unselected gaps at or above the join threshold while the video plays. Turn it off when you need to review the complete analysis window.",
          action: "Next: export video",
        };
      case "editor-export-video":
        return {
          label: `WELCOME TOUR · 12 OF ${TOTAL_TOUR_STEPS}`,
          title: "Export the final video",
          body: "This button creates the edited MP4 from the kept ranges, padding, suppression, and joined gaps. The export stays on this device at the original dimensions; reconnect the local source first if playback or export is unavailable.",
          action: "Next: video player",
        };
      case "editor-video":
        return {
          label: `WELCOME TOUR · 13 OF ${TOTAL_TOUR_STEPS}`,
          title: "Watch the source video",
          body: "This is the original local video. The timecode is limited to the marked game window. Use the browser video controls to play, pause, and scrub while checking a model range against the footage.",
          action: "Next: transport controls",
        };
      case "editor-transport":
        return {
          label: `WELCOME TOUR · 14 OF ${TOTAL_TOUR_STEPS}`,
          title: "Move frame by frame",
          body: "The transport buttons nudge the playhead by one second or one tenth of a second, and Play / Pause starts or stops playback. Playback speed changes how quickly the video runs without changing its saved boundaries.",
          action: "Next: game window",
        };
      case "editor-overview":
        return {
          label: `WELCOME TOUR · 15 OF ${TOTAL_TOUR_STEPS}`,
          title: "Read the GAME WINDOW rail",
          body: "The two rows show the first and second halves of the game, giving each rally more horizontal space on small screens. Tap or slide either row to seek, or select a colored range to focus it below. Review next moves through low-confidence or one-model disagreement ranges.",
          action: "Next: focused range",
        };
      case "editor-focus":
        return {
          label: `WELCOME TOUR · 16 OF ${TOTAL_TOUR_STEPS}`,
          title: "Refine the focused range",
          body: "The focused timeline enlarges the selected rally. Blue shows the output including padding, while the inner rally is marked separately. Use Previous and Next to move through the cut list, and Keep / Restore, Preview cut, Reset padding, or Mark reviewed to finish reviewing it.",
          action: "Next: change duration",
        };
      case "editor-change-duration":
        return {
          label: `WELCOME TOUR · 17 OF ${TOTAL_TOUR_STEPS}`,
          title: "Change a rally's duration",
          body: "Drag either orange handle to shorten or extend the rally. For frame-accurate edits, seek with the player or transport controls and choose Set rally start here or Set rally end here. Existing padding follows the corrected rally edges; Output edges adjust only the surrounding padding.",
          action: "Next: split a rally",
        };
      case "editor-split-rally":
        return {
          label: `WELCOME TOUR · 18 OF ${TOTAL_TOUR_STEPS}`,
          title: "Split one rally into two",
          body: "Move the playhead to the point where the rallies should separate, then choose Split at playhead. VolleyCut creates two ranges with the same padding and selects the new second part, so each side can be trimmed, kept, or removed independently.",
          action: "Next: marking tools",
        };
      case "editor-marking":
        return {
          label: `WELCOME TOUR · 19 OF ${TOTAL_TOUR_STEPS}`,
          title: "Add misses or ignore unusable footage",
          body: "Use Add a missed cut when the model missed a rally: mark its start, seek, then mark its end. Use Ignore source section for camera gaps or non-game footage; ignored time is excluded without becoming a negative label.",
          action: "Next: all cuts",
        };
      case "editor-cuts":
        return {
          label: `WELCOME TOUR · 20 OF ${TOTAL_TOUR_STEPS}`,
          title: "Use the all-cuts list",
          body: "Each card is one model prediction or manual addition. Click the time card to focus it, then use Keep, Removed, or Ignored to decide whether it contributes to the final edit. Check badges identify ranges that still need review.",
          action: "Finish tour",
        };
    }
  }, [editorStep, sourceReady, sourceStep, stage]);

  useEffect(() => {
    if (
      stage === "source" &&
      !sourceReady &&
      (tourState === "source-camera" ||
        tourState === "source-window" ||
        tourState === "source-create")
    ) {
      setHoldSourceSelect(false);
      writeTourState("source-select");
      setTourState("source-select");
      return;
    }
  }, [sourceReady, stage, tourState]);

  useEffect(() => {
    const current = readTourState();
    if (stage === "editor" && current !== "done" && current !== "dismissed") {
      const editorState: EditorStep = isEditorStep(current)
        ? current
        : EDITOR_STEPS[0];
      writeTourState(editorState);
      setTourState(editorState);
      return;
    }
    setTourState(current ?? "source-select");
  }, [stage]);

  useEffect(() => {
    if (
      stage !== "source" ||
      !sourceReady ||
      tourState !== "source-select" ||
      holdSourceSelect
    )
      return;
    writeTourState("source-camera");
    setTourState("source-camera");
  }, [holdSourceSelect, sourceReady, stage, tourState]);

  const visible =
    stage === "source"
      ? tourState === "source-select" ||
        (sourceReady &&
          (tourState === "source-camera" ||
            tourState === "source-window" ||
            tourState === "source-create"))
      : editorStep !== null;
  const restartLabel =
    stage === "source" ? "Restart setup tour" : "Restart editor tour";

  useEffect(() => {
    if (!visible) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        writeTourState("dismissed");
        setTourState("dismissed");
      }
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
      if (
        !revealed &&
        (stage === "editor" || (stage === "source" && sourceReady)) &&
        target &&
        rect &&
        (rect.top < 90 || rect.bottom > window.innerHeight - 90)
      ) {
        revealed = true;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      }
      setTargetRect(rect);
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
  }, [sourceReady, stage, targetName, visible]);

  function dismiss() {
    writeTourState("dismissed");
    setTourState("dismissed");
  }

  function restart() {
    const initialState: TourState =
      stage === "source" ? "source-select" : EDITOR_STEPS[0];
    setHoldSourceSelect(stage === "source");
    writeTourState(initialState);
    setTourState(initialState);
  }

  function advance() {
    if (sourceStep === "source-select" && !sourceReady) return;
    if (sourceStep === "source-select") setHoldSourceSelect(false);
    const nextState: TourState =
      stage === "editor"
        ? (() => {
            const index = editorStep ? EDITOR_STEPS.indexOf(editorStep) : 0;
            return index >= EDITOR_STEPS.length - 1
              ? "done"
              : EDITOR_STEPS[index + 1];
          })()
        : sourceStep === "source-select"
          ? "source-camera"
          : sourceStep === "source-camera"
            ? "source-window"
            : sourceStep === "source-window"
              ? "source-create"
              : EDITOR_STEPS[0];
    writeTourState(nextState);
    setTourState(nextState);
  }

  const activeStep =
    stage === "editor"
      ? SOURCE_STEP_COUNT + (editorStep ? EDITOR_STEPS.indexOf(editorStep) : 0)
      : sourceStep === "source-camera"
        ? 1
        : sourceStep === "source-window"
          ? 2
          : sourceStep === "source-create"
            ? 3
            : 0;
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
        aria-label={restartLabel}
      >
        <span aria-hidden="true">?</span>
        {restartLabel}
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
          aria-label="Skip tour"
        >
          ×
        </button>
        <span className={styles.kicker}>{copy.label}</span>
        <h2 id="volleycut-tour-title">{copy.title}</h2>
        <p id="volleycut-tour-body">{copy.body}</p>
        <div className={styles.progress} aria-hidden="true">
          {Array.from({ length: TOTAL_TOUR_STEPS }, (_, step) => (
            <i
              key={`tour-step-${step}`}
              data-active={activeStep === step || undefined}
            />
          ))}
        </div>
        <div className={styles.actions}>
          <button className={styles.skip} type="button" onClick={dismiss}>
            Skip tour
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
