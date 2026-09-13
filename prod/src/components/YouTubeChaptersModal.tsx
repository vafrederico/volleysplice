import { useMemo, useRef, useState } from "react";

import type { EditableCut, FinalCutInterval } from "@/lib/cut-draft";
import type { ScoreTracking } from "@/lib/score-tracking";
import {
  buildYouTubeChapters,
  defaultYouTubeChapterOptions,
  type YouTubeChapterOptions,
  youtubeChaptersFilename,
  youtubeChaptersText,
} from "@/lib/youtube-chapters";

import styles from "./YouTubeChaptersModal.module.css";

type YouTubeChaptersModalProps = {
  sourceFilename: string;
  intervals: readonly FinalCutInterval[];
  cuts: readonly EditableCut[];
  scoreTracking: ScoreTracking | null;
  hasSideSwitches: boolean;
  onClose: () => void;
};

type OutputMethod = "clipboard" | "text-file";

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("This browser did not allow clipboard access.");
}

function downloadText(text: string, filename: string): void {
  const url = URL.createObjectURL(
    new Blob([`${text}\n`], { type: "text/plain;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function YouTubeChaptersModal({
  sourceFilename,
  intervals,
  cuts,
  scoreTracking,
  hasSideSwitches,
  onClose,
}: YouTubeChaptersModalProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const hasScoreTracking = Boolean(scoreTracking?.serveMarkers.length);
  const [options, setOptions] = useState<YouTubeChapterOptions>(() =>
    defaultYouTubeChapterOptions(hasScoreTracking, hasSideSwitches),
  );
  const [outputMethod, setOutputMethod] = useState<OutputMethod>("clipboard");
  const [status, setStatus] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const chapters = useMemo(
    () => buildYouTubeChapters({ intervals, cuts, scoreTracking, options }),
    [cuts, intervals, options, scoreTracking],
  );
  const text = useMemo(
    () => youtubeChaptersText(chapters, options.includeCredit),
    [chapters, options.includeCredit],
  );

  function updateOption(option: keyof YouTubeChapterOptions, checked: boolean) {
    setOptions((current) => ({ ...current, [option]: checked }));
    setStatus(null);
  }

  async function exportChapters() {
    if (!text || isExporting) return;
    setIsExporting(true);
    setStatus(null);
    try {
      if (outputMethod === "clipboard") {
        await copyText(text);
        setStatus(
          "Copied. Paste these chapters into your YouTube description.",
        );
      } else {
        downloadText(text, youtubeChaptersFilename(sourceFilename));
        setStatus("Downloaded the YouTube chapters text file.");
      }
    } catch (cause) {
      setStatus(
        `Could not export the chapters: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    } finally {
      setIsExporting(false);
    }
  }

  return (
    <dialog
      ref={(dialog) => {
        dialogRef.current = dialog;
        if (dialog && !dialog.open) dialog.showModal();
      }}
      className={styles.dialog}
      aria-labelledby="youtube-chapters-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === dialogRef.current) onClose();
      }}
    >
      <div className={styles.modal}>
        <header>
          <div>
            <span>YOUTUBE EXPORT</span>
            <h2 id="youtube-chapters-title">Create video chapters</h2>
          </div>
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
          >
            Close
          </button>
        </header>

        <p className={styles.intro}>
          These timestamps match the final MP4 after removed footage is cut out.
          Choose what each chapter title should contain.
        </p>

        <fieldset className={styles.options}>
          <legend>Chapter title format</legend>
          <label>
            <input
              type="checkbox"
              checked={options.includeRallyNumber}
              onChange={(event) =>
                updateOption("includeRallyNumber", event.currentTarget.checked)
              }
            />
            <span>
              <strong>Rally number</strong>
              <small>Rally 1, Rally 2, …</small>
            </span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={options.includeServeNumber}
              onChange={(event) =>
                updateOption("includeServeNumber", event.currentTarget.checked)
              }
            />
            <span>
              <strong>Serve number</strong>
              <small>Uses your ordered serve markers.</small>
            </span>
          </label>
          {hasScoreTracking && (
            <>
              <label>
                <input
                  type="checkbox"
                  checked={options.includeScore}
                  onChange={(event) =>
                    updateOption("includeScore", event.currentTarget.checked)
                  }
                />
                <span>
                  <strong>Score at the serve</strong>
                  <small>Shown in Team 1 – Team 2 order.</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={options.includeServingTeam}
                  onChange={(event) =>
                    updateOption(
                      "includeServingTeam",
                      event.currentTarget.checked,
                    )
                  }
                />
                <span>
                  <strong>Serving team</strong>
                  <small>Shows the serving team by name.</small>
                </span>
              </label>
            </>
          )}
          {hasSideSwitches && (
            <label>
              <input
                type="checkbox"
                checked={options.includeSideSwitches}
                onChange={(event) =>
                  updateOption(
                    "includeSideSwitches",
                    event.currentTarget.checked,
                  )
                }
              />
              <span>
                <strong>Side-switch chapters</strong>
                <small>
                  Adds every applicable marker; switches in removed footage
                  attach to the next visible clip.
                </small>
              </span>
            </label>
          )}
        </fieldset>

        <fieldset className={styles.options}>
          <legend>Description</legend>
          <label>
            <input
              type="checkbox"
              checked={options.includeCredit}
              onChange={(event) =>
                updateOption("includeCredit", event.currentTarget.checked)
              }
            />
            <span>
              <strong>Include VolleySplice credit</strong>
              <small>Edited with https://volleysplice.com</small>
            </span>
          </label>
        </fieldset>

        <section
          className={styles.preview}
          aria-labelledby="chapter-preview-title"
        >
          <div>
            <h3 id="chapter-preview-title">Preview</h3>
            <span>
              {chapters.length} {chapters.length === 1 ? "chapter" : "chapters"}
            </span>
          </div>
          <pre>
            {text || "No visible rallies are available for chapter export."}
          </pre>
          {chapters.some((chapter) => chapter.title.includes("Re-do")) && (
            <small>
              “Re-do” means the following serve marker ignores the previous
              point; that point is not added to the score.
            </small>
          )}
        </section>

        <fieldset className={styles.destination}>
          <legend>Export to</legend>
          <label data-selected={outputMethod === "clipboard" || undefined}>
            <input
              type="radio"
              name="youtube-chapter-output"
              value="clipboard"
              checked={outputMethod === "clipboard"}
              onChange={() => {
                setOutputMethod("clipboard");
                setStatus(null);
              }}
            />
            <span>
              <strong>Clipboard</strong>
              <small>Ready to paste into YouTube.</small>
            </span>
          </label>
          <label data-selected={outputMethod === "text-file" || undefined}>
            <input
              type="radio"
              name="youtube-chapter-output"
              value="text-file"
              checked={outputMethod === "text-file"}
              onChange={() => {
                setOutputMethod("text-file");
                setStatus(null);
              }}
            />
            <span>
              <strong>Text file</strong>
              <small>Downloads a reusable .txt file.</small>
            </span>
          </label>
        </fieldset>

        <footer>
          <button
            type="button"
            className={styles.cancelButton}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className={styles.exportButton}
            disabled={!text || isExporting}
            onClick={() => void exportChapters()}
          >
            {isExporting
              ? "Exporting…"
              : outputMethod === "clipboard"
                ? "Copy chapters"
                : "Download text file"}
          </button>
        </footer>
        {status && (
          <p className={styles.status} role="status">
            {status}
          </p>
        )}
      </div>
    </dialog>
  );
}
