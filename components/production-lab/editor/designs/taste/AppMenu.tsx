import { useEffect, useId, useRef, useState } from "react";
import { GOOGLE_PLAY_URL } from "../../lib/android-app";
import { KEYBOARD_SHORTCUTS } from "./keyboard-shortcuts";

export function AppMenu({
  dark,
  onToggleTheme,
  onResetLayout,
  projectName,
  onResetProject,
}: {
  dark: boolean;
  onToggleTheme: () => void;
  onResetLayout: () => void;
  projectName: string;
  onResetProject: () => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const shortcuts = useRef<HTMLDialogElement>(null);
  const resetProject = useRef<HTMLDialogElement>(null);
  const cancelReset = useRef<HTMLButtonElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target))
        setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  const select = (action: () => void) => {
    action();
    setOpen(false);
    trigger.current?.focus();
  };
  return (
    <div
      ref={root}
      className="rd-app-menu"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) {
          event.preventDefault();
          event.stopPropagation();
          setOpen(false);
          trigger.current?.focus();
        }
      }}
    >
      <button
        ref={trigger}
        type="button"
        className="rd-menu-toggle"
        aria-label="App menu"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((current) => !current)}
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      {open && (
        <nav id={id} className="rd-menu-panel" aria-label="App options">
          <a href="https://www.volleysplice.com/privacy.html">Privacy</a>
          <a href="https://www.volleysplice.com/terms.html">Terms</a>
          <a href={GOOGLE_PLAY_URL}>Get it on Google Play</a>
          <button
            type="button"
            aria-pressed={dark}
            onClick={() => select(onToggleTheme)}
          >
            {dark ? "Light theme" : "Dark theme"}
          </button>
          <button type="button" onClick={() => select(onResetLayout)}>
            Reset layout
          </button>
          <button type="button" onClick={() => {
            setOpen(false);
            trigger.current?.focus();
            resetProject.current?.showModal();
            cancelReset.current?.focus();
          }}>
            Reset project changes
          </button>
          <button type="button" onClick={() => {
            setOpen(false);
            trigger.current?.focus();
            shortcuts.current?.showModal();
          }}>
            Keyboard shortcuts
          </button>
        </nav>
      )}
      <dialog
        ref={resetProject}
        className="rd-settings-dialog rd-reset-project-dialog"
        aria-labelledby={`${id}-reset-title`}
        aria-describedby={`${id}-reset-description`}
        onClose={() => trigger.current?.focus()}
      >
        <header><h2 id={`${id}-reset-title`}>Reset project changes?</h2></header>
        <p id={`${id}-reset-description`}>Restore “{projectName}” to its saved model results and default review settings? This clears your rally edits, keep/remove decisions, added markers, excluded footage, and score corrections. Your source video and analysis stay saved; inference will not run again.</p>
        <p>You can undo this reset with Ctrl+Z.</p>
        <footer>
          <button ref={cancelReset} className="td-secondary-button" type="button" onClick={() => resetProject.current?.close()}>Cancel</button>
          <button className="td-primary-button" type="button" onClick={() => {
            resetProject.current?.close();
            onResetProject();
          }}>Reset project changes</button>
        </footer>
      </dialog>
      <dialog
        ref={shortcuts}
        className="rd-settings-dialog rd-shortcuts-dialog"
        aria-labelledby={`${id}-shortcuts-title`}
        aria-describedby={`${id}-shortcuts-description`}
        onClose={() => trigger.current?.focus()}
        onClick={(event) => {
          if (event.target !== event.currentTarget) return;
          const rect = event.currentTarget.getBoundingClientRect();
          if (event.clientX < rect.left || event.clientX > rect.right ||
            event.clientY < rect.top || event.clientY > rect.bottom) shortcuts.current?.close();
        }}
      >
        <header>
          <h2 id={`${id}-shortcuts-title`}>Keyboard shortcuts</h2>
          <button type="button" aria-label="Close keyboard shortcuts" onClick={() => shortcuts.current?.close()}>×</button>
        </header>
        <p id={`${id}-shortcuts-description`}>Review shortcuts work in the review workspace. They pause while you type, use a dropdown, or open a modal. Escape closes this window.</p>
        <dl className="rd-shortcuts-list">
          {KEYBOARD_SHORTCUTS.map((shortcut) => (
            <div key={shortcut.action}><dt><kbd>{shortcut.label}</kbd></dt><dd>{shortcut.description}</dd></div>
          ))}
        </dl>
      </dialog>
    </div>
  );
}
