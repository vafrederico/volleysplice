import { useEffect, useId, useRef, useState } from "react";
import { GOOGLE_PLAY_URL } from "@/lib/android-app";

export function AppMenu({
  dark,
  onToggleTheme,
  onResetLayout,
}: {
  dark: boolean;
  onToggleTheme: () => void;
  onResetLayout: () => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
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
        </nav>
      )}
    </div>
  );
}
