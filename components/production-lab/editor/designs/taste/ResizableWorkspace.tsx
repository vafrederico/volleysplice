import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type Dispatch,
  type SetStateAction,
  type ReactNode,
} from "react";
import {
  clampLayout,
  readLayout,
  fittedVideoHeight,
  type WorkspaceLayout,
} from "./workspace-layout";

const STORAGE_KEY = "volleycut:production-lab:rally-desk-layout:v1";

function ResizeHandle({
  axis,
  label,
  value,
  min,
  max,
  direction = 1,
  onChange,
  onReset,
  className = "",
}: {
  axis: "horizontal" | "vertical";
  label: string;
  value: number;
  min: number;
  max: number;
  direction?: number;
  onChange: (value: number) => void;
  onReset: () => void;
  className?: string;
}) {
  const drag = useRef<{
    coordinate: number;
    value: number;
    pointerId: number;
  } | null>(null);
  const update = (next: number) => onChange(Math.max(min, Math.min(max, next)));
  return (
    <div
      className={`rd-resize-handle ${className}`}
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-orientation={axis === "horizontal" ? "vertical" : "horizontal"}
      aria-valuemin={Math.round(min)}
      aria-valuemax={Math.round(max)}
      aria-valuenow={Math.round(value)}
      aria-valuetext={`${Math.round(value)} pixels`}
      title={`${label}. Drag or use arrow keys; double-click to reset.`}
      onDoubleClick={onReset}
      onPointerDown={(event) => {
        if (!event.isPrimary || event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.focus();
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = {
          coordinate: axis === "horizontal" ? event.clientX : event.clientY,
          value,
          pointerId: event.pointerId,
        };
      }}
      onPointerMove={(event) => {
        const start = drag.current;
        if (!start || start.pointerId !== event.pointerId) return;
        const coordinate =
          axis === "horizontal" ? event.clientX : event.clientY;
        update(start.value + (coordinate - start.coordinate) * direction);
      }}
      onPointerUp={(event) => {
        drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId))
          event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
      onLostPointerCapture={() => {
        drag.current = null;
      }}
      onKeyDown={(event) => {
        const negative = axis === "horizontal" ? "ArrowLeft" : "ArrowUp";
        const positive = axis === "horizontal" ? "ArrowRight" : "ArrowDown";
        if (event.key === "Home" || event.key === "End") {
          event.preventDefault();
          update(event.key === "Home" ? min : max);
        } else if (event.key === negative || event.key === positive) {
          event.preventDefault();
          update(
            value +
              (event.key === positive ? 1 : -1) *
                direction *
                (event.shiftKey ? 40 : 10),
          );
        }
      }}
    />
  );
}

export function useWorkspaceLayout() {
  const [saved, setSaved] = useState<WorkspaceLayout>(() => {
    try {
      return readLayout(localStorage.getItem(STORAGE_KEY));
    } catch {
      return {};
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
    } catch {
      /* Resizing still works when storage is unavailable. */
    }
  }, [saved]);
  return { savedLayout: saved, onLayoutChange: setSaved };
}

export function ResizableWorkspace({
  ledger,
  children,
  inspector,
  mobileRegister,
  savedLayout: saved,
  onLayoutChange: setSaved,
}: {
  ledger: ReactNode;
  children: (videoResize: ReactNode) => ReactNode;
  inspector: ReactNode;
  mobileRegister: ReactNode;
  savedLayout: WorkspaceLayout;
  onLayoutChange: Dispatch<SetStateAction<WorkspaceLayout>>;
}) {
  const shell = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const update = () =>
      setViewport({
        width: shell.current?.clientWidth ?? 0,
        height: window.innerHeight,
      });
    const observer = new ResizeObserver(update);
    if (shell.current) observer.observe(shell.current);
    window.addEventListener("resize", update);
    update();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, []);
  const [autoHeight, setAutoHeight] = useState<number | null>(null);
  useLayoutEffect(() => {
    const root = shell.current;
    const video = root?.querySelector<HTMLElement>(".td-video-picture");
    const legend = root?.querySelector<HTMLElement>(".rd-timeline-key");
    if (!root || !video || !legend) return;
    let frame = 0;
    const measure = () => {
      frame = 0;
      if (window.innerWidth <= 960 || saved.video !== undefined) {
        setAutoHeight(null);
        return;
      }
      const picture = video.getBoundingClientRect();
      // Document coordinates keep fitting independent of the user's scroll position.
      const overhead =
        legend.getBoundingClientRect().bottom + window.scrollY - picture.height;
      const next = fittedVideoHeight(
        picture.width,
        window.innerHeight,
        overhead,
      );
      setAutoHeight((current) =>
        current !== null && Math.abs(current - next) < 0.5 ? current : next,
      );
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    const observer = new ResizeObserver(schedule);
    for (const element of [
      root,
      video,
      legend,
      root.querySelector(".rd-main"),
      root.querySelector(".td-transport"),
      root.parentElement?.querySelector(".rd-topbar"),
    ]) {
      if (element) observer.observe(element);
    }
    window.addEventListener("resize", schedule);
    measure();
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", schedule);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [saved.video]);
  const layout = clampLayout(saved, viewport.width, viewport.height);
  const change = (key: keyof WorkspaceLayout, value: number) =>
    setSaved((current) => ({ ...current, [key]: value }));
  const reset = (key: keyof WorkspaceLayout) =>
    setSaved((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  const videoHeight =
    saved.video !== undefined ? layout.video : (autoHeight ?? layout.video);
  const style = {
    ...(viewport.width > 0
      ? {
          "--rd-left-width": `${layout.left}px`,
          "--rd-right-width": `${layout.right}px`,
        }
      : {}),
    ...(saved.video !== undefined || autoHeight !== null
      ? { "--rd-video-height": `${videoHeight}px` }
      : {}),
  } as CSSProperties;
  return (
    <div ref={shell} className="rd-shell is-review rd-resizable" style={style}>
      <aside
        className="rd-ledger-column"
        aria-label="Clip and match-event ledger"
      >
        {ledger}
        <ResizeHandle
          axis="horizontal"
          label="Resize clip sidebar"
          className="rd-resize-left"
          value={layout.left}
          min={240}
          max={layout.leftMax}
          onChange={(value) => change("left", value)}
          onReset={() => reset("left")}
        />
      </aside>
      <div className="rd-main">
        {children(
          <ResizeHandle
            className="rd-video-resize"
            axis="vertical"
            label="Resize video player"
            value={videoHeight}
            min={180}
            max={layout.videoMax}
            onChange={(value) => change("video", value)}
            onReset={() => reset("video")}
          />,
        )}
      </div>
      <aside className="rd-inspector-stack">
        <ResizeHandle
          axis="horizontal"
          label="Resize inspector sidebar"
          className="rd-resize-right"
          direction={-1}
          value={layout.right}
          min={260}
          max={layout.rightMax}
          onChange={(value) => change("right", value)}
          onReset={() => reset("right")}
        />
        {inspector}
      </aside>
      <div className="rd-mobile-register">{mobileRegister}</div>
    </div>
  );
}
