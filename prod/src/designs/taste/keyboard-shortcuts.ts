export const KEYBOARD_SHORTCUTS = [
  { action: "undo", keys: ["ctrl+z"], label: "Ctrl + Z", description: "Undo last edit (up to 30 actions, saved per project)" },
  { action: "redo", keys: ["ctrl+y"], label: "Ctrl + Y", description: "Redo last undone edit" },
  { action: "playPause", keys: [" "], label: "Space", description: "Play / pause playback" },
  { action: "near", keys: ["n"], label: "N", description: "Set near serve" },
  { action: "far", keys: ["f"], label: "F", description: "Set far serve" },
  { action: "review", keys: ["r"], label: "R", description: "Next review after playhead; cycles cleaned up → clips → serves" },
  { action: "remove", keys: ["backspace"], label: "Backspace", description: "Remove current rally" },
  { action: "keep", keys: ["k"], label: "K", description: "Keep current rally" },
  { action: "back", keys: ["arrowleft"], label: "←", description: "Move playhead back 5 seconds" },
  { action: "forward", keys: ["arrowright"], label: "→", description: "Move playhead forward 5 seconds" },
  { action: "previousRally", keys: ["shift+arrowleft"], label: "Shift + ←", description: "Jump to previous rally" },
  { action: "nextRally", keys: ["shift+arrowright"], label: "Shift + →", description: "Jump to next rally" },
  { action: "faster", keys: ["+", "="], label: "+", description: "Increase playback speed" },
  { action: "slower", keys: ["-"], label: "−", description: "Decrease playback speed" },
  { action: "switch", keys: ["t"], label: "T", description: "Add team side switch at playhead" },
  { action: "serve", keys: ["s"], label: "S", description: "Add serve at playhead" },
  { action: "split", keys: ["shift+s"], label: "Shift + S", description: "Split current rally at playhead" },
  { action: "exclude", keys: ["e"], label: "E", description: "Mark excluded footage start / end" },
  { action: "missed", keys: ["shift+r"], label: "Shift + R", description: "Mark missed rally start / end" },
  { action: "cancelRange", keys: ["escape"], label: "Esc", description: "Cancel unfinished missed rally or excluded footage selection" },
  { action: "new", keys: ["shift+n"], label: "Shift + N", description: "Start a new project" },
  { action: "projects", keys: ["p"], label: "P", description: "Open and focus project list; use arrows to navigate" },
  { action: "export", keys: ["shift+e"], label: "Shift + E", description: "Open export" },
  { action: "removeEvent", keys: ["shift+backspace"], label: "Shift + Backspace", description: "Remove selected serve or team side switch marker" },
] as const;

export function shortcutAction(event: {
  key: string;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  repeat: boolean;
  isComposing: boolean;
}) {
  if (event.metaKey || event.altKey || event.isComposing) return null;
  const key = event.key.toLowerCase();
  const chord = `${event.ctrlKey ? "ctrl+" : ""}${event.shiftKey && key !== "+" ? "shift+" : ""}${key}`;
  const action = KEYBOARD_SHORTCUTS.find((shortcut) =>
    (shortcut.keys as readonly string[]).includes(chord),
  )?.action ?? null;
  if (event.repeat && action !== "back" && action !== "forward") return null;
  return action;
}

export type ReviewItem = { kind: "cleanup" | "clip" | "serve"; id: string; time: number };
const REVIEW_ORDER = { cleanup: 0, clip: 1, serve: 2 };
function compareReviewItems(left: ReviewItem, right: ReviewItem) {
  return REVIEW_ORDER[left.kind] - REVIEW_ORDER[right.kind] ||
    left.time - right.time || left.id.localeCompare(right.id);
}

// Remember the category, but derive progress from the timeline so seeking back
// revisits earlier pending items. Opening a rally at its padding must not repeat it.
export function nextReviewItem(
  items: ReviewItem[],
  category: ReviewItem["kind"] | null,
  playhead: number,
  currentId?: string,
) {
  const ordered = [...items].sort(compareReviewItems);
  const kinds = ["cleanup", "clip", "serve"] as const;
  const index = REVIEW_ORDER[category ?? "cleanup"];
  const next = ordered.find((item) => item.kind === kinds[index] && item.time > playhead && item.id !== currentId);
  if (next) return next;
  for (let offset = 1; offset <= kinds.length; offset++) {
    const first = ordered.find((item) => item.kind === kinds[(index + offset) % kinds.length]);
    if (first) return first;
  }
  return null;
}
