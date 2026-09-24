type NamedRally = { id: string; start: number; end: number };

function suffix(index: number): string {
  let value = index + 1;
  let text = "";
  while (value > 0) { value -= 1; text = String.fromCharCode(97 + value % 26) + text; value = Math.floor(value / 26); }
  return text;
}

/** Short presentation labels; immutable model IDs remain the identity for every edit. */
export function labRallyDisplayLabels(rallies: readonly NamedRally[]): Map<string, string> {
  const labels = new Map<string, string>();
  const groups = new Map<string, NamedRally[]>();
  const ordered = [...rallies].sort((a, b) => a.start - b.start || a.end - b.end || a.id.localeCompare(b.id));
  for (const [index, rally] of ordered.entries()) {
    const candidate = /^candidate:(R\d+)/.exec(rally.id);
    const compact = /^compact:(\d+)/.exec(rally.id);
    const base = candidate?.[1] ?? (compact ? "C" + compact[1].padStart(3, "0") : rally.id.startsWith("human:") ? rally.id.slice(6) : null);
    if (base) {
      const group = groups.get(base) ?? [];
      group.push(rally); groups.set(base, group);
    } else {
      const restored = /^restored:removed:(R\d+)/.exec(rally.id);
      labels.set(rally.id, restored ? restored[1] + " restored" : rally.id.includes(":") ? "Rally " + (index + 1) : rally.id);
    }
  }
  for (const [base, group] of groups) group.forEach((rally, index) => labels.set(rally.id, base + (group.length > 1 ? suffix(index) : "")));
  return labels;
}
