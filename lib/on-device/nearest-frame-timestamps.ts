/** Map sorted nominal grid times to sorted actual presentation times. */
export function nearestFrameTimestamps(source: readonly number[], targets: readonly number[]): number[] {
  if (!source.length) return [];
  let right = 0;
  return targets.map(target => {
    while (right < source.length && source[right]! < target) right++;
    const before = source[Math.max(0, right - 1)]!;
    const after = source[Math.min(right, source.length - 1)]!;
    return Math.abs(target - before) <= Math.abs(after - target) + 1e-10 ? before : after;
  });
}
