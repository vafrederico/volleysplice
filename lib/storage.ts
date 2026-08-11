import path from "node:path";

export function getDataRoot(): string {
  const configured = process.env.VOLLEYCUT_DATA_ROOT?.trim();
  return configured ? path.resolve(configured) : path.join(process.cwd(), "data");
}

export function getAnalysesRoot(): string {
  return path.join(getDataRoot(), "analyses");
}
