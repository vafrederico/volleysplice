export type PreparedVideoExport = {
  file: File;
  fileName: string;
};

export type VideoExportTarget = {
  createWritable(): Promise<WritableStream<unknown>>;
};

type SavePicker = (options: {
  suggestedName: string;
  types: Array<{ description: string; accept: Record<string, string[]> }>;
}) => Promise<VideoExportTarget>;

function exportFilename(sourceFilename: string): string {
  const baseName =
    sourceFilename
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "volleysplice";
  return `${baseName}-volleysplice.mp4`;
}

export function requestVideoExportTarget(
  sourceFilename: string,
): Promise<VideoExportTarget> | null {
  const candidate = (window as unknown as { showSaveFilePicker?: SavePicker })
    .showSaveFilePicker;
  if (!candidate) return null;
  const chooseTarget = candidate.bind(window);
  return chooseTarget({
    suggestedName: exportFilename(sourceFilename),
    types: [
      {
        description: "MP4 video",
        accept: { "video/mp4": [".mp4"] },
      },
    ],
  });
}

export function supportsOpfsExport(): boolean {
  return typeof navigator.storage?.getDirectory === "function";
}
