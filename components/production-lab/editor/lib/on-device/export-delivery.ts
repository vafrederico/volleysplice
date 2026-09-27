export type PreparedVideoExport = {
  file: File;
  fileName: string;
};

export type PreparedVideoDelivery = "shared" | "downloaded";

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

function downloadPreparedFile(prepared: PreparedVideoExport): void {
  const url = URL.createObjectURL(prepared.file);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = prepared.fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export async function deliverPreparedVideoExport(
  prepared: PreparedVideoExport,
): Promise<PreparedVideoDelivery> {
  const shareData: ShareData = {
    files: [prepared.file],
    title: prepared.fileName,
  };
  if (
    typeof navigator.share === "function" &&
    (typeof navigator.canShare !== "function" || navigator.canShare(shareData))
  ) {
    // Do not await anything before this call: Web Share consumes the current tap's activation.
    await navigator.share(shareData);
    return "shared";
  }

  downloadPreparedFile(prepared);
  return "downloaded";
}
