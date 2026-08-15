export type PreparedVideoExport = {
  file: File;
  fileName: string;
};

export type PreparedVideoDelivery = "shared" | "downloaded";

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
