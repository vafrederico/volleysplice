import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LabelDocument } from "../annotations.ts";
import type { ProductionEditorLabTask } from "../production-editor-lab.ts";
import { attachLabServing } from "../production-editor-lab-serving.ts";
import { SERVING_SIDE_MODEL_ID, SERVING_SIDE_MODEL_FINGERPRINT } from "../../prod/src/lib/on-device/serving-side-cache.ts";

/** Inference receipts live beside the externally configured comparison catalog. */
export async function loadEditorLabServing(task: ProductionEditorLabTask, recording: LabelDocument["recording"]) {
  const index = process.env.VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH?.trim();
  if (!index || !/^[\w-]+$/.test(recording.id)) return task;
  let raw: string;
  try { raw = await readFile(path.join(path.dirname(path.resolve(/* turbopackIgnore: true */ index)), "serving-v1", `${recording.id}.json`), "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return task; throw error; }
  const value = JSON.parse(raw);
  if (value.schemaVersion !== 1 || value.labelsUsed !== false || value.recordingId !== recording.id
    || value.contentSha256 !== recording.contentSha256 || value.duration !== recording.durationSeconds
    || value.roi?.x !== 0 || value.roi?.y !== 0 || value.roi?.width !== 1 || value.roi?.height !== 1
    || value.modelId !== SERVING_SIDE_MODEL_ID || value.modelFingerprint !== SERVING_SIDE_MODEL_FINGERPRINT || !value.configurations
    || Object.values(value.configurations).some((receipt: any) => receipt?.output?.modelId !== value.modelId
      || receipt.output.modelFingerprint !== value.modelFingerprint)) {
    throw new Error("Serving-side inference source identity differs");
  }
  return { ...task, configurations: task.configurations.map(configuration => value.configurations[configuration.id]
    ? attachLabServing(configuration, value.configurations[configuration.id], task.durationSeconds) : configuration) };
}
