import { privateValue } from "../lib/server/private-ledger.mjs";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { registerHooks } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Use the workstation's actual task identity, while retaining imported labels.
process.loadEnvFile(".env.local");
registerHooks({ resolve(specifier, context, nextResolve) {
  const resolved = specifier.startsWith("@/")
    ? pathToFileURL(path.resolve(`${specifier.slice(2)}.ts`)).href : specifier;
  return nextResolve(resolved, context);
} });
const root = privateValue("private-reference-0082");
const id = privateValue("recording-044");
const registration = JSON.parse(readFileSync(`${root}/labeling-registration.json`, "utf8"));
const before = readFileSync(registration.humanDraft.path);
const sha = (value: Buffer) => createHash("sha256").update(value).digest("hex");
if (sha(before) !== registration.humanDraft.sha256) throw new Error("Draft changed since import; preserve user edits");
const tasks = await import("../lib/server/labeling-tasks.ts");
const task = await tasks.getPreparedLabelingTask(id);
const document = JSON.parse(before.toString("utf8"));
document.labelingTaskAdaptation = {
  reason: "Match the labeling catalog's immutable task envelope; labels and their provenance remain unchanged",
  originalCreatedAt: document.createdAt,
  originalCapture: document.recording.capture,
  originalAnnotationPolicy: document.annotationPolicy,
};
document.createdAt = task.document.createdAt;
document.recording.capture = task.document.recording.capture;
document.annotationPolicy = task.document.annotationPolicy;
const result = await tasks.saveLabelingDraft(task, document);
const loaded = await tasks.getSavedLabelingDocument(await tasks.getPreparedLabelingTask(id));
const old = JSON.parse(before.toString("utf8"));
for (const field of ["rallies", "serveMarkers", "sideSwitches", "ignoredIntervals", "hardNegatives", "importProvenance"]) {
  if (JSON.stringify(old[field]) !== JSON.stringify(loaded.document[field])) throw new Error(`Imported ${field} changed`);
}
const after = readFileSync(registration.humanDraft.path);
const receipt = {
  passed: true, recordingId: id, source: loaded.source,
  savedAt: result.savedAt, priorRegistration: registration,
  immutableTaskIdentityValidated: true, importedLabelContentsUnchanged: true,
  humanDraft: { path: registration.humanDraft.path, sha256: sha(after), sizeBytes: after.length },
  sourceFile: { path: path.resolve("scripts/adapt-feedback-labeling-draft.mts"), sha256: sha(readFileSync("scripts/adapt-feedback-labeling-draft.mts")) },
};
writeFileSync(`${root}/labeling-registration-v2.json`, `${JSON.stringify(receipt, null, 2)}\n`, { flag: "wx" });
console.log(JSON.stringify({ passed: true, source: loaded.source, rallies: loaded.document.rallies.length, humanDraft: receipt.humanDraft }));
