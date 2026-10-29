import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

test("the retained lab snapshot matches its recorded file hashes", () => {
  const repository = new URL("../../", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("components/production-lab/editor/snapshot.json", repository), "utf8")) as {
    files: Array<{ destination: string; snapshotSha256: string }>;
    generatedHelpers: Array<{ file: string; sha256: string }>;
  };
  for (const entry of [...manifest.files.map(file => ({ file: file.destination, sha256: file.snapshotSha256 })), ...manifest.generatedHelpers]) {
    const source = readFileSync(new URL(entry.file, repository), "utf8").replaceAll("\r\n", "\n");
    assert.equal(createHash("sha256").update(source).digest("hex"), entry.sha256, entry.file);
  }
});

test("editor snapshots follow adapted type imports and remove stale owned files", () => {
  const root = mkdtempSync(path.join(tmpdir(), "editor-snapshot-"));
  function write(relative: string, source: string): void {
    const filename = path.join(root, relative);
    mkdirSync(path.dirname(filename), { recursive: true });
    writeFileSync(filename, source);
  }
  try {
    const editor = "components/production-lab/editor";
    write("prod/src/designs/taste/index.tsx", `import type { ReadyDesignReview } from "../useDesignReview";
export function RallyDesk(props: { review: ReadyDesignReview }) { return props.review; }
`);
    // These runtime imports must not be resolved or copied after adaptation.
    write("prod/src/designs/useDesignReview.ts", `import { runtimeHook } from "@/lib/not-in-the-lab";
export type DesignProjectOption = { id: string };
export type ReadyDesignReview = {
  state: "ready";
};
function reloadSelectedProject() { runtimeHook(); }
`);
    write("prod/src/lib/project-store.ts", `import { runtimeDatabase } from "@/lib/not-in-the-lab";
export type ProjectSource = { name: string };
export type VolleySpliceProject = { id: string; source: ProjectSource };
function migrateProjectInstance() { runtimeDatabase(); }
function requestResult() { runtimeDatabase(); }
`);
    write("prod/src/lib/cut-draft.ts", "export type CutDraft = {}; export type CutDraftSeed = {};\n");
    write("prod/src/lib/on-device/types.ts", "export type NormalizedRoi = {}; export type OnDeviceAnalysis = {}; export type OnDeviceMediaInfo = {}; export type OnDeviceSuppression = {};\n");
    write("prod/src/lib/on-device/analysis-window.ts", "export type AnalysisWindow = { start: number; end: number };\n");
    write("prod/src/lib/on-device/export.ts", "export type ExportInterval = {}; export type ScoreOverlayOptions = {};\n");
    write("prod/src/lib/product-analysis.ts", "export type ProductAnalysis = {};\n");
    for (const asset of ["volleysplice-logo.png", "volleysplice-icon-transparent.png"]) {
      write(`prod/public/runtime/${asset}`, "synthetic asset");
    }
    write(`${editor}/lib/stale.ts`, "export const obsolete = true;\n");
    write(`${editor}/notes.txt`, "Keep unmanaged files.\n");
    write(`${editor}/snapshot.json`, JSON.stringify({ files: [{ destination: `${editor}/lib/stale.ts` }] }));
    mkdirSync(path.join(root, "scripts"));
    const script = path.join(root, "scripts/snapshot-production-lab-editor.mjs");
    copyFileSync(new URL("../../scripts/snapshot-production-lab-editor.mjs", import.meta.url), script);
    const originalProjectSource = readFileSync(path.join(root, "prod/src/lib/project-store.ts"), "utf8");
    execFileSync(process.execPath, [script], { cwd: root });

    assert.equal(existsSync(path.join(root, editor, "lib/stale.ts")), false);
    assert.equal(readFileSync(path.join(root, editor, "notes.txt"), "utf8"), "Keep unmanaged files.\n");
    const adapter = readFileSync(path.join(root, editor, "designs/useDesignReview.ts"), "utf8");
    assert.match(adapter, /labMode\?: boolean/);
    assert.doesNotMatch(adapter, /runtimeHook|reloadSelectedProject/);
    const projectTypes = readFileSync(path.join(root, editor, "lib/project-store.ts"), "utf8");
    assert.match(projectTypes, /export type VolleySpliceProject/);
    assert.doesNotMatch(projectTypes, /runtimeDatabase|migrateProjectInstance|requestResult/);
    assert.equal(readFileSync(path.join(root, "prod/src/lib/project-store.ts"), "utf8"), originalProjectSource);

    const manifest = readFileSync(path.join(root, editor, "snapshot.json"), "utf8");
    assert.doesNotMatch(manifest, /stale\.ts|not-in-the-lab/);
    execFileSync(process.execPath, [script], { cwd: root });
    assert.equal(readFileSync(path.join(root, editor, "snapshot.json"), "utf8"), manifest);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
