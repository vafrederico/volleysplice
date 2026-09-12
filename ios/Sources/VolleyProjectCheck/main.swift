import Foundation
import VolleyCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Read-only validation: no source video, feature values, labels, names or identifiers are printed.
func checkProject(at path: String) throws -> JSONValue {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let root = try JSONDecoder().decode(JSONValue.self, from: data)
    let native = root["schema"]?.string == "volleycut-ios-project"
    let project: ProjectDocument
    if native {
        project = try JSONDecoder().decode(ProjectDocument.self, from: data)
        try project.validate()
    } else {
        project = try ProjectArchive.importData(data, expectedFeatureNames: FeatureSchema.base)
    }
    guard let original = project.feedback else { throw ProjectError.invalid("Missing retained feedback") }
    _ = try ProjectArchive.importBundle(original, expectedFeatureNames: FeatureSchema.base)
    var updated = original
    try ProjectArchive.updateCorrections(&updated, draft: project.draft)
    guard updated["initialInference"] == original["initialInference"], updated["features"] == original["features"] else {
        throw ProjectError.invalid("Correction export modified immutable analysis")
    }
    let restored = try ProjectArchive.importBundle(updated, expectedFeatureNames: FeatureSchema.base)
    guard restored.draft.cuts == project.draft.cuts,
          restored.draft.ignoredIntervals == project.draft.ignoredIntervals,
          restored.draft.chapterOptions == project.draft.chapterOptions else {
        throw ProjectError.invalid("Correction export lost editor changes")
    }
    let analysis = try ProjectArchive.retainedAnalysis(updated, expectedFeatureNames: FeatureSchema.base)
    let intervals = EditorMath.finalIntervals(project.draft, suppression: try ProjectArchive.suppressionRegions(updated), bounds: project.gameWindow)
    let score = try ScoreTracking.fromJSON(project.draft.scoreTracking, durationMs: project.durationMs)
    var summary: JSONValue = .object([
        "ok": .bool(true), "inputFormat": .string(native ? "native-project" : "feedback"),
        "feedbackSchemaVersion": updated["schemaVersion"] ?? .null,
        "sourceDurationMs": .number(Double(project.durationMs)),
        "gameStartMs": .number(Double(project.gameWindow.startMs)), "gameEndMs": .number(Double(project.gameWindow.endMs)),
        "cutCount": .number(Double(project.draft.cuts.count)),
        "includedCutCount": .number(Double(project.draft.cuts.filter(\.included).count)),
        "manualCutCount": .number(Double(project.draft.cuts.filter { $0.origin == .manual }.count)),
        "ignoredIntervalCount": .number(Double(project.draft.ignoredIntervals.count)),
        "initialRangeCount": .number(Double(original["initialInference"]?["ranges"]?.array?.count ?? 0)),
        "featureRows": .number(Double(analysis?.timestamps.count ?? 0)),
        "featureColumns": .number(Double(analysis?.featureNames.count ?? 0)),
        "serveMarkerCount": .number(Double(score.serveMarkers.count)),
        "sideSwitchMarkerCount": .number(Double(score.sideSwitchMarkers.count)),
        "finalIntervalCount": .number(Double(intervals.count)),
        "finalDurationMs": .number(Double(EditorMath.totalFinalMs(intervals))),
        "beforePaddingMs": .number(Double(project.draft.beforePaddingMs)),
        "afterPaddingMs": .number(Double(project.draft.afterPaddingMs)),
        "joinGapMs": .number(Double(project.draft.joinGapMs)),
        "immutableInferenceUnchanged": .bool(true), "immutableFeaturesUnchanged": .bool(true)
    ])
    summary["chapterOptions"] = updated["corrections"]?["ui"]?["preferences"]?["chapterOptions"] ?? .null
    return summary
}

func printJSON(_ value: JSONValue) {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    if let data = try? encoder.encode(value) {
        FileHandle.standardOutput.write(data); FileHandle.standardOutput.write(Data([10]))
    }
}

guard CommandLine.arguments.count == 2 else {
    printJSON(.object(["ok": .bool(false), "error": .string("usage"),
                       "usage": .string("VolleyProjectCheck <feedback-or-native-project.json>")]))
    exit(2)
}
do {
    printJSON(try checkProject(at: CommandLine.arguments[1]))
} catch {
    // Decoder and file errors can contain user paths or raw values: emit only a category.
    let category = error is DecodingError ? "invalid-json-or-schema" : error is ProjectError ? "project-contract-failed" : "input-or-validation-failed"
    printJSON(.object(["ok": .bool(false), "error": .string(category)]))
    exit(1)
}
