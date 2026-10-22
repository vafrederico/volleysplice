import Foundation

/// Bounded transport data is separate from the active native editor settings.
public enum PortableReviewPreferences {
    private static let booleans = ["cutPreviewEnabled", "renderScoreOnPreviewVideo", "renderScoreOnFinalVideo",
                                   "renderScoreTimeline", "fadeScoreOverlay"]
    private static let chapterKeys = ["includeRallyNumber", "includeServeNumber", "includeScore",
                                      "includeServingTeam", "includeSideSwitches", "includeCredit"]

    public static func isFuture(_ ui: JSONValue) -> Bool { (ui["schemaVersion"]?.double ?? 1) > 1 }

    public static func validate(_ ui: JSONValue, cuts: [EditableCut], checkIdentities: Bool = true) throws {
        var nodes = 0
        func visit(_ value: JSONValue, depth: Int) throws {
            nodes += 1
            guard nodes <= 4096, depth <= 16 else { throw ProjectError.invalid("corrections.ui exceeds JSON limits") }
            switch value {
            case .object(let values):
                for (key, child) in values {
                    guard !["__proto__", "constructor", "prototype"].contains(key) else {
                        throw ProjectError.invalid("corrections.ui contains an unsafe key")
                    }
                    try visit(child, depth: depth + 1)
                }
            case .array(let values): for child in values { try visit(child, depth: depth + 1) }
            case .string(let value):
                guard value.utf16.count <= 16384 else { throw ProjectError.invalid("corrections.ui string is too long") }
            case .number(let value):
                guard value.isFinite else { throw ProjectError.invalid("corrections.ui contains a nonfinite number") }
            case .bool, .null: break
            }
        }
        try visit(ui, depth: 0)
        guard case .object = ui else { throw ProjectError.invalid("corrections.ui must be an object") }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.withoutEscapingSlashes]
        guard try encoder.encode(ui).count <= 65536 else { throw ProjectError.invalid("corrections.ui exceeds 64 KiB") }
        if let version = ui["schemaVersion"] {
            guard let number = version.double, number >= 1, number <= 9007199254740991,
                  number.rounded(.towardZero) == number else {
                throw ProjectError.invalid("corrections.ui.schemaVersion must be a positive safe integer")
            }
        }
        let prefs = ui["preferences"] ?? .object([:])
        guard case .object = prefs else { throw ProjectError.invalid("corrections.ui.preferences must be an object") }
        for key in booleans where prefs[key] != nil {
            guard prefs[key]?.bool != nil else { throw ProjectError.invalid("Invalid UI boolean: \(key)") }
        }
        for (key, bounds) in [("playbackRate", 0.25...8.0), ("confidenceReviewThreshold", 0.0...1.0)] {
            if let value = prefs[key] {
                guard let number = value.double, bounds.contains(number) else { throw ProjectError.invalid("Invalid UI number: \(key)") }
            }
        }
        if let value = prefs["reviewedCutIds"] {
            guard let ids = value.array, ids.count <= 2048 else { throw ProjectError.invalid("Invalid reviewed cut IDs") }
            let inferred = Set(cuts.filter { $0.origin == .inferred }.map(\.id))
            var seen = Set<String>()
            for value in ids {
                guard let id = value.string, !id.isEmpty, id.utf16.count <= 256, seen.insert(id).inserted,
                      !checkIdentities || inferred.contains(id) else { throw ProjectError.invalid("Invalid reviewed cut identity") }
            }
        }
        // Future chapter settings are entirely opaque, including their known native keys.
        if !isFuture(ui), let chapters = prefs["chapterOptions"], chapters != .null {
            guard case .object = chapters else { throw ProjectError.invalid("Invalid chapter options") }
            for key in chapterKeys where chapters[key] != nil {
                guard chapters[key]?.bool != nil else { throw ProjectError.invalid("Invalid chapter option: \(key)") }
            }
        }
    }

    @discardableResult public static func finalScoreAlias(_ bundle: JSONValue) throws -> Bool? {
        guard let preferences = bundle["editorPreferences"] else { return nil }
        guard case .object = preferences else { throw ProjectError.invalid("editorPreferences must be an object") }
        guard let value = preferences["renderScoreOnFinalVideo"] else { return nil }
        guard let flag = value.bool else { throw ProjectError.invalid("Invalid final score compatibility preference") }
        return flag
    }

    public static func applying(_ ui: JSONValue?, alias: Bool?, to draft: EditorDraft) throws -> EditorDraft {
        var result = draft
        result.renderScoreOverlay = alias ?? result.renderScoreOverlay
        guard let ui else { return result }
        try validate(ui, cuts: draft.cuts)
        result.portableUi = ui
        guard !isFuture(ui) else { return result }
        let prefs = ui["preferences"] ?? .object([:])
        let v1 = ui["schemaVersion"] != nil
        result.finalPreviewEnabled = prefs["cutPreviewEnabled"]?.bool ?? (v1 ? false : draft.finalPreviewEnabled)
        result.renderScoreOverlay = prefs["renderScoreOnFinalVideo"]?.bool ?? alias ?? (v1 ? true : draft.renderScoreOverlay)
        result.renderScoreTimeline = prefs["renderScoreTimeline"]?.bool ?? (v1 ? false : draft.renderScoreTimeline)
        result.fadeScoreOverlay = prefs["fadeScoreOverlay"]?.bool ?? (v1 ? false : draft.fadeScoreOverlay)
        result.playbackRate = prefs["playbackRate"]?.double ?? (v1 ? 1 : draft.playbackRate)
        result.confidenceReviewThreshold = prefs["confidenceReviewThreshold"]?.double.map(Float.init) ?? (v1 ? 0.7 : draft.confidenceReviewThreshold)
        result.reviewedCutIds = prefs["reviewedCutIds"]?.array.map { Set($0.compactMap(\.string)) } ?? (v1 ? [] : draft.reviewedCutIds)
        if let chapters = prefs["chapterOptions"], chapterKeys.contains(where: { chapters[$0] != nil }) {
            let score = try ScoreTracking.fromJSON(draft.scoreTracking)
            let defaults = YouTubeChapters.defaultOptions(hasScoreTracking: score.enabled, hasSideSwitches: !score.sideSwitchMarkers.isEmpty)
            var merged = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(defaults))
            for key in chapterKeys { if let value = chapters[key] { merged[key] = value } }
            result.chapterOptions = try JSONDecoder().decode(YouTubeChapterOptions.self, from: JSONEncoder().encode(merged))
        }
        return result
    }

    /// A retained bundle is a fallback for legacy local drafts which predate portableUi.
    public static func compose(_ draft: EditorDraft, retained: JSONValue? = nil) throws -> JSONValue {
        if let retained {
            try validate(retained, cuts: draft.cuts, checkIdentities: false)
            // An update of a retained bundle must never downgrade a newer envelope.
            if isFuture(retained) { return retained }
        }
        var ui = draft.portableUi ?? retained ?? .object([:])
        try validate(ui, cuts: draft.cuts, checkIdentities: false)
        if isFuture(ui) { return ui }
        if draft.portableUi != nil, let retained {
            // Retain extensions added to either copy since import. Known transport values
            // belong to the draft, including deliberate removals made during reset.
            preserveUnknown(from: retained, in: &ui, excluding: ["schemaVersion", "preferences"])
            var prefs = ui["preferences"] ?? .object([:])
            let oldPrefs = retained["preferences"] ?? .object([:])
            preserveUnknown(from: oldPrefs, in: &prefs,
                excluding: booleans + ["playbackRate", "confidenceReviewThreshold", "reviewedCutIds", "chapterOptions"])
            if let oldChapters = oldPrefs["chapterOptions"], case .object = oldChapters {
                var chapters = prefs["chapterOptions"] ?? .object([:])
                if chapters == .null { chapters = .object([:]) }
                preserveUnknown(from: oldChapters, in: &chapters, excluding: chapterKeys)
                prefs["chapterOptions"] = chapters
            }
            ui["preferences"] = prefs
        }
        var prefs = ui["preferences"] ?? .object([:])
        // Keep the exact imported probability while its Float slider value is unchanged.
        let threshold = prefs["confidenceReviewThreshold"]?.double.flatMap {
            Float($0) == draft.confidenceReviewThreshold ? $0 : nil
        } ?? Double(String(draft.confidenceReviewThreshold)) ?? Double(draft.confidenceReviewThreshold)
        prefs["cutPreviewEnabled"] = .bool(draft.finalPreviewEnabled)
        // Native has no independent web preview score control.
        if prefs["renderScoreOnPreviewVideo"] == nil { prefs["renderScoreOnPreviewVideo"] = .bool(true) }
        prefs["renderScoreOnFinalVideo"] = .bool(draft.renderScoreOverlay)
        prefs["renderScoreTimeline"] = .bool(draft.renderScoreTimeline)
        prefs["fadeScoreOverlay"] = .bool(draft.fadeScoreOverlay ?? false)
        prefs["playbackRate"] = .number(draft.playbackRate)
        prefs["confidenceReviewThreshold"] = .number(threshold)
        prefs["reviewedCutIds"] = .array(draft.reviewedCutIds.sorted().map(JSONValue.string))
        if let options = draft.chapterOptions {
            let known = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(options))
            var chapters = prefs["chapterOptions"] ?? .object([:])
            if chapters == .null { chapters = .object([:]) }
            for key in chapterKeys { chapters[key] = known[key] }
            prefs["chapterOptions"] = chapters
        }
        ui["schemaVersion"] = .number(1); ui["preferences"] = prefs
        try validate(ui, cuts: draft.cuts)
        return ui
    }

    private static func preserveUnknown(from retained: JSONValue, in target: inout JSONValue, excluding: [String] = []) {
        guard case .object(let values) = retained else { return }
        for (key, value) in values where !excluding.contains(key) {
            if var current = target[key] {
                preserveUnknown(from: value, in: &current)
                target[key] = current
            } else { target[key] = value }
        }
    }

    public static func onOpen(_ draft: EditorDraft, deviceFade: Bool) -> EditorDraft {
        var result = draft
        result.fadeScoreOverlay = draft.fadeScoreOverlay ?? deviceFade
        return result
    }

    /// Restore ranges/review defaults while keeping PR28's score settings and unknown extensions.
    public static func retainingForReset(_ draft: EditorDraft, retained: JSONValue? = nil) throws -> JSONValue? {
        guard var ui = draft.portableUi ?? retained else { return nil }
        try validate(ui, cuts: draft.cuts, checkIdentities: false)
        if isFuture(ui) { return ui }
        var prefs = ui["preferences"] ?? .object([:])
        for key in booleans.filter({ $0 != "renderScoreOnPreviewVideo" }) + ["playbackRate", "confidenceReviewThreshold", "reviewedCutIds"] {
            prefs[key] = nil
        }
        if var chapters = prefs["chapterOptions"], chapters != .null {
            for key in chapterKeys { chapters[key] = nil }
            prefs["chapterOptions"] = chapters
        }
        ui["preferences"] = prefs
        return ui
    }

    public static func notice(_ draft: EditorDraft) -> String? {
        guard let ui = draft.portableUi, isFuture(ui) else { return nil }
        return "This project has a newer preferences version. Those settings are preserved but not applied; changes here stay local."
    }
    public static func playbackRates(_ active: Double) -> [Double] { Set([1, 1.5, 2, 4, 8, active]).sorted() }
    public static func playbackRateLabel(_ rate: Double) -> String {
        let text = String(rate)
        return (text.hasSuffix(".0") ? String(text.dropLast(2)) : text) + "x"
    }
}
