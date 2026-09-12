import Foundation

public enum GuidedTourStage: String, Codable, Sendable { case setup, editor }

/// Android GuidedTour.kt ordering, copy, and optional-score transition contract.
public enum GuidedTourStep: String, CaseIterable, Codable, Sendable {
    case setupSource = "SETUP_SOURCE"
    case setupWindow = "SETUP_WINDOW"
    case setupCreate = "SETUP_CREATE"
    case editorSettings = "EDITOR_SETTINGS"
    case editorExport = "EDITOR_EXPORT"
    case editorScoreToggle = "EDITOR_SCORE_TOGGLE"
    case editorScorePanel = "EDITOR_SCORE_PANEL"
    case editorVideo = "EDITOR_VIDEO"
    case editorReviewQueues = "EDITOR_REVIEW_QUEUES"
    case editorOverview = "EDITOR_OVERVIEW"
    case editorFocus = "EDITOR_FOCUS"
    case editorMarking = "EDITOR_MARKING"
    public var stage: GuidedTourStage { content.0 }
    public var target: String { content.1 }
    public var title: String { content.2 }
    public var body: String { content.3 }
    public var action: String { content.4 }
    private var content: (GuidedTourStage, String, String, String, String) {
        switch self {
        case .setupSource: return (.setup, "setup-source", "Choose your video", "Pick a game video from this device. Your video stays here and is not uploaded.", "Choose video first")
        case .setupWindow: return (.setup, "setup-window", "Choose the part with the game", "If the whole video is game footage, leave it set to the full video. Otherwise, move to the game’s start and end and mark them.", "Next")
        case .setupCreate: return (.setup, "setup-create", "Let VolleySplice find the rallies", "Tap Find rallies. Keep VolleySplice open while it prepares the suggested clips; the editor opens when they are ready.", "Find rallies first")
        case .editorSettings: return (.editor, "editor-settings", "Fine-tune only when you need to", "The Settings gear contains optional automatic cleanup, extra time around clips, short-break joining, and review sensitivity. The defaults are ready for most games.", "Next")
        case .editorExport: return (.editor, "editor-export", "Save the finished video", "Tap Save final video when the review looks right. Choose the original video first if VolleySplice asks you to reconnect it.", "Next")
        case .editorScoreToggle: return (.editor, "editor-score-toggle", "Add a scoreboard if you want one", "Score tracking is optional. Turn it on to check serve markers, add anything missing, and include a scoreboard in the saved video.", "Next: check the score")
        case .editorScorePanel: return (.editor, "editor-score-panel", "Check the score markers", "Correct which side serves, add missed serves, and add a side switch whenever the teams change court sides.", "Next")
        case .editorVideo: return (.editor, "editor-video", "Watch the video", "Play, pause, or move to any moment. Turn on Play only the final video above to skip every part that will not be saved.", "Next")
        case .editorReviewQueues: return (.editor, "editor-review-queues", "Start with what needs attention", "These queues contain automatic cleanups, clips, and serves that still need review. Ignored footage is skipped, and the counts shrink as you make decisions.", "Next")
        case .editorOverview: return (.editor, "editor-overview", "Review the suggested clips", "Each block is a clip planned for the final video. Start with Check next clip, then review anything else only if it looks wrong.", "Next")
        case .editorFocus: return (.editor, "editor-focus", "Fix one clip", "Include or leave out the selected clip, preview it, and adjust where it starts or ends. Tap Looks good when a flagged clip is correct.", "Next")
        case .editorMarking: return (.editor, "editor-marking", "Add anything VolleySplice missed", "For a missed rally, mark its start and end. Leave out a section for camera gaps, breaks, or other footage that should not appear in the final video.", "Finish tutorial")
        }
    }
    public func next(scoreTrackingEnabled: Bool = true) -> GuidedTourStep? {
        let steps = Self.allCases.filter { scoreTrackingEnabled || $0 != .editorScorePanel }
        guard let index = steps.firstIndex(of: self), index + 1 < steps.count else { return nil }
        return steps[index + 1]
    }
    public static func visibleSteps(stage: GuidedTourStage, scoreTrackingEnabled: Bool) -> [Self] {
        allCases.filter { $0.stage == stage && (scoreTrackingEnabled || $0 != .editorScorePanel) }
    }
}

public struct GuidedTourProgress: Equatable, Codable, Sendable {
    public var stored: String?
    public init(stored: String? = nil) { self.stored = stored }
    public mutating func restart(stage: GuidedTourStage) {
        stored = (stage == .setup ? GuidedTourStep.setupSource : .editorSettings).rawValue
    }
    public mutating func enter(stage: GuidedTourStage, scoreTrackingEnabled: Bool = true) {
        if stored == nil { restart(stage: stage) }
        if stage == .editor, let step = stored.flatMap(GuidedTourStep.init(rawValue:)), step.stage == .setup {
            restart(stage: .editor)
        }
        if !scoreTrackingEnabled && stored == GuidedTourStep.editorScorePanel.rawValue {
            stored = GuidedTourStep.editorVideo.rawValue
        }
    }
    public func current(stage: GuidedTourStage) -> GuidedTourStep? {
        guard let step = stored.flatMap(GuidedTourStep.init(rawValue:)), step.stage == stage else { return nil }
        return step
    }
    public mutating func dismiss() { stored = "dismissed" }
    public mutating func advance(stage: GuidedTourStage, sourceReady: Bool, scoreTrackingEnabled: Bool = true) {
        guard let step = current(stage: stage), step != .setupCreate,
              step != .setupSource || sourceReady else { return }
        stored = step.next(scoreTrackingEnabled: scoreTrackingEnabled)?.rawValue ?? "done"
    }
}
