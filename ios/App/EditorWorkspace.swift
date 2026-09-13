import SwiftUI
import AVFoundation
import Combine
import UIKit

enum EditorPalette {
    static let paper = Color(red: 0.973, green: 0.969, blue: 0.933)
    static let ink = Color(red: 0.09, green: 0.22, blue: 0.18)
    static let green = Color(red: 0.184, green: 0.408, blue: 0.333)
    static let rail = Color(red: 0.784, green: 0.839, blue: 0.792)
    static let orange = Color(red: 222 / 255.0, green: 121 / 255.0, blue: 89 / 255.0)
    static let acid = Color(red: 217 / 255.0, green: 238 / 255.0, blue: 158 / 255.0)
    static let kept = Color(red: 230 / 255.0, green: 244 / 255.0, blue: 223 / 255.0)
    static let removed = Color(red: 189 / 255.0, green: 24 / 255.0, blue: 38 / 255.0)
    static let warning = Color(red: 0.91, green: 0.639, blue: 0.09)
    static let danger = Color(red: 179 / 255.0, green: 38 / 255.0, blue: 30 / 255.0)
    static let muted = Color(red: 101 / 255.0, green: 118 / 255.0, blue: 111 / 255.0)
    static let suppression = Color(red: 209 / 255.0, green: 36 / 255.0, blue: 47 / 255.0)
    static let cleanupReview = Color(red: 244 / 255.0, green: 220 / 255.0, blue: 120 / 255.0)
    static let clipReview = Color(red: 243 / 255.0, green: 180 / 255.0, blue: 63 / 255.0)
    static let joinedGap = Color(red: 170 / 255.0, green: 169 / 255.0, blue: 162 / 255.0)
}

private struct CompactPhoneEditorKey: EnvironmentKey { static let defaultValue = false }
private extension EnvironmentValues {
    var compactPhoneEditor: Bool {
        get { self[CompactPhoneEditorKey.self] }
        set { self[CompactPhoneEditorKey.self] = newValue }
    }
}

private struct ScoreSectionHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}
private struct PlayerChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}
private struct OverviewHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}
private struct LegendHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 28
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

/// One source-time editor rearranged by usable window size, including iPad split view.
struct EditorWorkspace: View {
    @AppStorage("video-export-fade-scores") private var fadeScoreOverlay = false
    @Binding var project: ProjectDocument
    let player: AVPlayer
    let onSave: () -> Void
    let onExportProject: () -> Void
    let onExportVideo: () -> Void
    let onBack: () -> Void
    var savedProjects: [URL] = []
    var projectNames: [URL: String] = [:]
    var projectDates: [URL: Date] = [:]
    var onSelectProject: (URL) -> Void = { _ in }
    var onNewProject: () -> Void = {}
    var onManageStorage: () -> Void = {}
    var onDeleteProject: () -> Void = {}
    var onShowQueue: () -> Void = {}
    var onPrepareScore: (() -> Void)? = nil
    var onReconnectSource: (() -> Void)? = nil
    @State private var playheadMs: Int64 = 0
    @State private var playing = false
    @State private var settings = false
    @State private var exportTab = false
    @State private var compactChapters = false
    @State private var compactProjectOptions = false
    @State private var pendingCompactProjectExport = false
    @State private var trimWindow: TimeRange?
    @State private var resetConfirmation = false
    @State private var deleteConfirmation = false
    @State private var allSuggestions: [SuppressionRegion] = []
    @State private var selectedSuggestionId: String?
    @State private var presentation = EditorPresentation(draft: .init(sourceRevision: ""), suggestions: [], bounds: .init(startMs: 0, endMs: 1), durationMs: 1)
    @State private var preparedOverlay = PreparedScoreOverlay(tracking: ScoreTracking(enabled: false), rallyRanges: [])
    @State private var preparedInput: EditorDraft?
    @State private var message: String?
    @State private var lastPreviewSeek: Int64 = -1
    @State private var previewEndMs: Int64?
    @State private var leftDragStart: Double?
    @State private var rightDragStart: Double?
    @State private var videoDragStart: Double?
    @State private var scoreContentHeight: CGFloat = 760
    @State private var playerChromeHeight: CGFloat = 82
    @State private var overviewHeight: CGFloat = 212
    @State private var serveReviewReveal = 0
    @AppStorage(GuidedTourStore.key) private var tourState = ""
    @Environment(\.interfaceScale) private var interfaceScale
    @State private var deviceOrientation = UIDevice.current.orientation
    private var landscapeWindow: UIWindow? {
        guard isPhone else { return nil }
        let window = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            .flatMap(\.windows).first(where: \.isKeyWindow)
        guard let window, window.bounds.width > window.bounds.height else { return nil }
        return window
    }
    private var landscapeInsets: EdgeInsets {
        guard let window = landscapeWindow else { return EdgeInsets() }
        return EdgeInsets(top: 0, leading: window.safeAreaInsets.left / interfaceScale,
                          bottom: 0, trailing: window.safeAreaInsets.right / interfaceScale)
    }
    private var landscapeSpace: EdgeInsets {
        guard let window = landscapeWindow else { return EdgeInsets() }
        let orientation: UIInterfaceOrientation = deviceOrientation.isLandscape
            ? (deviceOrientation == .landscapeLeft ? .landscapeRight : .landscapeLeft)
            : window.windowScene?.interfaceOrientation ?? .unknown
        return EdgeInsets(top: 0,
            leading: orientation == .landscapeLeft ? max(0, window.safeAreaInsets.left - 12) / interfaceScale : 0,
            bottom: 0,
            trailing: orientation == .landscapeRight ? max(0, window.safeAreaInsets.right - 12) / interfaceScale : 0)
    }
    private var isPhone: Bool { UIDevice.current.userInterfaceIdiom == .phone }
    @AppStorage("desktop_editor_layout.leftSidebarWidth") private var leftWidth = 292.0
    @AppStorage("desktop_editor_layout.rightSidebarWidth") private var rightWidth = 292.0
    @AppStorage("desktop_editor_layout.phoneLeftSidebarWidth") private var phoneLeftWidth = 180.0
    @AppStorage("desktop_editor_layout.phoneRightSidebarWidth") private var phoneRightWidth = 220.0
    @AppStorage("desktop_editor_layout.desktopReviewPlayerHeight") private var desktopVideoHeight = 320.0
    @AppStorage("desktop_editor_layout.compactReviewPlayerHeight") private var compactVideoHeight = 176.0
    @AppStorage("desktop_editor_layout.phoneLandscapePlayerHeight") private var phoneVideoHeight = 240.0
    @AppStorage("desktop_editor_layout.phoneLandscapePlayerManual") private var phoneVideoManual = false
    private let clock = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    private var draft: EditorDraft { project.draft }
    private var bounds: TimeRange { project.gameWindow }
    private var suggestions: [SuppressionRegion] { presentation.activeSuggestions }
    private var intervals: [FinalCutInterval] { presentation.intervals }
    private var keptIds: Set<String> { presentation.keptIds }
    private var orderedCuts: [EditableCut] { presentation.orderedCuts }
    private var pendingConfidenceIds: Set<String> { presentation.pendingConfidenceIds }
    private var rallyGroups: [EditableRallyGroup] { presentation.groups }
    private var currentGroup: EditableRallyGroup? {
        if let selected = suggestions.first(where: { $0.id == selectedSuggestionId }),
           let group = rallyGroups.first(where: { $0.cuts.contains { $0.coreStartMs < selected.endMs && selected.startMs < $0.coreEndMs } }) { return group }
        return rallyGroups.first { $0.keepStartMs <= playheadMs && playheadMs < $0.coreEndMs } ??
        rallyGroups.last { $0.coreStartMs <= playheadMs } ?? rallyGroups.first
    }
    private var currentCut: EditableCut? { currentGroup?.asEditableCut() }
    private var currentGroupIndex: Int { rallyGroups.firstIndex { $0.id == currentGroup?.id } ?? -1 }
    private var keptRallyCount: Int { rallyGroups.filter { !$0.cutIds.isDisjoint(with: keptIds) }.count }
    private var reviewCuts: [EditableCut] { presentation.reviewCuts }
    private var score: ScoreTracking { presentation.score }
    private var scorePreparationNeeded: Bool {
        let output = project.feedback?["initialInference"]?["servingSide"]
        let switches = project.feedback?["initialInference"]?["sideSwitch"]
        let wantsSwitches = project.feedback?["source"]?["generateSideSwitchMarkers"]?.bool ?? true
        return output == nil || output == .null || (wantsSwitches && (switches == nil || switches == .null)) || project.feedback?["scorePreparationError"]?.string != nil
    }
    private var visibleScore: ScoreTracking { presentation.visibleScore }
    private var scoreTimestampMs: Int64 { preparedOverlay.snapshot(at: playheadMs).effectiveTimestampMs }
    private var derivedScore: DerivedScore { preparedOverlay.snapshot(at: playheadMs).score }
    private var currentServe: ServeMarker? {
        let id = preparedOverlay.snapshot(at: playheadMs).currentServeId
        return visibleScore.serveMarkers.first { $0.id == id } ?? visibleScore.serveMarkers.first
    }
    private var reviewServes: [ServeMarker] { presentation.reviewServes }
    private var selectedCleanup: SuppressionRegion? {
        suggestions.first { $0.id == selectedSuggestionId } ?? currentGroup.flatMap { group in
            suggestions.first { suggestion in group.cuts.contains { $0.coreStartMs < suggestion.endMs && suggestion.startMs < $0.coreEndMs } }
        }
    }

    var body: some View {
        GeometryReader { geometry in
            let safeWidth = geometry.size.width - landscapeInsets.leading - landscapeInsets.trailing
            let desktop = EditorLayout.mode(width: Double(safeWidth), height: Double(geometry.size.height)) == .desktop
            VStack(spacing: 0) {
                if desktop {
                    header(desktop: true, condensed: safeWidth < 1100).environment(\.compactPhoneEditor, false)
                        .padding(.leading, landscapeInsets.leading).padding(.trailing, landscapeInsets.trailing)
                    Divider()
                    if exportTab { exportWorkspace(desktop: true)
                        .padding(.leading, landscapeInsets.leading).padding(.trailing, landscapeInsets.trailing)
                    }
                    else {
                        desktopWorkspace(size: CGSize(width: safeWidth + landscapeSpace.leading + landscapeSpace.trailing, height: geometry.size.height))
                            .padding(.leading, landscapeInsets.leading - landscapeSpace.leading)
                            .padding(.trailing, landscapeInsets.trailing - landscapeSpace.trailing)
                    }
                } else { compactWorkspace
                    .padding(.leading, landscapeInsets.leading).padding(.trailing, landscapeInsets.trailing)
                }
            }
            .environment(\.compactPhoneEditor, isPhone)
            .background(EditorPalette.paper).foregroundStyle(EditorPalette.ink).tint(EditorPalette.green)
            .clipped()
        }
        // Give the editor real bounds across the horizontal window. Keep all
        // panes inside those bounds instead of offsetting scroll views beyond
        // a safe-area-sized root, which can retain a stale origin on rotation.
        .padding(.leading, -landscapeInsets.leading)
        .padding(.trailing, -landscapeInsets.trailing)
        .background(EditorPalette.paper.ignoresSafeArea())
        .onAppear {
            allSuggestions = (try? project.feedback.map(ProjectArchive.suppressionRegions)) ?? []
            refreshPresentation(); pollPlayback(); setTrimWindow()
        }
        .onChange(of: project.draft) { _, _ in refreshPresentation() }
        .onReceive(clock) { _ in pollPlayback() }
        .onReceive(NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification)) { _ in
            deviceOrientation = UIDevice.current.orientation
        }
        .onChange(of: currentCut?.id) { _, _ in setTrimWindow() }
        .onDisappear { player.pause(); onSave() }
        .sheet(isPresented: $settings) { settingsPanel.interfaceScaled() }
        .sheet(isPresented: $compactChapters) {
            NavigationStack {
                ScrollView { chapterExport.padding(20) }.navigationTitle("YouTube chapters")
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { compactChapters = false } } }
            }.interfaceScaled()
        }
        .sheet(isPresented: $compactProjectOptions, onDismiss: {
            if pendingCompactProjectExport { pendingCompactProjectExport = false; onSave(); onExportProject() }
        }) {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Project file").font(.headline)
                        Text("Features, model predictions, edits, ignored time and score markers. The source video is kept separate.").font(.callout)
                        Button("Save project") { pendingCompactProjectExport = true; compactProjectOptions = false }
                            .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).accessibilityIdentifier("exportProject")
                    }.padding(20)
                }.navigationTitle("Project options")
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { compactProjectOptions = false } } }
            }.interfaceScaled()
        }
        .alert("Delete current project?", isPresented: $deleteConfirmation) {
            Button("Delete", role: .destructive, action: onDeleteProject)
            Button("Cancel", role: .cancel) {}
        } message: { Text("Delete the saved project and edits for \(project.sourceName)? The source video and exported files will be kept.") }
        .alert("Restore model ranges?", isPresented: $resetConfirmation) {
            Button("Restore ranges", role: .destructive) { resetRanges() }
            Button("Cancel", role: .cancel) {}
        } message: { Text("This replaces clip boundaries, Keep/Remove decisions, manual ranges and ignored spans with the original analysis. Score markers are retained.") }
        .alert("Unable to edit", isPresented: Binding(get: { message != nil }, set: { if !$0 { message = nil } })) {
            Button("OK") { message = nil }
        } message: { Text(message ?? "") }
    }

    private func header(desktop: Bool, condensed: Bool) -> some View {
        Group {
            if desktop {
                HStack(spacing: 8) {
                    VolleySpliceBrand(compact: condensed).frame(width: condensed ? 40 : 154, height: 34)
                    Menu { projectMenuItems } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("CURRENT REVIEW").font(.system(size: 8, weight: .bold))
                            HStack(spacing: 4) { Text(project.sourceName).lineLimit(1).truncationMode(.middle); Image(systemName: "chevron.down") }
                                .font(.system(size: 11))
                        }.frame(maxWidth: condensed ? 125 : 190, alignment: .leading)
                    }.buttonStyle(.plain).accessibilityLabel("Projects").accessibilityIdentifier("projectSelector")
                    stageButton("Review", exported: false)
                    stageButton("Export", exported: true)
                    Spacer(minLength: 0)
                    queueCounter("Review cleanup", count: presentation.pendingCleanup.count, identifier: "reviewCleanup", condensed: condensed, action: nextCleanup).guidedTourTarget("editor-review-queues")
                    queueCounter("Review clips", count: reviewCuts.count, identifier: "reviewClips", condensed: condensed, action: nextReview)
                    queueCounter("Review serves", count: score.enabled ? reviewServes.count : 0, identifier: "reviewServes", condensed: condensed, action: nextServeReview)
                    if !condensed { headerStat("\(keptRallyCount)", label: "clips included") }
                    headerStat(time(EditorMath.totalFinalMs(intervals)), label: "planned video")
                        .accessibilityIdentifier("exportSummary").accessibilityValue("Wide workspace")
                    Button(action: { settings = true }) { Image(systemName: "gearshape").frame(width: 22, height: 24) }
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink, minimumHeight: 44))
                        .accessibilityLabel("Editor settings").accessibilityIdentifier("editorSettings").guidedTourTarget("editor-settings")
                    moreMenu
                }.padding(.horizontal, 8).padding(.vertical, 4)
                    .overlay { RoundedRectangle(cornerRadius: 10).stroke(EditorPalette.rail) }
                    .padding(.horizontal, 10).padding(.top, 4).padding(.bottom, 2)
            } else { compactHeader }
        }
    }
    private var projectMenuItems: some View {
        SavedProjectMenuItems(projects: savedProjects, names: projectNames, dates: projectDates,
            onNew: onNewProject, onSelect: onSelectProject)
    }
    private var moreMenu: some View {
        Menu("More") {
            Button("Processing queue", action: onShowQueue).accessibilityIdentifier("editorProcessingQueue")
            Button("Manage storage", action: onManageStorage).accessibilityIdentifier("manageStorage")
            Button("Restart guided tour") { exportTab = false; GuidedTourStore.restart(stage: .editor) }.accessibilityIdentifier("replayTutorial")
            Divider()
            Button("Delete current project", role: .destructive) { deleteConfirmation = true }.accessibilityIdentifier("deleteCurrentProject")
        }.accessibilityIdentifier("editorMore")
    }
    private var compactHeader: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                VolleySpliceBrand().fixedSize(horizontal: true, vertical: false).scaleEffect(0.78, anchor: .leading).frame(width: 134, height: 40, alignment: .leading)
                Text("VIDEO EDITOR · v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.1.0")")
                    .font(.system(size: 8, weight: .bold)).tracking(0.5).foregroundStyle(EditorPalette.muted)
                    .lineLimit(1).minimumScaleFactor(0.8).padding(5).background(EditorPalette.rail.opacity(0.2), in: RoundedRectangle(cornerRadius: 4))
                Spacer(minLength: 0)
                Button { settings = true } label: { Image(systemName: "gearshape.fill").font(.system(size: 22)).frame(width: 40, height: 40) }
                    .buttonStyle(.plain).accessibilityLabel("Editor settings").accessibilityIdentifier("editorSettings").guidedTourTarget("editor-settings")
            }
            HStack(alignment: .bottom, spacing: 8) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("CURRENT PROJECT").font(.system(size: 9, weight: .black)).tracking(0.9).foregroundStyle(EditorPalette.green)
                    Menu { projectMenuItems } label: {
                        Text("\(project.sourceName) · ready to review").font(.system(size: 13, weight: .semibold)).lineLimit(1).truncationMode(.tail)
                            .frame(maxWidth: .infinity, minHeight: 40).padding(.horizontal, 10)
                            .overlay { RoundedRectangle(cornerRadius: 6).stroke(EditorPalette.rail).allowsHitTesting(false) }.contentShape(Rectangle())
                    }.buttonStyle(.plain).accessibilityLabel("Current project, \(project.sourceName)").accessibilityIdentifier("projectSelector")
                }
                moreMenu.font(.system(size: 13, weight: .semibold)).frame(minWidth: 44, minHeight: 40)
            }
        }.padding(.horizontal, 12).padding(.vertical, 10)
            .overlay { RoundedRectangle(cornerRadius: 10).stroke(EditorPalette.rail).allowsHitTesting(false) }
    }
    private var reviewButton: some View {
        queueCounter("Review clips", count: reviewCuts.count, identifier: "reviewClips", action: nextReview)
    }
    private var reviewServesButton: some View {
        queueCounter("Review serves", count: score.enabled ? reviewServes.count : 0, identifier: "reviewServes", action: nextServeReview)
    }
    private func headerStat(_ value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 12, weight: .semibold, design: .monospaced))
            Text(label).font(.system(size: 8)).foregroundStyle(EditorPalette.muted)
        }.padding(.horizontal, 7).frame(minWidth: 68, minHeight: 40)
            .overlay { RoundedRectangle(cornerRadius: 4).stroke(EditorPalette.rail) }
            .accessibilityElement(children: .combine)
    }
    private func stageButton(_ title: String, exported: Bool) -> some View {
        Button { exportTab = exported } label: {
            VStack(spacing: 4) {
                Text(title).font(.system(size: 11, weight: .bold))
                    .foregroundStyle(exportTab == exported ? EditorPalette.ink : EditorPalette.muted)
                Rectangle().fill(exportTab == exported ? EditorPalette.green : EditorPalette.rail).frame(height: 2)
            }.frame(width: 60, height: 36).contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityIdentifier(exported ? "exportTab" : "reviewTab").guidedTourTarget(exported ? "editor-export" : "editor-review")
            .accessibilityAddTraits(exportTab == exported ? .isSelected : [])
    }
    private func desktopWorkspace(size: CGSize) -> some View {
        let phone = UIDevice.current.userInterfaceIdiom == .phone
        let rightMinimum = phone ? EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: landscapeSpace.trailing) : 220
        let widths = EditorLayout.sidebarWidths(availableWidth: Double(size.width - landscapeSpace.leading - landscapeSpace.trailing),
            left: phone ? phoneLeftWidth : leftWidth, right: phone ? phoneRightWidth : rightWidth, rightMinimum: rightMinimum)
        let left = widths.left
        let right = widths.right
        return HStack(spacing: 0) {
            GeometryReader { pane in
                let registerHeight = EditorLayout.clipRegisterHeight(availableHeight: Double(pane.size.height) - 24,
                                                                    scoreContentHeight: Double(scoreContentHeight))
                ScrollViewReader { proxy in
                    // Android scrolls the entire events column. Keep all score
                    // controls in its natural flow instead of clipping SELECTED
                    // SERVE beneath a 520-point inner viewport.
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            sectionHeading("MATCH EVENTS", color: EditorPalette.green).accessibilityIdentifier("eventsHeading").frame(maxWidth: .infinity, alignment: .leading)
                            scoreSection(desktop: true).guidedTourTarget("editor-score-panel")
                                .background { GeometryReader { content in
                                    Color.clear.preference(key: ScoreSectionHeightKey.self, value: content.size.height)
                                } }
                            Divider()
                            clipRegister(maximumHeight: registerHeight, desktop: true).frame(height: registerHeight)
                        }.padding(12)
                            .padding(.leading, landscapeSpace.leading > 0 ? 12 / interfaceScale : 0)
                    }
                        .contentMargins(.leading, 0, for: .scrollContent)
                        .onPreferenceChange(ScoreSectionHeightKey.self) { height in
                            if height > 0 && abs(height - scoreContentHeight) > 0.5 { scoreContentHeight = height }
                        }
                        .onChange(of: serveReviewReveal) { _, _ in proxy.scrollTo("selected-serve-controls", anchor: .bottom) }
                        .onChange(of: tourState) { _, _ in revealTutorial(using: proxy, targets: ["editor-score-toggle", "editor-score-panel"]) }
                        .onAppear { revealTutorial(using: proxy, targets: ["editor-score-toggle", "editor-score-panel"]) }
                }
            }.frame(width: left + landscapeSpace.leading).accessibilityIdentifier("eventsSidebar")
            resizeDivider(label: "Resize events sidebar", identifier: "resizeLeftSidebar", action: { translation, ended in
                if leftDragStart == nil { leftDragStart = left }
                let value = EditorLayout.resizedSidebarWidth(current: leftDragStart ?? left, translation: translation, side: .left, maximum: widths.leftMaximum)
                if phone { phoneLeftWidth = value } else { leftWidth = value }
                if ended { leftDragStart = nil }
            })
            GeometryReader { pane in
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: phone ? 4 : 12) {
                        videoSection(desktop: true, availableHeight: pane.size.height, phoneLandscape: phone)
                        gameTimeline(twoRows: true, trackHeight: phone ? 40 : 60)
                            .background { GeometryReader { content in Color.clear.preference(key: OverviewHeightKey.self, value: content.size.height) } }
                    }.padding(phone ? 6 : 12)
                }
                    .onPreferenceChange(OverviewHeightKey.self) { value in if value > 0 && abs(value - overviewHeight) > 0.5 { overviewHeight = value } }
                    .onPreferenceChange(PlayerChromeHeightKey.self) { value in if value > 0 && abs(value - playerChromeHeight) > 0.5 { playerChromeHeight = value } }
                    .onChange(of: tourState) { _, _ in revealTutorial(using: proxy, targets: ["editor-video", "editor-overview"]) }
                    .onAppear { revealTutorial(using: proxy, targets: ["editor-video", "editor-overview"]) }
            }
            }.frame(maxWidth: .infinity)
            resizeDivider(label: "Resize clip controls", identifier: "resizeRightSidebar", action: { translation, ended in
                if rightDragStart == nil { rightDragStart = right }
                let value = EditorLayout.resizedSidebarWidth(current: rightDragStart ?? right, translation: translation, side: .right, maximum: widths.rightMaximum, rightMinimum: rightMinimum)
                if phone { phoneRightWidth = value } else { rightWidth = value }
                if ended { rightDragStart = nil }
            })
            ScrollViewReader { proxy in
                ScrollView { VStack(alignment: .leading, spacing: phone ? 8 : 16) { currentClipControls(desktop: true); Divider(); rangeTools.guidedTourTarget("editor-marking"); Divider(); playbackOptions; VolleySpliceLegalFooter() }.padding(phone ? 8 : 12) }
                    .onChange(of: tourState) { _, _ in revealTutorial(using: proxy, targets: ["editor-focus", "editor-marking"]) }
                    .onAppear { revealTutorial(using: proxy, targets: ["editor-focus", "editor-marking"]) }
            }.frame(width: right + landscapeSpace.trailing).accessibilityIdentifier("rallySidebar")
        }.frame(maxHeight: .infinity)
    }
    private var compactWorkspace: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    compactHeader
                    compactWorkflow(onReview: { proxy.scrollTo("compact-review", anchor: .top) }, onExport: { proxy.scrollTo("compact-final-video", anchor: .top) })
                    compactSourceCard
                    compactFinalVideo.id("compact-final-video").guidedTourTarget("editor-export")
                    compactCard { scoreSection(desktop: false).guidedTourTarget("editor-score-panel") }.id("compact-review")
                    videoSection(desktop: false)
                    compactReviewQueues { target in proxy.scrollTo(target, anchor: .top) }.guidedTourTarget("editor-review-queues")
                    compactCard { gameTimeline(twoRows: true) }
                    compactCard { currentClipControls(desktop: false) }.id("compact-current-rally")
                    compactCard { rangeTools.guidedTourTarget("editor-marking") }
                    compactCard { clipRegister(maximumHeight: 360) }
                    VolleySpliceLegalFooter()
                }.padding(.horizontal, 16).padding(.vertical, 12)
            }
                .onChange(of: selectedSuggestionId) { _, id in if id != nil { proxy.scrollTo("compact-current-rally", anchor: .top) } }
                .onChange(of: tourState) { _, _ in revealTutorial(using: proxy) }
                .onAppear { revealTutorial(using: proxy) }
        }
    }
    private func revealTutorial(using proxy: ScrollViewProxy, targets: Set<String>? = nil) {
        guard let step = GuidedTourStep(rawValue: tourState), step.stage == .editor,
              targets == nil || targets!.contains(step.target) else { return }
        // The target anchors use the same stable IDs as the Android tour steps.
        // Centering leaves room for the tutorial card above or below the control.
        proxy.scrollTo(step.target, anchor: .center)
    }
    private func compactCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content().frame(maxWidth: .infinity, alignment: .leading).padding(12)
            .overlay { RoundedRectangle(cornerRadius: 10).stroke(EditorPalette.rail) }
    }
    private func compactWorkflow(onReview: @escaping () -> Void, onExport: @escaping () -> Void) -> some View {
        HStack(spacing: 12) {
            compactWorkflowStep(1, title: "Video", complete: true, current: false, identifier: "videoStage") { player.pause(); onSave(); onBack() }
            Rectangle().fill(EditorPalette.rail).frame(maxWidth: .infinity).frame(height: 1)
            compactWorkflowStep(2, title: "Review", complete: false, current: true, identifier: "reviewTab", action: onReview)
            Rectangle().fill(EditorPalette.rail).frame(maxWidth: .infinity).frame(height: 1)
            compactWorkflowStep(3, title: "Export", complete: false, current: false, identifier: "exportTab", action: onExport)
        }
    }
    private func compactWorkflowStep(_ number: Int, title: String, complete: Bool, current: Bool, identifier: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Text("\(number)").font(.system(size: 11, weight: .bold)).frame(width: 27, height: 27)
                    .background(complete || current ? EditorPalette.acid : .clear, in: Circle())
                    .overlay { Circle().stroke(current ? EditorPalette.green : EditorPalette.rail) }
                Text(title).font(.system(size: 10, weight: .semibold))
            }.frame(minHeight: 32)
        }.buttonStyle(.plain).fixedSize(horizontal: true, vertical: false).accessibilityIdentifier(identifier).accessibilityAddTraits(current ? .isSelected : [])
    }
    private var compactSourceCard: some View {
        compactCard {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 16) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("NOW REVIEWING").font(.system(size: 10, weight: .black)).tracking(0.8).foregroundStyle(EditorPalette.orange)
                        Text(project.sourceName).font(.system(size: 14, weight: .semibold)).lineLimit(2)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                    VStack(alignment: .leading, spacing: 10) {
                        Text("RALLIES FOUND").font(.system(size: 10, weight: .black)).tracking(0.8).foregroundStyle(EditorPalette.orange)
                        Text("\(project.feedback?["initialInference"]?["ranges"]?.array?.count ?? draft.cuts.filter { $0.origin == .inferred }.count) suggested clips").font(.system(size: 14, weight: .semibold))
                    }
                }
                Label("Saved on this device", systemImage: "circle.fill").font(.system(size: 10)).foregroundStyle(EditorPalette.green)
            }
        }
    }
    private var compactFinalVideo: some View {
        let pending = reviewCuts.count + presentation.pendingCleanup.count + (score.enabled ? reviewServes.count : 0)
        return compactCard {
            VStack(alignment: .leading, spacing: 12) {
                Text("YOUR FINAL VIDEO").font(.system(size: 10, weight: .black)).tracking(0.8).foregroundStyle(EditorPalette.orange)
                Text("\(time(EditorMath.totalFinalMs(intervals))) · \(keptRallyCount) clips included").font(.system(size: 16, weight: .semibold))
                    .accessibilityIdentifier("exportSummary").accessibilityValue("Compact workspace")
                Text(pending == 0 ? "Everything that needs attention has been checked." : "\(pending) \(pending == 1 ? "item needs" : "items need") attention · \(reviewCuts.count) clips · \(presentation.pendingCleanup.count) cleanup · \(score.enabled ? reviewServes.count : 0) serves")
                    .font(.system(size: 12, weight: .semibold)).foregroundStyle(pending == 0 ? EditorPalette.green : EditorPalette.ink)
                Text("Use the Settings gear to fine-tune automatic cleanup, extra time around clips, short breaks, and how many clips are flagged.").font(.system(size: 12)).foregroundStyle(EditorPalette.muted)
                Toggle(isOn: Binding(get: { draft.finalPreviewEnabled }, set: { value in edit { $0.finalPreviewEnabled = value }; lastPreviewSeek = -1 })) {
                    compactToggleLabel("Play only the final video", detail: "Skip every part that will not be saved")
                }.accessibilityIdentifier("finalCutPreview")
                if score.enabled {
                    Toggle(isOn: Binding(get: { draft.renderScoreOverlay }, set: { value in edit { $0.renderScoreOverlay = value; if !value { $0.renderScoreTimeline = false } } })) {
                        compactToggleLabel("Add scores to the final video", detail: "Preview the scoreboard and include it when you save")
                    }.accessibilityIdentifier("renderScores")
                    if draft.renderScoreOverlay {
                        scoreVisibilityPicker
                        Toggle(isOn: Binding(get: { draft.renderScoreTimeline }, set: { value in edit { $0.renderScoreTimeline = value } })) {
                            compactToggleLabel("Show the point history", detail: "Show the two team rails beside the score when a new point starts")
                        }.padding(.leading, 18).accessibilityIdentifier("renderPointTimeline")
                    }
                }
                Divider()
                Button("Save final video") { saveVideoWithOverlaySettings() }
                    .buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white, expand: true, minimumHeight: 48))
                    .disabled(intervals.isEmpty || player.currentItem == nil).accessibilityIdentifier("exportVideo")
                Button("YouTube chapters") { compactChapters = true }
                    .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink, expand: true)).accessibilityIdentifier("youtubeChaptersButton")
                Button("Project options") { compactProjectOptions = true }.font(.system(size: 12, weight: .semibold))
                    .frame(maxWidth: .infinity, minHeight: 36).accessibilityIdentifier("projectOptions")
                Text("The finished video and optional project files stay on this device.").font(.system(size: 10)).foregroundStyle(EditorPalette.muted)
            }
        }
    }
    private func compactToggleLabel(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.system(size: 14, weight: .semibold))
            Text(detail).font(.system(size: 12)).foregroundStyle(EditorPalette.muted)
        }
    }
    private func compactReviewQueues(scrollTo: @escaping (String) -> Void) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) { compactQueueButtons(scrollTo: scrollTo) }
            VStack(spacing: 8) { compactQueueButtons(scrollTo: scrollTo) }
        }
    }
    @ViewBuilder private func compactQueueButtons(scrollTo: @escaping (String) -> Void) -> some View {
        compactQueueButton("Review cleanup", count: presentation.pendingCleanup.count, identifier: "reviewCleanup") { nextCleanup(); scrollTo("compact-current-rally") }
        compactQueueButton("Review clips", count: reviewCuts.count, identifier: "reviewClips") { nextReview(); scrollTo("compact-current-rally") }
        compactQueueButton("Review serves", count: score.enabled ? reviewServes.count : 0, identifier: "reviewServes") { nextServeReview(); scrollTo("compact-review") }
    }
    private func compactQueueButton(_ title: String, count: Int, identifier: String, action: @escaping () -> Void) -> some View {
        Button("\(title) · \(count)", action: action).font(.system(size: 11, weight: .semibold)).fixedSize(horizontal: true, vertical: false)
            .buttonStyle(EditorActionButtonStyle(background: count > 0 ? (identifier == "reviewCleanup" ? EditorPalette.cleanupReview : EditorPalette.clipReview) : EditorPalette.rail.opacity(0.4), foreground: EditorPalette.ink, minimumHeight: 44))
            .disabled(count == 0).accessibilityLabel("\(title) \(count)").accessibilityIdentifier(identifier)
    }
    private func resizeDivider(label: String, identifier: String, action: @escaping (Double, Bool) -> Void) -> some View {
        Color.clear.frame(width: 16).overlay { Rectangle().fill(EditorPalette.rail).frame(width: 1) }
            .overlay { VStack(spacing: 3) { ForEach(0..<3) { _ in Rectangle().fill(EditorPalette.muted).frame(width: 6, height: 2) } } }
            // The divider moves during resizing. Local coordinates feed that
            // movement back into translation, making the bar lag behind touch.
            .contentShape(Rectangle()).gesture(DragGesture(coordinateSpace: .global)
                .onChanged { action($0.translation.width / interfaceScale, false) }
                .onEnded { action($0.translation.width / interfaceScale, true) })
            .accessibilityLabel(label).accessibilityIdentifier(identifier)
    }
    private func videoSection(desktop: Bool, availableHeight: Double = 400, phoneLandscape: Bool = false) -> some View {
        let preferredHeight = desktopVideoHeight > 0 ? desktopVideoHeight : 320
        let height = phoneLandscape ? EditorLayout.phoneLandscapePlayerHeight(preferred: phoneVideoHeight,
            paneHeight: availableHeight, timelineHeight: overviewHeight, chromeHeight: playerChromeHeight, manual: phoneVideoManual)
            : EditorLayout.playerHeight(desktop ? preferredHeight : compactVideoHeight, desktop: desktop)
        return VStack(spacing: phoneLandscape ? 2 : 8) {
            ZStack(alignment: .topTrailing) {
                SourcePlayerView(player: player).background(.black).contentShape(Rectangle()).onTapGesture { togglePlay() }
                    .accessibilityLabel(playing ? "Pause video" : "Play video").accessibilityAddTraits(.isButton).accessibilityIdentifier("editorVideo").guidedTourTarget("editor-video")
                if player.currentItem == nil {
                    VStack(spacing: 12) {
                        Text("Reconnect the source video to enable playback.").font(.callout).foregroundStyle(.white).accessibilityIdentifier("sourceDisconnected")
                        if let onReconnectSource {
                            Button("Reconnect source", action: onReconnectSource).buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).accessibilityIdentifier("reconnectSource")
                        }
                    }.frame(maxWidth: .infinity, maxHeight: .infinity).padding(16)
                }
                if score.enabled && draft.renderScoreOverlay {
                    GeometryReader { geometry in
                        let sourceSize = player.currentItem?.presentationSize ?? .zero
                        let aspect = sourceSize.width > 0 && sourceSize.height > 0 ? sourceSize.width / sourceSize.height : 16.0 / 9.0
                        let width = min(geometry.size.width, geometry.size.height * aspect)
                        let height = width / aspect
                        ScoreOverlayPreview(prepared: preparedOverlay, sourceTimestampMs: playheadMs, renderTimeline: draft.renderScoreTimeline, fadeScoreOverlay: fadeScoreOverlay)
                            .frame(width: width, height: height)
                            .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                    }.allowsHitTesting(false)
                }
            }.frame(height: height)
            VStack(spacing: 8) {
            if !phoneLandscape { videoResizeHandle(height: height, preferredHeight: preferredHeight, desktop: desktop, availableHeight: availableHeight, phoneLandscape: false) }
            transport(desktop: desktop, phoneLandscape: phoneLandscape) {
                videoResizeHandle(height: height, preferredHeight: preferredHeight, desktop: desktop, availableHeight: availableHeight, phoneLandscape: phoneLandscape)
            }
            }.background { GeometryReader { content in Color.clear.preference(key: PlayerChromeHeightKey.self, value: content.size.height) } }
        }
    }
    private func videoResizeHandle(height: Double, preferredHeight: Double, desktop: Bool, availableHeight: Double, phoneLandscape: Bool) -> some View {
        Capsule().fill(EditorPalette.rail).frame(width: 54, height: 6).frame(maxWidth: .infinity).frame(height: phoneLandscape ? 28 : 18).contentShape(Rectangle())
                .gesture(DragGesture(coordinateSpace: .global).onChanged { event in
                    let current = phoneLandscape ? height : (desktop ? preferredHeight : compactVideoHeight)
                    if videoDragStart == nil { videoDragStart = current }
                    let requested = (videoDragStart ?? current) + Double(event.translation.height) / interfaceScale
                    if phoneLandscape {
                        phoneVideoHeight = EditorLayout.phoneLandscapePlayerHeight(preferred: requested, paneHeight: availableHeight,
                            timelineHeight: overviewHeight, chromeHeight: playerChromeHeight, manual: true)
                        phoneVideoManual = true
                    } else {
                        let value = EditorLayout.playerHeight(requested, desktop: desktop)
                        if desktop { desktopVideoHeight = value } else { compactVideoHeight = value }
                    }
                }.onEnded { _ in videoDragStart = nil }).accessibilityLabel("Resize video").accessibilityIdentifier("resizeVideo")
                .contextMenu {
                    if phoneLandscape {
                        Button("Fit video and timeline") { phoneVideoManual = false }
                            .accessibilityIdentifier("fitVideoAndTimeline")
                    }
                }
                .accessibilityValue(phoneLandscape ? (phoneVideoManual ? "Manual height" : "Fit video and timeline") : "Resizable")
    }
    @ViewBuilder private func transport<Resize: View>(desktop: Bool, phoneLandscape: Bool, @ViewBuilder resize: () -> Resize) -> some View {
        if desktop {
        HStack(spacing: 8) {
            Text("\(time(playheadMs)) / \(time(project.durationMs))").font(isPhone ? .system(size: 10, design: .monospaced) : .system(.caption, design: .monospaced)).lineLimit(1).accessibilityIdentifier("playheadTime")
            if phoneLandscape { resize().frame(minWidth: 40, maxWidth: .infinity) }
            else {
                SourceSeekSlider(value: Double(playheadMs), range: Double(bounds.startMs)...Double(max(bounds.startMs + 1, bounds.endMs))) { seek(Int64($0)) }
                    .frame(minWidth: 40, maxWidth: .infinity).frame(height: 36).accessibilityLabel("Seek source video").accessibilityIdentifier("sourceSeek")
            }
            Picker("Speed", selection: Binding(get: { draft.playbackRate }, set: { value in
                edit { $0.playbackRate = value }; if playing { player.rate = value }
            })) { ForEach([Float(1), 2, 4, 8], id: \.self) { Text("\(Int($0))×").tag($0) } }
                .pickerStyle(.menu).font(isPhone ? .system(size: 11) : .body).controlSize(isPhone ? .mini : .regular).accessibilityIdentifier("playbackRate")
            Button(playing ? "Pause" : "Play", action: togglePlay)
                .buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white)).disabled(player.currentItem == nil).accessibilityIdentifier("playPause")
        }
        } else {
            VStack(spacing: 6) {
                HStack(spacing: 8) {
                    Text("\(time(playheadMs)) / \(time(project.durationMs))").font(.system(.caption, design: .monospaced)).lineLimit(1).accessibilityIdentifier("playheadTime")
                    SourceSeekSlider(value: Double(playheadMs), range: Double(bounds.startMs)...Double(max(bounds.startMs + 1, bounds.endMs))) { seek(Int64($0)) }
                        .frame(minWidth: 40, maxWidth: .infinity).frame(height: 36).accessibilityLabel("Seek source video").accessibilityIdentifier("sourceSeek")
                }
                HStack(spacing: 8) {
                    Menu {
                        ForEach([Float(1), 2, 4, 8], id: \.self) { rate in
                            Button("\(Int(rate))x") {
                                edit { $0.playbackRate = rate }; if playing { player.rate = rate }
                            }
                        }
                    } label: { Text("Speed \u{00B7} \(Int(draft.playbackRate))x") }
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink, expand: true, minimumHeight: 48))
                        .accessibilityIdentifier("playbackRate")
                    Button(playing ? "Pause" : "Play", action: togglePlay)
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white, expand: true, minimumHeight: 48))
                        .disabled(player.currentItem == nil).accessibilityIdentifier("playPause")
                }
            }
        }
    }
    private func gameTimeline(twoRows: Bool, trackHeight: CGFloat = 60) -> some View {
        let midpoint = bounds.startMs + (bounds.endMs - bounds.startMs) / 2
        return VStack(alignment: .leading, spacing: 8) {
            sectionHeading("GAME TIMELINE")
            if twoRows {
                timeline(range: .init(startMs: bounds.startMs, endMs: midpoint), identifier: "gameTimelineFirst", trackHeight: trackHeight).guidedTourTarget("editor-overview")
                timeline(range: .init(startMs: midpoint, endMs: bounds.endMs), identifier: "gameTimelineSecond", trackHeight: trackHeight)
            } else { timeline(range: bounds, identifier: "gameTimeline", trackHeight: trackHeight).guidedTourTarget("editor-overview") }
            TimelineLegend().accessibilityIdentifier("timelineLegend")
        }
    }
    private func timeline(range: TimeRange, identifier: String, trackHeight: CGFloat = 60) -> some View {
        VStack(spacing: 3) {
            TimelineTrack(cuts: draft.cuts, ignored: draft.ignoredIntervals, intervals: intervals,
                reviewIds: pendingConfidenceIds, serves: score.enabled ? visibleScore.serveMarkers : [], switches: score.enabled ? visibleScore.sideSwitchMarkers : [],
                suggestions: suggestions, appliedSuggestionIds: presentation.appliedSuggestionIds,
                removedSuggestionIds: presentation.userRemovedSuggestionIds, removedCutIds: presentation.userRemovedCleanupCutIds,
                keptIds: keptIds, selectedCutIds: currentGroup?.cutIds ?? [], selectedSuggestionId: selectedSuggestionId, selectedServeId: currentServe?.id,
                range: range, playheadMs: playheadMs, onSeek: seek, onSelectCleanup: selectCleanup)
                .frame(height: trackHeight).accessibilityIdentifier(identifier)
            HStack { Text(time(range.startMs)); Spacer(); Text(time(range.endMs)) }.font(.system(.caption2, design: .monospaced))
        }
    }
    private func sectionHeading(_ title: String, color: Color = EditorPalette.orange) -> some View {
        Text(title).font(.system(size: isPhone ? 10 : 11, weight: .bold)).foregroundStyle(color)
    }
    private func currentClipControls(desktop: Bool) -> some View {
        VStack(alignment: .leading, spacing: isPhone ? 6 : 10) {
            HStack { sectionHeading("CURRENT RALLY", color: EditorPalette.green); Spacer(); Text(currentCut?.id ?? "—").font(.system(size: 11, weight: .bold, design: .monospaced)).foregroundStyle(EditorPalette.orange) }
            HStack(spacing: isPhone ? 4 : 8) {
                if isPhone && desktop {
                    Text(currentCut.map { "\(time($0.keepStartMs, precise: true))–\(time($0.keepEndMs, precise: true))" } ?? "—")
                        .font(.system(size: 10, design: .monospaced)).lineLimit(1).accessibilityIdentifier("currentRallyRange")
                    Spacer(minLength: 0)
                }
                Button(desktop ? "‹" : "Previous") { navigateRally(-1) }.disabled(currentGroupIndex <= 0).accessibilityLabel("Previous rally").accessibilityIdentifier("previousRally")
                if !(isPhone && desktop) {
                    Spacer()
                    Text("\(max(0, currentGroupIndex + 1))/\(rallyGroups.count)").font(.caption.monospaced())
                    Spacer()
                }
                Button(desktop ? "›" : "Next") { navigateRally(1) }.disabled(currentGroupIndex < 0 || currentGroupIndex >= rallyGroups.count - 1).accessibilityLabel("Next rally").accessibilityIdentifier("nextRally")
            }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).font(isPhone ? .system(size: 11) : .caption)
            if let cleanup = selectedCleanup { cleanupControls(cleanup) }
            if let cut = currentCut {
                if desktop && !isPhone {
                    Text("\(time(cut.keepStartMs, precise: true))–\(time(cut.keepEndMs, precise: true))")
                        .font(.system(.caption, design: .monospaced)).accessibilityIdentifier("currentRallyRange")
                } else if !desktop {
                    Text("\(time(cut.coreStartMs, precise: true)) – \(time(cut.coreEndMs, precise: true)) · \(Int(cut.confidence * 100))%")
                        .font(.system(.caption, design: .monospaced)).accessibilityIdentifier("currentRallyRange")
                    if let agreement = cut.agreement { Text(agreement.replacingOccurrences(of: "-", with: " ")).font(isPhone ? .system(size: 11) : .caption) }
                }
                if !keptIds.contains(cut.id) { Text(cut.included ? "Removed by cleanup or ignored time" : "Removed by you").font(.caption.bold()).foregroundStyle(EditorPalette.danger) }
                if selectedCleanup == nil { HStack(spacing: 6) {
                    Button(keptIds.contains(cut.id) ? "✓ Keep" : "Keep") { include(cut, true) }.frame(maxWidth: .infinity)
                        .buttonStyle(EditorActionButtonStyle(background: keptIds.contains(cut.id) ? EditorPalette.kept : .white, foreground: keptIds.contains(cut.id) ? EditorPalette.green : EditorPalette.ink, expand: true, border: keptIds.contains(cut.id) ? EditorPalette.green : EditorPalette.rail)).accessibilityIdentifier("keepRally")
                        .accessibilityValue(!cut.included ? "removed" : keptIds.contains(cut.id) ? "included" : "automatically excluded")
                    Button(!cut.included ? "✓ Remove" : "Remove") { include(cut, false) }.frame(maxWidth: .infinity)
                        .buttonStyle(EditorActionButtonStyle(background: !cut.included ? EditorPalette.removed : .white, foreground: !cut.included ? .white : EditorPalette.suppression, expand: true, border: !cut.included ? EditorPalette.suppression : EditorPalette.rail)).accessibilityIdentifier("removeRally")
                } }
                focusedTimeline(cut, desktop: desktop)
                HStack {
                    Button(desktop ? "Set start" : "Set rally start here") { trim(cut, start: playheadMs, end: cut.coreEndMs) }.disabled(playheadMs > cut.coreEndMs - 100).accessibilityIdentifier("setRallyStart")
                    Button(desktop ? "Set end" : "Set rally end here") { trim(cut, start: cut.coreStartMs, end: playheadMs) }.disabled(playheadMs < cut.coreStartMs + 100).accessibilityIdentifier("setRallyEnd")
                }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink, expand: true))
                if !desktop {
                HStack {
                    Button("Start −0.1") { trim(cut, start: cut.coreStartMs - 100, end: cut.coreEndMs) }.accessibilityIdentifier("nudgeStartEarlier")
                    Button("Start +0.1") { trim(cut, start: cut.coreStartMs + 100, end: cut.coreEndMs) }.accessibilityIdentifier("nudgeStartLater")
                }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).font(isPhone ? .system(size: 11) : .caption)
                HStack {
                    Button("End −0.1") { trim(cut, start: cut.coreStartMs, end: cut.coreEndMs - 100) }.accessibilityIdentifier("nudgeEndEarlier")
                    Button("End +0.1") { trim(cut, start: cut.coreStartMs, end: cut.coreEndMs + 100) }.accessibilityIdentifier("nudgeEndLater")
                }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).font(isPhone ? .system(size: 11) : .caption)
                }
                Button("Split at playhead") {
                    if let member = currentGroup?.cuts.first(where: { playheadMs >= $0.coreStartMs + 100 && playheadMs <= $0.coreEndMs - 100 }),
                       var split = EditorMath.splitCut(draft, cutId: member.id, positionMs: playheadMs, bounds: bounds) {
                        split.reviewedCutIds.remove(member.id); replaceDraft(split)
                    }
                }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white, expand: true)).disabled(!(currentGroup?.cuts.contains { playheadMs >= $0.coreStartMs + 100 && playheadMs <= $0.coreEndMs - 100 } ?? false)).accessibilityIdentifier("splitRally")
                if !desktop {
                Button("Preview rally") { previewRally(cut) }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).disabled(player.currentItem == nil).accessibilityIdentifier("previewRally")
                if cut.origin == .inferred {
                    Button("Reset extra time") { replaceDraft(EditorMath.resetExtraTime(draft, cutIds: currentGroup?.cutIds ?? [cut.id], bounds: bounds)) }
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).accessibilityIdentifier("resetExtraTime")
                } else {
                    Button("Delete manual rally", role: .destructive) { replaceDraft(EditorMath.deleteManualCuts(draft, cutIds: currentGroup?.cutIds ?? [cut.id])) }
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).accessibilityIdentifier("deleteManualRally")
                }
                }
            } else if selectedCleanup == nil { Text("No rallies. Add a clip below.").font(.callout) }
        }
    }
    private func cleanupControls(_ suggestion: SuppressionRegion) -> some View {
        let removed = EditorMath.effectiveDecision(draft, suggestion: suggestion) == "suppress"
        let explicit = draft.suppressionDecisionOverrides[suggestion.logicalId] != nil
        return VStack(alignment: .leading, spacing: 8) {
            Text("AUTOMATIC CLEANUP").font(.caption.bold()).foregroundStyle(EditorPalette.suppression)
                .accessibilityIdentifier("cleanupSelection").accessibilityValue(suggestion.id)
            Text(!removed ? "Kept · \(time(suggestion.startMs))–\(time(suggestion.endMs)) stays in the video." : explicit ?
                 "Removed by you · this cleanup section will be left out." :
                 "Suggested removal · VolleySplice will leave out this likely non-play section unless you keep it.").font(isPhone ? .system(size: 11) : .caption)
            HStack(spacing: 6) {
                Button(!removed ? "✓ Keep" : "Keep") { decideCleanup(suggestion, keep: true) }
                    .buttonStyle(EditorActionButtonStyle(background: !removed ? EditorPalette.kept : .white, foreground: !removed ? EditorPalette.green : EditorPalette.ink, expand: true, border: !removed ? EditorPalette.green : EditorPalette.rail))
                    .accessibilityIdentifier("keepRally").accessibilityValue(removed ? (explicit ? "removed" : "automatically excluded") : "included")
                Button(removed && !explicit ? "Remove · auto" : removed ? "✓ Remove" : "Remove") { decideCleanup(suggestion, keep: false) }
                    .buttonStyle(EditorActionButtonStyle(background: removed ? (explicit ? EditorPalette.removed : Color(red: 1, green: 229 / 255.0, blue: 231 / 255.0)) : .white,
                                                        foreground: removed && explicit ? .white : EditorPalette.suppression, expand: true, border: removed ? EditorPalette.suppression : EditorPalette.rail))
                    .accessibilityIdentifier("removeRally")
            }
        }
    }
    private func focusedTimeline(_ cut: EditableCut, desktop: Bool) -> some View {
        let window = trimWindow ?? bounds
        return VStack(alignment: .leading, spacing: 4) {
            HStack { Text(time(window.startMs)); Spacer(); Text(time((window.startMs + window.endMs) / 2)); Spacer(); Text(time(window.endMs)) }
                .font(.system(size: 10, design: .monospaced)).foregroundStyle(EditorPalette.muted)
            GeometryReader { geometry in
                TimelineTrack(cuts: [cut], ignored: draft.ignoredIntervals, intervals: intervals,
                    reviewIds: Set(reviewCuts.map(\.id)), serves: score.enabled ? visibleScore.serveMarkers : [], switches: score.enabled ? visibleScore.sideSwitchMarkers : [],
                    suggestions: suggestions, appliedSuggestionIds: presentation.appliedSuggestionIds,
                    removedSuggestionIds: presentation.userRemovedSuggestionIds, removedCutIds: presentation.userRemovedCleanupCutIds,
                    keptIds: keptIds, selectedCutIds: currentGroup?.cutIds ?? [], selectedSuggestionId: selectedSuggestionId, selectedServeId: currentServe?.id,
                    range: window, playheadMs: playheadMs, onSeek: seek, onSelectCleanup: selectCleanup)
                    .accessibilityIdentifier("focusedTimeline").guidedTourTarget("editor-focus")
                trimHandle(cut, isStart: true, width: geometry.size.width, window: window)
                trimHandle(cut, isStart: false, width: geometry.size.width, window: window)
                if cut.origin == .inferred {
                    paddingHandle(cut, isStart: true, width: geometry.size.width, window: window)
                    paddingHandle(cut, isStart: false, width: geometry.size.width, window: window)
                }
            }.frame(height: isPhone ? 48 : 68).coordinateSpace(name: "trimTrack")
            if !desktop { Text(cut.origin == .inferred ? "Orange outer handles adjust padding · green inner handles adjust rally length." : "Drag the green handles to adjust this rally's exact start and end.").font(isPhone ? .system(size: 10) : .caption2).foregroundStyle(EditorPalette.muted) }
        }
    }
    private func trimHandle(_ cut: EditableCut, isStart: Bool, width: Double, window: TimeRange) -> some View {
        let position = isStart ? cut.coreStartMs : cut.coreEndMs
        let fraction = Double(position - window.startMs) / Double(max(1, window.endMs - window.startMs))
        return RoundedRectangle(cornerRadius: 3).fill(EditorPalette.ink).frame(width: 16, height: isPhone ? 24 : 34)
            .overlay { Text(isStart ? "‹" : "›").foregroundStyle(.white).font(.headline) }
            .position(x: min(width - 8, max(8, fraction * width)), y: isPhone ? 18 : 24)
            .gesture(DragGesture(coordinateSpace: .named("trimTrack")).onChanged { event in

                let value = window.startMs + Int64(min(1, max(0, event.location.x / max(1, width))) * Double(window.endMs - window.startMs))
                trim(cut, start: isStart ? value : cut.coreStartMs, end: isStart ? cut.coreEndMs : value)
            }).accessibilityLabel(isStart ? "Rally core start" : "Rally core end").accessibilityIdentifier(isStart ? "trimStartHandle" : "trimEndHandle")
            .accessibilityValue(String(format: "%.3f seconds", Double(position) / 1_000))
    }
    private func paddingHandle(_ cut: EditableCut, isStart: Bool, width: Double, window: TimeRange) -> some View {
        let position = isStart ? cut.keepStartMs : cut.keepEndMs
        let fraction = Double(position - window.startMs) / Double(max(1, window.endMs - window.startMs))
        return RoundedRectangle(cornerRadius: 3).fill(EditorPalette.warning).frame(width: 18, height: isPhone ? 14 : 20)
            .overlay { Text(isStart ? "‹" : "›").font(.headline).foregroundStyle(EditorPalette.ink) }
            .position(x: min(width - 9, max(9, fraction * width)), y: isPhone ? 39 : 54)
            .gesture(DragGesture(coordinateSpace: .named("trimTrack")).onChanged { event in

                let value = window.startMs + Int64(min(1, max(0, event.location.x / max(1, width))) * Double(window.endMs - window.startMs))
                setPaddingEdge(isStart: isStart, value: value)
            }).accessibilityLabel(isStart ? "Extra time start" : "Extra time end").accessibilityIdentifier(isStart ? "paddingStartHandle" : "paddingEndHandle")
            .accessibilityValue(String(format: "%.3f seconds", Double(position) / 1_000))
            .accessibilityAdjustableAction { direction in
                setPaddingEdge(isStart: isStart, value: position + (direction == .increment ? 100 : -100))
            }
    }
    private var rangeTools: some View {
        VStack(alignment: .leading, spacing: isPhone ? 6 : 10) {
            sectionHeading("FIX MISSED OR EXTRA FOOTAGE")
            HStack(spacing: 8) {
            Button(draft.pendingManualStartMs == nil ? "Add rally start" : "Add rally end") {
                if let start = draft.pendingManualStartMs { replaceDraft(EditorMath.addManual(draft, startMs: start, endMs: playheadMs, bounds: bounds)) }
                else { edit { $0.pendingManualStartMs = playheadMs; $0.pendingIgnoreStartMs = nil } }
            }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white, expand: true)).accessibilityIdentifier("manualRange")
            Button(draft.pendingIgnoreStartMs == nil ? "Leave out start" : "Leave out end") {
                if let start = draft.pendingIgnoreStartMs { replaceDraft(EditorMath.addIgnored(draft, startMs: start, endMs: playheadMs, durationMs: project.durationMs, reason: "non-game-content")) }
                else { edit { $0.pendingIgnoreStartMs = playheadMs; $0.pendingManualStartMs = nil } }
            }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.orange, foreground: .white, expand: true)).accessibilityIdentifier("ignoredRange")
            }
            if let start = draft.pendingManualStartMs {
                HStack { Text("From \(time(start))").font(.caption.monospaced()); Button("Cancel") { edit { $0.pendingManualStartMs = nil } } }
            }
            if let start = draft.pendingIgnoreStartMs {
                HStack { Text("From \(time(start))").font(.caption.monospaced()); Button("Cancel") { edit { $0.pendingIgnoreStartMs = nil } } }
            }
            ForEach(draft.ignoredIntervals.filter { $0.startMs < bounds.endMs && $0.endMs > bounds.startMs }) { ignored in
                HStack {
                    Button("\(time(ignored.startMs))–\(time(ignored.endMs))") { seek(ignored.startMs) }.font(isPhone ? .system(size: 11) : .caption).multilineTextAlignment(.leading)
                    Spacer(minLength: 0)
                    if ignored.reason != "outside-game-window" {
                        Button(role: .destructive) { edit { $0.ignoredIntervals.removeAll { $0.id == ignored.id } } } label: { Image(systemName: "trash") }
                            .accessibilityLabel("Remove ignored range \(ignored.id)")
                    }
                }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink))
            }
        }
    }
    private var scoreVisibilityPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Score visibility", selection: $fadeScoreOverlay) {
                Text("Always visible").tag(false)
                Text("Fade in and out").tag(true)
            }.pickerStyle(.menu).accessibilityIdentifier("scoreVisibility")
            Text("Fading follows the point timeline. The timeline always fades in and out.")
                .font(.caption).foregroundStyle(EditorPalette.muted)
        }
    }
    private func saveVideoWithOverlaySettings() {
        edit { $0.fadeScoreOverlay = fadeScoreOverlay }
        onSave()
        onExportVideo()
    }
    private var playbackOptions: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeading("PLAYBACK")
            Toggle("Play only final cut", isOn: Binding(get: { draft.finalPreviewEnabled }, set: { value in edit { $0.finalPreviewEnabled = value }; lastPreviewSeek = -1 }))
                .accessibilityIdentifier("finalCutPreview")
            Divider().padding(.vertical, 6)
            sectionHeading("VIDEO OVERLAY")
            Toggle("Render scores", isOn: Binding(get: { draft.renderScoreOverlay }, set: { value in edit { $0.renderScoreOverlay = value; if !value { $0.renderScoreTimeline = false } } }))
                .accessibilityIdentifier("renderScores")
            if draft.renderScoreOverlay && score.enabled { scoreVisibilityPicker }
            Toggle("Render point timeline", isOn: Binding(get: { draft.renderScoreTimeline }, set: { value in edit { $0.renderScoreTimeline = value } }))
                .disabled(!draft.renderScoreOverlay).accessibilityIdentifier("renderPointTimeline")
        }.font(.system(size: 11, weight: .semibold))
    }
    private func clipRegister(maximumHeight: Double?, desktop: Bool = false) -> some View {
        let selectedId = currentCut?.id
        let reviewIds = Set(reviewCuts.map(\.id))
        return VStack(alignment: .leading, spacing: 8) {
            HStack { sectionHeading("CLIP REGISTER"); Spacer(); Text("\(keptRallyCount) / \(rallyGroups.count) included").font(.system(size: 10, design: .monospaced)).foregroundStyle(EditorPalette.muted) }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 3) {
                        ForEach(rallyGroups, id: \.id) { group in
                            let cut = group.asEditableCut()
                            let included = !group.cutIds.isDisjoint(with: keptIds)
                            if desktop {
                                let checked = group.cutIds.isSubset(of: draft.reviewedCutIds)
                                let status = !included ? "Left out" : checked ? "Checked" : reviewIds.contains(cut.id) ? "Needs review" : "Ready"
                                HStack(spacing: 4) {
                                    Button { seek(cut.keepStartMs) } label: {
                                        HStack(spacing: 7) {
                                            Text(cut.id).font(.system(size: 10, weight: .bold, design: .monospaced))
                                            VStack(alignment: .leading, spacing: 3) {
                                                Text("\(time(cut.keepStartMs, precise: true))–\(time(cut.keepEndMs, precise: true))")
                                                    .font(.system(size: 10, design: .monospaced)).lineLimit(1)
                                                Text(status).font(.system(size: 9))
                                                    .foregroundStyle(reviewIds.contains(cut.id) && !checked ? EditorPalette.warning : EditorPalette.muted)
                                            }
                                            Spacer(minLength: 0)
                                        }.frame(maxWidth: .infinity, minHeight: 44).contentShape(Rectangle())
                                    }.buttonStyle(.plain).accessibilityIdentifier("clip-\(cut.id)")
                                        .accessibilityValue(status)
                                    Button {
                                        // Android ledger toggles inclusion only; it does not
                                        // resolve an independent cleanup or review decision.
                                        edit { state in
                                            for index in state.cuts.indices where group.cutIds.contains(state.cuts[index].id) {
                                                state.cuts[index].included = !included
                                            }
                                        }
                                    } label: {
                                        Text(included ? "On" : "Off").font(.system(size: 10))
                                            .foregroundStyle(included ? EditorPalette.green : EditorPalette.danger)
                                            .frame(minWidth: 36, minHeight: 44).contentShape(Rectangle())
                                    }.buttonStyle(.plain)
                                        .accessibilityLabel(included ? "On" : "Off")
                                        .accessibilityIdentifier("toggleClip-\(cut.id)")
                                }.padding(.horizontal, 7)
                                    .background(selectedId == cut.id ? EditorPalette.acid.opacity(0.5) : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                                    .id(cut.id)
                                Divider()
                            } else {
                                Button { seek(cut.coreStartMs) } label: {
                                    HStack {
                                        Image(systemName: included ? "checkmark.square.fill" : "xmark.square")
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(cut.id).bold(); Text("\(time(cut.coreStartMs))–\(time(cut.coreEndMs))").font(isPhone ? .system(size: 10) : .caption2)
                                        }
                                        Spacer(minLength: 2)
                                        if reviewIds.contains(cut.id) { Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(EditorPalette.warning) }
                                    }.font(.system(.caption, design: .monospaced)).padding(8).frame(maxWidth: .infinity, minHeight: 44)
                                        .background(selectedId == cut.id ? EditorPalette.acid.opacity(0.45) : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                                        .foregroundStyle(included ? EditorPalette.ink : EditorPalette.danger)
                                }.buttonStyle(.plain).id(cut.id).accessibilityIdentifier("clip-\(cut.id)")
                            }
                        }
                    }
                }.frame(maxHeight: maximumHeight ?? .infinity)
                    // Offscreen rows must not accept taps in the score controls
                    // above this nested scroll viewport after automatic scrolling.
                    .clipped().contentShape(Rectangle())
                    .onChange(of: currentCut?.id) { _, id in if let id { withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(id, anchor: .center) } } }
            }
        }
    }

    private func scoreSection(desktop: Bool) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle("SCOREBOARD · BETA", isOn: Binding(get: { score.enabled }, set: { value in
                editScore { $0.enabled = value }
                if value && scorePreparationNeeded {
                    if player.currentItem == nil { message = "Re-link the source video to prepare score tracking" }
                    else { onPrepareScore?() }
                }
            }))
                .font(.system(size: 11, weight: .bold)).foregroundStyle(EditorPalette.orange).accessibilityIdentifier("scoreTracking").guidedTourTarget("editor-score-toggle")
            if score.enabled {
                if let error = project.feedback?["scorePreparationError"]?.string {
                    VStack(alignment: .leading, spacing: 4) {
                        sectionHeading("SCORE MARKERS NEED ATTENTION")
                        Text(error).font(.system(size: 12)).foregroundStyle(EditorPalette.danger)
                        Text("Turn score tracking off and on to try again.").font(.system(size: 12)).foregroundStyle(EditorPalette.muted)
                    }.accessibilityIdentifier("scorePreparationError")
                }
                markerList
                pointTimeline
                HStack(alignment: .top) { teamField(first: true); teamField(first: false) }
                if let serve = currentServe { selectedServe(serve, desktop: desktop).id("selected-serve-controls") }
                HStack {
                    Button("+ Add serve") { editScore { $0 = ScoreReducer.addServe($0, timestampMs: playheadMs, side: .review) } }
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink, expand: true)).accessibilityIdentifier("addServe")
                    Button("⇄ Add team switch") { editScore { $0 = ScoreReducer.addSideSwitch($0, timestampMs: playheadMs) } }
                        .buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white, expand: true)).accessibilityIdentifier("addSideSwitch")
                }.font(isPhone ? .system(size: 11) : .caption)
                if derivedScore.reviewPointCount > 0 { Text("\(derivedScore.reviewPointCount) points need review").font(.caption.bold()).foregroundStyle(EditorPalette.warning) }
            }
        }
    }
    private func teamField(first: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            EditableTeamName(name: first ? score.team1Name : score.team2Name, label: first ? "Team 1" : "Team 2", identifier: first ? "team1Name" : "team2Name") { value in
                editScore { if first { $0.team1Name = value } else { $0.team2Name = value } }
            }
            Text("\(first ? derivedScore.team1Score : derivedScore.team2Score)")
                .font(.system(size: isPhone ? 22 : 27, weight: .bold, design: .monospaced)).foregroundStyle(first ? EditorPalette.green : EditorPalette.orange)
                .frame(maxWidth: .infinity, minHeight: isPhone ? 40 : 60)
                .overlay { RoundedRectangle(cornerRadius: 4).stroke(EditorPalette.rail) }
                .accessibilityIdentifier(first ? "team1Score" : "team2Score")
        }.frame(maxWidth: .infinity)
    }
    private func selectedServe(_ serve: ServeMarker, desktop: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeading("SELECTED SERVE")
            Text("\(time(serve.timestampMs, precise: true)) · \(serve.side.rawValue)").font(.system(size: isPhone ? 13 : 18, weight: .semibold, design: .monospaced))
            if desktop {
                HStack(spacing: 4) {
                    ForEach([ServingSide.near, .far], id: \.rawValue) { side in
                        Button {
                            editScore { state in
                                if let index = state.serveMarkers.firstIndex(where: { $0.id == serve.id }) { state.serveMarkers[index].side = side }
                            }
                        } label: {
                            Text(side == .near ? "Near" : "Far").font(.system(size: 10))
                                .frame(maxWidth: .infinity, minHeight: 36)
                                .background(serve.side == side ? EditorPalette.acid.opacity(0.5) : .clear, in: RoundedRectangle(cornerRadius: 5))
                                .overlay { RoundedRectangle(cornerRadius: 5).stroke(EditorPalette.rail) }
                                .contentShape(Rectangle())
                        }.buttonStyle(.plain)
                            .accessibilityAddTraits(serve.side == side ? .isSelected : [])
                    }
                    Button {
                        editScore { state in
                            if let index = state.serveMarkers.firstIndex(where: { $0.id == serve.id }) { state.serveMarkers[index].ignorePreviousPoint.toggle() }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: serve.ignorePreviousPoint ? "checkmark.square.fill" : "square").font(.system(size: 16))
                            Text("Ignore / replay previous point").font(.system(size: 9)).fixedSize(horizontal: false, vertical: true)
                        }.frame(maxWidth: .infinity, minHeight: 44).contentShape(Rectangle())
                    }.buttonStyle(.plain).disabled(visibleScore.serveMarkers.first?.id == serve.id)
                        .accessibilityLabel("Ignore / replay previous point").accessibilityValue(serve.ignorePreviousPoint ? "1" : "0")
                        .accessibilityIdentifier("ignorePreviousPoint")
                }.accessibilityElement(children: .contain).accessibilityIdentifier("selectedServeSide")
                    .accessibilityValue(serve.side.rawValue)
            } else {
                HStack(spacing: 4) {
                    ForEach([ServingSide.near, .far], id: \.rawValue) { side in
                        Button {
                            editScore { state in
                                if let index = state.serveMarkers.firstIndex(where: { $0.id == serve.id }) { state.serveMarkers[index].side = side }
                            }
                        } label: {
                            Text(side == .near ? "Near" : "Far").font(.system(size: 12))
                                .frame(maxWidth: .infinity, minHeight: 36)
                                .background(serve.side == side ? EditorPalette.acid.opacity(0.5) : .clear, in: RoundedRectangle(cornerRadius: 5))
                                .overlay { RoundedRectangle(cornerRadius: 5).stroke(EditorPalette.rail) }
                                .contentShape(Rectangle())
                        }.buttonStyle(.plain)
                            .accessibilityAddTraits(serve.side == side ? .isSelected : [])
                    }
                }.accessibilityElement(children: .contain).accessibilityIdentifier("selectedServeSide")
                    .accessibilityValue(serve.side.rawValue)
                Toggle("Ignore / replay previous point", isOn: Binding(get: { serve.ignorePreviousPoint }, set: { value in editScore { state in
                    if let index = state.serveMarkers.firstIndex(where: { $0.id == serve.id }) { state.serveMarkers[index].ignorePreviousPoint = value }
                } })).font(isPhone ? .system(size: 11) : .caption).disabled(visibleScore.serveMarkers.first?.id == serve.id).accessibilityIdentifier("ignorePreviousPoint")
                Button("Remove serve", role: .destructive) { editScore { $0 = ScoreReducer.removeServe($0, markerId: serve.id) } }
                    .buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).accessibilityIdentifier("removeServe")
            }
        }
    }
    private var settingsPanel: some View {
        NavigationStack {
            Form {
                Section { InterfaceScaleControl() }
                Section("Padding and gaps") {
                    settingSlider("Extra time before each clip", value: draft.beforePaddingMs, identifier: "beforePadding") { before in
                        replaceDraft(EditorMath.applyPadding(draft, beforeMs: before, afterMs: draft.afterPaddingMs, durationMs: project.durationMs, gameWindow: bounds))
                    }
                    settingSlider("Extra time after each clip", value: draft.afterPaddingMs, identifier: "afterPadding") { after in
                        replaceDraft(EditorMath.applyPadding(draft, beforeMs: draft.beforePaddingMs, afterMs: after, durationMs: project.durationMs, gameWindow: bounds))
                    }
                    settingSlider("Keep short breaks under", value: draft.joinGapMs, identifier: "joinGap") { value in edit { $0.joinGapMs = value } }
                }
                Section("Review") {
                    HStack { Text("Clips to check"); Spacer(); Text("\(Int(draft.confidenceReviewThreshold * 100))%") }
                    Slider(value: Binding(get: { Double(draft.confidenceReviewThreshold) }, set: { value in edit { $0.confidenceReviewThreshold = Float(value) } }), in: 0...1, step: 0.05)
                        .accessibilityIdentifier("reviewThreshold")
                }
                Section("Automatic cleanup") {
                    Picker("Policy", selection: Binding(get: { draft.selectedSuppressionPolicy }, set: { value in edit { $0.selectedSuppressionPolicy = value; if value != "none" { $0.suppressionInitialBehavior = "disable-initially" } } })) {
                        ForEach(["none", "conservative", "balanced", "aggressive"], id: \.self) { Text(["none": "Off", "conservative": "Light", "balanced": "Recommended", "aggressive": "Strong"][$0] ?? $0).tag($0) }
                    }.accessibilityIdentifier("suppressionPolicy")

                }
                Section("Restore") { Button("Restore original model ranges", role: .destructive) { settings = false; resetConfirmation = true }.accessibilityIdentifier("resetRanges") }
            }.scrollContentBackground(.hidden).background(EditorPalette.paper).navigationTitle("Editor settings")
                .navigationBarTitleDisplayMode(.inline).tint(EditorPalette.green)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { settings = false }.accessibilityIdentifier("settingsDone") } }
        }
    }
    private func settingSlider(_ label: String, value: Int64, identifier: String, update: @escaping (Int64) -> Void) -> some View {
        VStack(alignment: .leading) {
            HStack {
                Text(label); Spacer()
                EditorTimeSetting(valueMs: value, label: label, identifier: identifier) { value in update(value) }
            }
            Slider(value: Binding(get: { Double(value) }, set: { update(Int64($0)) }), in: 0...10_000, step: 100).accessibilityLabel(label).accessibilityIdentifier(identifier)
        }
    }
    private func exportWorkspace(desktop: Bool) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Export").font(.title2.bold())
                Text("\(intervals.count) joined clips · \(time(EditorMath.totalFinalMs(intervals)))").font(.system(.body, design: .monospaced))
                Divider()
                let layout = desktop ? AnyLayout(HStackLayout(alignment: .top, spacing: 24)) : AnyLayout(VStackLayout(alignment: .leading, spacing: 24))
                layout {
                    videoExportPanel.frame(maxWidth: .infinity, alignment: .leading)
                    chapterExport.frame(maxWidth: .infinity, alignment: .leading)
                    projectExportPanel.frame(maxWidth: .infinity, alignment: .leading)
                }
            }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    private func queueCounter(_ label: String, count: Int, identifier: String, condensed: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 1) {
                Text("\(count)").font(.system(size: 10, weight: .bold, design: .monospaced))
                Text(label.replacingOccurrences(of: " ", with: "\n")).font(.system(size: condensed ? 7 : 8)).lineLimit(2).multilineTextAlignment(.center)
            }.frame(width: condensed ? 68 : 92, height: 44)
                .foregroundStyle(count > 0 ? EditorPalette.ink : EditorPalette.muted)
                .background(count > 0 ? EditorPalette.acid : EditorPalette.rail.opacity(0.4), in: RoundedRectangle(cornerRadius: 5))
        }.buttonStyle(.plain)
            .disabled(count == 0).accessibilityLabel("\(label) \(count)").accessibilityIdentifier(identifier)
    }
    private var videoExportPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
                sectionHeading("FINAL VIDEO")
                Toggle("Render scores", isOn: Binding(get: { draft.renderScoreOverlay }, set: { value in edit { $0.renderScoreOverlay = value } })).disabled(!score.enabled)
                    .accessibilityIdentifier("renderScoreOverlay")
                if draft.renderScoreOverlay && score.enabled { scoreVisibilityPicker }
                Button("Save final video") { saveVideoWithOverlaySettings() }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.green, foreground: .white)).disabled(intervals.isEmpty || player.currentItem == nil).accessibilityIdentifier("exportVideo")
                if player.currentItem == nil { Text("Reconnect the source video to export MP4.").font(isPhone ? .system(size: 11) : .caption) }
        }
    }
    private var projectExportPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
                Text("Project file").font(.headline)
                Text("Features, model predictions, edits, ignored time and score markers. The source video is kept separate.").font(.callout)
                Button("Save project") { onSave(); onExportProject() }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).accessibilityIdentifier("exportProject")
        }
    }
    private var chapters: [YouTubeChapter] {
        YouTubeChapters.build(intervals: intervals, cuts: draft.cuts, scoreTracking: score.enabled ? visibleScore : nil, options: chapterOptions)
    }
    private var chapterOptions: YouTubeChapterOptions {
        draft.chapterOptions ?? YouTubeChapters.defaultOptions(hasScoreTracking: score.enabled, hasSideSwitches: !score.sideSwitchMarkers.isEmpty)
    }
    private func chapterBinding(_ keyPath: WritableKeyPath<YouTubeChapterOptions, Bool>) -> Binding<Bool> {
        Binding(get: { chapterOptions[keyPath: keyPath] }, set: { value in
            var options = chapterOptions; options[keyPath: keyPath] = value
            edit { $0.chapterOptions = options }
        })
    }
    private var chapterExport: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("YouTube chapters").font(.headline)
            Text("\(chapters.count) chapters on the exported timeline").font(.caption.monospaced())
            Toggle("Rally numbers", isOn: chapterBinding(\.includeRallyNumber)).accessibilityIdentifier("chapterRallyNumbers")
            Toggle("Serve numbers", isOn: chapterBinding(\.includeServeNumber)).accessibilityIdentifier("chapterServeNumbers")
            Toggle("Scores", isOn: chapterBinding(\.includeScore)).disabled(!score.enabled).accessibilityIdentifier("chapterScores")
            Toggle("Serving team", isOn: chapterBinding(\.includeServingTeam)).disabled(!score.enabled).accessibilityIdentifier("chapterServingTeam")
            Toggle("Side switches", isOn: chapterBinding(\.includeSideSwitches)).disabled(score.sideSwitchMarkers.isEmpty).accessibilityIdentifier("chapterSideSwitches")
            Toggle("Include VolleySplice credit", isOn: chapterBinding(\.includeCredit)).accessibilityIdentifier("chapterCredit")
            Text("Edited with https://volleysplice.com").font(.caption).foregroundStyle(.secondary)
            ShareLink("Share chapters", item: YouTubeChapters.text(chapters, includeCredit: chapterOptions.includeCredit)).buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).disabled(chapters.isEmpty).accessibilityIdentifier("exportChapters")
            if !chapters.isEmpty {
                DisclosureGroup("Preview chapters") {
                    Text(YouTubeChapters.text(chapters, includeCredit: chapterOptions.includeCredit)).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                }
            }
        }
    }
    private var pointTimeline: some View {
        SidebarPointTimeline(score: derivedScore, tracking: visibleScore, onSeek: seek)
    }
    private var markerList: some View {
        SidebarMarkerList(tracking: visibleScore, effectiveTimestampMs: scoreTimestampMs, onSeek: seek) { id, isServe in
            editScore { $0 = isServe ? ScoreReducer.removeServe($0, markerId: id) : ScoreReducer.removeSideSwitch($0, markerId: id) }
        }
    }

    private func edit(_ operation: (inout EditorDraft) -> Void) { var value = draft; operation(&value); replaceDraft(value) }
    private func replaceDraft(_ value: EditorDraft) {
        var value = value; value.updatedAtMs = Int64(Date().timeIntervalSince1970 * 1_000)
        do { try value.validate(durationMs: project.durationMs); project.draft = value; onSave() }
        catch { message = error.localizedDescription }
    }
    private func refreshPresentation() {
        var input = draft
        // Save timestamps, transport and export checkboxes cannot affect review
        // materialization. Do not rebuild its caches when those controls change.
        input.updatedAtMs = 0; input.pendingManualStartMs = nil; input.pendingIgnoreStartMs = nil
        input.ignoreReason = ""; input.finalPreviewEnabled = false; input.playbackRate = 1
        input.renderScoreOverlay = false; input.renderScoreTimeline = false; input.fadeScoreOverlay = nil; input.chapterOptions = nil
        guard input != preparedInput else { return }
        preparedInput = input
        let updated = EditorPresentation(draft: draft, suggestions: allSuggestions, bounds: bounds, durationMs: project.durationMs)
        presentation = updated
        preparedOverlay = PreparedScoreOverlay(tracking: updated.visibleScore, rallyRanges: updated.scoreRallyRanges,
            mergedRanges: updated.intervals.map { .init(startMs: $0.startMs, endMs: $0.endMs) })
    }
    private func editScore(_ operation: (inout ScoreTracking) -> Void) {
        var value = score; operation(&value)
        do { let json = try value.jsonValue(); edit { $0.scoreTracking = json } } catch { message = error.localizedDescription }
    }
    private func include(_ cut: EditableCut, _ keep: Bool) {
        let members = rallyGroups.first { $0.cutIds.contains(cut.id) }?.cuts ?? [cut]
        var value = draft
        for member in members { value = EditorMath.setIncluded(value, cutId: member.id, included: keep) }
        value.reviewedCutIds.formUnion(members.map(\.id))
        replaceDraft(value)
    }
    private func decideCleanup(_ suggestion: SuppressionRegion, keep: Bool) {
        replaceDraft(EditorReview.decideCleanup(draft, suggestion: suggestion, keep: keep, reviewedCutIds: currentGroup?.cutIds ?? []))
    }
    private func trim(_ cut: EditableCut, start: Int64, end: Int64) {
        let members = rallyGroups.first { $0.cutIds.contains(cut.id) }?.cuts ?? [cut]
        var value = draft
        if start != cut.coreStartMs, let first = members.min(by: { $0.coreStartMs < $1.coreStartMs }) {
            value = EditorMath.setCoreStart(value, cutId: first.id, valueMs: start, minimumMs: bounds.startMs)
        }
        if end != cut.coreEndMs, let last = members.max(by: { $0.coreEndMs < $1.coreEndMs }) {
            value = EditorMath.setCoreEnd(value, cutId: last.id, valueMs: end, maximumMs: bounds.endMs)
        }
        replaceDraft(value)
    }
    private func setPaddingEdge(isStart: Bool, value: Int64) {
        guard let group = currentGroup else { return }
        let member = isStart ? group.cuts.min(by: { $0.keepStartMs < $1.keepStartMs }) : group.cuts.max(by: { $0.keepEndMs < $1.keepEndMs })
        guard let member else { return }
        replaceDraft(EditorMath.setKeepBoundary(draft, cutId: member.id, valueMs: value, isStart: isStart, bounds: bounds))
    }
    private func navigateRally(_ direction: Int) {
        let target = currentGroupIndex + direction
        guard rallyGroups.indices.contains(target) else { return }
        seek(rallyGroups[target].keepStartMs)
    }
    private func previewRally(_ cut: EditableCut) {
        edit { $0.finalPreviewEnabled = false }
        seek(cut.keepStartMs); previewEndMs = cut.keepEndMs
        player.playImmediately(atRate: draft.playbackRate); playing = true
    }
    private func nextReview() {
        let current = reviewCuts.first { $0.id == currentCut?.id }
        let remaining = reviewCuts.filter { $0.id != current?.id }
        if let current, let group = rallyGroups.first(where: { $0.id == current.id }) {
            edit { $0.reviewedCutIds.formUnion(group.cutIds) }
        }
        guard let next = remaining.first(where: { $0.keepStartMs >= playheadMs }) ?? remaining.first,
              let timestamp = EditorReview.firstReviewableTime(startMs: next.keepStartMs, endMs: next.keepEndMs, ignored: draft.ignoredIntervals) else { return }
        exportTab = false; seek(timestamp)
    }
    private func nextCleanup() {
        guard let suggestion = EditorReview.nextCleanup(presentation.pendingCleanup, positionMs: playheadMs, selectedId: selectedSuggestionId) else { return }
        selectCleanup(suggestion)
    }
    private func selectCleanup(_ suggestion: SuppressionRegion) {
        player.pause(); playing = false; exportTab = false
        let start = max(bounds.startMs, suggestion.startMs - 2_000)
        seek(EditorReview.firstReviewableTime(startMs: start, endMs: suggestion.endMs, ignored: draft.ignoredIntervals) ?? start)
        selectedSuggestionId = suggestion.id
    }
    private func nextServeReview() {
        guard !reviewServes.isEmpty else { return }
        let next: ServeMarker
        if let index = reviewServes.firstIndex(where: { $0.id == currentServe?.id }) { next = reviewServes[(index + 1) % reviewServes.count] }
        else { next = reviewServes.first { $0.timestampMs >= playheadMs } ?? reviewServes[0] }
        exportTab = false; seek(next.timestampMs); serveReviewReveal += 1
    }
    private func setTrimWindow() {
        guard let cut = currentCut else { trimWindow = bounds; return }
        trimWindow = .init(startMs: max(bounds.startMs, cut.keepStartMs - 5_000), endMs: min(bounds.endMs, cut.keepEndMs + 5_000))
    }
    private func seek(_ value: Int64) {
        selectedSuggestionId = nil
        previewEndMs = nil
        let time = min(project.durationMs, max(0, value)); playheadMs = time
        player.seek(to: CMTime(value: time, timescale: 1_000), toleranceBefore: .zero, toleranceAfter: .zero)
    }
    private func togglePlay() {
        guard player.currentItem != nil else { return }
        previewEndMs = nil
        if player.rate > 0 { player.pause(); playing = false }
        else { if playheadMs >= project.durationMs { seek(bounds.startMs) }; player.playImmediately(atRate: draft.playbackRate); playing = true }
    }
    private func pollPlayback() {
        let seconds = player.currentTime().seconds
        if seconds.isFinite && seconds >= 0 {
            let value = Int64((seconds * 1_000).rounded())
            if value != playheadMs { playheadMs = value }
        }
        let isPlaying = player.rate > 0
        if playing != isPlaying { playing = isPlaying }
        if playing, let selected = suggestions.first(where: { $0.id == selectedSuggestionId }),
           playheadMs < max(bounds.startMs, selected.startMs - 2_000) || playheadMs > selected.endMs { selectedSuggestionId = nil }
        if playing, let end = previewEndMs, playheadMs >= end {
            player.pause(); playing = false; previewEndMs = nil; seek(end); return
        }
        guard playing, draft.finalPreviewEnabled else { lastPreviewSeek = -1; return }
        if intervals.contains(where: { $0.startMs <= playheadMs && playheadMs < $0.endMs }) { lastPreviewSeek = -1; return }
        if let next = intervals.first(where: { $0.startMs > playheadMs }) {
            if lastPreviewSeek != next.startMs { lastPreviewSeek = next.startMs; seek(next.startMs) }
        } else { player.pause(); playing = false }
    }
    private func resetRanges() {
        guard let ranges = project.feedback?["initialInference"]?["ranges"]?.array else { message = "This project has no retained original analysis."; return }
        let cuts = ranges.enumerated().compactMap { index, range -> EditableCut? in
            guard let start = range["start"]?.double, let end = range["end"]?.double else { return nil }
            return .init(id: String(format: "R%03d", index + 1), coreStartMs: Int64((start * 1_000).rounded()), coreEndMs: Int64((end * 1_000).rounded()), confidence: Float(range["confidence"]?.double ?? 1), agreement: range["agreement"]?.string)
        }
        var value = EditorMath.newDraft(ranges: cuts, durationMs: project.durationMs, gameWindow: bounds, sourceRevision: draft.sourceRevision)
        value.scoreTracking = draft.scoreTracking; value.renderScoreOverlay = draft.renderScoreOverlay; value.renderScoreTimeline = draft.renderScoreTimeline
        value.fadeScoreOverlay = draft.fadeScoreOverlay
        replaceDraft(EditorMath.alignRallyServeMarkers(value)); setTrimWindow()
    }
    private func time(_ value: Int64, precise: Bool = false) -> String {
        let seconds = max(0, value) / 1_000
        let base = seconds >= 3_600 ? String(format: "%lld:%02lld:%02lld", seconds / 3_600, (seconds / 60) % 60, seconds % 60) : String(format: "%lld:%02lld", seconds / 60, seconds % 60)
        return precise ? base + String(format: ".%02lld", (max(0, value) % 1_000) / 10) : base
    }
}

private struct SidebarPointTimeline: View {
    @Environment(\.compactPhoneEditor) private var phone
    let score: DerivedScore
    let tracking: ScoreTracking
    let onSeek: (Int64) -> Void
    var body: some View {
        let awarded = score.points.enumerated().filter { $0.element.status == .counted && $0.element.winnerTeamId != nil }
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("POINT TIMELINE").font(phone ? .system(size: 10, weight: .bold) : .caption.bold()).foregroundStyle(EditorPalette.orange)
                if !phone { Spacer(); Text("\(awarded.count) awarded").font(.caption2).foregroundStyle(EditorPalette.muted).accessibilityIdentifier("awardedCount") }
            }
            if awarded.isEmpty { Text("No points awarded yet").font(phone ? .system(size: 10) : .caption).foregroundStyle(EditorPalette.muted) }
            else {
                ScrollView(.horizontal) {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 0) {
                            Text("RALLY").font(.system(size: 9, weight: .bold)).frame(width: 72, alignment: .leading)
                            ForEach(awarded, id: \.element.serveMarkerId) { entry in
                                Text("\(entry.offset + 1)").font(.system(size: 9, design: .monospaced)).frame(width: 32)
                            }
                        }.foregroundStyle(EditorPalette.muted)
                        pointRow(name: tracking.team1Name, team: .team1, points: awarded.map(\.element), color: EditorPalette.green)
                        pointRow(name: tracking.team2Name, team: .team2, points: awarded.map(\.element), color: EditorPalette.orange)
                    }
                }
            }
        }.accessibilityElement(children: .contain).accessibilityIdentifier("pointTimeline")
    }
    private func pointRow(name: String, team: ScoreTeamId, points: [DerivedScorePoint], color: Color) -> some View {
        HStack(spacing: 0) {
            Text(name).font(.system(size: 10)).lineLimit(1).frame(width: 72, alignment: .leading)
            ForEach(points, id: \.serveMarkerId) { point in
                Group {
                    if point.winnerTeamId == team {
                        Button { onSeek(point.timestampMs) } label: {
                            Text("\(team == .team1 ? point.team1ScoreAfter : point.team2ScoreAfter)")
                                .font(.system(size: 9, design: .monospaced)).foregroundStyle(EditorPalette.ink)
                                .frame(width: 24, height: 24).background(color.opacity(0.14), in: Circle())
                                .overlay { Circle().stroke(color, lineWidth: 2) }
                        }.buttonStyle(.plain).accessibilityLabel("Seek to point")
                            .accessibilityIdentifier("point-\(point.serveMarkerId)")
                    } else { Rectangle().fill(EditorPalette.rail).frame(width: 26, height: 1) }
                }.frame(width: 32, height: 28)
            }
        }
    }
}

private struct SidebarMarkerList: View {
    @Environment(\.compactPhoneEditor) private var phone
    let tracking: ScoreTracking
    let effectiveTimestampMs: Int64
    let onSeek: (Int64) -> Void
    let onRemove: (String, Bool) -> Void
    private struct Row: Identifiable {
        let id: String; let timeMs: Int64; let isServe: Bool; let needsReview: Bool; let detail: String
    }
    private var rows: [Row] {
        (tracking.serveMarkers.enumerated().map { index, marker in
            Row(id: marker.id, timeMs: marker.timestampMs, isServe: true, needsReview: marker.side == .review,
                detail: "\(marker.side == .review ? "Needs review" : index == 0 ? "First serve" : marker.side.rawValue) · \(marker.origin.rawValue)")
        } + tracking.sideSwitchMarkers.map { Row(id: $0.id, timeMs: $0.timestampMs, isServe: false, needsReview: false, detail: "Team side switch") })
            .sorted { ($0.timeMs, $0.id) < ($1.timeMs, $1.id) }
    }
    var body: some View {
        let rows = self.rows
        let selectedId = rows.last { $0.timeMs <= effectiveTimestampMs }?.id ?? rows.first?.id
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("MARKERS").font(phone ? .system(size: 10, weight: .bold) : .caption.bold()).foregroundStyle(EditorPalette.orange).accessibilityIdentifier("scoreMarkerList")
                if !phone { Spacer(); Text("\(rows.count) · tap to select").font(.caption2).foregroundStyle(EditorPalette.muted).accessibilityIdentifier("markerCount") }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(rows) { row in
                            VStack(spacing: 0) {
                            HStack(spacing: 4) {
                                Button { onSeek(row.timeMs) } label: {
                                    VStack(alignment: .leading, spacing: 3) {
                                        HStack(spacing: 6) {
                                            Text(row.isServe ? "🏐" : "⇄").font(.system(size: phone ? 10 : 13))
                                                .frame(width: phone ? 16 : 18)
                                                .accessibilityIdentifier("markerIcon-\(row.id)")
                                            Text("\(row.timeMs / 60_000):\(String(format: "%04.1f", Double(row.timeMs % 60_000) / 1_000))")
                                                .font(phone ? .system(size: 11, weight: .semibold, design: .monospaced) : .caption.monospaced().bold())
                                                .accessibilityIdentifier("markerTime-\(row.id)")
                                        }.fixedSize(horizontal: true, vertical: false)
                                        Text(row.detail).font(phone ? .system(size: 9) : .caption2).foregroundStyle(EditorPalette.muted).lineLimit(1)
                                    }.frame(maxWidth: .infinity, minHeight: phone ? 32 : 44, alignment: .leading).padding(.horizontal, phone ? 2 : 6).contentShape(Rectangle())
                                }.buttonStyle(.plain).accessibilityIdentifier(row.isServe ? "serve-\(row.id)" : "switch-\(row.id)")
                                Button("Remove") { onRemove(row.id, row.isServe) }.font(phone ? .system(size: 9) : .caption2).buttonStyle(.plain).foregroundStyle(EditorPalette.suppression)
                                    .frame(minWidth: phone ? 34 : 44, minHeight: phone ? 32 : 44).accessibilityIdentifier("removeMarker-\(row.id)")
                            }.background(selectedId == row.id ? EditorPalette.acid.opacity(0.35) : row.needsReview ? EditorPalette.clipReview.opacity(0.14) : Color.clear)
                            Divider()
                            }.frame(height: phone ? 33 : nil)
                                .accessibilityElement(children: .contain).accessibilityAddTraits(selectedId == row.id ? .isSelected : []).id(row.id)
                        }
                    }
                }.frame(height: phone ? min(4, max(1, Double(rows.count))) * 33 : min(220, max(44, Double(rows.count) * 45)))
                    .accessibilityIdentifier("markerViewport")
                    .onChange(of: selectedId, initial: true) { _, id in if let id { proxy.scrollTo(id, anchor: .center) } }
            }
        }
    }
}

private struct TimelineLegend: View {
    @State private var contentHeight: CGFloat = 28
    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 14) {
                item("Rally core") { block(EditorPalette.green) }
                item("Padding") { block(EditorPalette.rail) }
                item("Needs review") { block(EditorPalette.orange) }
                item("Serve") { Text("🏐").font(.system(size: 13)) }
                item("Side switch") { Text("⇄").font(.system(size: 13, weight: .bold, design: .monospaced)) }
                item("Excluded") {
                    Canvas { context, size in
                        var x = -size.height
                        while x < size.width {
                            var line = Path(); line.move(to: CGPoint(x: x, y: size.height)); line.addLine(to: CGPoint(x: x + size.height, y: 0))
                            context.stroke(line, with: .color(EditorPalette.suppression), lineWidth: 2); x += 6
                        }
                    }.frame(width: 20, height: 10).clipped()
                }
                item("Joined gap") {
                    HStack(spacing: 3) { ForEach(0..<4) { _ in Capsule().fill(EditorPalette.joinedGap).frame(width: 2, height: 10) } }
                }
            }.padding(.vertical, 4)
                .background { GeometryReader { content in
                    Color.clear.preference(key: LegendHeightKey.self, value: content.size.height)
                } }
        }
        // A horizontal ScrollView inside a vertical one otherwise proposes a
        // 10pt cross-axis height. Reserve the legend's actual padded content so
        // the phone player budget includes it and keeps it above the home edge.
        .frame(height: contentHeight)
        .onPreferenceChange(LegendHeightKey.self) { value in
            if value > 0 && abs(value - contentHeight) > 0.5 { contentHeight = value }
        }
    }
    private func block(_ color: Color) -> some View { RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 20, height: 10) }
    private func item<Symbol: View>(_ title: String, @ViewBuilder symbol: () -> Symbol) -> some View {
        HStack(spacing: 5) { symbol(); Text(title).font(.system(size: 10, weight: .semibold)).foregroundStyle(EditorPalette.muted) }
            .accessibilityElement(children: .ignore).accessibilityLabel(title)
    }
}

private struct TimelineTrack: View {
    let cuts: [EditableCut]
    let ignored: [IgnoredSourceInterval]
    let intervals: [FinalCutInterval]
    let reviewIds: Set<String>
    let serves: [ServeMarker]
    let switches: [SideSwitchMarker]
    let suggestions: [SuppressionRegion]
    let appliedSuggestionIds: Set<String>
    let removedSuggestionIds: Set<String>
    let removedCutIds: Set<String>
    let keptIds: Set<String>
    let selectedCutIds: Set<String>
    let selectedSuggestionId: String?
    let selectedServeId: String?
    let range: TimeRange
    let playheadMs: Int64
    let onSeek: (Int64) -> Void
    let onSelectCleanup: (SuppressionRegion) -> Void
    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                // Preserve complete outlines on the shorter phone overview rows.
                let laneHeight = max(1, min(30, size.height - 24))
                let suppressionHeight = max(1, size.height - 16)
                func x(_ time: Int64) -> Double { Double(time - range.startMs) / Double(max(1, range.endMs - range.startMs)) * size.width }
                func rectangle(_ start: Int64, _ end: Int64, y: Double, height: Double, color: Color) {
                    let a = max(0, x(start)); let b = min(size.width, x(end))
                    if b > a { context.fill(Path(CGRect(x: a, y: y, width: b - a, height: height)), with: .color(color)) }
                }
                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(EditorPalette.rail.opacity(0.32)))
                for cut in cuts {
                    let userRemoved = !cut.included || removedCutIds.contains(cut.id)
                    let needsReview = reviewIds.contains(cut.id)
                    let padding = userRemoved ? EditorPalette.muted.opacity(0.55) : !keptIds.contains(cut.id) ? EditorPalette.danger.opacity(0.65) : needsReview ? EditorPalette.warning : EditorPalette.rail
                    let core = userRemoved ? EditorPalette.muted : needsReview ? EditorPalette.orange : EditorPalette.green
                    rectangle(cut.keepStartMs, cut.keepEndMs, y: 18, height: laneHeight, color: padding)
                    rectangle(cut.coreStartMs, cut.coreEndMs, y: 18, height: laneHeight, color: core)
                }
                for interval in intervals { for gap in interval.joinedGaps { rectangle(gap.startMs, gap.endMs, y: 18, height: laneHeight, color: EditorPalette.joinedGap) } }
                let selectedCuts = cuts.filter { selectedCutIds.contains($0.id) }
                if let start = selectedCuts.map(\.keepStartMs).min(), let end = selectedCuts.map(\.keepEndMs).max(), end > range.startMs, start < range.endMs {
                    context.stroke(RoundedRectangle(cornerRadius: 4).path(in: CGRect(x: max(0, x(start)), y: 15, width: max(1, min(size.width, x(end)) - max(0, x(start))), height: laneHeight + 6)), with: .color(EditorPalette.orange), lineWidth: 2)
                }
                for region in ignored { rectangle(region.startMs, region.endMs, y: 0, height: size.height, color: EditorPalette.muted.opacity(0.5)) }
                for suggestion in suggestions where suggestion.startMs < range.endMs && suggestion.endMs > range.startMs {
                    let a = max(0, x(suggestion.startMs)), b = min(size.width, x(suggestion.endMs))
                    let rect = CGRect(x: a, y: 8, width: max(2, b - a), height: suppressionHeight)
                    let applied = appliedSuggestionIds.contains(suggestion.id)
                    let color = removedSuggestionIds.contains(suggestion.id) ? EditorPalette.muted : EditorPalette.suppression
                    context.fill(Path(rect), with: .color(color.opacity(applied ? 0.62 : 0.2)))
                    if !applied { context.stroke(Path(rect), with: .color(color), lineWidth: 2) }
                    var hatch = context
                    hatch.clip(to: Path(rect))
                    var position = a - suppressionHeight
                    while position < b {
                        var line = Path(); line.move(to: CGPoint(x: position, y: rect.maxY)); line.addLine(to: CGPoint(x: position + suppressionHeight, y: 8))
                        hatch.stroke(line, with: .color(color.opacity(applied ? 0.95 : 0.55)), lineWidth: 2)
                        position += 10
                    }
                    if selectedSuggestionId == suggestion.id { context.stroke(Path(rect.insetBy(dx: 1, dy: 2)), with: .color(.white), lineWidth: 1.5) }
                }
                if playheadMs >= range.startMs && playheadMs <= range.endMs {
                    context.fill(Path(CGRect(x: x(playheadMs) - 1, y: 0, width: 2, height: size.height)), with: .color(EditorPalette.orange))
                }
                for serve in serves where serve.timestampMs >= range.startMs && serve.timestampMs <= range.endMs {
                    let center = CGPoint(x: x(serve.timestampMs), y: 10)
                    context.fill(Path(CGRect(x: center.x - 0.5, y: 18, width: 1, height: size.height - 18)), with: .color(EditorPalette.ink))
                    let circle = Path(ellipseIn: CGRect(x: center.x - 8, y: 2, width: 16, height: 16))
                    context.fill(circle, with: .color(serve.side == .review ? Color(red: 1, green: 216 / 255.0, blue: 77 / 255.0) : .white))
                    context.stroke(circle, with: .color(EditorPalette.ink), lineWidth: 1)
                    context.draw(Text("🏐").font(.system(size: 12)), at: center)
                    if serve.id == selectedServeId { context.stroke(Path(ellipseIn: CGRect(x: center.x - 10, y: 0, width: 20, height: 20)), with: .color(EditorPalette.orange), lineWidth: 2) }
                }
                for marker in switches where marker.timestampMs >= range.startMs && marker.timestampMs <= range.endMs {
                    let center = CGPoint(x: x(marker.timestampMs), y: 10)
                    context.fill(Path(CGRect(x: center.x - 0.5, y: 18, width: 1, height: size.height - 18)), with: .color(EditorPalette.ink))
                    let box = Path(CGRect(x: center.x - 8, y: 2, width: 16, height: 16))
                    context.fill(box, with: .color(Color(red: 223 / 255.0, green: 1, blue: 53 / 255.0)))
                    context.stroke(box, with: .color(EditorPalette.ink), lineWidth: 1)
                    context.draw(Text("⇄").font(.system(size: 12, weight: .bold, design: .monospaced)), at: center)
                }
            }.contentShape(Rectangle()).gesture(DragGesture(minimumDistance: 0).onChanged { event in
                let fraction = min(1, max(0, event.location.x / max(1, geometry.size.width)))
                let time = range.startMs + Int64(fraction * Double(range.endMs - range.startMs))
                let tolerance = Int64(10 / max(1, Double(geometry.size.width)) * Double(range.endMs - range.startMs))
                let markers = serves.map(\.timestampMs) + switches.map(\.timestampMs)
                if event.location.y <= 24, let marker = markers.min(by: { abs($0 - time) < abs($1 - time) }), abs(marker - time) <= tolerance { onSeek(marker) }
                else if let suggestion = suggestions.first(where: { $0.startMs <= time && time < $0.endMs }) { onSelectCleanup(suggestion) }
                else { onSeek(time) }
            }).accessibilityLabel("Source timeline, tap or drag to seek")
                .accessibilityChildren {
                    ForEach(serves.filter { $0.timestampMs >= range.startMs && $0.timestampMs <= range.endMs }) { marker in
                        Button("Serve at \(marker.timestampMs / 1000) seconds") { onSeek(marker.timestampMs) }.accessibilityIdentifier("timelineServe-\(marker.id)")
                    }
                    ForEach(switches.filter { $0.timestampMs >= range.startMs && $0.timestampMs <= range.endMs }) { marker in
                        Button("Side switch at \(marker.timestampMs / 1000) seconds") { onSeek(marker.timestampMs) }.accessibilityIdentifier("timelineSwitch-\(marker.id)")
                    }
                    ForEach(suggestions.filter { $0.startMs < range.endMs && $0.endMs > range.startMs }, id: \.id) { region in
                        Button("Cleanup at \(region.startMs / 1000) seconds") { onSelectCleanup(region) }.accessibilityIdentifier("timelineCleanup-\(region.id)")
                    }
                }
        }.coordinateSpace(name: "trimTrack").clipped()
    }
}

private final class SourcePlayerLayerView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}
private struct EditableTeamName: View {
    let name: String
    let label: String
    let identifier: String
    let onCommit: (String) -> Void
    @State private var text = ""
    var body: some View {
        TextField(label, text: $text).textFieldStyle(.roundedBorder).font(.callout).accessibilityIdentifier(identifier)
            .onAppear { text = name }
            .onChange(of: name) { _, value in if text != value && !text.isEmpty { text = value } }
            .onChange(of: text) { _, value in if !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && value != name { onCommit(value) } }
            .onSubmit { if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { text = name } }
    }
}
/// Exact entry complements the coarse drag gesture and supports hardware keyboards and VoiceOver.
private struct EditorTimeSetting: View {
    let valueMs: Int64
    let label: String
    let identifier: String
    let onCommit: (Int64) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool
    private var enteredMs: Int64? {
        guard let seconds = Double(text.replacingOccurrences(of: ",", with: ".")),
              seconds.isFinite, (0...10).contains(seconds) else { return nil }
        return Int64((seconds * 10).rounded()) * 100
    }
    var body: some View {
        HStack(spacing: 6) {
            TextField("0–10", text: $text).textFieldStyle(.roundedBorder)
                .keyboardType(.numbersAndPunctuation).submitLabel(.done).multilineTextAlignment(.trailing).monospacedDigit()
                .frame(width: 72).focused($focused)
                .accessibilityLabel("\(label), seconds").accessibilityIdentifier(identifier + "Seconds")
                .onSubmit { commit() }
            Text("s").font(.caption)
            Button("Apply") { commit() }.buttonStyle(EditorActionButtonStyle(background: EditorPalette.paper, foreground: EditorPalette.ink)).disabled(enteredMs == nil)
                .accessibilityIdentifier(identifier + "Apply")
        }
        .onAppear { text = String(format: "%.1f", Double(valueMs) / 1_000) }
        .onChange(of: valueMs) { _, value in
            if !focused { text = String(format: "%.1f", Double(value) / 1_000) }
        }
    }
    private func commit() {
        guard let value = enteredMs else { return }
        if value != valueMs { onCommit(value) }
        text = String(format: "%.1f", Double(value) / 1_000); focused = false
    }
}
private struct SourcePlayerView: UIViewRepresentable {
    let player: AVPlayer
    func makeUIView(context: Context) -> SourcePlayerLayerView {
        let view = SourcePlayerLayerView(); view.playerLayer.videoGravity = .resizeAspect; view.playerLayer.player = player; return view
    }
    func updateUIView(_ uiView: SourcePlayerLayerView, context: Context) { uiView.playerLayer.player = player }
}

private struct EditorActionButtonStyle: ButtonStyle {
    let background: Color
    let foreground: Color
    var expand = false
    var minimumHeight: CGFloat = 42
    var border: Color = EditorPalette.rail
    var cornerRadius: CGFloat = 24
    @Environment(\.isEnabled) private var enabled
    @Environment(\.compactPhoneEditor) private var phone
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.system(size: phone ? 11 : 12, weight: .semibold))
            .foregroundStyle(enabled ? (configuration.role == .destructive ? EditorPalette.danger : foreground) : EditorPalette.muted.opacity(0.5))
            .padding(.horizontal, phone ? 6 : 10).frame(maxWidth: expand ? .infinity : nil, minHeight: phone ? min(32, minimumHeight) : minimumHeight)
            .background(background.opacity(enabled ? 1 : 0.15), in: RoundedRectangle(cornerRadius: cornerRadius))
            .overlay { RoundedRectangle(cornerRadius: cornerRadius).stroke(border.opacity(enabled ? 1 : 0.25), lineWidth: 1) }
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.8 : 1)
    }
}

/// UIKit gives the inactive track an explicit contrast color and keeps native slider accessibility.
private struct SourceSeekSlider: UIViewRepresentable {
    let value: Double
    let range: ClosedRange<Double>
    let onChange: (Double) -> Void
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIView(context: Context) -> UISlider {
        let slider = UISlider()
        slider.minimumTrackTintColor = UIColor(EditorPalette.green)
        slider.maximumTrackTintColor = UIColor(EditorPalette.ink.opacity(0.40))
        slider.thumbTintColor = UIColor(EditorPalette.ink)
        slider.accessibilityIdentifier = "sourceSeek"
        slider.accessibilityLabel = "Seek source video"
        slider.addTarget(context.coordinator, action: #selector(Coordinator.changed(_:)), for: .valueChanged)
        return slider
    }
    func updateUIView(_ slider: UISlider, context: Context) {
        context.coordinator.owner = self
        slider.minimumValue = Float(range.lowerBound); slider.maximumValue = Float(range.upperBound)
        if !slider.isTracking { slider.value = Float(min(range.upperBound, max(range.lowerBound, value))) }
    }
    final class Coordinator: NSObject {
        var owner: SourceSeekSlider
        init(_ owner: SourceSeekSlider) { self.owner = owner }
        @objc func changed(_ sender: UISlider) { owner.onChange(Double(sender.value)) }
    }
}
