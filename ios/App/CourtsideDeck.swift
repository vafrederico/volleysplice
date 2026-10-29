import SwiftUI
import AVFoundation

enum EditorStage: String, CaseIterable { case rallies = "Rallies", editor = "Editor", export = "Export" }

/// Equal side lanes keep the pills geometrically centered, independently of the
/// project name and metrics. The trailing lane scrolls inside its own bounds.
struct CourtsideWorkspaceBar<ProjectItems: View>: View {
    let name: String
    @Binding var stage: EditorStage
    let width: CGFloat
    let portrait: Bool
    let included: Int
    let duration: String
    let layoutDescription: String
    let cleanup: Int, clips: Int, serves: Int
    let onCleanup: () -> Void, onClips: () -> Void, onServes: () -> Void
    let onSettings: () -> Void, onDelete: () -> Void
    @ViewBuilder var projectItems: () -> ProjectItems
    private var wrapped: Bool { portrait && width < 600 }
    var body: some View {
        VStack(spacing: 2) {
            if wrapped {
                HStack(spacing: 6) { projectControl; Spacer(minLength: 0); settings }
                HStack { Spacer(minLength: 0); pills; Spacer(minLength: 0) }
                if stage != .rallies {
                    HStack(spacing: 4) {
                        stat("\(included)", "clips included", expanded: true)
                        stat(duration, "total time", expanded: true)
                            .accessibilityIdentifier("exportSummary").accessibilityValue(layoutDescription)
                    }
                    HStack(spacing: 4) { reviewCounters(expanded: true) }
                        .guidedTourTarget("editor-review-queues")
                }
            } else {
                let pillWidth: CGFloat = 198
                let side = max(0, (width - 16 - pillWidth) / 2)
                HStack(spacing: 0) {
                    projectControl.frame(width: side, alignment: .leading).clipped()
                    pills.frame(width: pillWidth)
                    HStack(spacing: 2) {
                        Spacer(minLength: 0)
                        if stage != .rallies { metrics }
                        settings
                    }.frame(width: side, alignment: .trailing)
                }
            }
        }.padding(.horizontal, 8).padding(.vertical, 4)
            .foregroundStyle(stage == .rallies ? CourtsidePalette.ink : EditorPalette.ink)
    }
    private var projectControl: some View {
        HStack(spacing: 3) {
            VolleySpliceBrand(compact: true).frame(width: 30, height: 34).accessibilityLabel("VolleySplice")
            Menu(content: projectItems) {
                HStack(spacing: 4) {
                    Text(name).font(.system(size: 13, weight: .semibold)).lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    Image(systemName: "chevron.down").font(.system(size: 10, weight: .semibold))
                }.frame(minHeight: 44, alignment: .leading).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityLabel("Projects, \(name)").accessibilityIdentifier("projectSelector")
            Button(action: onDelete) { Image(systemName: "trash").frame(width: 44, height: 44) }
                .buttonStyle(.plain).foregroundStyle(EditorPalette.danger)
                .accessibilityLabel("Delete current project").accessibilityIdentifier("deleteCurrentProject")
        }
    }
    private var pills: some View {
        HStack(spacing: 0) {
            ForEach(EditorStage.allCases, id: \.self) { item in
                Button { stage = item } label: {
                    Text(item.rawValue).font(.system(size: 12, weight: stage == item ? .semibold : .regular))
                        .frame(maxWidth: .infinity, minHeight: 44)
                        .background(stage == item ? Color.white : Color.clear, in: Capsule())
                }.buttonStyle(.plain).accessibilityAddTraits(stage == item ? .isSelected : [])
                    .accessibilityIdentifier("workspace-\(item.rawValue.lowercased())-tab")
                    .guidedTourTarget(item == .export ? "deck-export" : item == .editor ? "deck-editor" : "deck-rallies")
            }
        }.padding(3).frame(width: 198).background(CourtsidePalette.control.opacity(0.25), in: Capsule())
    }
    private var settings: some View {
        Button(action: onSettings) { Image(systemName: "gearshape").font(.system(size: 22)).frame(width: 44, height: 44) }
            .buttonStyle(.plain).accessibilityLabel("Editor settings").accessibilityIdentifier("editorSettings").guidedTourTarget("editor-settings")
    }
    private var metrics: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                stat("\(included)", "clips included")
                stat(duration, "total time").accessibilityIdentifier("exportSummary").accessibilityValue(layoutDescription)
                reviewCounters(expanded: false)
            }.fixedSize(horizontal: true, vertical: false)
        }.frame(maxWidth: 314).defaultScrollAnchor(.trailing).guidedTourTarget("editor-review-queues")
    }
    @ViewBuilder private func reviewCounters(expanded: Bool) -> some View {
        counter(cleanup, "cleanup", expanded: expanded, onCleanup)
        counter(clips, "clips", expanded: expanded, onClips)
        counter(serves, "serves", expanded: expanded, onServes)
    }
    private func stat(_ value: String, _ label: String, expanded: Bool = false) -> some View {
        VStack(spacing: 1) { Text(value).fontWeight(.semibold).monospacedDigit(); Text(label).font(.system(size: 11)).lineLimit(2) }
            .font(.system(size: 12)).multilineTextAlignment(.center)
            .frame(width: expanded ? nil : 57, height: 44).frame(maxWidth: expanded ? .infinity : nil)
            .accessibilityElement(children: .combine)
    }
    private func counter(_ count: Int, _ label: String, expanded: Bool, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 0) {
                Text("\(count)").font(.system(size: 12, weight: .semibold))
                Text("Review\n\(label)").font(.system(size: 11)).lineLimit(2).fixedSize(horizontal: false, vertical: true)
            }.multilineTextAlignment(.center).frame(width: expanded ? nil : 60, height: 48)
                .frame(maxWidth: expanded ? .infinity : nil)
                .background(count > 0 ? EditorPalette.acid : CourtsidePalette.raised, in: RoundedRectangle(cornerRadius: 5))
        }.buttonStyle(.plain).disabled(count == 0).accessibilityLabel("Review \(label), \(count)")
            .accessibilityIdentifier("review\(label.capitalized)")
    }
}

/// Layout and controls consume real presentation data. AVPlayer and draft writes
/// stay in EditorWorkspace; synthetic previews need only replace the video slot.
struct CourtsideDeckWorkspace<Video: View>: View {
    @Binding var navigation: RallyDeckNavigation
    let draft: EditorDraft
    let presentation: EditorPresentation
    let queue: [RallyReviewStop]
    let bounds: TimeRange
    let playhead: Int64
    let playing: Bool
    let sourceAvailable: Bool
    let overlay: PreparedScoreOverlay
    let onSeek: (Int64) -> Void
    let onPlay: () -> Void
    let onRate: (Double) -> Void
    let onDecision: (Bool) -> Void
    let onServe: (ServeMarker?, ServingSide?) -> Void
    let onTrim: (Bool, Int64) -> Void
    let onNeighbor: (Int) -> Void
    let onReview: () -> Void
    let onResumeReview: () -> Void
    let onSelect: (EditableRallyGroup) -> Void
    let onAdd: (TimeRange) -> Void
    @ViewBuilder var video: () -> Video
    @State private var listCenters: [String: CGFloat] = [:]
    @State private var listInteractionObserved = false
    @State private var splitDragStart: Double?
    @AppStorage("courtside.videoFraction") private var videoFraction = 0.56
    @Environment(\.compactPhoneEditor) private var phone
    @Environment(\.interfaceScale) private var interfaceScale
    @AppStorage(GuidedTourStore.key) private var tourState = ""
    private var rallies: [EditableRallyGroup] { presentation.groups }
    private var rally: EditableRallyGroup? { rallies.first { $0.id == navigation.selectedId } }
    private var index: Int { rallies.firstIndex { $0.id == rally?.id } ?? -1 }
    private var mode: RallyDeckMode { navigation.mode(selected: rally, queue: queue) }
    private var score: ScoreTracking { presentation.score }
    private var activeServes: [ServeMarker] {
        if let rally { return RallyReview.serves(for: rally, in: presentation.visibleScore.serveMarkers) }
        return presentation.visibleScore.serveMarkers.filter { navigation.focus?.serveIds.contains($0.id) == true }
    }
    private var occupied: [TimeRange] {
        rallies.map { .init(startMs: $0.keepStartMs, endMs: $0.keepEndMs) } + draft.ignoredIntervals.map { .init(startMs: $0.startMs, endMs: $0.endMs) }
    }
    private var missing: TimeRange? { RallyReview.missingRange(at: playhead, bounds: bounds, occupied: occupied) }
    var body: some View {
        GeometryReader { geometry in
            let landscape = geometry.size.width > geometry.size.height && geometry.size.width >= 560
            VStack(spacing: 0) {
                reviewHeader
                if landscape {
                    let available = max(1, geometry.size.width - (phone ? 8 : 24))
                    let playerWidth = max(240, min(available - 300, available * videoFraction))
                    HStack(spacing: 0) {
                        playerPane.frame(width: phone ? max(240, geometry.size.width * 0.56) : playerWidth).contentShape(Rectangle()).clipped()
                        if !phone {
                            videoDivider(available: available, current: playerWidth)
                                .opacity(navigation.addingMissing ? 0 : 1)
                                .allowsHitTesting(!navigation.addingMissing)
                                .accessibilityHidden(navigation.addingMissing)
                        }
                        else { Color.clear.frame(width: 8) }
                        detailPane.frame(maxWidth: .infinity).contentShape(Rectangle()).clipped()
                    }
                } else {
                    let playerHeight = phone || geometry.size.width < 600
                        ? min(360, max(170, geometry.size.height * 0.40))
                        : max(170, min(geometry.size.height * 0.62, geometry.size.height - 280))
                    playerPane.frame(height: playerHeight).contentShape(Rectangle()).clipped()
                    detailPane.frame(maxHeight: .infinity)
                }
            }
        }.background(CourtsidePalette.ground).foregroundStyle(CourtsidePalette.ink)
    }
    private var reviewHeader: some View {
        HStack {
            Text(headerTitle).font(.system(size: 16, weight: .semibold)).monospacedDigit()
                .accessibilityIdentifier("deckReviewCount")
            Spacer(minLength: 8)
            Button {
                if mode == .review {
                    navigation.browsing = true; navigation.editingRally = false
                    navigation.follow(time: playhead, rallies: rallies)
                }
                else { onResumeReview() }
            } label: {
                HStack(spacing: 6) {
                    if mode != .review { Image(systemName: "arrow.left") }
                    Text(mode == .review ? "All rallies (\(rallies.count))" : "Review rallies")
                    if mode == .review { Image(systemName: "arrow.right") }
                }.frame(minHeight: 44)
            }.buttonStyle(.plain).foregroundStyle(CourtsidePalette.accent).accessibilityIdentifier("deckBrowse")
        }.font(.system(size: 13)).foregroundStyle(CourtsidePalette.muted)
            .padding(.horizontal, 12).guidedTourTarget("deck-progress")
    }
    private var headerTitle: String {
        switch mode {
        case .adding: return "Add missing rally"
        case .allRallies: return "All rallies"
        case .editing: return "Editing rally \(index + 1)"
        case .review: return "Reviewing \(queue.count) \(queue.count == 1 ? "rally" : "rallies")"
        }
    }
    private func videoDivider(available: CGFloat, current: CGFloat) -> some View {
        Color.clear.frame(width: 24).overlay { Capsule().fill(CourtsidePalette.control).frame(width: 4, height: 44) }
            .contentShape(Rectangle()).gesture(DragGesture(coordinateSpace: .global).onChanged { event in
                if splitDragStart == nil { splitDragStart = current }
                videoFraction = min(available - 300, max(240, (splitDragStart ?? current) + event.translation.width / interfaceScale)) / available
            }.onEnded { _ in splitDragStart = nil })
            .accessibilityElement().accessibilityLabel("Resize rally video").accessibilityIdentifier("deckVideoDivider")
            .accessibilityValue("\(Int(videoFraction * 100)) percent")
            .accessibilityAdjustableAction { direction in
                videoFraction = min((available - 300) / available, max(240 / available, videoFraction + (direction == .increment ? 0.05 : -0.05)))
            }
    }
    private var playerPane: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .top) {
                video().frame(maxWidth: .infinity, maxHeight: .infinity)
            }.background(.black).guidedTourTarget("deck-video")
            playbackControls
        }
    }
    private var playbackControls: some View {
        HStack(spacing: 10) {
            CourtsideActionButton(title: playing ? "Pause video" : "Play video",
                imageName: playing ? "pause.fill" : "play.fill", identifier: "deckPlay",
                enabled: sourceAvailable, selected: false, bordered: false, action: onPlay)
                .frame(width: 44, height: 44)
            Text(deckTime(playhead)).monospacedDigit().font(.system(size: 13))
                .accessibilityIdentifier("deckPlayhead")
            Spacer(minLength: 0)
            Text(playbackDuration).font(.system(size: 11)).foregroundStyle(.white.opacity(0.7)).lineLimit(1)
            playbackSpeed
        }.padding(.horizontal, 10).foregroundStyle(.white).background(Color(white: 0.09))
    }
    private var playbackDuration: String {
        if navigation.browsing || navigation.addingMissing { return "\(deckTime(bounds.endMs)) source" }
        let start = rally?.coreStartMs ?? navigation.focus?.startMs ?? 0
        let end = rally?.coreEndMs ?? navigation.focus?.endMs ?? 0
        let seconds = Double(end - start) / 1_000
        return String(format: "%.1f s", seconds)
    }
    private var playbackSpeed: some View {
        Menu {
            ForEach(EditorPlaybackPolicy.options(for: draft.playbackRate), id: \.self) { rate in
                Button("\(rate.formatted())×") { onRate(rate) }
            }
        } label: {
            Text("\(draft.playbackRate.formatted())×").font(.system(size: 13, weight: .semibold))
                .frame(minWidth: 44, minHeight: 44)
        }
        .accessibilityLabel("Playback speed, \(draft.playbackRate.formatted()) times").accessibilityIdentifier("deckSpeed")
    }
    @ViewBuilder private var detailPane: some View {
        if navigation.addingMissing {
            MissingRallyPane(selection: $navigation.missingSelection, rallies: rallies, effectiveIds: presentation.keptIds,
                reviewIds: presentation.pendingConfidenceIds, ignored: draft.ignoredIntervals,
                bounds: bounds, occupied: occupied, playhead: playhead, onSeek: onSeek,
                onCancel: { navigation.addingMissing = false }, onAdd: onAdd)
        } else if navigation.browsing { browser.guidedTourTarget("deck-list") }
        else {
            ScrollViewReader { proxy in
                VStack(spacing: 0) {
                    ScrollView {
                        VStack(spacing: 10) { focusControls }.padding(10)
                    }
                    if rally != nil || navigation.focus != nil {
                        navigationButtons.padding(10)
                    }
                }.onChange(of: navigation.focus?.id) { _, _ in proxy.scrollTo("deck-trim", anchor: .top) }
                    .onChange(of: tourState) { _, value in
                        if let step = GuidedTourStep(rawValue: value) { proxy.scrollTo(step.target, anchor: .center) }
                    }
            }
        }
    }
    @ViewBuilder private var focusControls: some View {
        if let rally {
            VStack(spacing: 8) {
                HStack {
                    nudgeControls(start: true, value: rally.coreStartMs)
                    Spacer(minLength: 0)
                    Text("Rally \(index + 1)").font(.system(size: 15, weight: .semibold))
                    Spacer(minLength: 0)
                    nudgeControls(start: false, value: rally.coreEndMs)
                }
                CourtsideTimeline(rallies: rallies, effectiveIds: presentation.keptIds,
                    reviewIds: presentation.pendingConfidenceIds, removedIds: presentation.userRemovedCleanupCutIds, ignored: draft.ignoredIntervals,
                    bounds: bounds, playhead: playhead, selected: rally, onSeek: onSeek, onTrim: onTrim)
                HStack { Text("Start \(deckTime(rally.coreStartMs))"); Spacer(); Text("End \(deckTime(rally.coreEndMs))") }
                    .font(.system(size: 12)).monospacedDigit().foregroundStyle(CourtsidePalette.muted)
            }.guidedTourTarget("deck-trim")
            decisions
            serveChoices.guidedTourTarget("deck-tagging")
        } else if let focus = navigation.focus {
            Text(focus.cleanupIds.isEmpty ? "Serve review" : "Cleanup review").font(.headline)
            Text("\(deckTime(focus.startMs)) – \(deckTime(focus.endMs))").monospacedDigit()
            decisions
            serveChoices
        } else {
            Text("No rallies yet").font(.headline)
            Button("Browse source video") { navigation.browsing = true }.buttonStyle(CourtsideButtonStyle())
        }
    }
    private func nudgeControls(start: Bool, value: Int64) -> some View {
        CourtsideTrimNudges(start: start) { direction in onTrim(start, value + Int64(direction) * 1_000) }
    }
    private var decisions: some View {
        CourtsideDecisionRow(selected: decisionSelected(keep: true) ? true : decisionSelected(keep: false) ? false : nil, onDecision: onDecision)
            .accessibilityIdentifier("deckDecisions")
    }
    private func decisionSelected(keep: Bool) -> Bool {
        guard let rally, rally.cutIds.isSubset(of: draft.reviewedCutIds) else { return false }
        return keep ? rally.cuts.allSatisfy(\.included) : rally.cuts.allSatisfy { !$0.included }
    }
    @ViewBuilder private var serveChoices: some View {
        if score.enabled {
            if activeServes.isEmpty {
                servingRow(nil)
            } else {
                ForEach(activeServes) { marker in
                    VStack(spacing: 3) {
                        if activeServes.count > 1 { Text("Serve \(deckTime(marker.timestampMs))").font(.system(size: 11)) }
                        servingRow(marker)
                    }
                }
            }
        } else { Text("Enable score tracking in Editor to tag serves.").font(.system(size: 12)).foregroundStyle(CourtsidePalette.muted) }
    }
    private func servingRow(_ marker: ServeMarker?) -> some View {
        CourtsideServeRow(selection: marker == nil ? 2 : marker?.side == .near ? 0 : marker?.side == .far ? 1 : nil,
            canAdd: rally != nil, hasMarker: marker != nil) { choice in
                onServe(marker, choice == 0 ? .near : choice == 1 ? .far : nil)
            }
    }
    private var navigationButtons: some View {
        CourtsideReviewNavigation(previous: neighbor(-1) != nil,
            pending: RallyReview.canAdvance(focus: navigation.focus, selected: rally, queue: queue), next: neighbor(1) != nil,
            onNeighbor: onNeighbor, onReview: onReview)
    }
    private func neighbor(_ direction: Int) -> EditableRallyGroup? {
        rally.flatMap { RallyReview.neighbor(of: $0, direction: direction, rallies: rallies, effectiveIds: presentation.keptIds) }
    }
    private var browser: some View {
        VStack(spacing: 4) {
            HStack {
                Text(RallyReview.at(playhead, in: rallies) == nil ? "Between rallies \u{00B7} \(deckTime(playhead))" : "Rally \((rallies.firstIndex { $0.id == navigation.selectedId } ?? 0) + 1) \u{00B7} \(deckTime(playhead))")
                    .lineLimit(1).accessibilityIdentifier("deckSourceStatus")
                Spacer(minLength: 8)
                Button(navigation.showTimeline ? "Hide timeline" : "Show timeline") { navigation.showTimeline.toggle() }
                    .frame(minHeight: 44).accessibilityIdentifier("deckToggleSourceTimeline")
            }.font(.system(size: 12)).foregroundStyle(CourtsidePalette.muted)
            HStack(spacing: 8) {
                if navigation.showTimeline {
                    CourtsideSourceRail(rallies: rallies, effectiveIds: presentation.keptIds, bounds: bounds, playhead: playhead,
                        listCenter: rallies.first { $0.id == navigation.listCenterId }?.coreStartMs, onSeek: onSeek)
                        .frame(width: 96)
                }
                VStack(spacing: 4) {
                    HStack {
                        Label("Play", systemImage: "circle.fill").foregroundStyle(CourtsidePalette.accent)
                        Spacer(minLength: 8)
                        HStack(spacing: 6) {
                            RallyListMarker()
                            Text(rallies.firstIndex { $0.id == navigation.listCenterId }.map { "List \u{00B7} \($0 + 1)" } ?? "List —")
                        }.foregroundStyle(CourtsidePalette.muted)
                    }.font(.system(size: 12)).accessibilityIdentifier("deckListLegend")
                    rallyList
                }
            }.frame(maxHeight: .infinity)
            HStack(spacing: 8) {
                CourtsideActionButton(title: index >= 0 ? "Edit rally \(index + 1)" : "Edit selected rally",
                    identifier: "deckEditSelected", enabled: rally != nil, selected: false) {
                        if let rally { navigation.edit(rally, queue: queue) }
                    }
                    .frame(maxWidth: .infinity).frame(height: 44)
                CourtsideActionButton(title: "+ Add missing rally", identifier: "deckAddMissing",
                    enabled: true, selected: missing != nil) {
                        navigation.missingSelection.reset(); navigation.addingMissing = true; onSeek(playhead)
                    }
                    .frame(maxWidth: .infinity).frame(height: 44).guidedTourTarget("deck-add")
            }
        }.padding(10).background(.white)
    }
    private var rallyList: some View {
        ScrollViewReader { proxy in
            GeometryReader { geometry in
                ScrollView {
                    LazyVStack(spacing: 3) {
                        ForEach(Array(rallies.enumerated()), id: \.element.id) { rowIndex, value in
                            Button {
                                if value.id == navigation.selectedId { navigation.edit(value, queue: queue) }
                                else { onSelect(value) }
                            } label: {
                                let snapshot = overlay.snapshot(at: value.coreStartMs).score
                                VStack(spacing: 6) {
                                    HStack {
                                        HStack(spacing: 6) {
                                            Text("Rally \(rowIndex + 1)").fontWeight(.semibold)
                                            if value.id == navigation.listCenterId { RallyListMarker() }
                                        }
                                        Spacer()
                                        Text(score.enabled ? "\(snapshot.team1Score) : \(snapshot.team2Score)" : "—")
                                    }
                                    HStack {
                                        Text("\(deckTime(value.coreStartMs)) – \(deckTime(value.coreEndMs))")
                                        Spacer()
                                        Text(RallyReview.retained(value, effectiveIds: presentation.keptIds) ? "Keep" : "Remove")
                                            .foregroundStyle(RallyReview.retained(value, effectiveIds: presentation.keptIds) ? CourtsidePalette.muted : EditorPalette.danger)
                                    }.font(.system(size: 11)).foregroundStyle(CourtsidePalette.muted)
                                }.font(.system(size: 13)).monospacedDigit().padding(10).frame(minHeight: 62)
                                    .background(value.id == navigation.selectedId ? CourtsidePalette.control.opacity(0.3) : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                                    .contentShape(Rectangle())
                            }.buttonStyle(.plain).id(value.id)
                                .accessibilityIdentifier("deckRallyRow-\(value.id)")
                                .accessibilityAddTraits(value.id == navigation.selectedId ? .isSelected : [])
                                .accessibilityValue(value.id == navigation.listCenterId ? "List center" : "")
                                .background { GeometryReader { row in Color.clear.preference(key: RallyRowCenters.self, value: [value.id: row.frame(in: .named("deck-list-scroll")).midY]) } }
                        }
                    }
                }.coordinateSpace(name: "deck-list-scroll")
                    .modifier(RallyListInteraction {
                        listInteractionObserved = true
                        navigation.following = false
                        navigation.listCenterId = listCenters.min { abs($0.value - geometry.size.height / 2) < abs($1.value - geometry.size.height / 2) }?.key
                    })
                    .onPreferenceChange(RallyRowCenters.self) { centers in
                        listCenters = centers
                        guard !navigation.following, listInteractionObserved else { return }
                        let id = centers.min { abs($0.value - geometry.size.height / 2) < abs($1.value - geometry.size.height / 2) }?.key
                        if id != navigation.listCenterId { navigation.listCenterId = id }
                    }
                    .onChange(of: navigation.followRequest) { _, _ in
                        if let id = navigation.listCenterId { proxy.scrollTo(id, anchor: .center) }
                    }
                    .onChange(of: playhead) { _, value in
                        if let active = RallyReview.at(value, in: rallies) { navigation.selectedId = active.id }
                        guard navigation.following, let next = RallyReview.nearest(value, in: rallies), next.id != navigation.listCenterId else { return }
                        navigation.listCenterId = next.id; proxy.scrollTo(next.id, anchor: .center)
                    }
                    .onAppear {
                        listInteractionObserved = false
                        if navigation.following { navigation.follow(time: playhead, rallies: rallies) }
                        if let id = navigation.listCenterId ?? navigation.selectedId { proxy.scrollTo(id, anchor: .center) }
                    }
                    .onDisappear { listInteractionObserved = false }
            }
        }
    }

}

private struct RallyRowCenters: PreferenceKey {
    static let defaultValue: [String: CGFloat] = [:]
    static func reduce(value: inout [String: CGFloat], nextValue: () -> [String: CGFloat]) { value.merge(nextValue(), uniquingKeysWith: { _, new in new }) }
}

/// Observe native scrolling without competing with the scroll view's pan gesture.
private struct RallyListInteraction: ViewModifier {
    let onInteraction: () -> Void
    @ViewBuilder func body(content: Content) -> some View {
        if #available(iOS 18, *) {
            content.onScrollPhaseChange { _, phase in
                if phase == .tracking || phase == .interacting { onInteraction() }
            }
        } else {
            content.simultaneousGesture(DragGesture(minimumDistance: 10).onChanged { _ in onInteraction() })
        }
    }
}

struct CourtsideTimeline: View {
    let rallies: [EditableRallyGroup]
    let effectiveIds: Set<String>
    var reviewIds: Set<String> = []
    var removedIds: Set<String> = []
    let ignored: [IgnoredSourceInterval]
    let bounds: TimeRange
    let playhead: Int64
    let selected: EditableRallyGroup
    let onSeek: (Int64) -> Void
    let onTrim: (Bool, Int64) -> Void
    var body: some View {
        VStack(spacing: 3) {
            CoreTrimSurface(core: .init(startMs: selected.coreStartMs, endMs: selected.coreEndMs),
                kept: .init(startMs: selected.keepStartMs, endMs: selected.keepEndMs), bounds: bounds, playhead: playhead,
                startLabel: "Trim rally start", endLabel: "Trim rally end", startIdentifier: "deckTrimStart", endIdentifier: "deckTrimEnd",
                contextWindow: RallyReview.context(for: selected, rallies: rallies, effectiveIds: effectiveIds, bounds: bounds),
                halfHeightHandles: true,
                onTrim: onTrim) { scale in
                    track(scale)
                }
                .id(selected.cutIds.sorted())
            HStack {
                if let previous = RallyReview.neighbor(of: selected, direction: -1, rallies: rallies, effectiveIds: effectiveIds) {
                    Button("Previous · \(deckTime(previous.coreStartMs))") { onSeek(previous.keepStartMs) }
                        .accessibilityLabel("Previous retained rally at \(deckTime(previous.coreStartMs))")
                }
                Spacer(minLength: 0)
                if let next = RallyReview.neighbor(of: selected, direction: 1, rallies: rallies, effectiveIds: effectiveIds) {
                    Button("Next · \(deckTime(next.coreStartMs))") { onSeek(next.keepStartMs) }
                        .accessibilityLabel("Next retained rally at \(deckTime(next.coreStartMs))")
                }
            }.font(.system(size: 11)).buttonStyle(.plain)
        }
    }
    private func track(_ scale: RallyTrimScale) -> some View {
        let window = scale.window
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 8).fill(EditorPalette.rail.opacity(0.45)).contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { gesture in onSeek(scale.time(atX: gesture.location.x)) })
            ForEach(rallies.filter { $0.id == selected.id ||
                $0.id == RallyReview.neighbor(of: selected, direction: -1, rallies: rallies, effectiveIds: effectiveIds)?.id ||
                $0.id == RallyReview.neighbor(of: selected, direction: 1, rallies: rallies, effectiveIds: effectiveIds)?.id }, id: \.id) { rally in
                let active = rally.id == selected.id
                ForEach(rally.cuts, id: \.id) { cut in
                    let removed = !cut.included || removedIds.contains(cut.id)
                    let review = reviewIds.contains(cut.id)
                    let padding = removed ? EditorPalette.muted.opacity(0.55) : !effectiveIds.contains(cut.id) ? EditorPalette.danger.opacity(0.65) : review ? EditorPalette.warning : EditorPalette.rail
                    let core = removed ? EditorPalette.muted : review ? EditorPalette.orange : EditorPalette.green
                    RoundedRectangle(cornerRadius: 4).fill(padding)
                        .frame(width: max(0, scale.x(min(window.endMs, cut.keepEndMs)) - scale.x(max(window.startMs, cut.keepStartMs))), height: 30)
                        .offset(x: scale.x(max(window.startMs, cut.keepStartMs)), y: 18).allowsHitTesting(false)
                    RoundedRectangle(cornerRadius: 4).fill(core)
                        .frame(width: max(0, scale.x(min(window.endMs, cut.coreEndMs)) - scale.x(max(window.startMs, cut.coreStartMs))), height: 30)
                        .offset(x: scale.x(max(window.startMs, cut.coreStartMs)), y: 18).allowsHitTesting(false)
                }
                if active {
                    RoundedRectangle(cornerRadius: 4).stroke(EditorPalette.orange, lineWidth: 2)
                        .frame(width: max(1, scale.x(min(window.endMs, rally.keepEndMs)) - scale.x(max(window.startMs, rally.keepStartMs))), height: 36)
                        .offset(x: scale.x(max(window.startMs, rally.keepStartMs)), y: 15).allowsHitTesting(false)
                }
            }
            ForEach(ignored) { span in
                if span.startMs < window.endMs && window.startMs < span.endMs {
                    Rectangle().fill(CourtsidePalette.control).frame(width: max(0, scale.x(min(window.endMs, span.endMs)) - scale.x(max(window.startMs, span.startMs))), height: 64)
                        .offset(x: scale.x(max(window.startMs, span.startMs))).allowsHitTesting(false)
                }
            }
            if playhead >= window.startMs && playhead <= window.endMs {
                Rectangle().fill(CourtsidePalette.accent).frame(width: 2, height: 64).offset(x: scale.x(playhead)).allowsHitTesting(false)
            }
        }
    }
}

private struct RallyListMarker: View {
    var body: some View {
        Canvas { context, size in
            let x = size.width / 2, y = size.height / 2
            var diamond = Path()
            diamond.move(to: .init(x: x, y: y - 5))
            diamond.addLine(to: .init(x: x + 5, y: y))
            diamond.addLine(to: .init(x: x, y: y + 5))
            diamond.addLine(to: .init(x: x - 5, y: y))
            diamond.closeSubpath()
            context.fill(diamond, with: .color(.white))
            context.stroke(diamond, with: .color(CourtsidePalette.ink), lineWidth: 2)
        }.frame(width: 14, height: 14).accessibilityHidden(true)
    }
}

private struct CourtsideSourceRail: View {
    let rallies: [EditableRallyGroup]
    let effectiveIds: Set<String>
    let bounds: TimeRange
    let playhead: Int64
    let listCenter: Int64?
    let onSeek: (Int64) -> Void
    var body: some View {
        let columns = SourceTimelineColumns(bounds: bounds)
        HStack(spacing: 8) {
            ForEach(0..<2, id: \.self) { index in
                let window = columns.ranges[index]
                VStack(spacing: 2) {
                    Text(deckTime(window.startMs))
                    rail(window: window, index: index, columns: columns)
                    Text(deckTime(window.endMs))
                }.font(.system(size: 11)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.7)
                    .frame(width: 44)
            }
        }
    }
    private func rail(window: TimeRange, index: Int, columns: SourceTimelineColumns) -> some View {
        GeometryReader { geometry in
            let height = max(1, geometry.size.height - 16)
            let duration = Double(max(1, window.endMs - window.startMs))
            let y: (Int64) -> CGFloat = { 8 + CGFloat(min(1, max(0, Double($0 - window.startMs) / duration))) * height }
            ZStack(alignment: .topLeading) {
                Capsule().fill(CourtsidePalette.control).frame(width: 3, height: height).offset(x: 20.5, y: 8)
                ForEach(rallies.filter { $0.keepStartMs < window.endMs && window.startMs < $0.keepEndMs }, id: \.id) { rally in
                    Rectangle().fill(RallyReview.retained(rally, effectiveIds: effectiveIds) ? EditorPalette.green : EditorPalette.danger)
                        .frame(width: 18, height: max(3, y(min(window.endMs, rally.keepEndMs)) - y(max(window.startMs, rally.keepStartMs))))
                        .offset(x: 13, y: y(max(window.startMs, rally.keepStartMs)))
                        .accessibilityIdentifier("deckSourceRally-\(index)-\(rally.id)")
                }
                if columns.column(at: playhead) == index {
                    Image(systemName: "circle.fill").font(.system(size: 8)).foregroundStyle(CourtsidePalette.accent)
                        .offset(x: 0, y: y(playhead) - 4)
                    Rectangle().fill(CourtsidePalette.accent).frame(width: 22, height: 2).offset(y: y(playhead))
                }
                if let listCenter, columns.column(at: listCenter) == index {
                    RallyListMarker().offset(x: 29, y: y(listCenter) - 7)
                }
            }.frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading)
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 1)
                    .onChanged { onSeek(columns.time(in: index, fraction: Double(($0.location.y - 8) / height))) }
                    .onEnded { onSeek(columns.time(in: index, fraction: Double(($0.location.y - 8) / height))) }
                    .exclusively(before: SpatialTapGesture().onEnded {
                        onSeek(columns.time(in: index, fraction: Double(($0.location.y - 8) / height)))
                    }))
        }.accessibilityElement().accessibilityLabel("Source timeline column \(index + 1), \(deckTime(window.startMs)) to \(deckTime(window.endMs)). Circle: video. Diamond: list center.")
            .accessibilityIdentifier("deckSourceTimelineColumn-\(index)")
            .accessibilityValue("Video \(deckTime(playhead))" + (listCenter.map { "; List center \(deckTime($0))" } ?? ""))
            .accessibilityAdjustableAction { direction in
                onSeek(min(window.endMs, max(window.startMs, playhead + (direction == .increment ? 1_000 : -1_000))))
            }
    }
}

private struct MissingRallyPane: View {
    @Binding var selection: MissingRallySelection
    let rallies: [EditableRallyGroup]
    let effectiveIds: Set<String>
    let reviewIds: Set<String>
    let ignored: [IgnoredSourceInterval]
    let bounds: TimeRange
    let occupied: [TimeRange]
    let playhead: Int64
    let onSeek: (Int64) -> Void
    let onCancel: () -> Void
    let onAdd: (TimeRange) -> Void
    private var range: TimeRange? { selection.range(endingAt: playhead, bounds: bounds, occupied: occupied) }
    private var canMark: Bool { selection.canMarkStart(at: playhead, bounds: bounds, occupied: occupied) }
    private var instruction: String {
        if let start = selection.startMs {
            if playhead < start + RallyReview.minimumDurationMs { return "Seek at least 0.5 seconds after the start." }
            if range == nil { return "Choose an end before the next rally or ignored footage." }
            return "Seek to the rally end, then mark and add it below."
        }
        return canMark ? "Seek to the rally start, then mark it below." : "Choose a start in the empty footage between rallies."
    }
    var body: some View {
        GeometryReader { _ in
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(instruction).font(.system(size: 14)).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Button("Cancel", action: onCancel).frame(minHeight: 44).accessibilityIdentifier("deckAddCancel")
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(selection.startMs.map { "Start \(deckTime($0))" } ?? "Start —")
                                .accessibilityIdentifier("deckAddStartTime")
                            Spacer(minLength: 8)
                            Text(selection.startMs == nil ? "End —" : "End \(deckTime(playhead))")
                                .accessibilityIdentifier("deckAddPosition")
                            Spacer(minLength: 8)
                            Button("Reset") { selection.reset() }.disabled(selection.startMs == nil)
                                .frame(minWidth: 44, minHeight: 44).accessibilityIdentifier("deckAddReset")
                        }.font(.system(size: 14, weight: .medium)).monospacedDigit().foregroundStyle(CourtsidePalette.accent)
                        MissingRallyTimeline(rallies: rallies, effectiveIds: effectiveIds, reviewIds: reviewIds,
                            ignored: ignored, bounds: bounds, playhead: playhead, startMs: selection.startMs,
                            valid: range != nil, onSeek: onSeek)
                            .frame(height: 204)
                    }.padding(.bottom, 12)
                }
                CourtsideActionButton(title: selection.startMs == nil ? "Mark start at \(deckTime(playhead))" : "Mark end & add rally", identifier: "deckAddMark",
                    enabled: selection.startMs == nil ? canMark : range != nil, selected: true) {
                    if selection.startMs == nil { selection.markStart(at: playhead, bounds: bounds, occupied: occupied) }
                    else if let range { onSeek(playhead); onAdd(range) }
                }.frame(maxWidth: .infinity).frame(height: 44)
            }.padding(10)
        }.background(.white)
    }
}

/// Consecutive rows share a linear source-time scale. Playback and dragging
/// update the marker from the player's position without a separate seek state.
private struct MissingRallyTimeline: View {
    let rallies: [EditableRallyGroup]
    let effectiveIds: Set<String>
    let reviewIds: Set<String>
    let ignored: [IgnoredSourceInterval]
    let bounds: TimeRange
    let playhead: Int64
    let startMs: Int64?
    let valid: Bool
    let onSeek: (Int64) -> Void
    @GestureState private var horizontalScrub: Bool?
    var body: some View {
        GeometryReader { _ in
            let rows = 3
            VStack(spacing: 6) {
                ForEach(0..<rows, id: \.self) { index in
                    let duration = bounds.endMs - bounds.startMs
                    let window = TimeRange(startMs: bounds.startMs + duration * Int64(index) / Int64(rows),
                        endMs: bounds.startMs + duration * Int64(index + 1) / Int64(rows))
                    VStack(spacing: 2) {
                        HStack { Text(deckTime(window.startMs)); Spacer(); Text(deckTime(window.endMs)) }
                            .font(.system(size: 11)).monospacedDigit().foregroundStyle(CourtsidePalette.muted)
                        timelineRow(window, last: index == rows - 1)
                            .accessibilityIdentifier("deckAddTimelineRow-\(index)")
                    }.frame(maxHeight: .infinity)
                }
            }
        }
    }
    private func timelineRow(_ window: TimeRange, last: Bool) -> some View {
        Canvas { context, size in
            func x(_ time: Int64) -> CGFloat {
                CGFloat(Double(time - window.startMs) / Double(max(1, window.endMs - window.startMs))) * size.width
            }
            func bar(_ range: TimeRange, color: Color, y: CGFloat = 6, height: CGFloat? = nil) {
                let left = max(window.startMs, range.startMs), right = min(window.endMs, range.endMs)
                guard right > left else { return }
                let rect = CGRect(x: x(left), y: y, width: max(1, x(right) - x(left)), height: height ?? max(1, size.height - 12))
                context.fill(Path(roundedRect: rect, cornerRadius: 3), with: .color(color))
            }
            for rally in rallies {
                for cut in rally.cuts {
                    let kept = cut.included && effectiveIds.contains(cut.id)
                    let color = !kept ? CourtsidePalette.muted : reviewIds.contains(cut.id) ? EditorPalette.orange : EditorPalette.green
                    bar(.init(startMs: cut.keepStartMs, endMs: cut.keepEndMs), color: color.opacity(0.25))
                    bar(.init(startMs: cut.coreStartMs, endMs: cut.coreEndMs), color: color)
                }
            }
            for span in ignored { bar(.init(startMs: span.startMs, endMs: span.endMs), color: CourtsidePalette.control, y: 0, height: size.height) }
            if let startMs, playhead > startMs {
                bar(.init(startMs: startMs, endMs: playhead), color: (valid ? CourtsidePalette.accent : EditorPalette.danger).opacity(0.25), y: 0, height: size.height)
            }
            if let startMs, startMs >= window.startMs, startMs < window.endMs || last && startMs == window.endMs {
                let position = min(size.width - 2, max(0, x(startMs)))
                context.fill(Path(CGRect(x: position, y: 0, width: 2, height: size.height)), with: .color(CourtsidePalette.ink))
                context.fill(Path(CGRect(x: position, y: 0, width: min(9, size.width - position), height: 5)), with: .color(CourtsidePalette.ink))
            }
            if playhead >= window.startMs, playhead < window.endMs || last && playhead == window.endMs {
                let position = min(size.width - 2, max(0, x(playhead)))
                context.fill(Path(CGRect(x: position, y: 0, width: 2, height: size.height)), with: .color(CourtsidePalette.accent))
                context.fill(Path(ellipseIn: CGRect(x: position - 3, y: 0, width: 8, height: 8)), with: .color(CourtsidePalette.accent))
            }
        }.frame(minHeight: 44).background(CourtsidePalette.raised, in: RoundedRectangle(cornerRadius: 5))
            .overlay {
                GeometryReader { geometry in
                    let seek: (CGFloat) -> Void = { x in
                        let fraction = min(1, max(0, x / max(1, geometry.size.width)))
                        onSeek(window.startMs + Int64((fraction * Double(window.endMs - window.startMs)).rounded()))
                    }
                    Color.clear.contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded { seek($0.location.x) })
                        .simultaneousGesture(DragGesture(minimumDistance: 8)
                            .updating($horizontalScrub) { value, horizontal, _ in
                                if horizontal == nil { horizontal = abs(value.translation.width) > abs(value.translation.height) }
                            }
                            .onChanged { value in
                                if horizontalScrub ?? (abs(value.translation.width) > abs(value.translation.height)) {
                                    seek(value.location.x)
                                }
                            })
                }
            }
            .accessibilityElement().accessibilityLabel("Source timeline, \(deckTime(window.startMs)) to \(deckTime(window.endMs))")
            .accessibilityValue("Video \(deckTime(playhead))")
            .accessibilityAdjustableAction { direction in
                onSeek(min(window.endMs, max(window.startMs, playhead + (direction == .increment ? 1_000 : -1_000))))
            }
    }
}

func deckTime(_ value: Int64) -> String {
    let seconds = max(0, value) / 1_000
    return String(format: "%lld:%02lld.%01lld", seconds / 60, seconds % 60, (max(0, value) % 1_000) / 100)
}
