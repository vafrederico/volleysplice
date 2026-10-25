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
    let onRate: (Float) -> Void
    let onTeam: (Bool, String) -> Void
    let onDecision: (Bool) -> Void
    let onServe: (ServeMarker?, ServingSide?) -> Void
    let onTrim: (Bool, Int64) -> Void
    let onNeighbor: (Int) -> Void
    let onReview: () -> Void
    let onSelect: (EditableRallyGroup) -> Void
    let onAdd: (TimeRange) -> Void
    @ViewBuilder var video: () -> Video
    @State private var adding = false
    @State private var teamEditing = false
    @State private var firstTeam = true
    @State private var teamName = ""
    @State private var listCenters: [String: CGFloat] = [:]
    @State private var splitDragStart: Double?
    @AppStorage("courtside.videoFraction") private var videoFraction = 0.56
    @Environment(\.compactPhoneEditor) private var phone
    @Environment(\.interfaceScale) private var interfaceScale
    @AppStorage(GuidedTourStore.key) private var tourState = ""
    private var rallies: [EditableRallyGroup] { presentation.groups }
    private var rally: EditableRallyGroup? { rallies.first { $0.id == navigation.selectedId } }
    private var index: Int { rallies.firstIndex { $0.id == rally?.id } ?? -1 }
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
                progress
                if landscape {
                    let available = max(1, geometry.size.width - (phone ? 8 : 24))
                    let playerWidth = max(240, min(available - 300, available * videoFraction))
                    HStack(spacing: 0) {
                        playerPane(compactScores: false).frame(width: phone ? max(240, geometry.size.width * 0.56) : playerWidth).contentShape(Rectangle()).clipped()
                        if !phone { videoDivider(available: available, current: playerWidth) }
                        else { Color.clear.frame(width: 8) }
                        detailPane.frame(maxWidth: .infinity).contentShape(Rectangle()).clipped()
                    }
                } else {
                    let playerHeight = phone || geometry.size.width < 600
                        ? min(360, max(170, geometry.size.height * 0.40))
                        : max(170, min(geometry.size.height * 0.62, geometry.size.height - 280))
                    playerPane(compactScores: phone).frame(height: playerHeight).contentShape(Rectangle()).clipped()
                    detailPane.frame(maxHeight: .infinity)
                }
            }
        }.background(CourtsidePalette.ground).foregroundStyle(CourtsidePalette.ink)
            .alert(firstTeam ? "Team 1" : "Team 2", isPresented: $teamEditing) {
                TextField("Team name", text: $teamName)
                Button("Save") { onTeam(firstTeam, teamName) }
                Button("Cancel", role: .cancel) {}
            }
            .sheet(isPresented: $adding) {
                MissingRallySheet(selection: .init(positionMs: playhead, bounds: bounds, occupied: occupied), bounds: bounds,
                    occupied: occupied, onSeek: onSeek, onCancel: { adding = false }, onAdd: { onAdd($0); adding = false })
                    .presentationDetents([.medium, .large]).interfaceScaled()
            }
    }
    private var progress: some View {
        let pending = Set(queue.map(\.id))
        let total = navigation.initialReviewKeys.union(pending).count
        let checked = max(0, total - pending.count)
        return VStack(spacing: 5) {
            HStack {
                Text(queue.isEmpty ? "Review complete" : "\(queue.count) need review")
                Spacer()
                Text("\(checked) / \(total) checked").monospacedDigit()
                Button(navigation.browsing ? "Review" : "All \(rallies.count)") { navigation.browsing.toggle() }
                    .frame(minWidth: 60, minHeight: 44).accessibilityIdentifier("deckBrowse")
            }.font(.system(size: 13)).foregroundStyle(CourtsidePalette.muted)
            ProgressView(value: total == 0 ? 1 : Double(checked) / Double(total)).tint(CourtsidePalette.accent)
        }.padding(.horizontal, 12).padding(.bottom, 6).guidedTourTarget("deck-progress")
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
    private func playerPane(compactScores: Bool) -> some View {
        VStack(spacing: 0) {
            if score.enabled { scoreBar(compact: compactScores) }
            ZStack(alignment: .top) {
                video().frame(maxWidth: .infinity, maxHeight: .infinity)
            }.background(.black).guidedTourTarget("deck-video")
            HStack(spacing: 10) {
                CourtsideActionButton(title: playing ? "Pause video" : "Play video",
                    imageName: playing ? "pause.fill" : "play.fill", identifier: "deckPlay",
                    enabled: sourceAvailable, selected: false, bordered: false, action: onPlay)
                    .frame(width: 44, height: 44)
                Text(deckTime(playhead)).monospacedDigit().font(.system(size: 13))
                Spacer(minLength: 0)
                Text(navigation.browsing ? "\(deckTime(bounds.endMs)) source" : String(format: "%.1f s", Double((rally?.coreEndMs ?? navigation.focus?.endMs ?? 0) - (rally?.coreStartMs ?? navigation.focus?.startMs ?? 0)) / 1_000))
                    .font(.system(size: 11)).foregroundStyle(.white.opacity(0.7)).lineLimit(1)
                Menu {
                    ForEach(EditorPlaybackPolicy.rates, id: \.self) { rate in
                        Button("\(rate.formatted())×") { onRate(rate) }
                    }
                } label: { Text("\(draft.playbackRate.formatted())×").font(.system(size: 13, weight: .semibold)).frame(minWidth: 44, minHeight: 44) }
                    .accessibilityLabel("Playback speed, \(draft.playbackRate.formatted()) times").accessibilityIdentifier("deckSpeed")
            }.padding(.horizontal, 10).foregroundStyle(.white).background(Color(white: 0.09))
        }
    }
    private func scoreBar(compact: Bool) -> some View {
        let snapshot = overlay.snapshot(at: playhead).score
        return HStack(spacing: 6) {
            teamButton(score.team1Name, first: true, compact: compact).frame(maxWidth: .infinity, alignment: .leading)
            Text("\(snapshot.team1Score) : \(snapshot.team2Score)").font(.system(size: compact ? 14 : 20, weight: .semibold)).monospacedDigit()
            teamButton(score.team2Name, first: false, compact: compact).frame(maxWidth: .infinity, alignment: .trailing)
        }.padding(.horizontal, 10).padding(.vertical, compact ? 0 : 10).foregroundStyle(.white)
            .background(Color(white: 0.08)).accessibilityIdentifier("deckScoreBar")
    }
    private func teamButton(_ name: String, first: Bool, compact: Bool) -> some View {
        Button { firstTeam = first; teamName = name; teamEditing = true } label: { Text(name).font(.system(size: compact ? 12 : 13)).lineLimit(compact ? 1 : 2).frame(minHeight: compact ? 30 : 44).contentShape(Rectangle()) }
            .buttonStyle(.plain).accessibilityLabel("Edit \(name)")
    }
    @ViewBuilder private var detailPane: some View {
        if navigation.browsing { browser.guidedTourTarget("deck-list") }
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
        CourtsideReviewNavigation(previous: neighbor(-1) != nil, pending: !queue.isEmpty, next: neighbor(1) != nil,
            onNeighbor: onNeighbor, onReview: onReview)
    }
    private func neighbor(_ direction: Int) -> EditableRallyGroup? {
        rally.flatMap { RallyReview.neighbor(of: $0, direction: direction, rallies: rallies, effectiveIds: presentation.keptIds) }
    }
    private var browser: some View {
        VStack(spacing: 4) {
            HStack {
                Text("\(rallies.count) rallies").font(.system(size: 16, weight: .semibold))
                Spacer()
                Button(navigation.showTimeline ? "Hide timeline" : "Show timeline") { navigation.showTimeline.toggle() }.frame(minHeight: 44)
            }.font(.system(size: 12))
            HStack {
                Text(RallyReview.at(playhead, in: rallies) == nil ? "Between rallies · \(deckTime(playhead))" : "Rally \((rallies.firstIndex { $0.id == navigation.selectedId } ?? 0) + 1) · \(deckTime(playhead))")
                    .lineLimit(1).frame(width: 125, alignment: .leading)
                Slider(value: Binding(get: { Double(playhead) }, set: { onSeek(Int64($0)) }),
                    in: Double(bounds.startMs)...Double(max(bounds.startMs + 1, bounds.endMs)))
                    .frame(minWidth: 60).tint(CourtsidePalette.accent)
                    .accessibilityLabel("Seek source video").accessibilityValue(deckTime(playhead)).accessibilityIdentifier("deckSourceSeek")
                Button("Follow video") { navigation.follow(time: playhead, rallies: rallies) }.disabled(navigation.following)
            }.font(.system(size: 11)).foregroundStyle(CourtsidePalette.muted)
            HStack(spacing: 8) {
                if navigation.showTimeline {
                    VStack(spacing: 2) {
                        Text(deckTime(bounds.startMs)).font(.system(size: 11))
                        CourtsideSourceRail(rallies: rallies, effectiveIds: presentation.keptIds, bounds: bounds, playhead: playhead,
                            listCenter: rallies.first { $0.id == navigation.listCenterId }?.coreStartMs, onSeek: onSeek)
                        Text(deckTime(bounds.endMs)).font(.system(size: 11))
                    }.frame(width: 62).accessibilityIdentifier("deckSourceTimeline")
                }
                rallyList
            }.frame(maxHeight: .infinity)
            HStack(spacing: 8) {
                CourtsideActionButton(title: index >= 0 ? "Edit rally \(index + 1)" : "Edit selected rally",
                    identifier: "deckEditSelected", enabled: rally != nil, selected: false) { navigation.browsing = false }
                    .frame(maxWidth: .infinity).frame(height: 44)
                CourtsideActionButton(title: "+ Add missing rally", identifier: "deckAddMissing",
                    enabled: true, selected: missing != nil) { adding = true }
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
                            Button { onSelect(value) } label: {
                                let snapshot = overlay.snapshot(at: value.coreStartMs).score
                                VStack(spacing: 6) {
                                    HStack { Text("Rally \(rowIndex + 1)").fontWeight(.semibold); Spacer(); Text(score.enabled ? "\(snapshot.team1Score) : \(snapshot.team2Score)" : "—") }
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
                                .accessibilityAddTraits(value.id == navigation.selectedId ? .isSelected : [])
                                .background { GeometryReader { row in Color.clear.preference(key: RallyRowCenters.self, value: [value.id: row.frame(in: .named("deck-list-scroll")).midY]) } }
                        }
                    }
                }.coordinateSpace(name: "deck-list-scroll")
                    .modifier(RallyListInteraction {
                        navigation.following = false
                        navigation.listCenterId = listCenters.min { abs($0.value - geometry.size.height / 2) < abs($1.value - geometry.size.height / 2) }?.key
                    })
                    .onPreferenceChange(RallyRowCenters.self) { centers in
                        listCenters = centers
                        guard !navigation.following else { return }
                        let id = centers.min { abs($0.value - geometry.size.height / 2) < abs($1.value - geometry.size.height / 2) }?.key
                        if id != navigation.listCenterId { navigation.listCenterId = id }
                    }
                    .onChange(of: navigation.followRequest) { _, _ in
                        if let id = navigation.listCenterId { proxy.scrollTo(id, anchor: .center) }
                    }
                    .onChange(of: playhead) { _, value in
                        guard navigation.following, let next = RallyReview.nearest(value, in: rallies), next.id != navigation.listCenterId else { return }
                        navigation.listCenterId = next.id; proxy.scrollTo(next.id, anchor: .center)
                    }
                    .onAppear { if let id = navigation.listCenterId ?? navigation.selectedId { proxy.scrollTo(id, anchor: .center) } }
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

private struct CourtsideSourceRail: View {
    let rallies: [EditableRallyGroup]
    let effectiveIds: Set<String>
    let bounds: TimeRange
    let playhead: Int64
    let listCenter: Int64?
    let onSeek: (Int64) -> Void
    var body: some View {
        GeometryReader { geometry in
            let height = max(1, geometry.size.height - 16)
            let duration = Double(max(1, bounds.endMs - bounds.startMs))
            let y: (Int64) -> CGFloat = { 8 + CGFloat(min(1, max(0, Double($0 - bounds.startMs) / duration))) * height }
            ZStack(alignment: .topLeading) {
                Capsule().fill(CourtsidePalette.control).frame(width: 3, height: height).offset(x: 25, y: 8)
                ForEach(rallies, id: \.id) { rally in
                    Rectangle().fill(RallyReview.retained(rally, effectiveIds: effectiveIds) ? EditorPalette.green : EditorPalette.danger)
                        .frame(width: 18, height: max(3, y(rally.keepEndMs) - y(rally.keepStartMs))).offset(x: 18, y: y(rally.keepStartMs))
                }
                Image(systemName: "circle.fill").font(.system(size: 9)).foregroundStyle(CourtsidePalette.accent).offset(x: 1, y: y(playhead) - 5)
                Rectangle().fill(CourtsidePalette.accent).frame(width: 26, height: 2).offset(y: y(playhead))
                if let listCenter {
                    Image(systemName: "diamond").font(.system(size: 12, weight: .bold)).offset(x: 40, y: y(listCenter) - 6)
                }
            }.frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading)
                .contentShape(Rectangle()).coordinateSpace(name: "deck-source-rail")
                .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("deck-source-rail")).onChanged { gesture in
                onSeek(bounds.startMs + Int64(min(1, max(0, Double(gesture.location.y - 8) / Double(height))) * duration))
            })
        }.accessibilityElement().accessibilityLabel("Source timeline. Circle: video. Diamond: list center.")
            .accessibilityValue("Video \(deckTime(playhead))" + (listCenter.map { "; List center \(deckTime($0))" } ?? ""))
            .accessibilityAdjustableAction { direction in onSeek(playhead + (direction == .increment ? 1_000 : -1_000)) }
    }
}

private struct MissingRallySheet: View {
    @State var selection: MissingRallySelection
    let bounds: TimeRange
    let occupied: [TimeRange]
    let onSeek: (Int64) -> Void
    let onCancel: () -> Void
    let onAdd: (TimeRange) -> Void
    private var valid: Bool {
        guard let range = selection.range else { return false }
        return range.endMs - range.startMs >= RallyReview.minimumDurationMs &&
            !occupied.contains { $0.startMs < range.endMs && range.startMs < $0.endMs }
    }
    var body: some View {
        NavigationStack {
            Form {
                Section("Source video") {
                    Text(deckTime(selection.positionMs)).monospacedDigit()
                    Slider(value: Binding(get: { Double(selection.positionMs) }, set: { value in
                        selection.seek(to: Int64(value), bounds: bounds, occupied: occupied)
                        onSeek(selection.positionMs)
                    }), in: Double(bounds.startMs)...Double(max(bounds.startMs + 1, bounds.endMs)))
                        .accessibilityLabel("Seek source video").accessibilityValue(deckTime(selection.positionMs))
                        .accessibilityIdentifier("deckAddSourceSeek")
                    Text("Seek to a gap, check the range, then confirm Add rally.").font(.callout)
                }
                Section("New rally") {
                    if let range = selection.range {
                    Text("Start \(deckTime(range.startMs))").monospacedDigit()
                    Slider(value: Binding(get: { Double(selection.range?.startMs ?? range.startMs) }, set: { selection.range?.startMs = Int64($0) }), in: Double(bounds.startMs)...Double(max(bounds.startMs + 1, range.endMs - 500)), step: 100)
                        .accessibilityLabel("New rally start").accessibilityValue(deckTime(range.startMs))
                    Text("End \(deckTime(range.endMs))").monospacedDigit()
                    Slider(value: Binding(get: { Double(selection.range?.endMs ?? range.endMs) }, set: { selection.range?.endMs = Int64($0) }), in: Double(min(bounds.endMs - 1, range.startMs + 500))...Double(bounds.endMs), step: 100)
                        .accessibilityLabel("New rally end").accessibilityValue(deckTime(range.endMs))
                    }
                    if !valid { Text("Choose a gap outside existing rallies and ignored footage.").foregroundStyle(EditorPalette.danger) }
                }
            }.navigationTitle("Add missing rally").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: onCancel) }
                    ToolbarItem(placement: .confirmationAction) { Button("Add rally") { if let range = selection.range { onAdd(range) } }.disabled(!valid) }
                }
        }
    }
}

func deckTime(_ value: Int64) -> String {
    let seconds = max(0, value) / 1_000
    return String(format: "%lld:%02lld.%01lld", seconds / 60, seconds % 60, (max(0, value) % 1_000) / 100)
}
