import SwiftUI
import AVKit

enum SetupPalette {
    static let paper = Color(red: 248 / 255, green: 247 / 255, blue: 238 / 255)
    static let ink = Color(red: 23 / 255, green: 56 / 255, blue: 46 / 255)
    static let muted = Color(red: 101 / 255, green: 118 / 255, blue: 111 / 255)
    static let green = Color(red: 47 / 255, green: 104 / 255, blue: 85 / 255)
    static let acid = Color(red: 217 / 255, green: 238 / 255, blue: 158 / 255)
    static let rail = Color(red: 200 / 255, green: 214 / 255, blue: 202 / 255)
    static let orange = Color(red: 222 / 255, green: 121 / 255, blue: 89 / 255)
}

/// Android Brand.kt uses the transparent icon and text, never the white splash logo.
struct VolleySpliceBrand: View {
    var compact = false
    var body: some View {
        HStack(spacing: 4) {
            Image("BrandMark").resizable().scaledToFit().frame(width: compact ? 28 : 36, height: compact ? 28 : 36)
            if !compact {
                (Text("volley").foregroundColor(Color(red: 5 / 255, green: 43 / 255, blue: 80 / 255)) +
                 Text("splice").foregroundColor(Color(red: 8 / 255, green: 117 / 255, blue: 245 / 255)))
                    .font(.system(size: 20, weight: .black)).italic().tracking(-0.75).lineLimit(1)
            }
        }.accessibilityElement(children: .ignore).accessibilityLabel("VolleySplice")
    }
}

/// The selector opens existing projects directly; importing a project file is a separate action.
struct SavedProjectMenuItems: View {
    let projects: [URL]
    let names: [URL: String]
    let dates: [URL: Date]
    let onNew: () -> Void
    let onSelect: (URL) -> Void
    var body: some View {
        Button("Start a new video", action: onNew).accessibilityIdentifier("startNewVideo")
        Divider()
        if projects.isEmpty { Text("No projects yet") }
        ForEach(projects, id: \.self) { url in
            Button { onSelect(url) } label: {
                if let date = dates[url] {
                    Text("\(names[url] ?? url.lastPathComponent) \u{00B7} \(date.formatted(date: .abbreviated, time: .shortened))")
                } else { Text(names[url] ?? url.lastPathComponent) }
            }.accessibilityIdentifier("project-\(url.lastPathComponent)")
        }
    }
}

struct NewProjectWorkspace: View {
    @ObservedObject var model: WorkspaceModel
    let chooseVideo: () -> Void, openSaved: () -> Void
    @State private var settings = false
    @State private var awaitingAnalysis = false
    @AppStorage(GuidedTourStore.key) private var tourState = ""
    var body: some View {
        GeometryReader { geometry in
            let desktop = EditorLayout.mode(width: Double(geometry.size.width), height: Double(geometry.size.height)) == .desktop
            VStack(spacing: 0) {
                ScrollViewReader { scroll in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 18) {
                            header(compact: !desktop)
                            if awaitingAnalysis {
                                SetupAnalysisProgress(queue: model.queue, sourceName: model.sourceName, sourceFingerprint: model.selectedFingerprint)
                                Button("Back to video setup") { awaitingAnalysis = false }
                                    .font(.system(size: 12)).frame(maxWidth: .infinity, minHeight: 40)
                            } else {
                                intro
                                sourceSetup(desktop: desktop)
                            }
                            VolleySpliceLegalFooter()
                        }.padding(desktop ? 24 : 14)
                            .frame(maxWidth: .infinity)
                    }
                    .onChange(of: tourState) { _, state in
                        if let step = GuidedTourStep(rawValue: state), step.stage == .setup {
                            scroll.scrollTo(step.target, anchor: .center)
                        }
                    }
                }
                if model.busy {
                    HStack { ProgressView(); Text(model.status); Spacer(); Button("Cancel", action: model.cancel) }
                        .font(.callout).padding(12)
                }
                Text(model.status).font(.caption).foregroundStyle(SetupPalette.muted)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 14).padding(.bottom, 6)
                    .accessibilityIdentifier("analysisStatus")
            }
            .background(SetupPalette.paper).foregroundStyle(SetupPalette.ink).tint(SetupPalette.green)
        }
        .onChange(of: model.source) { _, _ in awaitingAnalysis = false }
        .sheet(isPresented: $settings) { SetupSettings(model: model, dismiss: { settings = false }).interfaceScaled() }
    }
    private func header(compact: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                VolleySpliceBrand()
                Text("VIDEO EDITOR · v\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.1.0")")
                    .font(.system(size: 8, weight: .bold)).tracking(0.7).foregroundStyle(SetupPalette.muted)
                    .padding(5).background(SetupPalette.rail.opacity(0.25), in: RoundedRectangle(cornerRadius: 4))
                    .lineLimit(1).minimumScaleFactor(0.8)
                Spacer(minLength: 0)
                Button { settings = true } label: { Image(systemName: "gearshape").frame(width: 40, height: 40) }
                    .accessibilityLabel("Settings").accessibilityIdentifier("setupSettings")
            }
            HStack(alignment: .bottom, spacing: 8) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("CURRENT PROJECT").font(.system(size: 9, weight: .black)).tracking(0.9).foregroundStyle(SetupPalette.green)
                    Menu {
                        SavedProjectMenuItems(projects: model.projects, names: model.projectNames,
                            dates: model.projectDates, onNew: model.beginNewProject, onSelect: model.openProject)
                    } label: {
                        Text(model.sourceName ?? "Start a new video").font(.system(size: 13)).lineLimit(1)
                            .frame(maxWidth: .infinity, minHeight: 40)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(SetupPalette.rail))
                    }.accessibilityLabel("Current project, \(model.sourceName ?? "Start a new video")").accessibilityIdentifier("projectSelector")
                }
                Menu {
                    Button("＋ Start a new video…", action: model.beginNewProject)
                    Button("Open saved projects", action: openSaved)
                    Button("Processing queue") { model.showQueue = true }
                    Button("Manage storage") { if model.flushProject() { model.showStorage = true } }.accessibilityIdentifier("manageStorage")
                    Button("Restart guided tour") { GuidedTourStore.restart(stage: .setup) }
                        .accessibilityIdentifier("replayTutorial")
                } label: { Text("More").font(.system(size: 13)).frame(minWidth: 44, minHeight: 40) }
                    .accessibilityIdentifier("setupMore")
            }
        }.padding(.horizontal, 12).padding(.vertical, 10)
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(SetupPalette.rail))
    }
    private var intro: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Bump. Set. Splice.").font(.system(size: 10, weight: .black, design: .monospaced)).tracking(1).foregroundStyle(SetupPalette.green)
            Text("Choose the game.\nVolleySplice finds the rallies.")
                .font(.system(size: 30, weight: .black)).tracking(-0.8).fixedSize(horizontal: false, vertical: true)
            Text("Choose a game video, confirm the part to analyze, then let VolleySplice prepare the review timeline.")
                .font(.system(size: 13)).foregroundStyle(SetupPalette.muted)
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow { workflow(1, "Choose video"); workflow(2, "Set game window") }
                Divider().overlay(SetupPalette.rail).gridCellColumns(2)
                GridRow { workflow(3, "Analyze locally"); workflow(4, "Review rallies") }
            }.overlay(RoundedRectangle(cornerRadius: 6).stroke(SetupPalette.rail))
            Text("Your video stays on this device.").font(.system(size: 11)).foregroundStyle(SetupPalette.muted)
        }.padding(18).frame(maxWidth: .infinity, alignment: .leading)
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(SetupPalette.rail))
    }
    private func workflow(_ number: Int, _ label: String) -> some View {
        HStack(spacing: 7) {
            Text("\(number)").font(.system(size: 10, weight: .black, design: .monospaced))
                .frame(width: 24, height: 24).background(SetupPalette.acid, in: RoundedRectangle(cornerRadius: 5))
            Text(label).font(.system(size: 11, weight: .semibold))
        }.frame(maxWidth: .infinity, minHeight: 36, alignment: .leading).padding(.horizontal, 10)
    }
    private func sourceSetup(desktop: Bool) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("STEP 1 OF 3").font(.system(size: 11, weight: .bold)).foregroundStyle(SetupPalette.orange)
            Text(model.sourceName ?? "Choose your game video").font(.headline).fixedSize(horizontal: false, vertical: true)
            if model.sourceAccess?.photo != nil {
                Text("Linked to the original recording in Camera Roll. Photos trims and filters are not applied.")
                    .font(.footnote).foregroundStyle(SetupPalette.muted).accessibilityIdentifier("photoSourceLinked")
            }
            Text(model.media == nil ? "Pick a video from this device. Nothing will be uploaded." : "Ready to set up · your video stays on this device.")
                .font(.system(size: 12)).foregroundStyle(SetupPalette.muted)
            Button(model.media == nil ? "Choose video" : "Choose a different video", action: chooseVideo)
                .buttonStyle(SetupActionButtonStyle(primary: model.media == nil))
                .disabled(model.busy).accessibilityIdentifier("chooseRecording").guidedTourTarget("setup-source")
            if let media = model.media {
                SetupGameWindow(player: model.player, duration: media.duration, sourceID: model.sourceName ?? "",
                    start: $model.start, end: $model.end, desktop: desktop, enabled: !model.busy)
                    .guidedTourTarget("setup-window")
                Toggle(isOn: $model.generateSideSwitchMarkers) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Teams change court sides").font(.system(size: 13, weight: .semibold))
                        Text("VolleySplice will find side switches for the optional scoreboard.").font(.system(size: 12)).foregroundStyle(SetupPalette.muted)
                    }
                }.padding(10).overlay(RoundedRectangle(cornerRadius: 6).stroke(SetupPalette.rail))
                    .disabled(model.busy).accessibilityIdentifier("generateSideSwitchMarkers")
                Button("Find the rallies") { model.player.pause(); model.analyze(); awaitingAnalysis = model.error == nil }
                    .buttonStyle(SetupActionButtonStyle(primary: true))
                    .disabled(model.busy || !model.start.isFinite || !model.end.isFinite || model.end - model.start < 1)
                    .accessibilityIdentifier("analyzeRecording").guidedTourTarget("setup-create")
                Text("This may take a while for a long video. You can review another project while it runs.")
                    .font(.system(size: 11)).foregroundStyle(SetupPalette.muted)
            } else {
                Button("Open saved projects", action: openSaved).font(.system(size: 12))
                    .frame(maxWidth: .infinity, minHeight: 40).accessibilityIdentifier("openSavedProjects")
            }
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(SetupPalette.rail))
    }
}

struct VolleySpliceLegalFooter: View {
    @State private var showNotices = false
    var body: some View {
        HStack(spacing: 10) {
            Link("Privacy", destination: URL(string: "https://www.volleysplice.com/privacy.html")!)
            Text("·").foregroundStyle(SetupPalette.muted)
            Button("Open source") { showNotices = true }.accessibilityIdentifier("openSourceNotices")
        }.font(.system(size: 12)).frame(maxWidth: .infinity, minHeight: 44)
            .sheet(isPresented: $showNotices) {
                NavigationStack {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 14) {
                            Text("VolleySplice includes the following open-source components. Their licenses remain available to you under their original terms.")
                            Text("VolleySplice project code and project-owned assets — MIT License")
                            Text("OpenCV 4.12.0 — Apache License 2.0")
                            Text("Apple system frameworks, device codecs, and other platform components are provided under their respective system licenses.")
                                .font(.system(size: 12)).foregroundStyle(SetupPalette.muted)
                            Link("View VolleySplice MIT License", destination: URL(string: "https://github.com/vafrederico/volleysplice/blob/main/LICENSE")!)
                            Link("View all third-party notices", destination: URL(string: "https://github.com/vafrederico/volleysplice/blob/main/THIRD_PARTY_NOTICES.md")!)
                            Link("View Apache License 2.0", destination: URL(string: "https://www.apache.org/licenses/LICENSE-2.0")!)
                        }.frame(maxWidth: .infinity, alignment: .leading).padding()
                    }.background(SetupPalette.paper).foregroundStyle(SetupPalette.ink)
                        .navigationTitle("Open-source notices").navigationBarTitleDisplayMode(.inline)
                        .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Close") { showNotices = false } } }
                }.tint(SetupPalette.green).interfaceScaled()
            }
    }
}

private struct SetupActionButtonStyle: ButtonStyle {
    let primary: Bool
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.system(size: 13, weight: .semibold))
            .frame(maxWidth: .infinity, minHeight: 44)
            .foregroundStyle(primary ? SetupPalette.ink : SetupPalette.green)
            .background(primary ? SetupPalette.acid : SetupPalette.paper, in: Capsule())
            .overlay(Capsule().stroke(primary ? Color.clear : SetupPalette.rail))
            .opacity(enabled ? (configuration.isPressed ? 0.7 : 1) : 0.4)
    }
}

private struct SetupGameWindow: View {
    let player: AVPlayer, duration: Double, sourceID: String
    @Binding var start: Double
    @Binding var end: Double
    let desktop: Bool, enabled: Bool
    @State private var playhead = 0.0
    @State private var playing = false
    @State private var dragStart: Double?
    @AppStorage("desktop_editor_layout.setup_player_compact_height_dp") private var compactHeight = 176.0
    @AppStorage("desktop_editor_layout.setup_player_desktop_height_dp") private var desktopHeight = 320.0
    private let timer = Timer.publish(every: 0.2, on: .main, in: .common).autoconnect()
    private var height: Double { EditorLayout.playerHeight(desktop ? desktopHeight : compactHeight, desktop: desktop, setup: true) }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VideoPlayer(player: player).frame(height: height).background(.black).accessibilityIdentifier("videoPreview")
            Capsule().fill(SetupPalette.rail).frame(width: 70, height: 4).frame(maxWidth: .infinity, minHeight: 24)
                .contentShape(Rectangle()).gesture(DragGesture(minimumDistance: 2).onChanged { value in
                    if dragStart == nil { dragStart = height }
                    setHeight((dragStart ?? height) + Double(value.translation.height))
                }.onEnded { _ in dragStart = nil })
                .accessibilityLabel("Resize setup video player height")
                .accessibilityAdjustableAction { direction in setHeight(height + (direction == .increment ? 40 : -40)) }
            HStack(spacing: 8) {
                Button(playing ? "Pause" : "Play") { if playing { player.pause() } else { player.play() } }
                    .accessibilityIdentifier("setupPlayPause")
                Button("−10s") { seek(playhead - 10) }
                Button("+10s") { seek(playhead + 10) }
                Spacer(minLength: 0)
                Text("\(time(playhead)) / \(time(duration))").font(.system(size: 11, design: .monospaced))
            }.buttonStyle(.bordered).font(.system(size: 12)).disabled(!enabled)
            Slider(value: Binding(get: { min(duration, max(0, playhead)) }, set: seek), in: 0...max(0.001, duration))
                .disabled(!enabled).accessibilityLabel("Video position").accessibilityIdentifier("setupSeek")
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("STEP 2 OF 3").font(.system(size: 11, weight: .bold)).foregroundStyle(SetupPalette.orange)
                    Text("Choose the part with the game").font(.system(size: 14, weight: .bold))
                }
                Spacer(minLength: 4)
                Button("Use full video") { start = 0; end = duration }.font(.system(size: 12)).disabled(!enabled)
                    .accessibilityIdentifier("useFullVideo")
            }
            Text("Leave the full video selected, or mark where the game starts and ends.").font(.system(size: 12)).foregroundStyle(SetupPalette.muted)
            ViewThatFits(in: .horizontal) {
                HStack { boundsButtons }
                VStack(alignment: .leading) { boundsButtons }
            }.buttonStyle(.bordered).font(.system(size: 12)).disabled(!enabled)
            Text("\(time(end - start)) selected · \(time(duration - (end - start))) left out")
                .font(.system(size: 12, weight: .semibold)).foregroundStyle(SetupPalette.green)
        }
        .onReceive(timer) { _ in
            guard player.currentTime().seconds.isFinite else { return }
            playhead = min(duration, max(0, player.currentTime().seconds)); playing = player.rate != 0
        }
        .onChange(of: sourceID) { _, _ in playhead = 0; playing = false }
        .onDisappear { player.pause() }
    }
    @ViewBuilder private var boundsButtons: some View {
        Button("Game starts · \(time(start))") { start = min(max(0, playhead), max(0, end - 1)) }.accessibilityIdentifier("gameStart")
        Button("Game ends · \(time(end))") { end = duration >= start + 1 ? min(duration, max(start + 1, playhead)) : duration }.accessibilityIdentifier("gameEnd")
    }
    private func setHeight(_ value: Double) {
        let resolved = EditorLayout.playerHeight(value, desktop: desktop, setup: true)
        if desktop { desktopHeight = resolved } else { compactHeight = resolved }
    }
    private func seek(_ seconds: Double) { playhead = min(duration, max(0, seconds)); player.seek(to: CMTime(seconds: playhead, preferredTimescale: 60000), toleranceBefore: .zero, toleranceAfter: .zero) }
    private func time(_ seconds: Double) -> String { let value = max(0, seconds); return String(format: "%d:%05.2f", Int(value) / 60, value.truncatingRemainder(dividingBy: 60)) }
}

private struct SetupSettings: View {
    @ObservedObject var model: WorkspaceModel
    let dismiss: () -> Void
    @State private var showStorage = false
    @AppStorage("show_analysis_measurements") private var showMeasurements = false
    var body: some View {
        NavigationStack {
            Form {
                Section { InterfaceScaleControl() }
                Section {
                    Button("Manage storage") {
                        if model.flushProject() { showStorage = true }
                    }.accessibilityIdentifier("manageStorage")
                }
                Section {
                    Toggle("Show analysis measurements", isOn: $showMeasurements)
                    if showMeasurements, let result = model.result {
                        ForEach(result.stageSeconds.keys.sorted(), id: \.self) { name in
                            HStack { Text(name); Spacer(); Text("\(result.stageSeconds[name] ?? 0, specifier: "%.2f") s").monospacedDigit() }
                        }
                    }
                    Button("Replay tutorial") { dismiss(); GuidedTourStore.restart(stage: .setup) }.accessibilityIdentifier("replayTutorial")
                    Button("Clear feature cache", action: model.clearFeatureCache).disabled(model.busy || model.queue.hasPendingWork).accessibilityIdentifier("clearFeatureCache")
                }
                #if DEBUG
                Section {
                    DisclosureGroup("Developer checks") {
                        Button("Check Android rotation fixtures") { dismiss(); model.mediaCheck() }.accessibilityIdentifier("mediaCheck")
                        Button("Check canonical inference") { dismiss(); model.golden() }.accessibilityIdentifier("goldenCheck")
                        if model.source != nil { Button("Check native video export") { dismiss(); model.exportCheck() }.accessibilityIdentifier("videoExportCheck") }
                    }.accessibilityIdentifier("developerChecks").disabled(model.busy || model.queue.hasPendingWork)
                }
                #endif
            }.scrollContentBackground(.hidden).background(SetupPalette.paper)
                .navigationTitle("Settings").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done", action: dismiss) } }
        }.tint(SetupPalette.green)
            .sheet(isPresented: $showStorage) { AppStorageView(model: model).interfaceScaled() }
    }
}
