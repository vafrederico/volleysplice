import SwiftUI

enum GuidedTourStore {
    static let key = "volleycut-guided-tour-v2.state"
    static func restart(stage: GuidedTourStage) {
        var progress = GuidedTourProgress(); progress.restart(stage: stage)
        UserDefaults.standard.set(progress.stored, forKey: key)
    }
}

struct GuidedTourTargets: PreferenceKey {
    static var defaultValue: [String: Anchor<CGRect>] = [:]
    static func reduce(value: inout [String: Anchor<CGRect>], nextValue: () -> [String: Anchor<CGRect>]) {
        value.merge(nextValue(), uniquingKeysWith: { _, latest in latest })
    }
}
extension View {
    func guidedTourTarget(_ name: String) -> some View {
        anchorPreference(key: GuidedTourTargets.self, value: .bounds) { [name: $0] }
            .id(name)
    }
    func guidedTour(stage: GuidedTourStage, sourceReady: Bool, scoreTrackingEnabled: Bool) -> some View {
        modifier(GuidedTourHost(stage: stage, sourceReady: sourceReady, scoreTrackingEnabled: scoreTrackingEnabled))
    }
}

private struct GuidedTourHost: ViewModifier {
    let stage: GuidedTourStage
    let sourceReady: Bool
    let scoreTrackingEnabled: Bool
    @AppStorage(GuidedTourStore.key) private var stored = ""
    private var progress: GuidedTourProgress { GuidedTourProgress(stored: stored.isEmpty ? nil : stored) }
    func body(content: Content) -> some View {
        content.overlayPreferenceValue(GuidedTourTargets.self) { targets in
            GeometryReader { geometry in
                if let step = progress.current(stage: stage) {
                    let frame = targets[step.target].map { geometry[$0] }
                    GuidedTourCard(step: step, frame: frame, size: geometry.size,
                        sourceReady: sourceReady, scoreTrackingEnabled: scoreTrackingEnabled,
                        close: { stored = "dismissed" }, next: {
                            var next = progress
                            next.advance(stage: stage, sourceReady: sourceReady, scoreTrackingEnabled: scoreTrackingEnabled)
                            stored = next.stored ?? "done"
                        })
                }
            }
        }
        .onAppear(perform: enter)
        .onChange(of: stage) { _, _ in enter() }
        .onChange(of: scoreTrackingEnabled) { _, _ in enter() }
    }
    private func enter() {
        var next = progress; next.enter(stage: stage, scoreTrackingEnabled: scoreTrackingEnabled)
        if stored != next.stored { stored = next.stored ?? "" }
    }
}

private struct GuidedTourCard: View {
    let step: GuidedTourStep
    let frame: CGRect?
    let size: CGSize
    let sourceReady: Bool, scoreTrackingEnabled: Bool
    let close: () -> Void, next: () -> Void
    private let orange = Color(red: 239 / 255, green: 91 / 255, blue: 53 / 255)
    private var steps: [GuidedTourStep] { GuidedTourStep.visibleSteps(stage: step.stage, scoreTrackingEnabled: scoreTrackingEnabled) }
    private var canAdvance: Bool { step != .setupCreate && (step != .setupSource || sourceReady) }
    var body: some View {
        ZStack(alignment: .topLeading) {
            if let frame, frame.intersects(CGRect(origin: .zero, size: size)) {
                RoundedRectangle(cornerRadius: 4).stroke(orange, lineWidth: 2)
                    .frame(width: max(0, frame.width + 6), height: max(0, frame.height + 6))
                    .position(x: frame.midX, y: frame.midY).allowsHitTesting(false)
            }
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("TUTORIAL · \((steps.firstIndex(of: step) ?? 0) + 1) OF \(steps.count)")
                        .font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(orange)
                    Spacer()
                    Button("Close", action: close).accessibilityIdentifier("tutorialClose")
                }
                Text(step.title).font(.headline).accessibilityIdentifier("tutorialTitle")
                Text(step.body).font(.system(size: 13)).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 3) {
                    ForEach(steps.indices, id: \.self) { index in
                        Rectangle().fill(index <= (steps.firstIndex(of: step) ?? 0) ? orange : Color.gray.opacity(0.2)).frame(height: 3)
                    }
                }
                HStack {
                    Button("Skip tour", action: close).accessibilityIdentifier("tutorialSkip")
                    Spacer()
                    Button(step == .setupSource && sourceReady ? "Next" : step.action, action: next)
                        .disabled(!canAdvance).buttonStyle(.borderedProminent).tint(orange)
                        .accessibilityIdentifier("tutorialNext")
                }.font(.system(size: 12, weight: .semibold))
            }
            .padding(14).frame(width: min(350, max(250, size.width - 24)))
            .background(Color(red: 248 / 255, green: 247 / 255, blue: 238 / 255))
            .foregroundStyle(Color(red: 32 / 255, green: 32 / 255, blue: 30 / 255))
            .overlay(Rectangle().stroke(.primary, lineWidth: 1))
            .shadow(color: .black.opacity(0.12), radius: 8, y: 3)
            .padding(12).frame(maxWidth: .infinity, maxHeight: .infinity,
                               alignment: (frame?.midY ?? 0) > size.height * 0.45 ? .topTrailing : .bottomTrailing)
        }
    }
}
