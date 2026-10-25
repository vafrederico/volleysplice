import SwiftUI

private struct CoreTrimHeight: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

/// The deck supplies a uniform source-time context including retained neighbors.
/// The full editor can still request its existing core detail scale.
struct CoreTrimSurface<Track: View>: View {
    let core: TimeRange
    let kept: TimeRange
    let bounds: TimeRange
    let playhead: Int64
    let startLabel: String, endLabel: String
    let startIdentifier: String, endIdentifier: String
    var color = EditorPalette.green
    var contextWindow: TimeRange? = nil
    var halfHeightHandles = false
    let onTrim: (Bool, Int64) -> Void
    @ViewBuilder var track: (RallyTrimScale) -> Track
    @State private var drag: RallyTrimDrag?
    @State private var contentHeight: CGFloat = 100
    var body: some View {
        GeometryReader { geometry in
            let scale = drag?.scale ?? RallyTrimScale(core: core, bounds: bounds, width: geometry.size.width, context: contextWindow)
            let handles = scale.handles(for: core, spreadForTouch: contextWindow == nil)
            let hidden = scale.hiddenPadding(kept)
            VStack(spacing: 4) {
                HStack {
                    Text(deckTime(scale.window.startMs)); Spacer()
                    Text(deckTime(scale.window.endMs))
                }.font(.system(size: 11)).monospacedDigit().foregroundStyle(EditorPalette.muted)
                ZStack(alignment: .topLeading) {
                    track(scale).frame(height: 64).clipped()
                    if handles.spreadForTouch {
                        Canvas { context, _ in
                            for (time, target) in [(core.startMs, handles.startX), (core.endMs, handles.endX)] {
                                let exact = min(scale.width - scale.inset, max(scale.inset, scale.x(time)))
                                var line = Path()
                                line.move(to: CGPoint(x: target, y: 32)); line.addLine(to: CGPoint(x: exact, y: 58))
                                line.move(to: CGPoint(x: exact, y: 52)); line.addLine(to: CGPoint(x: exact, y: 64))
                                context.stroke(line, with: .color(color), lineWidth: 1.5)
                            }
                        }.allowsHitTesting(false)
                    }
                    ForEach([true, false], id: \.self) { start in
                        let value = start ? core.startMs : core.endMs
                        trimHandle(start: start, value: value, x: start ? handles.startX : handles.endX, scale: scale)
                    }
                }.frame(height: 64).coordinateSpace(name: "coreTrimSurface")
                if handles.spreadForTouch {
                    note("Handles spread for touch; ticks show exact bounds.")
                }
                if hidden.beforeMs > 0 || hidden.afterMs > 0 {
                    note(String(format: "Padding outside view: %.2fs before · %.2fs after", Double(hidden.beforeMs) / 1_000, Double(hidden.afterMs) / 1_000))
                }
                if core.startMs < scale.window.startMs || core.endMs > scale.window.endMs {
                    note("Core continues outside this view; release to refit.")
                }
                if playhead < scale.window.startMs || playhead > scale.window.endMs {
                    note("Video \(playhead < scale.window.startMs ? "before" : "after") view · \(deckTime(playhead))")
                }
            }.frame(maxWidth: .infinity).fixedSize(horizontal: false, vertical: true)
                .background { GeometryReader { size in Color.clear.preference(key: CoreTrimHeight.self, value: size.size.height) } }
        }.frame(minWidth: 88).frame(height: contentHeight)
            .onPreferenceChange(CoreTrimHeight.self) { value in
                if value > 0, abs(value - contentHeight) > 0.5 { contentHeight = value }
            }
    }
    private func trimHandle(start: Bool, value: Int64, x: Double, scale: RallyTrimScale) -> some View {
        RoundedRectangle(cornerRadius: 4).fill(color)
            .frame(width: 10, height: halfHeightHandles ? 28 : 44)
            .overlay { RoundedRectangle(cornerRadius: 4).stroke(.white, lineWidth: 2) }
            .padding(.bottom, halfHeightHandles ? 2 : 0)
            .frame(width: RallyTrimScale.handleWidth, height: 64, alignment: halfHeightHandles ? .bottom : .center).contentShape(Rectangle())
            .offset(x: x - scale.inset)
            .highPriorityGesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("coreTrimSurface")).onChanged { gesture in
                if drag == nil { drag = .init(scale: scale, initialValueMs: value, isStart: start) }
                if let drag { onTrim(drag.isStart, drag.value(translationX: gesture.translation.width)) }
            }.onEnded { _ in drag = nil })
            .accessibilityElement().accessibilityLabel(start ? startLabel : endLabel)
            .accessibilityIdentifier(start ? startIdentifier : endIdentifier)
            .accessibilityValue(String(format: "%.3f seconds", Double(value) / 1_000))
            .accessibilityAdjustableAction { direction in onTrim(start, value + (direction == .increment ? 100 : -100)) }
    }
    private func note(_ text: String) -> some View {
        Text(text).font(.system(size: 11)).foregroundStyle(EditorPalette.muted)
            .frame(maxWidth: .infinity, alignment: .leading).fixedSize(horizontal: false, vertical: true)
    }
}
