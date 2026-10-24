#if canImport(SwiftUI)
import SwiftUI
#else
import TokamakCore
import TokamakStaticHTML
#endif

// Layout-only views shared with the local compatible-renderer preview. They do
// not import AVPlayer, UIKit, project storage, or synthetic production data.
enum CourtsidePalette {
    static let ground = Color(red: 238 / 255.0, green: 243 / 255.0, blue: 244 / 255.0)
    static let ink = Color(red: 19 / 255.0, green: 20 / 255.0, blue: 22 / 255.0)
    static let muted = Color(red: 93 / 255.0, green: 95 / 255.0, blue: 96 / 255.0)
    static let accent = Color(red: 27 / 255.0, green: 108 / 255.0, blue: 128 / 255.0)
    static let control = Color(red: 205 / 255.0, green: 206 / 255.0, blue: 209 / 255.0)
    static let raised = Color(red: 244 / 255.0, green: 246 / 255.0, blue: 248 / 255.0)
}

struct CourtsideButtonStyle: ButtonStyle {
    var selected = false
    var destructive = false
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.system(size: 13, weight: .semibold))
            .frame(maxWidth: .infinity, minHeight: 44)
            .foregroundColor(selected ? .white : destructive ? Color(red: 179 / 255.0, green: 38 / 255.0, blue: 30 / 255.0) : CourtsidePalette.ink)
            .background(RoundedRectangle(cornerRadius: 7).fill(selected ? CourtsidePalette.ink : .white))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(CourtsidePalette.control, lineWidth: selected ? 0 : 1))
            .opacity(enabled ? (configuration.isPressed ? 0.65 : 1) : 0.35)
    }
}

struct CourtsideDecisionRow: View {
    let selected: Bool?
    let onDecision: (Bool) -> Void
    var body: some View {
        HStack(spacing: 8) {
            Button("Keep") { onDecision(true) }.buttonStyle(CourtsideButtonStyle(selected: selected == true))
            Button("Remove") { onDecision(false) }.buttonStyle(CourtsideButtonStyle(selected: selected == false, destructive: true))
        }
    }
}

struct CourtsideServeRow: View {
    /// 0: Near, 1: Far, 2: Not a serve; nil: unresolved.
    let selection: Int?
    let canAdd: Bool
    let hasMarker: Bool
    let onSelect: (Int) -> Void
    var body: some View {
        HStack(spacing: 6) {
            Button("Near") { onSelect(0) }.buttonStyle(CourtsideButtonStyle(selected: selection == 0)).disabled(!hasMarker && !canAdd)
            Button("Far") { onSelect(1) }.buttonStyle(CourtsideButtonStyle(selected: selection == 1)).disabled(!hasMarker && !canAdd)
            Button("Not a serve") { onSelect(2) }.buttonStyle(CourtsideButtonStyle(selected: selection == 2)).disabled(!hasMarker)
        }
    }
}

struct CourtsideReviewNavigation: View {
    let previous: Bool, pending: Bool, next: Bool
    let onNeighbor: (Int) -> Void
    let onReview: () -> Void
    var body: some View {
        HStack(spacing: 6) {
            Button("Previous rally") { onNeighbor(-1) }.buttonStyle(CourtsideButtonStyle()).disabled(!previous)
            Button("Next review", action: onReview).buttonStyle(CourtsideButtonStyle(selected: pending)).disabled(!pending)
            Button("Next rally") { onNeighbor(1) }.buttonStyle(CourtsideButtonStyle()).disabled(!next)
        }
    }
}

struct CourtsideTrimNudges: View {
    let start: Bool
    let onNudge: (Int) -> Void
    var body: some View {
        VStack(alignment: start ? .leading : .trailing, spacing: 2) {
            Text(start ? "Start" : "End").font(.system(size: 11)).foregroundColor(CourtsidePalette.muted)
            HStack(spacing: 4) {
                ForEach([-1, 1], id: \.self) { direction in
                    Button(direction < 0 ? "−1s" : "+1s") { onNudge(direction) }
                        .buttonStyle(CourtsideButtonStyle()).frame(width: 44)
                }
            }
        }
    }
}
