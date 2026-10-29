#if canImport(SwiftUI)
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif
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
    var expand = true
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(.system(size: 13, weight: .semibold))
            .frame(maxWidth: expand ? .infinity : nil, minHeight: 44)
            .foregroundColor(selected ? .white : destructive ? Color(red: 179 / 255.0, green: 38 / 255.0, blue: 30 / 255.0) : CourtsidePalette.ink)
            .background(RoundedRectangle(cornerRadius: 7).fill(selected ? CourtsidePalette.ink : .white))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(CourtsidePalette.control, lineWidth: selected ? 0 : 1))
            .opacity(enabled ? (configuration.isPressed ? 0.65 : 1) : 0.35)
            .contentShape(Rectangle())
    }
}

struct CourtsideDecisionRow: View {
    let selected: Bool?
    let onDecision: (Bool) -> Void
    var body: some View {
        HStack(spacing: 8) {
            Button("Keep") { onDecision(true) }.buttonStyle(CourtsideButtonStyle(selected: selected == true)).courtsideSelected(selected == true)
            Button("Remove") { onDecision(false) }.buttonStyle(CourtsideButtonStyle(selected: selected == false, destructive: true)).courtsideSelected(selected == false)
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
            Button("Near") { onSelect(0) }.buttonStyle(CourtsideButtonStyle(selected: selection == 0)).disabled(!hasMarker && !canAdd).courtsideSelected(selection == 0)
            Button("Far") { onSelect(1) }.buttonStyle(CourtsideButtonStyle(selected: selection == 1)).disabled(!hasMarker && !canAdd).courtsideSelected(selection == 1)
            Button("Not a serve") { onSelect(2) }.buttonStyle(CourtsideButtonStyle(selected: selection == 2)).disabled(!hasMarker).courtsideSelected(selection == 2)
        }
    }
}

private extension View {
    @ViewBuilder func courtsideSelected(_ selected: Bool) -> some View {
        #if canImport(SwiftUI)
        accessibilityAddTraits(selected ? .isSelected : [])
        #else
        self
        #endif
    }
}

struct CourtsideReviewNavigation: View {
    let previous: Bool, pending: Bool, next: Bool
    let onNeighbor: (Int) -> Void
    let onReview: () -> Void
    var body: some View {
        HStack(spacing: 6) {
            navigationButton("Previous rally", enabled: previous) { onNeighbor(-1) }
            navigationButton("Next review", enabled: pending, selected: pending, action: onReview)
            navigationButton("Next rally", enabled: next) { onNeighbor(1) }
        }
    }
    private func navigationButton(_ title: String, enabled: Bool, selected: Bool = false, action: @escaping () -> Void) -> some View {
        #if canImport(UIKit)
        CourtsideActionButton(title: title, enabled: enabled, selected: selected, action: action)
            .disabled(!enabled)
            .frame(maxWidth: .infinity).frame(height: 44)
        #else
        Button(action: action) {
            Text(title).font(.system(size: 13, weight: .semibold))
                .frame(maxWidth: .infinity, minHeight: 44)
                .foregroundColor(selected ? .white : CourtsidePalette.ink)
                .background(RoundedRectangle(cornerRadius: 7).fill(selected ? CourtsidePalette.ink : .white))
                .overlay(RoundedRectangle(cornerRadius: 7).stroke(CourtsidePalette.control, lineWidth: selected ? 0 : 1))
                .contentShape(Rectangle())
        }.buttonStyle(PlainButtonStyle()).disabled(!enabled).opacity(enabled ? 1 : 0.35)
        #endif
    }
}

#if canImport(UIKit)
/// Native controls keep their full action targets under interface scaling.
struct CourtsideActionButton: UIViewRepresentable {
    let title: String
    var imageName: String? = nil
    var identifier: String? = nil
    let enabled: Bool, selected: Bool
    var bordered = true
    let action: () -> Void
    func makeCoordinator() -> Coordinator { Coordinator(action) }
    func makeUIView(context: Context) -> UIButton {
        let button = UIButton(type: .custom)
        button.titleLabel?.font = .systemFont(ofSize: 13, weight: .semibold)
        button.titleLabel?.numberOfLines = 2
        button.titleLabel?.textAlignment = .center
        button.layer.cornerRadius = 7
        button.layer.borderColor = UIColor(CourtsidePalette.control).cgColor
        button.setPreferredSymbolConfiguration(.init(pointSize: 20), forImageIn: .normal)
        button.setContentHuggingPriority(.defaultLow, for: .horizontal)
        button.addTarget(context.coordinator, action: #selector(Coordinator.pressed), for: .touchUpInside)
        return button
    }
    func updateUIView(_ button: UIButton, context: Context) {
        context.coordinator.action = action
        if button.accessibilityLabel != title {
            button.setTitle(imageName == nil ? title : nil, for: .normal)
            button.setImage(imageName.flatMap { UIImage(systemName: $0) }, for: .normal)
            button.accessibilityLabel = title
        }
        button.accessibilityIdentifier = identifier
        button.isEnabled = enabled
        button.alpha = enabled ? 1 : 0.35
        button.setTitleColor(selected ? .white : UIColor(CourtsidePalette.ink), for: .normal)
        button.tintColor = .white
        button.backgroundColor = bordered ? (selected ? UIColor(CourtsidePalette.ink) : .white) : .clear
        button.layer.borderWidth = bordered && !selected ? 1 : 0
    }
    @MainActor final class Coordinator: NSObject {
        var action: () -> Void
        init(_ action: @escaping () -> Void) { self.action = action }
        @objc func pressed() { action() }
    }
}
#endif

struct CourtsideTrimNudges: View {
    let start: Bool
    let onNudge: (Int) -> Void
    var body: some View {
        VStack(alignment: start ? .leading : .trailing, spacing: 2) {
            Text(start ? "Start" : "End").font(.system(size: 11)).foregroundColor(CourtsidePalette.muted)
            HStack(spacing: 4) {
                nudgeButton(-1)
                nudgeButton(1)
            }
        }
    }
    private func nudgeButton(_ direction: Int) -> some View {
        Button { onNudge(direction) } label: {
            Text(direction < 0 ? "−1s" : "+1s").frame(width: 44, height: 44).contentShape(Rectangle())
        }.buttonStyle(CourtsideButtonStyle(expand: false)).frame(width: 44).contentShape(Rectangle())
        #if canImport(SwiftUI)
            .accessibilityLabel("Move rally \(start ? "start" : "end") one second \(direction < 0 ? "earlier" : "later")")
            .accessibilityIdentifier("deck\(start ? "Start" : "End")\(direction < 0 ? "Earlier" : "Later")")
        #endif
    }
}
