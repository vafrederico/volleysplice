import SwiftUI
import UIKit

private struct InterfaceScaleKey: EnvironmentKey { static let defaultValue = 1.0 }
private struct InterfaceDefaultScaleKey: EnvironmentKey { static let defaultValue = 1.0 }
extension EnvironmentValues {
    var interfaceDefaultScale: Double {
        get { self[InterfaceDefaultScaleKey.self] }
        set { self[InterfaceDefaultScaleKey.self] = newValue }
    }
    var interfaceScale: Double {
        get { self[InterfaceScaleKey.self] }
        set { self[InterfaceScaleKey.self] = newValue }
    }
}

/// Like Android's density setting, change the logical layout proposal as well
/// as the rendered size. Local gesture coordinates stay in those logical points.
/// Apply once per app-owned presentation root; system pickers remain native.
private struct InterfaceScaleModifier: ViewModifier {
    var isWindow = false
    @AppStorage(InterfaceScale.preferenceKey) private var preference = InterfaceScale.defaultValue
    @AppStorage(InterfaceScale.customPreferenceKey) private var custom = false
    @Environment(\.interfaceDefaultScale) private var inheritedDefault
    func body(content: Content) -> some View {
        GeometryReader { geometry in
            // Keyboard avoidance changes the content height, not the window's
            // orientation. Keep typing from changing the automatic UI size.
            let windowSize = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
                .flatMap(\.windows).first(where: \.isKeyWindow)?.bounds.size ?? geometry.size
            let automatic = isWindow ? InterfaceScale.automatic(isPhone: UIDevice.current.userInterfaceIdiom == .phone,
                width: windowSize.width, height: windowSize.height) : inheritedDefault
            let scale = InterfaceScale.resolved(preference, custom: custom, automatic: automatic)
            content
                .environment(\.interfaceDefaultScale, automatic)
                .environment(\.interfaceScale, scale)
                .frame(width: geometry.size.width / scale, height: geometry.size.height / scale)
                .scaleEffect(scale, anchor: .topLeading)
                .frame(width: geometry.size.width, height: geometry.size.height, alignment: .topLeading)
        }
    }
}

extension View {
    func interfaceScaled(isWindow: Bool = false) -> some View { modifier(InterfaceScaleModifier(isWindow: isWindow)) }
}

struct InterfaceScaleControl: View {
    @AppStorage(InterfaceScale.preferenceKey) private var preference = InterfaceScale.defaultValue
    @AppStorage(InterfaceScale.customPreferenceKey) private var custom = false
    @Environment(\.interfaceDefaultScale) private var automatic
    @State private var pendingScale: Double?
    private var scale: Double { pendingScale ?? InterfaceScale.resolved(preference, custom: custom, automatic: automatic) }
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("UI size").fontWeight(.semibold)
                Spacer()
                Text("\(Int((scale * 100).rounded()))%").font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundStyle(EditorPalette.green).accessibilityIdentifier("uiScalePercent")
            }
            Text("Scales the interface on top of iOS screen sizing. Use a smaller size to fit more in landscape.")
                .font(.system(size: 10)).foregroundStyle(EditorPalette.muted)
            // Keep the sheet and its presenting window still until release.
            InterfaceScaleSlider(value: scale, onChange: { pendingScale = InterfaceScale.normalized($0) }, onCommit: {
                preference = InterfaceScale.normalized($0); custom = true; pendingScale = nil
            }, onCancel: { pendingScale = nil })
                .frame(maxWidth: .infinity).frame(height: 31)
            HStack {
                Text("Smaller")
                Spacer()
                Button("Default") { custom = false; preference = InterfaceScale.defaultValue }
                    .buttonStyle(.borderless).disabled(!InterfaceScale.hasOverride(preference, custom: custom))
                    .accessibilityIdentifier("uiScaleDefault")
                Spacer()
                Text("Larger")
            }.font(.system(size: 10)).foregroundStyle(EditorPalette.muted)
        }.onDisappear { pendingScale = nil }
    }
}

/// UIKit reports touch-up outside and cancellation explicitly. SwiftUI Slider
/// can lose its editing-end callback as the surrounding interface resizes.
private struct InterfaceScaleSlider: UIViewRepresentable {
    let value: Double
    let onChange: (Double) -> Void
    let onCommit: (Double) -> Void
    let onCancel: () -> Void
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIView(context: Context) -> UISlider {
        let slider = UISlider()
        slider.minimumValue = Float(InterfaceScale.range.lowerBound)
        slider.maximumValue = Float(InterfaceScale.range.upperBound)
        slider.tintColor = UIColor(EditorPalette.green)
        slider.accessibilityIdentifier = "uiScale"
        slider.accessibilityLabel = "UI size"
        slider.setContentHuggingPriority(.defaultLow, for: .horizontal)
        slider.addTarget(context.coordinator, action: #selector(Coordinator.began), for: .touchDown)
        slider.addTarget(context.coordinator, action: #selector(Coordinator.changed), for: .valueChanged)
        slider.addTarget(context.coordinator, action: #selector(Coordinator.ended), for: [.touchUpInside, .touchUpOutside])
        slider.addTarget(context.coordinator, action: #selector(Coordinator.cancelled), for: .touchCancel)
        return slider
    }
    func updateUIView(_ slider: UISlider, context: Context) {
        context.coordinator.parent = self
        if !slider.isTracking { slider.value = Float(value) }
        slider.accessibilityValue = "\(Int((value * 100).rounded()))%"
    }
    @MainActor final class Coordinator: NSObject {
        var parent: InterfaceScaleSlider
        init(_ parent: InterfaceScaleSlider) { self.parent = parent }
        @objc func began(_ slider: UISlider) { parent.onChange(Double(slider.value)) }
        @objc func changed(_ slider: UISlider) {
            if slider.isTracking { parent.onChange(Double(slider.value)) }
            else { parent.onCommit(Double(slider.value)) } // VoiceOver adjustment
        }
        @objc func ended(_ slider: UISlider) {
            slider.value = Float(InterfaceScale.normalized(Double(slider.value)))
            parent.onCommit(Double(slider.value))
        }
        @objc func cancelled(_ slider: UISlider) { parent.onCancel() }
    }
}
