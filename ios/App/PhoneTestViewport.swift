import SwiftUI
import UIKit

private struct PhoneTestLayoutKey: EnvironmentKey { static let defaultValue: PhoneTestLayout? = nil }
extension EnvironmentValues {
    var phoneTestLayout: PhoneTestLayout? {
        get { self[PhoneTestLayoutKey.self] }
        set { self[PhoneTestLayoutKey.self] = newValue }
    }
}

/// Keep one content identity while changing its layout proposal, so toggling
/// testing mode preserves the selected project, editor stage and playback state.
struct PhoneTestViewport<Content: View>: View {
    @ViewBuilder var content: () -> Content
    #if DEBUG
    @AppStorage("debug.testAsIPhone") private var enabled = false
    @Environment(\.horizontalSizeClass) private var horizontal
    @Environment(\.verticalSizeClass) private var vertical
    var body: some View {
        GeometryReader { geometry in
            let window = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
                .flatMap(\.windows).first(where: \.isKeyWindow)
            let windowSize = window?.bounds.size ?? geometry.size
            let testing = enabled && UIDevice.current.userInterfaceIdiom == .pad
            let layout: PhoneTestLayout? = testing ? .init(landscape: windowSize.width > windowSize.height) : nil
            let width = layout?.width ?? geometry.size.width
            let height = layout?.height ?? geometry.size.height
            let fit = layout?.fit(width: geometry.size.width, height: geometry.size.height) ?? 1
            ZStack(alignment: .bottom) {
                (testing ? Color.black : Color.clear).ignoresSafeArea()
                content()
                    .environment(\.phoneTestLayout, layout)
                    .environment(\.horizontalSizeClass, testing ? .compact : horizontal)
                    .environment(\.verticalSizeClass, testing ? (layout?.landscape == true ? .compact : .regular) : vertical)
                    .frame(width: layout?.contentWidth ?? width, height: layout?.contentHeight ?? height)
                    .padding(EdgeInsets(top: layout?.top ?? 0, leading: layout?.leading ?? 0,
                                        bottom: layout?.bottom ?? 0, trailing: layout?.trailing ?? 0))
                    .frame(width: width, height: height)
                    .background(testing ? Color.black : Color.clear)
                    .contentShape(Rectangle())
                    .clipped().scaleEffect(fit)
                    .frame(width: geometry.size.width, height: geometry.size.height)
                if let layout {
                    Text("iPhone test · \(Int(layout.width)) × \(Int(layout.height)) pt")
                        .font(.system(size: 11)).foregroundStyle(.white).padding(4).background(.black)
                        .accessibilityIdentifier("phoneTestViewportStatus")
                        .allowsHitTesting(false)
                }
            }
        }
    }
    #else
    var body: some View { content() }
    #endif
}

#if DEBUG
struct PhoneTestControls: View {
    @AppStorage("debug.testAsIPhone") private var enabled = false
    var body: some View {
        if UIDevice.current.userInterfaceIdiom == .pad {
            Section("Developer testing") {
                Toggle("Test as iPhone", isOn: $enabled).accessibilityIdentifier("testAsIPhone")
                if enabled {
                    Text("390 × 844 pt. Rotate the iPad to test landscape.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
        }
    }
}
#endif
