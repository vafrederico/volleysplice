import Foundation

/// App display preference; deliberately independent of project/export state.
public enum InterfaceScale {
    public static let preferenceKey = "interface_scale"
    public static let customPreferenceKey = "interface_scale_is_custom"
    public static let range = 0.60...1.25
    public static let step = 0.05
    public static let defaultValue = 1.0
    public static func automatic(isPhone: Bool, width: Double, height: Double) -> Double {
        isPhone && width > height ? 0.75 : defaultValue
    }
    public static func hasOverride(_ preference: Double, custom: Bool) -> Bool {
        custom || abs(normalized(preference) - defaultValue) > 0.001
    }
    public static func resolved(_ preference: Double, custom: Bool, automatic: Double) -> Double {
        hasOverride(preference, custom: custom) ? normalized(preference) : automatic
    }
    public static func normalized(_ value: Double) -> Double {
        guard value.isFinite else { return defaultValue }
        let clamped = min(range.upperBound, max(range.lowerBound, value))
        return min(range.upperBound, max(range.lowerBound, (clamped / step).rounded() * step))
    }
}
