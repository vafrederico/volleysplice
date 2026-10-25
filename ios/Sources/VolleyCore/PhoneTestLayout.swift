import Foundation

/// Standard display-zoom logical sizes. Insets approximate each hardware family;
/// this remains an iPad-hosted app viewport rather than an iPhone OS runtime.
public enum PhoneTestDevice: String, CaseIterable, Sendable {
    case se3, iphone13, iphone16Pro, iphone16ProMax
    public static let defaultDevice = PhoneTestDevice.iphone13
    public var name: String {
        switch self {
        case .se3: return "iPhone SE (3rd gen)"
        case .iphone13: return "iPhone 13 / 14"
        case .iphone16Pro: return "iPhone 16 Pro"
        case .iphone16ProMax: return "iPhone 16 Pro Max"
        }
    }
    public var portraitWidth: Double {
        switch self { case .se3: return 375; case .iphone13: return 390; case .iphone16Pro: return 402; case .iphone16ProMax: return 440 }
    }
    public var portraitHeight: Double {
        switch self { case .se3: return 667; case .iphone13: return 844; case .iphone16Pro: return 874; case .iphone16ProMax: return 956 }
    }
    public var cameraInset: Double {
        switch self { case .se3: return 20; case .iphone13: return 47; case .iphone16Pro, .iphone16ProMax: return 62 }
    }
    public init(storedValue: String) { self = Self(rawValue: storedValue) ?? Self.defaultDevice }
}

/// A logical phone viewport for debug device testing, independent of iPad pixels.
public struct PhoneTestLayout: Equatable, Sendable {
    public let landscape: Bool
    public let device: PhoneTestDevice
    public var width: Double { landscape ? device.portraitHeight : device.portraitWidth }
    public var height: Double { landscape ? device.portraitWidth : device.portraitHeight }
    public var top: Double { landscape ? 0 : device.cameraInset }
    public var bottom: Double { device == .se3 ? 0 : landscape ? 21 : 34 }
    public var leading: Double { landscape && device != .se3 ? device.cameraInset : 0 }
    public var trailing: Double { leading }
    public var contentWidth: Double { width - leading - trailing }
    public var contentHeight: Double { height - top - bottom }
    public init(landscape: Bool, device: PhoneTestDevice = .defaultDevice) { self.landscape = landscape; self.device = device }
    public func fit(width availableWidth: Double, height availableHeight: Double) -> Double {
        guard availableWidth.isFinite, availableHeight.isFinite else { return 1 }
        return max(0.1, min(1, availableWidth / width, availableHeight / height))
    }
}
