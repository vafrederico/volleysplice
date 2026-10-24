import Foundation

/// A logical phone viewport for debug device testing, independent of iPad pixels.
public struct PhoneTestLayout: Equatable, Sendable {
    public let landscape: Bool
    public var width: Double { landscape ? 844 : 390 }
    public var height: Double { landscape ? 390 : 844 }
    public var top: Double { landscape ? 0 : 47 }
    public var bottom: Double { landscape ? 21 : 34 }
    public var leading: Double { landscape ? 47 : 0 }
    public var trailing: Double { leading }
    public var contentWidth: Double { width - leading - trailing }
    public var contentHeight: Double { height - top - bottom }
    public init(landscape: Bool) { self.landscape = landscape }
    public func fit(width availableWidth: Double, height availableHeight: Double) -> Double {
        guard availableWidth.isFinite, availableHeight.isFinite else { return 1 }
        return max(0.1, min(1, availableWidth / width, availableHeight / height))
    }
}
