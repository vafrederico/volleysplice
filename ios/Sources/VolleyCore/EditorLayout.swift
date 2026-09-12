import Foundation

/// Android EditorActivity layout bounds expressed in SwiftUI logical points.
/// SwiftUI drag translations already use points; do not divide them by display scale.
public enum EditorLayout {
    public enum Mode: Equatable, Sendable { case compact, desktop }
    public enum Sidebar: Equatable, Sendable { case left, right }
    public struct SidebarWidths: Equatable, Sendable {
        public var left: Double, right: Double, leftMaximum: Double, rightMaximum: Double
    }
    public static func mode(width: Double, height: Double) -> Mode {
        let wideWindow = width >= 840 && height >= 360
        // Landscape phones lose usable width to the camera/home safe areas.
        // Keep portrait compact, but allow their short, wide editor workspace.
        let landscapePhone = width > height && width >= 640 && height >= 300
        return wideWindow || landscapePhone ? .desktop : .compact
    }
    public static func playerBounds(desktop: Bool, setup: Bool = false) -> ClosedRange<Double> {
        (desktop && !setup ? 180.0 : 140.0)...(desktop ? 1200.0 : 720.0)
    }
    public static func playerHeight(_ height: Double, desktop: Bool, setup: Bool = false) -> Double {
        clamp(height, to: playerBounds(desktop: desktop, setup: setup))
    }
    /// Automatic sizing keeps the timeline visible; an explicit manual size
    /// may make the center pane scroll without changing the initial layout.
    public static func phoneLandscapePlayerHeight(preferred: Double, paneHeight: Double, timelineHeight: Double, chromeHeight: Double, manual: Bool = false) -> Double {
        if manual { return clamp(preferred, to: 48...1200) }
        // Phone center: 12pt outer padding, 4pt section gap, 2pt video/row gap.
        let available = max(48, paneHeight - timelineHeight - chromeHeight - 18)
        return min(max(48, preferred), available)
    }
    public static func resizedPlayerHeight(current: Double, translation: Double, desktop: Bool, setup: Bool = false) -> Double {
        playerHeight(current + translation, desktop: desktop, setup: setup)
    }
    /// The opposite-notch extension is added after resolving the saved base
    /// width. Count it toward the phone minimum, retaining a 100-point base.
    public static func phoneRightSidebarMinimum(reclaimedSpace: Double) -> Double {
        max(100, 180 - max(0, reclaimedSpace))
    }
    /// Preserve Android's center/chrome budget and iPad right minimum by default.
    /// Callers pass stored widths without persisting these viewport-only clamps.
    public static func sidebarWidths(availableWidth: Double, left: Double, right: Double, rightMinimum: Double = 220) -> SidebarWidths {
        let minimum = clamp(rightMinimum, to: 0...560)
        let budget = max(180 + minimum, availableWidth - 220 - 64)
        // Subtracting a fractional notch allowance can round just below the
        // lower bound. Keep every closed range valid at the exact minimum.
        let resolvedLeft = clamp(left, to: 180...max(180, min(520, budget - minimum)))
        let resolvedRight = clamp(right, to: minimum...max(minimum, min(560, budget - resolvedLeft)))
        return SidebarWidths(left: resolvedLeft, right: resolvedRight,
                             leftMaximum: max(180, min(520, budget - resolvedRight)),
                             rightMaximum: max(minimum, min(560, budget - resolvedLeft)))
    }
    /// The right handle's positive translation shrinks its pane; the left expands.
    public static func resizedSidebarWidth(current: Double, translation: Double, side: Sidebar, maximum: Double, rightMinimum: Double = 220) -> Double {
        let minimum = side == .left ? 180.0 : clamp(rightMinimum, to: 0...560)
        return clamp(current + (side == .left ? translation : -translation),
                     to: minimum...max(minimum, maximum))
    }
    /// The Android-style events column scrolls as a whole. Give score controls
    /// their measured natural height, then fill remaining room with clips. The
    /// register retains its heading and at least three 44-point rows when the
    /// whole column must scroll in a short window. Available height excludes
    /// the column's outer padding; 51 points cover its heading/divider/spacings.
    public static func clipRegisterHeight(availableHeight: Double, scoreContentHeight: Double) -> Double {
        max(180, availableHeight - max(0, scoreContentHeight) - 51)
    }
    private static func clamp(_ value: Double, to range: ClosedRange<Double>) -> Double {
        min(range.upperBound, max(range.lowerBound, value))
    }
}
