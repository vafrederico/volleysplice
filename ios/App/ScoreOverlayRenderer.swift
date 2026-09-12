import Foundation
import SwiftUI
import UIKit

/// Immutable, top-row-first premultiplied BGRA shared by SwiftUI and the YUV
/// compositor. No UIKit, Core Animation or drawing occurs on an export frame.
final class ScoreOverlayRaster: NSObject, @unchecked Sendable {
    let pixels: Data
    let image: CGImage
    let width: Int, height: Int, scoreWidth: Int
    var rowBytes: Int { width * 4 }
    init(pixels: Data, image: CGImage, width: Int, height: Int, scoreWidth: Int) {
        self.pixels = pixels; self.image = image; self.width = width; self.height = height; self.scoreWidth = scoreWidth
    }
}

@MainActor enum ScoreOverlayRenderer {
    private static let cache: NSCache<NSString, ScoreOverlayRaster> = {
        let cache = NSCache<NSString, ScoreOverlayRaster>(); cache.totalCostLimit = 24 * 1024 * 1024; cache.countLimit = 64
        return cache
    }()

    static func raster(snapshot: ScoreOverlaySnapshot, videoWidth: Int, videoHeight: Int, renderTimeline: Bool) throws -> ScoreOverlayRaster {
        guard videoWidth > 0, videoHeight > 0 else { throw ProjectError.invalid("Invalid score overlay size") }
        let layout = ScoreOverlay.layout(videoWidth: videoWidth, videoHeight: videoHeight, snapshot: snapshot) { text, size in
            Double((text as NSString).size(withAttributes: [.font: UIFont.boldSystemFont(ofSize: CGFloat(size))]).width)
        }
        let points = renderTimeline ? ScoreOverlay.visiblePoints(videoWidth: videoWidth, layout: layout, points: snapshot.points) : []
        let pointLayout = ScoreOverlay.pointLayout(videoWidth: videoWidth, videoHeight: videoHeight, score: layout, pointCount: points.count)
        // Use JSON escaping so arbitrary team labels cannot collide with delimiters.
        let fields = [String(videoWidth), String(videoHeight), snapshot.team1Name, snapshot.team2Name,
                      String(snapshot.score.team1Score), String(snapshot.score.team2Score), snapshot.score.servingTeamId?.rawValue ?? ""] +
            points.flatMap { [$0.serveMarkerId, $0.winnerTeamId.rawValue, String($0.teamPointNumber)] }
        let key = String(decoding: try JSONEncoder().encode(fields), as: UTF8.self) as NSString
        if let cached = cache.object(forKey: key) { return cached }
        let width = min(videoWidth, max(layout.width, layout.width + Int(ceil(pointLayout.columnSpacing * Float(points.count)))))
        let height = min(videoHeight, layout.height)
        var pixels = Data(count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        try pixels.withUnsafeMutableBytes { bytes in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo) else {
                throw ProjectError.invalid("Cannot allocate score overlay pixels")
            }
            // UIKit text/path drawing uses top-left coordinates. The resulting
            // bytes have that same row order; the compositor never flips graphics.
            context.translateBy(x: 0, y: CGFloat(height)); context.scaleBy(x: 1, y: -1)
            UIGraphicsPushContext(context); defer { UIGraphicsPopContext() }
            drawScore(context, snapshot: snapshot, layout: layout)
            drawPoints(context, points: points, layout: pointLayout)
        }
        guard let provider = CGDataProvider(data: pixels as CFData), let image = CGImage(width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw ProjectError.invalid("Cannot create score overlay image")
        }
        let result = ScoreOverlayRaster(pixels: pixels, image: image, width: width, height: height, scoreWidth: layout.width)
        cache.setObject(result, forKey: key, cost: pixels.count)
        return result
    }

    private static func teamColor(_ team: ScoreTeamId) -> UIColor {
        let rgb = team == .team1 ? ScoreOverlay.team1RGB : ScoreOverlay.team2RGB
        return UIColor(red: CGFloat((rgb >> 16) & 255) / 255, green: CGFloat((rgb >> 8) & 255) / 255,
                       blue: CGFloat(rgb & 255) / 255, alpha: 1)
    }
    private static func drawScore(_ context: CGContext, snapshot: ScoreOverlaySnapshot, layout: ScoreOverlayLayout) {
        let border = CGFloat(layout.borderWidth), inset = border / 2
        let right = CGFloat(layout.width) - inset, bottom = CGFloat(layout.height) - inset, radius = CGFloat(layout.radius)
        // Match Android Canvas' quadratic curve, with only the bottom-right corner rounded.
        let path = UIBezierPath()
        path.move(to: CGPoint(x: inset, y: inset)); path.addLine(to: CGPoint(x: right, y: inset))
        path.addLine(to: CGPoint(x: right, y: bottom - radius))
        path.addQuadCurve(to: CGPoint(x: right - radius, y: bottom), controlPoint: CGPoint(x: right, y: bottom))
        path.addLine(to: CGPoint(x: inset, y: bottom)); path.close()
        context.saveGState(); path.addClip()
        let widths = [layout.team1Width, layout.scoreWidth, layout.team2Width, layout.scoreWidth]
        let labels = [ScoreOverlay.formatTeamLabel(snapshot.team1Name, serving: snapshot.score.servingTeamId == .team1),
                      ScoreOverlay.formatScore(snapshot.score.team1Score),
                      ScoreOverlay.formatTeamLabel(snapshot.team2Name, serving: snapshot.score.servingTeamId == .team2),
                      ScoreOverlay.formatScore(snapshot.score.team2Score)]
        var left: CGFloat = 0
        let font = UIFont.boldSystemFont(ofSize: CGFloat(layout.fontSize))
        for index in 0..<4 {
            let rect = CGRect(x: left, y: 0, width: CGFloat(widths[index]), height: CGFloat(layout.height))
            context.setFillColor((index == 0 ? teamColor(.team1) : index == 2 ? teamColor(.team2) : .white).cgColor)
            context.fill(rect)
            text(labels[index], rect: rect.insetBy(dx: CGFloat(layout.horizontalPadding), dy: 0), font: font,
                 color: index % 2 == 0 ? .white : .black)
            left += CGFloat(widths[index])
        }
        context.restoreGState()
        context.setStrokeColor(UIColor.black.cgColor); context.setLineWidth(border)
        left = 0
        for width in widths.dropLast() {
            left += CGFloat(width); context.move(to: CGPoint(x: left, y: 0)); context.addLine(to: CGPoint(x: left, y: CGFloat(layout.height)))
        }
        context.strokePath(); UIColor.black.setStroke(); path.lineWidth = border; path.stroke()
    }
    private static func drawPoints(_ context: CGContext, points: [ScorePointTimelineEntry], layout: ScorePointTimelineLayout) {
        guard !points.isEmpty, layout.columnSpacing > 0, layout.circleRadius > 0 else { return }
        func y(_ team: ScoreTeamId) -> CGFloat { CGFloat(team == .team1 ? layout.team1CenterY : layout.team2CenterY) }
        for team in [ScoreTeamId.team1, .team2] {
            guard let last = points.lastIndex(where: { $0.winnerTeamId == team }) else { continue }
            context.setStrokeColor(teamColor(team).cgColor); context.setLineWidth(CGFloat(layout.lineWidth))
            context.move(to: CGPoint(x: CGFloat(layout.startX), y: y(team)))
            context.addLine(to: CGPoint(x: CGFloat(layout.startX + layout.columnSpacing * (Float(last) + 0.5)), y: y(team)))
            context.strokePath()
        }
        let radius = CGFloat(layout.circleRadius), font = UIFont.boldSystemFont(ofSize: CGFloat(layout.fontSize))
        for (index, point) in points.enumerated() {
            let x = CGFloat(layout.startX + layout.columnSpacing * (Float(index) + 0.5))
            let circle = CGRect(x: x - radius, y: y(point.winnerTeamId) - radius, width: radius * 2, height: radius * 2)
            context.setFillColor(teamColor(point.winnerTeamId).cgColor); context.fillEllipse(in: circle)
            context.setStrokeColor(UIColor.black.cgColor); context.setLineWidth(CGFloat(layout.lineWidth)); context.strokeEllipse(in: circle)
            text(String(point.teamPointNumber), rect: circle, font: font, color: .white)
        }
    }
    private static func text(_ value: String, rect: CGRect, font: UIFont, color: UIColor) {
        let paragraph = NSMutableParagraphStyle(); paragraph.alignment = .center; paragraph.lineBreakMode = .byTruncatingTail
        let line = CGRect(x: rect.minX, y: rect.midY - font.lineHeight / 2, width: max(1, rect.width), height: font.lineHeight)
        (value as NSString).draw(in: line, withAttributes: [.font: font, .foregroundColor: color, .paragraphStyle: paragraph])
    }

    #if DEBUG
    /// Android's Canvas pixel cases; runs without decoding video.
    static func checkPixels() throws {
        let empty = PreparedScoreOverlay(tracking: .init(team1Name: "Falcons", team2Name: "Wolves"), rallyRanges: [])
        let score = try raster(snapshot: empty.snapshot(at: 0), videoWidth: 640, videoHeight: 480, renderTimeline: true)
        func rgb(_ raster: ScoreOverlayRaster, _ x: Int, _ y: Int) -> UInt32 {
            guard x < raster.width, y < raster.height else { return 0 }
            let i = y * raster.rowBytes + x * 4
            return UInt32(raster.pixels[i + 2]) << 16 | UInt32(raster.pixels[i + 1]) << 8 | UInt32(raster.pixels[i])
        }
        guard rgb(score, 10, 5) == ScoreOverlay.team1RGB, rgb(score, 90, 5) == 0xffffff,
              rgb(score, 138, 5) == ScoreOverlay.team2RGB else { throw ProjectError.invalid("Score overlay cell pixel check failed") }
        let prepared = PreparedScoreOverlay(tracking: .init(serveMarkers: [.init(id: "S1", timestampMs: 1000, side: .near),
            .init(id: "S2", timestampMs: 5000, side: .near)]),
            rallyRanges: [.init(coreStartMs: 5000, coreEndMs: 7000, keepStartMs: 4000, keepEndMs: 8000)])
        let point = try raster(snapshot: prepared.snapshot(at: 4250), videoWidth: 640, videoHeight: 480, renderTimeline: true)
        let x = point.scoreWidth + Int(36 * 0.58 * 0.5 + 36 * 0.18 * 0.65), y = Int(36 * 0.28)
        guard rgb(point, x, y) == ScoreOverlay.team1RGB else { throw ProjectError.invalid("Point timeline pixel check failed") }
        let off = try raster(snapshot: prepared.snapshot(at: 4250), videoWidth: 640, videoHeight: 480, renderTimeline: false)
        guard off.width == off.scoreWidth else { throw ProjectError.invalid("Disabled point timeline still has pixels") }
    }
    #endif
}

/// The caller supplies the aspect-fit video rectangle, excluding letterboxing.
/// Layout uses physical display pixels like Android's Compose preview.
@MainActor struct ScoreOverlayPreview: View {
    let prepared: PreparedScoreOverlay
    let sourceTimestampMs: Int64
    let renderTimeline: Bool
    @Environment(\.displayScale) private var displayScale
    @Environment(\.interfaceScale) private var interfaceScale
    var body: some View {
        let previewScale = displayScale * interfaceScale
        GeometryReader { geometry in
            let snapshot = prepared.snapshot(at: sourceTimestampMs)
            if let raster = try? ScoreOverlayRenderer.raster(snapshot: snapshot,
                videoWidth: max(1, Int((geometry.size.width * previewScale).rounded())),
                videoHeight: max(1, Int((geometry.size.height * previewScale).rounded())), renderTimeline: renderTimeline) {
                let scoreWidth = min(raster.scoreWidth, raster.width)
                HStack(spacing: 0) {
                    if let score = raster.image.cropping(to: CGRect(x: 0, y: 0, width: scoreWidth, height: raster.height)) {
                        Image(decorative: score, scale: previewScale).resizable().frame(width: CGFloat(scoreWidth) / previewScale, height: CGFloat(raster.height) / previewScale)
                    }
                    if raster.width > scoreWidth, let points = raster.image.cropping(to: CGRect(x: scoreWidth, y: 0, width: raster.width - scoreWidth, height: raster.height)) {
                        Image(decorative: points, scale: previewScale).resizable().frame(width: CGFloat(raster.width - scoreWidth) / previewScale, height: CGFloat(raster.height) / previewScale)
                            .opacity(Double(snapshot.opacity))
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(snapshot.team1Name) \(ScoreOverlay.formatScore(snapshot.score.team1Score)), \(snapshot.team2Name) \(ScoreOverlay.formatScore(snapshot.score.team2Score))")
            }
        }.allowsHitTesting(false).accessibilityIdentifier("scoreOverlayPreview")
    }
}
