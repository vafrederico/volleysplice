#!/usr/bin/env swift
// Run on the Mac: swift ios/scripts/render-app-icon.swift
// Mechanical reuse of Android's transparent brand mark, not regenerated artwork.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let sourceURL = root.appendingPathComponent("App/Assets.xcassets/BrandMark.imageset/brand.png")
let outputURL = root.appendingPathComponent("App/Assets.xcassets/AppIcon.appiconset/AppIcon.png")
guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
      let mark = CGImageSourceCreateImageAtIndex(source, 0, nil),
      let context = CGContext(data: nil, width: 1024, height: 1024, bitsPerComponent: 8,
                              bytesPerRow: 4096, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
    fatalError("Cannot load the canonical transparent Android brand mark")
}
// colors.xml launcher_background #F7F4EE; launcher_foreground.xml inset 18/108.
context.setFillColor(CGColor(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                             components: [CGFloat(247) / 255, CGFloat(244) / 255, CGFloat(238) / 255, 1])!)
context.fill(CGRect(x: 0, y: 0, width: 1024, height: 1024))
context.interpolationQuality = .high
let inset = 1024.0 * 18.0 / 108.0
context.draw(mark, in: CGRect(x: inset, y: inset, width: 1024 - 2 * inset, height: 1024 - 2 * inset))
guard let image = context.makeImage(),
      let destination = CGImageDestinationCreateWithURL(outputURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    fatalError("Cannot create the opaque launcher icon")
}
CGImageDestinationAddImage(destination, image, nil)
guard CGImageDestinationFinalize(destination) else { fatalError("Cannot write \(outputURL.path)") }
print(outputURL.path)
