// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Point at an external local clone; no app source is submitted to a service.
guard let renderer = ProcessInfo.processInfo.environment["VOLLEYCUT_TOKAMAK_ROOT"] else {
    fatalError("Set VOLLEYCUT_TOKAMAK_ROOT to a local Tokamak checkout")
}
let package = Package(name: "CourtsidePreview", dependencies: [.package(path: renderer)], targets: [
    .executableTarget(name: "CourtsidePreview", dependencies: [.product(name: "TokamakStaticHTML", package: "Tokamak")])
])
